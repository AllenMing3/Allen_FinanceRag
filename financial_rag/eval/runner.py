"""
L1 检索层离线评估运行器 — 分通道 × 标准 IR 指标 × macro 平均。

设计原则（对应 plan C 层）：
- 显式本地模式装配 HybridRetriever(embedder=None, reranker=None)：不读 env、
  不依赖 conftest、不用 factory（factory 会读 DASHSCOPE_API_KEY 且把 chroma/bm25
  指向生产 kb_dir），保证 CLI 与 pytest 两种入口都离线、确定、免 API。
- 临时目录隔离 chroma/bm25 落盘，绝不污染生产知识库。
- 分通道评估（吸取 Step3 教训：避免 BM25 改善被 Jaccard 融合淹没）：
    bm25   — 纯关键词通道（_bm25.search，jieba 分词，不含 QueryParser）
    vector — 纯 Jaccard 通道（_vector.search_jaccard）
    fusion — 完整链路（retriever.search，含 QueryParser 残余分词 + RRF 融合）
- 每通道去重为逻辑 doc_id 序列 → 每 query 算全套指标 → macro-avg over queries。
- Step 5 覆盖面（filter_cases/gate_cases golden）：
    filter — search(filters=...) 全链路过滤：泄漏率（返回项违反过滤条件）+
             trap 命中（主题强相关但应被过滤排除的文档）+ IR 指标
    gate   — 零结果拦截（真实检索路径）+ 分数拦截（decide_quality_gate 纯函数
             喂合成 rerank 分数，覆盖离线不可达的 quality_gate 分支）+ 诊断完整性

CLI: python -m financial_rag.eval.runner
"""
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from financial_rag.retrievers.hybrid_engine import (
    HybridRetriever, decide_quality_gate, jieba_tokenizer,
)
from financial_rag.retrievers.chunker import TextChunker
from financial_rag.retrievers.filters import apply_filters
from financial_rag.retrievers.query_parser import QueryParser
from financial_rag.eval.metrics import (
    METRIC_NAMES, evaluate_query, dedupe_ranked, relevant_ids,
)

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_EVAL_KB = _GOLDEN_DIR / "eval_kb.jsonl"
_EVAL_KB_LONG = _GOLDEN_DIR / "eval_kb_long.jsonl"
_QRELS = _GOLDEN_DIR / "qrels.jsonl"
_FILTER_CASES = _GOLDEN_DIR / "filter_cases.jsonl"
_GATE_CASES = _GOLDEN_DIR / "gate_cases.jsonl"

CHANNELS = ["bm25", "vector", "fusion"]
DEFAULT_TOP_K = 5

# 过滤评估指标（leakage_rate/trap_hits 越小越好，其余同 IR 指标越大越好）
FILTER_METRIC_NAMES = METRIC_NAMES + ["leakage_rate", "trap_hits"]
# 门控评估指标（false_block_rate/false_pass_rate 越小越好）
GATE_METRIC_NAMES = ["accuracy", "false_block_rate", "false_pass_rate",
                     "diagnostics_complete"]
# 合成日期：golden 语料本身无 publish_date，仅为日期类过滤 case 确定性合成
# （不改 eval_kb.jsonl，避免动已锁基线的通道指标）
SYNTHETIC_PUBLISH_DATES = {
    "d_nvidia_blackwell": "2024-03-18",
    "d_amd_mi300": "2023-12-06",
    "d_tsmc_cowos": "2024-06-15",
    "d_hbm_memory": "2024-09-10",
    "d_moutai_2024": "2025-03-28",
}


# ===================== golden 数据加载 =====================

def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_golden_docs() -> List[Dict]:
    """加载短文基准库 + 长文场景库，转为入库入参形状 {text, meta}。"""
    docs = []
    for path in (_EVAL_KB, _EVAL_KB_LONG):
        for d in _load_jsonl(path):
            meta = {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "doc_type": d["doc_type"],
                "source": d["source"],
            }
            if d["doc_id"] in SYNTHETIC_PUBLISH_DATES:
                meta["publish_date"] = SYNTHETIC_PUBLISH_DATES[d["doc_id"]]
            docs.append({"text": d["text"], "meta": meta})
    return docs


def load_qrels() -> List[Dict]:
    """加载 qrels（分级正例 grade>=1 + 显式负例 grade=0 + info_needed 锚点）。"""
    return _load_jsonl(_QRELS)


def load_filter_cases() -> List[Dict]:
    """加载过滤评估 case（filters + relevant_docs + trap_docs）。"""
    return _load_jsonl(_FILTER_CASES) if _FILTER_CASES.exists() else []


def load_gate_cases() -> List[Dict]:
    """加载门控评估 case（pass_in_domain / block_no_result / score_case）。"""
    return _load_jsonl(_GATE_CASES) if _GATE_CASES.exists() else []


# ===================== 本地检索器装配 =====================

def build_local_retriever(workdir: str) -> HybridRetriever:
    """显式本地模式装配（embedder/reranker=None），临时目录隔离落盘。

    刻意不用 factory.create_hybrid_retriever()：它读 DASHSCOPE_API_KEY 决定模式，
    且把 chroma/bm25 指向生产 kb_dir。这里显式本地 + 临时目录，保证离线确定、免 API、
    不碰生产库。chunker 参数与 factory 完全对齐（skip_threshold=500）。
    """
    chunker = TextChunker(chunk_size=1500, chunk_overlap=100,
                          min_chunk_size=80, skip_threshold=500)
    return HybridRetriever(
        embedder=None,
        reranker=None,
        tokenizer=jieba_tokenizer(),
        chunker=chunker,
        parser=QueryParser(),
        chroma_persist_dir=str(Path(workdir) / "chroma"),
        bm25_db_path=str(Path(workdir) / "bm25.db"),
    )


def _channel_ranked(retriever: HybridRetriever, channel: str,
                    query: str, top_k: int) -> List[str]:
    """按通道取排序结果，去重为逻辑 doc_id 序列（多 chunk 保留最高 rank）。"""
    if channel == "bm25":
        items = retriever._bm25.search(retriever.documents, query, top_k)
    elif channel == "vector":
        items = retriever._vector.search_jaccard(
            retriever.documents, query, top_k,
            tokenize_fn=retriever._bm25.tokenize,
        )
    elif channel == "fusion":
        items = retriever.search(query, top_k=top_k, use_rerank=False)
    else:
        raise ValueError(f"未知通道: {channel}")
    return dedupe_ranked(items)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# ===================== 过滤评估（Step 5 Filter） =====================

def evaluate_filters(retriever: HybridRetriever, cases: List[Dict],
                     top_k: int = DEFAULT_TOP_K) -> Dict:
    """对每个 filter case 走 search(filters=...) 全链路，量化过滤正确性。

    指标（每 case）：
    - IR 指标：按 filter-aware qrels 算（trap 文档不在 relevant_docs 里，
      被过滤排除是正确行为，不惩罚 recall）
    - leakage_rate：返回项中违反过滤条件的比例（用生产 apply_filters 复判，
      复用同一套匹配语义，不重写判断逻辑）
    - trap_hits：返回的逻辑 doc 中命中 trap_docs 的数量（硬违规，整数）
    """
    per_case: Dict[str, Dict[str, float]] = {}
    for c in cases:
        k = c.get("top_k", top_k)
        items = retriever.search(c["query"], top_k=k, use_rerank=False,
                                 filters=c["filters"])
        # 泄漏判定：生产 apply_filters 复判后数量缩水 → 有违规项
        kept = apply_filters(items, c["filters"])
        leakage = (len(items) - len(kept)) / len(items) if items else 0.0
        ranked = dedupe_ranked(items)
        traps = set(c.get("trap_docs", []))
        metrics = evaluate_query(ranked, c["relevant_docs"], k)
        metrics["leakage_rate"] = leakage
        metrics["trap_hits"] = float(sum(1 for d in ranked if d in traps))
        per_case[c["case_id"]] = metrics

    macro = {m: _mean(pm[m] for pm in per_case.values()) for m in FILTER_METRIC_NAMES} \
        if per_case else {m: 0.0 for m in FILTER_METRIC_NAMES}
    return {"case_count": len(cases), "macro": macro, "per_case": per_case}


# ===================== 门控评估（Step 5 Gate） =====================

# 拦截诊断必备字段（对齐“硬拦截+全量诊断反馈”规范）
_GATE_DIAG_COMMON = ("stage", "reason", "query", "threshold")
_GATE_DIAG_SCORE_EXTRA = ("top_score", "top_text_preview")


def _gate_diag_complete(info: Dict) -> bool:
    """拦截诊断完整性：公共字段 + quality_gate 阶段额外的分数字段。"""
    required = list(_GATE_DIAG_COMMON)
    if info.get("stage") == "quality_gate":
        required += list(_GATE_DIAG_SCORE_EXTRA)
    return all(k in info for k in required)


def _gate_case_outcome(case: Dict, retriever: HybridRetriever,
                       threshold: float) -> Dict:
    """单 case 门控判定 → {correct, blocked, stage, diagnostics_complete}。

    三条路径：
    - pass_in_domain / block_no_result：真实检索链路 search()，读 last_gate_info
    - score_case：喂合成 rerank 分数给 decide_quality_gate 纯函数，覆盖
      离线（reranker=None）永不触发的分数门控分支
    """
    kind = case["kind"]
    expected_block = case["expected"] == "block"

    if kind == "score_case":
        blocked, info = decide_quality_gate(
            case["items"], threshold, has_rerank=True,
            query=case.get("query", case["case_id"]),
        )
    else:
        retriever.last_gate_info = None
        items = retriever.search(case["query"], top_k=DEFAULT_TOP_K,
                                 use_rerank=False)
        info = retriever.last_gate_info
        blocked = not items

    correct = blocked == expected_block
    if blocked and case.get("expected_stage"):
        correct = correct and (info or {}).get("stage") == case["expected_stage"]
    diag_ok = _gate_diag_complete(info) if blocked and info else (not blocked)
    return {
        "correct": correct,
        "blocked": blocked,
        "stage": (info or {}).get("stage"),
        "diagnostics_complete": bool(diag_ok),
    }


def evaluate_gate(retriever: HybridRetriever, cases: List[Dict],
                  threshold: float) -> Dict:
    """门控决策评估：准确率 + 误拦/误放率 + 诊断完整性。

    带 known_limitation=True 的 case 不纳入 macro 指标与回归锁（当前链路
    已知拦不住，锁了必红），但保留在 per_case/known_limitation_cases 里
    可见，配合测试层棘轮断言：修复翻绿后必须摘标纳入强制锁。
    """
    outcomes = {c["case_id"]: _gate_case_outcome(c, retriever, threshold)
                for c in cases}
    enforced = [c for c in cases if not c.get("known_limitation")]
    kl_cases = [c for c in cases if c.get("known_limitation")]
    n = len(enforced)
    if n == 0:
        return {"case_count": 0, "threshold": threshold,
                "macro": {m: 0.0 for m in GATE_METRIC_NAMES},
                "per_case": outcomes,
                "known_limitation_cases": [c["case_id"] for c in kl_cases]}
    false_block = sum(1 for c in enforced
                      if c["expected"] == "pass" and outcomes[c["case_id"]]["blocked"])
    false_pass = sum(1 for c in enforced
                     if c["expected"] == "block" and not outcomes[c["case_id"]]["blocked"])
    pass_total = sum(1 for c in enforced if c["expected"] == "pass")
    block_total = n - pass_total
    macro = {
        "accuracy": _mean(outcomes[c["case_id"]]["correct"] for c in enforced),
        "false_block_rate": false_block / pass_total if pass_total else 0.0,
        "false_pass_rate": false_pass / block_total if block_total else 0.0,
        "diagnostics_complete": _mean(outcomes[c["case_id"]]["diagnostics_complete"]
                                      for c in enforced),
    }
    return {"case_count": n, "threshold": threshold,
            "macro": macro, "per_case": outcomes,
            "known_limitation_cases": [c["case_id"] for c in kl_cases]}


# ===================== 报告结构 =====================

@dataclass
class ChannelResult:
    macro: Dict[str, float]                    # {metric: macro-avg over all queries}
    per_query: Dict[str, Dict[str, float]]     # {query_id: {metric: score}}
    per_tag: Dict[str, Dict[str, float]]       # {tag: {metric: macro over subset}}


@dataclass
class EvalReport:
    top_k: int
    query_count: int
    doc_count: int
    chunk_count: int
    channels: Dict[str, ChannelResult]
    filter_eval: Dict = None    # evaluate_filters() 结果（无 case 时为空 dict）
    gate_eval: Dict = None      # evaluate_gate() 结果

    def macro(self, channel: str, metric: str) -> float:
        """便捷取某通道某指标的 macro-avg（供断言/对比）。"""
        return self.channels[channel].macro[metric]

    def to_dict(self) -> Dict:
        return {
            "top_k": self.top_k,
            "query_count": self.query_count,
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "channels": {
                ch: {"macro": cr.macro, "per_tag": cr.per_tag, "per_query": cr.per_query}
                for ch, cr in self.channels.items()
            },
            "filter": self.filter_eval or {},
            "gate": self.gate_eval or {},
        }


# ===================== 评估主流程 =====================

def evaluate(top_k: int = DEFAULT_TOP_K) -> EvalReport:
    """装配本地检索器 → 索引 golden → 分通道分 query 算指标 → macro/per-tag 汇总
    → 过滤/门控 case 评估（与通道评估同构：临时目录、离线、确定）。"""
    docs = load_golden_docs()
    qrels = load_qrels()
    filter_cases = load_filter_cases()
    gate_cases = load_gate_cases()

    with tempfile.TemporaryDirectory(prefix="fineval_") as workdir:
        retriever = build_local_retriever(workdir)
        retriever.index(docs, precompute_embeddings=False, use_chunker=True)
        chunk_count = len(retriever.documents)

        # 每通道 × 每 query 的指标（在临时目录生命周期内完成检索）
        per_channel_query: Dict[str, Dict[str, Dict[str, float]]] = {ch: {} for ch in CHANNELS}
        for q in qrels:
            qid = q["query_id"]
            k = q.get("top_k", top_k)
            grades = q["relevant_docs"]
            for ch in CHANNELS:
                ranked = _channel_ranked(retriever, ch, q["query"], k)
                per_channel_query[ch][qid] = evaluate_query(ranked, grades, k)

        # Step 5：过滤 + 门控评估（复用同一索引，避免重复装配）
        filter_eval = evaluate_filters(retriever, filter_cases, top_k)
        gate_eval = evaluate_gate(retriever, gate_cases, retriever.gate_threshold)

    # 汇总（脱离临时目录后纯计算）
    all_tags = sorted({t for q in qrels for t in q.get("tags", [])})
    tag_to_qids = {tag: [q["query_id"] for q in qrels if tag in q.get("tags", [])]
                   for tag in all_tags}

    channels: Dict[str, ChannelResult] = {}
    for ch in CHANNELS:
        qm = per_channel_query[ch]
        macro = {m: _mean(qm[qid][m] for qid in qm) for m in METRIC_NAMES}
        per_tag = {
            tag: {m: _mean(qm[qid][m] for qid in qids) for m in METRIC_NAMES}
            for tag, qids in tag_to_qids.items()
        }
        channels[ch] = ChannelResult(macro=macro, per_query=qm, per_tag=per_tag)

    return EvalReport(top_k=top_k, query_count=len(qrels),
                      doc_count=len(docs), chunk_count=chunk_count,
                      channels=channels,
                      filter_eval=filter_eval, gate_eval=gate_eval)


def format_report(report: EvalReport) -> str:
    """分通道 × 指标 macro 表格 + fusion 关键子集，供 CLI/调试打印。"""
    k = report.top_k
    lines = [
        f"L1 检索评估  |  {report.doc_count} 篇文档 -> {report.chunk_count} chunks  "
        f"|  {report.query_count} 条 query  |  top_k={k}",
        "",
        f"{'channel':<8}" + "".join(f"{m:>11}" for m in METRIC_NAMES),
        "-" * (8 + 11 * len(METRIC_NAMES)),
    ]
    for ch in CHANNELS:
        lines.append(f"{ch:<8}" + "".join(f"{report.channels[ch].macro[m]:>11.4f}"
                                          for m in METRIC_NAMES))
    lines += ["", f"fusion 通道关键子集 @{k} (recall / mrr / ndcg):"]
    for tag in ("bug_scene", "unlogged_term", "long_doc", "non_ai", "seed"):
        pt = report.channels["fusion"].per_tag.get(tag)
        if pt:
            lines.append(f"  {tag:<14} recall={pt['recall']:.4f}  "
                         f"mrr={pt['mrr']:.4f}  ndcg={pt['ndcg']:.4f}")

    fe = report.filter_eval or {}
    if fe.get("case_count"):
        fm = fe["macro"]
        lines += ["", f"过滤评估  |  {fe['case_count']} 个 case  "
                      f"(leakage/trap_hits 越小越好，其余越大越好):"]
        lines.append("  " + "  ".join(f"{m}={fm[m]:.4f}" for m in FILTER_METRIC_NAMES))

    ge = report.gate_eval or {}
    if ge.get("case_count"):
        gm = ge["macro"]
        lines += ["", f"门控评估  |  {ge['case_count']} 个强制 case  "
                      f"|  threshold={ge['threshold']}"]
        lines.append("  " + "  ".join(f"{m}={gm[m]:.4f}" for m in GATE_METRIC_NAMES))
        if ge.get("known_limitation_cases"):
            lines.append(f"  已知局限（不纳入指标，待修复后棘轮纳入）: "
                         f"{', '.join(ge['known_limitation_cases'])}")
    return "\n".join(lines)


# ===================== 基线对比 =====================

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"
# fusion 通道纳入回归锁的核心指标（plan D：回退超阈值即失败）
GUARDED_METRICS = ["recall", "mrr", "ndcg"]
DEFAULT_TOL = 0.02
# Step 5 回归锁：(section, metric, 方向) —— lower 指标上升超阈值 / higher 指标下降超阈值均告警
GUARDED_STEP5 = [
    ("filter", "leakage_rate", "lower"),
    ("filter", "trap_hits", "lower"),
    ("filter", "recall", "higher"),
    ("gate", "accuracy", "higher"),
    ("gate", "false_block_rate", "lower"),
    ("gate", "false_pass_rate", "lower"),
]


def save_baseline(report: EvalReport, path: Path = BASELINE_PATH) -> None:
    """写入 baseline.json（各通道 macro + per_tag + filter/gate 段；
    不含 per_query/时间戳，保证可比、确定）。"""
    data = {
        "top_k": report.top_k,
        "query_count": report.query_count,
        "doc_count": report.doc_count,
        "chunk_count": report.chunk_count,
        "channels": {
            ch: {"macro": cr.macro, "per_tag": cr.per_tag}
            for ch, cr in report.channels.items()
        },
        "filter": {
            "case_count": (report.filter_eval or {}).get("case_count", 0),
            "macro": (report.filter_eval or {}).get("macro", {}),
            "per_case": (report.filter_eval or {}).get("per_case", {}),
        },
        "gate": {
            "case_count": (report.gate_eval or {}).get("case_count", 0),
            "threshold": (report.gate_eval or {}).get("threshold"),
            "macro": (report.gate_eval or {}).get("macro", {}),
            "known_limitation_cases": (report.gate_eval or {}).get(
                "known_limitation_cases", []),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_baseline(path: Path = BASELINE_PATH) -> Optional[Dict]:
    """读取 baseline.json；不存在返回 None。"""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Comparison:
    deltas: Dict[tuple, float]      # {(channel, metric) 或 ("filter"/"gate", metric): current - baseline}
    regressions: List[Dict]         # 核心指标回退超阈值的明细
    tol: float

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(report: EvalReport, baseline: Dict, tol: float = DEFAULT_TOL) -> Comparison:
    """对比当前 report 与 baseline：通道指标 + Step 5 filter/gate 指标。

    baseline 缺 filter/gate 段（旧基线）时该段 delta=0 不误报，
    提示重跑 --update-baseline 升级基线即可纳入守护。
    """
    deltas: Dict[tuple, float] = {}
    regressions: List[Dict] = []
    for ch in CHANNELS:
        base_macro = baseline.get("channels", {}).get(ch, {}).get("macro", {})
        for m in METRIC_NAMES:
            cur = report.channels[ch].macro[m]
            base = base_macro.get(m, cur)   # baseline 缺该指标 -> delta=0，不误报
            delta = cur - base
            deltas[(ch, m)] = delta
            if ch == "fusion" and m in GUARDED_METRICS and delta < -tol:
                regressions.append({
                    "channel": ch, "metric": m,
                    "baseline": round(base, 4), "current": round(cur, 4),
                    "delta": round(delta, 4),
                })

    # Step 5：filter/gate 段（按方向守护：lower 指标不得上升，higher 指标不得下降）
    cur_sections = {"filter": report.filter_eval or {}, "gate": report.gate_eval or {}}
    for section, metric, direction in GUARDED_STEP5:
        base_macro = baseline.get(section, {}).get("macro", {})
        cur = cur_sections[section].get("macro", {}).get(metric)
        if cur is None or metric not in base_macro:
            continue
        base = base_macro[metric]
        delta = cur - base
        deltas[(section, metric)] = delta
        bad = delta > tol if direction == "lower" else delta < -tol
        if bad:
            regressions.append({
                "channel": section, "metric": metric,
                "baseline": round(base, 4), "current": round(cur, 4),
                "delta": round(delta, 4),
            })
    return Comparison(deltas=deltas, regressions=regressions, tol=tol)


def format_comparison(cmp: Comparison) -> str:
    """打印 delta 表（带符号）+ Step 5 delta + 回归结论。"""
    lines = ["", f"对比 baseline (tol={cmp.tol}，数值为 current - baseline)：",
             f"{'channel':<8}" + "".join(f"{m:>11}" for m in METRIC_NAMES),
             "-" * (8 + 11 * len(METRIC_NAMES))]
    for ch in CHANNELS:
        lines.append(f"{ch:<8}" + "".join(f"{cmp.deltas[(ch, m)]:>+11.4f}" for m in METRIC_NAMES))
    step5 = [(k, v) for k, v in cmp.deltas.items() if k[0] in ("filter", "gate")]
    if step5:
        lines.append("")
        lines.append("Step 5 (filter/gate) delta：")
        for (section, metric), delta in sorted(step5):
            lines.append(f"  {section}.{metric:<20} {delta:>+.4f}")
    lines.append("")
    if cmp.regressions:
        lines.append("回归告警（核心指标回退超阈值）：")
        for r in cmp.regressions:
            lines.append(f"  {r['channel']}.{r['metric']}: "
                         f"{r['baseline']} -> {r['current']} (delta {r['delta']})")
    else:
        lines.append("无回归：fusion 核心指标 (recall/mrr/ndcg) 与 Step 5 "
                     "filter/gate 守护指标均未回退超阈值。")
    return "\n".join(lines)


# ===================== CLI =====================

def main(argv=None) -> int:
    import argparse
    import logging
    logging.getLogger("financial_rag").setLevel(logging.WARNING)
    logging.getLogger("jieba").setLevel(logging.ERROR)

    ap = argparse.ArgumentParser(
        prog="python -m financial_rag.eval.runner",
        description="L1 检索层离线评估（分通道 IR 指标 + 基线对比）")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="fusion 核心指标回退容忍阈值（默认 0.02）")
    ap.add_argument("--update-baseline", action="store_true",
                    help="用本次结果覆盖生成 baseline.json")
    args = ap.parse_args(argv)

    report = evaluate(top_k=args.top_k)
    print(format_report(report))

    if args.update_baseline:
        save_baseline(report)
        print(f"\nbaseline 已写入 {BASELINE_PATH.name}")
        return 0

    baseline = load_baseline()
    if baseline is None:
        print("\n（尚无 baseline.json，用 --update-baseline 生成首个基线）")
        return 0

    cmp = compare(report, baseline, tol=args.tol)
    print(format_comparison(cmp))
    return 0 if cmp.ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
