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

CLI: python -m financial_rag.eval.runner
"""
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from financial_rag.retrievers.hybrid_engine import HybridRetriever, jieba_tokenizer
from financial_rag.retrievers.chunker import TextChunker
from financial_rag.retrievers.query_parser import QueryParser
from financial_rag.eval.metrics import METRIC_NAMES, evaluate_query, dedupe_ranked

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_EVAL_KB = _GOLDEN_DIR / "eval_kb.jsonl"
_EVAL_KB_LONG = _GOLDEN_DIR / "eval_kb_long.jsonl"
_QRELS = _GOLDEN_DIR / "qrels.jsonl"

CHANNELS = ["bm25", "vector", "fusion"]
DEFAULT_TOP_K = 5


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
            docs.append({
                "text": d["text"],
                "meta": {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "doc_type": d["doc_type"],
                    "source": d["source"],
                },
            })
    return docs


def load_qrels() -> List[Dict]:
    """加载 qrels（分级正例 grade>=1 + 显式负例 grade=0 + info_needed 锚点）。"""
    return _load_jsonl(_QRELS)


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
        }


# ===================== 评估主流程 =====================

def evaluate(top_k: int = DEFAULT_TOP_K) -> EvalReport:
    """装配本地检索器 → 索引 golden → 分通道分 query 算指标 → macro/per-tag 汇总。"""
    docs = load_golden_docs()
    qrels = load_qrels()

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
                      channels=channels)


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
    return "\n".join(lines)


# ===================== 基线对比 =====================

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"
# fusion 通道纳入回归锁的核心指标（plan D：回退超阈值即失败）
GUARDED_METRICS = ["recall", "mrr", "ndcg"]
DEFAULT_TOL = 0.02


def save_baseline(report: EvalReport, path: Path = BASELINE_PATH) -> None:
    """写入 baseline.json（各通道 macro + per_tag；不含 per_query/时间戳，保证可比、确定）。"""
    data = {
        "top_k": report.top_k,
        "query_count": report.query_count,
        "doc_count": report.doc_count,
        "chunk_count": report.chunk_count,
        "channels": {
            ch: {"macro": cr.macro, "per_tag": cr.per_tag}
            for ch, cr in report.channels.items()
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
    deltas: Dict[tuple, float]      # {(channel, metric): current - baseline}
    regressions: List[Dict]         # fusion 核心指标回退超阈值的明细
    tol: float

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(report: EvalReport, baseline: Dict, tol: float = DEFAULT_TOL) -> Comparison:
    """对比当前 report 与 baseline，算每指标 delta，锁定 fusion 核心指标回退。"""
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
    return Comparison(deltas=deltas, regressions=regressions, tol=tol)


def format_comparison(cmp: Comparison) -> str:
    """打印 delta 表（带符号）+ 回归结论。"""
    lines = ["", f"对比 baseline (tol={cmp.tol}，数值为 current - baseline)：",
             f"{'channel':<8}" + "".join(f"{m:>11}" for m in METRIC_NAMES),
             "-" * (8 + 11 * len(METRIC_NAMES))]
    for ch in CHANNELS:
        lines.append(f"{ch:<8}" + "".join(f"{cmp.deltas[(ch, m)]:>+11.4f}" for m in METRIC_NAMES))
    lines.append("")
    if cmp.regressions:
        lines.append("回归告警（fusion 核心指标回退超阈值）：")
        for r in cmp.regressions:
            lines.append(f"  {r['metric']}: {r['baseline']} -> {r['current']} (delta {r['delta']})")
    else:
        lines.append("无回归：fusion 核心指标 (recall/mrr/ndcg) 未回退超阈值。")
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
