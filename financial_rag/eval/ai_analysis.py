"""
质量评估 AI 预分析（辅助观测）— LLM 消化离线 eval 报告，产出归因结论与修复线索。

定位（辅，非门禁）：
- L1/L2 离线尺子（runner / output_runner）的数字输出与退出码语义完全不变；
  本模块只在 CLI 显式 --ai-analysis 时追加一段解读：总体判断 → 关键发现
  （带证据数字）→ 可能根因 → 建议下一步，让人不用自己啃原始表格。
- 与 ragas_judge 的区分：ragas_judge 是 LLM 参与"打分"（judge 主流程，
  结构性故障硬报错）；本模块是 LLM 参与"解读已有分数"（附加层，失败仅
  告警跳过，不改变评估退出码）。

设计原则：
- digest 构建是纯函数（离线确定、可单测），只从 report/comparison 提取
  紧凑摘要：macro、回归明细、低分 case（截断控长），不喂全量 per_case。
- LLM 走现有基建 get_llm() + LLMCaller.call_json()，零新依赖；
  use_cache=False 避免进程级共享缓存跨运行串扰。
- prompt 留在 eval 模块内部（沿用 ragas_judge 先例：eval 内部实现不进
  prompts.py，业务生成 prompt 不应引入 eval 依赖）。
- 结果落盘 ai_analysis_results.json（带时间戳），不进 baseline，
  不破坏 pytest 离线确定性；CI 不跑本模块（需 API key）。

CLI: python -m financial_rag.eval.runner --ai-analysis
     python -m financial_rag.eval.output_runner --ai-analysis
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "ai_analysis_results.json"

# 低分 case / query 入选数量上限（控 prompt 长度）
_LOW_CASE_LIMIT = 5
# 证据文本截断长度
_TEXT_PREVIEW_LIMIT = 80

_OVERALL_LABELS = {"ok": "正常", "watch": "需关注", "regressed": "存在回归"}

_ANALYSIS_SYSTEM = """你是 RAG 系统的质量评估分析师。用户给出一份离线质量评估报告的 JSON 摘要，请你预消化后给出归因结论与修复线索，供工程师快速定位问题。

纪律：
1. 所有结论只能基于报告中的数字与明细，严禁编造任何指标、case 或数值。
2. findings 的 evidence 字段必须引用报告中真实出现的数值或 case_id。
3. 区分事实与推测：已确认的回退/低分放 findings，对原因的猜测放 possible_causes。
4. 信息不足以归因时明确说明，不要强行下结论。
5. baseline_comparison 为 null 表示无基线对比，只能评估当前绝对水平，不要谈"回退"。
6. 用中文输出。

overall 判定参考：regressions 非空或关键用例破线 → regressed；无回归但存在明显低分 case、弱区分度或守护指标贴近阈值 → watch；其余 → ok。

只输出合法 JSON，schema：
{"overall": "ok|watch|regressed", "summary": "一句话总体判断", "findings": [{"severity": "high|medium|low", "finding": "发现描述", "evidence": "报告中的证据数值/case_id"}], "possible_causes": ["可能根因（推测）"], "next_steps": ["建议的下一步动作"]}"""


# ===================== digest 构建（纯函数，离线可测） =====================

def _round_map(d: Dict) -> Dict:
    """数值统一 round 4 位，保证 digest 确定且紧凑。"""
    return {k: (round(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in (d or {}).items()}


def _truncate(text: str, limit: int = _TEXT_PREVIEW_LIMIT) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _comparison_digest(cmp) -> Optional[Dict]:
    """Comparison → 紧凑对比段；cmp=None（无基线）返回 None。

    L1 deltas 键是 (channel, metric) 元组，L2 是 str，统一序列化为 "a.b"。
    """
    if cmp is None:
        return None
    deltas = {
        (f"{k[0]}.{k[1]}" if isinstance(k, tuple) else str(k)): round(v, 4)
        for k, v in cmp.deltas.items()
    }
    out = {"tol": cmp.tol, "deltas": deltas, "regressions": cmp.regressions}
    if hasattr(cmp, "gap_delta"):   # L2 Comparison 额外带区分度 delta
        out["gap_delta"] = round(cmp.gap_delta, 4)
    return out


def build_digest_l1(report, cmp=None) -> Dict:
    """L1 检索评估报告 → LLM 可消化的紧凑摘要。

    入选内容：分通道 macro、fusion per_tag、filter/gate macro、
    基线 delta/回归明细、fusion recall 最低的若干 query、门控判错 case。
    """
    channels = report.channels or {}
    fusion = channels.get("fusion")

    low_queries: List[Dict] = []
    if fusion and fusion.per_query:
        ranked = sorted(fusion.per_query.items(),
                        key=lambda kv: kv[1].get("recall", 0.0))
        low_queries = [
            {"query_id": qid, **_round_map({
                m: scores.get(m, 0.0) for m in ("recall", "mrr", "ndcg")})}
            for qid, scores in ranked[:_LOW_CASE_LIMIT]
        ]

    ge = report.gate_eval or {}
    gate_failed = [
        {"case_id": cid, "blocked": oc.get("blocked"), "stage": oc.get("stage")}
        for cid, oc in (ge.get("per_case") or {}).items()
        if not oc.get("correct")
    ]

    fe = report.filter_eval or {}
    digest = {
        "kind": "l1_retrieval",
        "overview": {
            "top_k": report.top_k, "query_count": report.query_count,
            "doc_count": report.doc_count, "chunk_count": report.chunk_count,
        },
        "channels_macro": {ch: _round_map(cr.macro) for ch, cr in channels.items()},
        "fusion_per_tag": {tag: _round_map(ms)
                           for tag, ms in ((fusion.per_tag if fusion else {}) or {}).items()},
        "filter": {"case_count": fe.get("case_count", 0),
                   "macro": _round_map(fe.get("macro", {}))} if fe.get("case_count") else None,
        "gate": {"case_count": ge.get("case_count", 0),
                 "threshold": ge.get("threshold"),
                 "macro": _round_map(ge.get("macro", {})),
                 "known_limitation_cases": ge.get("known_limitation_cases", []),
                 "failed_cases": gate_failed} if ge.get("case_count") else None,
        "low_recall_queries": low_queries,
        "baseline_comparison": _comparison_digest(cmp),
    }
    return digest


def build_digest_l2(report, cmp=None, case_meta: Optional[Dict] = None) -> Dict:
    """L2 输出评估报告 → LLM 可消化的紧凑摘要。

    入选内容：macro/per_tier/per_tag、区分度 gap、基线 delta/回归明细、
    output_quality 最低的若干 case（含缺陷证据：未溯源数字/无据句/缺失字段）。

    case_meta: 可选 {case_id: {"quality_tier":..., "tags":[...]}}，来自 golden
    expected（per_case 本身不含分层标签）；缺省时 low case 不带标签。
    """
    case_meta = case_meta or {}
    per_case = report.per_case or {}
    ranked = sorted(per_case.items(),
                    key=lambda kv: kv[1].get("output_quality", 0.0))
    low_cases = []
    for cid, scores in ranked[:_LOW_CASE_LIMIT]:
        unanchored = scores.get("unanchored", []) or []
        low_cases.append({
            "case_id": cid,
            **_round_map({m: scores.get(m, 0.0) for m in (
                "output_quality", "numerical_fidelity",
                "source_grounding", "field_completeness")}),
            "quality_tier": case_meta.get(cid, {}).get("quality_tier"),
            "tags": case_meta.get(cid, {}).get("tags", []),
            "missing_fields": scores.get("missing_fields", []),
            "unmatched_numbers": [_truncate(n) for n in
                                  (scores.get("unmatched_numbers", []) or [])[:5]],
            "unanchored_count": len(unanchored),
            "unanchored_preview": [_truncate(s) for s in unanchored[:2]],
        })

    return {
        "kind": "l2_output",
        "overview": {"case_count": report.case_count,
                     "good_count": report.good_count, "bad_count": report.bad_count},
        "macro": _round_map(report.macro),
        "per_tier": {tier: _round_map(ms) for tier, ms in (report.per_tier or {}).items()},
        "per_tag": {tag: _round_map(ms) for tag, ms in (report.per_tag or {}).items()},
        "separation_gap": round(report.separation_gap, 4),
        "low_quality_cases": low_cases,
        "baseline_comparison": _comparison_digest(cmp),
    }


# ===================== LLM 分析 =====================

def _has_api_key() -> bool:
    """API key 可用性检查（config 导入时已 load_dotenv，此处兜底再查一次）。"""
    return bool(os.getenv("DASHSCOPE_API_KEY", ""))


def analyze(digest: Dict, caller=None) -> Optional[Dict]:
    """把 digest 喂给 LLM 产出结构化预分析；任何失败返回 None（辅助层不阻断）。

    caller 可注入（测试 stub）；缺省时用 get_llm() + LLMCaller 现构。
    失败语义：缺 key / 调用异常 / JSON 解析失败 → 告警 + None，
    由调用方决定提示文案，评估退出码不受影响。
    """
    if not _has_api_key():
        logger.warning("[AI预分析] 缺少 DASHSCOPE_API_KEY，跳过预分析")
        return None

    if caller is None:
        try:
            from financial_rag.llm import get_llm
            from financial_rag.llm.caller import LLMCaller
            caller = LLMCaller(get_llm())
        except Exception as e:
            logger.warning(f"[AI预分析] LLM 构造失败，跳过预分析: {e}")
            return None

    kind = digest.get("kind", "unknown")
    user = (f"评估类型: {kind}\n"
            f"报告 JSON 摘要:\n{json.dumps(digest, ensure_ascii=False, indent=1)}")
    try:
        result = caller.call_json(
            user, system=_ANALYSIS_SYSTEM,
            max_tokens=1500, temperature=0.0, use_cache=False,
        )
    except Exception as e:
        logger.warning(f"[AI预分析] LLM 调用失败，跳过预分析: {e}")
        return None

    if not isinstance(result, dict) or "summary" not in result:
        logger.warning(f"[AI预分析] LLM 输出不符合 schema（解析结果: {result!r}），跳过预分析")
        return None
    return result


def format_analysis(result: Dict) -> str:
    """结构化分析 → 中文 CLI 报告段。"""
    overall = result.get("overall", "watch")
    lines = [
        "",
        f"══ AI 预分析（辅助观测，不改变评估结论）  |  总体: "
        f"{_OVERALL_LABELS.get(overall, overall)} ══",
        result.get("summary", ""),
    ]
    findings = result.get("findings") or []
    if findings:
        lines.append("")
        lines.append("关键发现：")
        for f in findings:
            lines.append(f"  [{f.get('severity', '?')}] {f.get('finding', '')}")
            if f.get("evidence"):
                lines.append(f"      证据: {f['evidence']}")
    causes = result.get("possible_causes") or []
    if causes:
        lines.append("")
        lines.append("可能根因（推测）：")
        lines.extend(f"  - {c}" for c in causes)
    steps = result.get("next_steps") or []
    if steps:
        lines.append("")
        lines.append("建议下一步：")
        lines.extend(f"  - {s}" for s in steps)
    return "\n".join(lines)


def save_analysis(result: Dict, kind: str, path: Path = None) -> None:
    """落盘 ai_analysis_results.json（带时间戳；辅助观测，不参与离线基线）。

    path 延迟绑定模块全局 RESULTS_PATH（而非默认参数），便于测试重定向落盘路径。
    """
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "role": "auxiliary",
        **result,
    }
    with open(path or RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_print(text: str) -> None:
    """print 兼容 Windows GBK 控制台：LLM 输出可能含 U+2212 等无法编码字符，
    流未 reconfigure 成 utf-8 时降级 replace，不让辅助分析炸掉评估 CLI。"""
    try:
        print(text)
    except UnicodeEncodeError:
        import sys
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def analyze_and_report(digest: Dict) -> None:
    """CLI 便捷入口：analyze → 打印 → 落盘；失败打印跳过提示（不影响退出码）。"""
    result = analyze(digest)
    if result is None:
        _safe_print("\n[AI预分析] 已跳过：缺少 DASHSCOPE_API_KEY 或 LLM 调用/解析失败"
                    "（详见日志）。不影响评估结论与退出码。")
        return
    _safe_print(format_analysis(result))
    save_analysis(result, digest.get("kind", "unknown"))
    _safe_print(f"\nAI 预分析已写入 {RESULTS_PATH.name}")