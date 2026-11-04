"""
L1 检索层离线评估子系统 — golden qrels + 标准 IR 指标 + 分通道 + 基线对比
+ Step 5 过滤/质量门控评估（filter_cases/gate_cases）。

对外公共接口：
- 指标纯函数（metrics）：recall/precision/mrr/ndcg/map，零外部依赖、离线确定
- 评估运行器（runner）：evaluate() 装配本地检索器分通道评估，产出 EvalReport
- 过滤/门控评估（runner）：evaluate_filters/evaluate_gate，泄漏率 + trap 命中 +
  门控准确率/误拦误放率/诊断完整性
- 基线对比（runner）：save/load_baseline、compare，锁定 fusion 核心指标与
  Step 5 守护指标回归

CLI: python -m financial_rag.eval.runner [--top-k N] [--tol X] [--update-baseline]
"""
from .metrics import (
    METRIC_NAMES,
    relevant_ids,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    map_at_k,
    evaluate_query,
    dedupe_ranked,
)
from .runner import (
    CHANNELS,
    DEFAULT_TOP_K,
    DEFAULT_TOL,
    FILTER_METRIC_NAMES,
    GATE_METRIC_NAMES,
    GUARDED_METRICS,
    GUARDED_STEP5,
    ChannelResult,
    Comparison,
    EvalReport,
    compare,
    evaluate,
    evaluate_filters,
    evaluate_gate,
    format_comparison,
    format_report,
    load_baseline,
    load_filter_cases,
    load_gate_cases,
    load_golden_docs,
    load_qrels,
    save_baseline,
)

__all__ = [
    # Metrics（纯函数）
    "METRIC_NAMES", "relevant_ids",
    "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "map_at_k",
    "evaluate_query", "dedupe_ranked",
    # Runner（评估 + 报告结构）
    "evaluate", "EvalReport", "ChannelResult",
    "load_golden_docs", "load_qrels",
    "format_report",
    # Step 5（过滤 + 门控评估）
    "evaluate_filters", "evaluate_gate",
    "load_filter_cases", "load_gate_cases",
    "FILTER_METRIC_NAMES", "GATE_METRIC_NAMES", "GUARDED_STEP5",
    # Baseline（对比 + 回归锁）
    "save_baseline", "load_baseline", "compare", "Comparison", "format_comparison",
    # 常量
    "CHANNELS", "DEFAULT_TOP_K", "DEFAULT_TOL", "GUARDED_METRICS",
]
