"""
L1 检索层离线评估子系统 — golden qrels + 标准 IR 指标 + 分通道 + 基线对比。

对外公共接口：
- 指标纯函数（metrics）：recall/precision/mrr/ndcg/map，零外部依赖、离线确定
- 评估运行器（runner）：evaluate() 装配本地检索器分通道评估，产出 EvalReport
- 基线对比（runner）：save/load_baseline、compare，锁定 fusion 核心指标回归

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
    GUARDED_METRICS,
    ChannelResult,
    Comparison,
    EvalReport,
    compare,
    evaluate,
    format_comparison,
    format_report,
    load_baseline,
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
    # Baseline（对比 + 回归锁）
    "save_baseline", "load_baseline", "compare", "Comparison", "format_comparison",
    # 常量
    "CHANNELS", "DEFAULT_TOP_K", "DEFAULT_TOL", "GUARDED_METRICS",
]
