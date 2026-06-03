"""
Tools 包 — 能力注册中心 + 业务工具模块

核心基础设施来自 core.py（原 tools.py），子模块提供可注册的业务能力。
LLM 通过 Function Calling 可直接调起这些能力。

子模块:
- core:          FunctionDef, FunctionRegistry, ToolExecutor, ToolCallSession 等基础设施
- news_tools:    新闻搜索、拉取、保存为 Markdown 报告 (fetch_news_report)
- kline_tools:   ETF K 线数据获取、统计、保存为分析报告 (fetch_etf_kline_report)
"""

# 核心基础设施 — 全部从 core.py 中转导出
from financial_rag.tools.core import (
    # 数据类
    FunctionDef,
    ToolCallRequest,
    ToolCallResult,
    ToolCallStats,
    # 注册中心 + 执行器
    FunctionRegistry,
    ToolExecutor,
    # 会话管理
    ToolCallSession,
    # 工厂函数
    create_financial_registry,
    create_tool_session,
    # 内置能力函数
    calculate_growth_rate,
    calculate_financial_ratio,
    compare_metrics,
    summarize_financials,
    # 常量
    CATEGORIES,
)

# 业务工具模块
from financial_rag.tools.news_tools import (
    fetch_news_report,
    NEWS_REPORT_TOOL,
)
from financial_rag.tools.kline_tools import (
    fetch_etf_kline_report,
    KLINE_REPORT_TOOL,
)

__all__ = [
    # core
    "FunctionDef",
    "ToolCallRequest",
    "ToolCallResult",
    "ToolCallStats",
    "FunctionRegistry",
    "ToolExecutor",
    "ToolCallSession",
    "create_financial_registry",
    "create_tool_session",
    "calculate_growth_rate",
    "calculate_financial_ratio",
    "compare_metrics",
    "summarize_financials",
    "CATEGORIES",
    # news_tools
    "fetch_news_report",
    "NEWS_REPORT_TOOL",
    # kline_tools
    "fetch_etf_kline_report",
    "KLINE_REPORT_TOOL",
]
