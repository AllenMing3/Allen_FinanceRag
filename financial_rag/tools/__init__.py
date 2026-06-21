"""
Tools 包 — 能力注册中心 + 业务工具模块

核心基础设施来自 core.py（原 tools.py），子模块提供可注册的业务能力。
LLM 通过 Function Calling 可直接调起这些能力。

子模块:
- core:              FunctionDef, FunctionRegistry, ToolExecutor, ToolCallSession 等基础设施
- extraction_tools:  信息抽取能力 (extract_financial_metrics, extract_entities, ...)
- news_tools:        新闻搜索、拉取、保存为 Markdown 报告 (fetch_news_report) [feedparser RSS]
- kline_tools:       股票/ETF K 线数据获取、统计、保存为分析报告 (fetch_etf_kline_report) [Tushare]
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

# 抽取工具模块
from financial_rag.tools.extraction_tools import (
    extract_financial_metrics,
    extract_entities,
    extract_document_metadata,
    detect_document_type,
    generate_search_queries,
    EXTRACTION_TOOLS,
    inject_extraction_llm,
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
from financial_rag.tools.event_impact_tools import (
    fetch_date_events,
    fetch_kline_context,
    assess_event_impact,
    inject_event_llm,
    EVENT_IMPACT_TOOLS,
)
from financial_rag.tools.scoring_tools import (
    evaluate_pipeline_quality,
    check_hallucination,
    generate_score_report,
    SCORING_TOOLS,
)
from financial_rag.tools.coordinator_tools import (
    classify_query_intent,
    select_agent_chain,
    COORDINATOR_TOOLS,
)
from financial_rag.tools.kline_tools import (
    analyze_kline,
    generate_kline_analysis,
    inject_kline_llm,
    KLINE_ANALYSIS_TOOLS,
    STOCK_MAP,
    KLINE_ANALYSIS_SYSTEM,
    KLINE_ANALYSIS_PROMPT,
)
from financial_rag.tools.report_tools import (
    synthesize_report,
    inject_report_llm,
    REPORT_TOOLS,
)
from financial_rag.tools.analysis_tools import (
    analyze_news_deep,
    analyze_topic_deep,
    inject_analysis_deps,
    ANALYSIS_TOOLS,
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
    # extraction_tools
    "extract_financial_metrics",
    "extract_entities",
    "extract_document_metadata",
    "detect_document_type",
    "generate_search_queries",
    "EXTRACTION_TOOLS",
    "inject_extraction_llm",
    # news_tools
    "fetch_news_report",
    "NEWS_REPORT_TOOL",
    # kline_tools
    "fetch_etf_kline_report",
    "KLINE_REPORT_TOOL",
    "analyze_kline",
    "generate_kline_analysis",
    "inject_kline_llm",
    "KLINE_ANALYSIS_TOOLS",
    "STOCK_MAP",
    "KLINE_ANALYSIS_SYSTEM",
    "KLINE_ANALYSIS_PROMPT",
    # report_tools
    "synthesize_report",
    "inject_report_llm",
    "REPORT_TOOLS",
    # scoring_tools
    "evaluate_pipeline_quality",
    "check_hallucination",
    "generate_score_report",
    "SCORING_TOOLS",
    # coordinator_tools
    "classify_query_intent",
    "select_agent_chain",
    "COORDINATOR_TOOLS",
    # analysis_tools
    "analyze_news_deep",
    "analyze_topic_deep",
    "inject_analysis_deps",
    "ANALYSIS_TOOLS",
]
