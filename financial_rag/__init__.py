"""
Financial RAG — AI/科技行业智能分析 RAG 系统

架构分层:
  1. core/base.py        — 基础抽象 (BaseAgent, AgentContext, AgentResult)
  2. core/orchestrator.py — Agent 调度编排 (SEQUENTIAL / PARALLEL / CONDITIONAL)
  3. core/pipeline.py    — 5 阶段流水线 (Fetch → Index → Process → Output → Evolve)
  4. core/router.py      — CLI 命令路由 (CommandRouter)
  5. core/factory.py     — 工厂函数 (create_orchestrator, setup_environment)
  6. core/indexer.py     — 混合检索 + RRF 融合
  7. core/reflector.py   — ReAct 反思 + 防幻觉
  8. core/scorer.py      — 全链路打分卡
  9. core/protocol.py    — Agent 间消息总线 (MessageBus)

三个专业化 Agent:
  - IngestionAgent   AI 行业文档摄取 + 元数据提取 (via tool calling)
  - ExtractionAgent  AI 行业指标抽取 + 实体识别 (via tool calling)
  - ReportAgent      LLM 驱动的新闻综合分析 + 引用报告

Function Calling 工具系统:
  - tools/extraction_tools.py — 5 个抽取工具 (LLM-first + regex fallback)
  - tools/news_tools.py — 新闻搜索工具
  - tools/kline_tools.py — K线分析工具

与业务完全脱钩 — 所有核心架构通过抽象接口定义，可替换任意领域
"""
from financial_rag.config import config, AppConfig, MCPConfig
from financial_rag.core import (
    AgentOrchestrator, ExecutionMode,
    PipelineOrchestrator, PipelineConfig, PipelineStatus,
    ReflectionLoop, ReflectionConfig, HallucinationGuard,
    PipelineScoreCard, StageScore, ScoreGrade, create_scorecard,
)
from financial_rag.agents import (
    IngestionAgent, ExtractionAgent,
    ReportAgent, KLineAgent,
)
from financial_rag.retrievers import HybridRetriever, jieba_tokenizer
from financial_rag.templates import (
    SlottedTemplate, SlotDef,
    FINANCIAL_REPORT_TEMPLATE, NEWS_BRIEF_TEMPLATE,
    QUICK_QA_TEMPLATE, DEEP_ANALYSIS_TEMPLATE,
    get_template, ALL_TEMPLATES,
)
from financial_rag.slot_filler import SlotFiller, SlotResult, FillStats, create_slot_filler
from financial_rag.tools import (
    FunctionRegistry, FunctionDef, ToolExecutor, ToolCallSession,
    ToolCallStats, ToolCallResult, ToolCallRequest,
    CATEGORIES, create_financial_registry, create_tool_session,
)
from financial_rag.rss_fetcher import (
    search_news, fetch_all_news,
    fetch_ths_news, fetch_sina_finance, fetch_eastmoney_search,
)
from financial_rag.tushare_client import (
    fetch_stock_kline, fetch_etf_kline, compute_kline_stats,
    compute_technical_indicators, search_stock, search_etf,
    fetch_financial_indicators,
)
from financial_rag.mcp_client import MCPClient, NewsMCPClient

__version__ = "2.0.0"
__all__ = [
    # 配置
    "config", "AppConfig", "MCPConfig",
    # Core - Orchestration
    "AgentOrchestrator", "ExecutionMode",
    # Core - Indexer
    "PipelineOrchestrator", "PipelineConfig", "PipelineStatus",
    # Core - Reflection
    "ReflectionLoop", "ReflectionConfig", "HallucinationGuard",
    # Core - Scorer
    "PipelineScoreCard", "StageScore", "ScoreGrade", "create_scorecard",
    # Agents
    "IngestionAgent", "ExtractionAgent",
    "ReportAgent", "KLineAgent",
    # Retrievers
    "HybridRetriever", "jieba_tokenizer",
    # Templates & Slot-Filling
    "SlottedTemplate", "SlotDef",
    "FINANCIAL_REPORT_TEMPLATE", "NEWS_BRIEF_TEMPLATE",
    "QUICK_QA_TEMPLATE", "DEEP_ANALYSIS_TEMPLATE",
    "get_template", "ALL_TEMPLATES",
    "SlotFiller", "SlotResult", "FillStats", "create_slot_filler",
    # Function Registry & Tool Calling
    "FunctionRegistry", "FunctionDef", "ToolExecutor", "ToolCallSession",
    "ToolCallStats", "ToolCallResult", "ToolCallRequest",
    "CATEGORIES", "create_financial_registry", "create_tool_session",
    # News Fetcher (domestic APIs)
    "search_news", "fetch_all_news",
    "fetch_ths_news", "fetch_sina_finance", "fetch_eastmoney_search",
    # Tushare Client
    "fetch_stock_kline", "fetch_etf_kline", "compute_kline_stats",
    "compute_technical_indicators", "search_stock", "search_etf",
    "fetch_financial_indicators",
    # MCP Client
    "MCPClient", "NewsMCPClient",
]
