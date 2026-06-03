"""
Financial RAG — 财报/经济新闻智能分析系统

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

五个专业化 Agent:
  - IngestionAgent   财报/新闻数据摄取
  - ExtractionAgent  关键财务指标抽取
  - AnalysisAgent    多维度财务分析
  - ForecastAgent    趋势预测与情景分析
  - ReportAgent      多格式报告生成

MCP 集成:
  - mcp_client/  连接第三方 MCP 服务器 (如 china-stock-mcp) 获取新闻/行情数据
  - 未启用 MCP 时自动回退到 akshare 直连

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
    AnalysisAgent, ForecastAgent, ReportAgent,
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
from financial_rag.news_fetcher import (
    fetch_stock_news, fetch_financial_news, fetch_announcements,
    get_sample_news_for_rag, NewsItem, NewsResult, HAS_AKSHARE,
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
    "AnalysisAgent", "ForecastAgent", "ReportAgent",
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
    # News Fetcher (akshare-based fallback)
    "fetch_stock_news", "fetch_financial_news", "fetch_announcements",
    "get_sample_news_for_rag", "NewsItem", "NewsResult", "HAS_AKSHARE",
    # MCP Client
    "MCPClient", "NewsMCPClient",
]
