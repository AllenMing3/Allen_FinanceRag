"""
Financial RAG — 财报/经济新闻智能分析系统

三大核心架构：
  1. Coordinate  — 多 Agent 协调调度
  2. Indexer     — 多文本索引流水线
  3. Reflection  — ReAct 反思 + 六层防幻觉

五个专业化 Agent：
  - IngestionAgent   财报/新闻数据摄取
  - ExtractionAgent  关键财务指标抽取
  - AnalysisAgent    多维度财务分析
  - ForecastAgent    趋势预测与情景分析
  - ReportAgent      多格式报告生成

与业务完全脱钩 — 所有核心架构通过抽象接口定义，可替换任意领域
"""
from financial_rag.config import config, AppConfig
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
from financial_rag.middleware import FinancialHallucinationGuard
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

__version__ = "1.3.0"
__all__ = [
    # 配置
    "config", "AppConfig",
    # Core - Coordinate
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
    # Retrievers / Middleware
    "HybridRetriever", "jieba_tokenizer", "FinancialHallucinationGuard",
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
]
