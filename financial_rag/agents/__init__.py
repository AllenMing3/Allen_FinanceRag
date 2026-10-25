"""
金融 RAG Agents — 精简 3-Agent 架构

- IngestionAgent: 财报/新闻数据摄取 + 元数据提取
- AnalysisAgent: 统一分析 (指标抽取 + K线分析 + 事件影响 + 报告生成)
- ScoringAgent: 全链路评分 (各阶段打分 + 防幻觉校验)

意图分类与 Agent 链路由由 core/agent_router.py (AgentRouter) 承担，不设专职调度 Agent。
"""
from .ingestion_agent import IngestionAgent
from .analysis_agent import AnalysisAgent
from .scoring_agent import ScoringAgent

__all__ = [
    "IngestionAgent",
    "AnalysisAgent",
    "ScoringAgent",
]
