"""
金融 RAG Agents — 精简 4-Agent 架构

- CoordinatorAgent: 智能调度 (意图分类 + Agent 链选择)
- IngestionAgent: 财报/新闻数据摄取 + 元数据提取
- AnalysisAgent: 统一分析 (指标抽取 + K线分析 + 事件影响 + 报告生成)
- ScoringAgent: 全链路评分 (各阶段打分 + 防幻觉校验)
"""
from .coordinator_agent import CoordinatorAgent
from .ingestion_agent import IngestionAgent
from .analysis_agent import AnalysisAgent
from .scoring_agent import ScoringAgent

__all__ = [
    "CoordinatorAgent",
    "IngestionAgent",
    "AnalysisAgent",
    "ScoringAgent",
]
