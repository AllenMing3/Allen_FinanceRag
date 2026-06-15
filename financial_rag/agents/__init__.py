"""
金融 RAG Agents — 基于 Coordinate 架构

七个专业化 Agent:
- CoordinatorAgent: 智能调度 (意图分类 + Agent 链选择)
- IngestionAgent: 财报/新闻数据摄取 + 元数据提取
- ExtractionAgent: 关键财务指标抽取 + 实体识别
- ReportAgent: LLM 驱动的新闻综合分析 + 引用报告
- KLineAgent: K 线技术分析 (委托 analyze_kline 工具)
- EventImpactAgent: 事件影响分析 (日期事件 → 利好/利空 + 影响因子)
- ScoringAgent: 全链路评分 (各阶段打分 + 防幻觉校验)
"""
from .coordinator_agent import CoordinatorAgent
from .ingestion_agent import IngestionAgent
from .extraction_agent import ExtractionAgent
from .report_agent import ReportAgent
from .kline_agent import KLineAgent
from .event_impact_agent import EventImpactAgent
from .scoring_agent import ScoringAgent

__all__ = [
    "CoordinatorAgent",
    "IngestionAgent",
    "ExtractionAgent",
    "ReportAgent",
    "KLineAgent",
    "EventImpactAgent",
    "ScoringAgent",
]
