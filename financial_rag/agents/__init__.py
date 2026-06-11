"""
金融 RAG Agents — 基于 Coordinate 架构

四个专业化 Agent:
- IngestionAgent: 财报/新闻数据摄取 + 元数据提取
- ExtractionAgent: 关键财务指标抽取 + 实体识别
- ReportAgent: LLM 驱动的新闻综合分析 + 引用报告
- KLineAgent: K 线技术分析（按需查询，不进 KB）
"""
from .ingestion_agent import IngestionAgent
from .extraction_agent import ExtractionAgent
from .report_agent import ReportAgent
from .kline_agent import KLineAgent

__all__ = [
    "IngestionAgent",
    "ExtractionAgent",
    "ReportAgent",
    "KLineAgent",
]
