"""
金融 RAG Agents — 基于 Coordinate 架构

五个专业化 Agent:
- IngestionAgent: 财报/新闻数据摄取
- ExtractionAgent: 关键财务指标抽取
- AnalysisAgent: 多维度财务分析
- ForecastAgent: 趋势预测
- ReportAgent: 多格式报告生成
"""
from .ingestion_agent import IngestionAgent
from .extraction_agent import ExtractionAgent
from .analysis_agent import AnalysisAgent
from .forecast_agent import ForecastAgent
from .report_agent import ReportAgent

__all__ = [
    "IngestionAgent",
    "ExtractionAgent",
    "AnalysisAgent",
    "ForecastAgent",
    "ReportAgent",
]
