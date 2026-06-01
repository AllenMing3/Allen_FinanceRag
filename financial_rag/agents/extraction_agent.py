"""
ExtractionAgent — 关键财务指标抽取

功能:
- 从财报中抽取: 营收、利润、毛利率、现金流等
- 从新闻中抽取: 事件、影响、关联公司
- 结构化输出供后续分析使用
- 【打分】关键词抽取质量 & 查询改写质量评分
"""
from typing import Dict, Any, List

from financial_rag.core.coordinator import BaseAgent, AgentContext, AgentResult


class ExtractionAgent(BaseAgent):
    """
    Agent 2: 信息抽取

    从原始文档中抽取结构化财务数据
    """

    # 财务指标定义（纯结构，不绑定实现）
    FINANCIAL_METRICS = [
        "revenue",           # 营业收入
        "net_income",        # 净利润
        "gross_margin",      # 毛利率
        "operating_cash_flow",  # 经营现金流
        "total_assets",      # 总资产
        "total_liabilities", # 总负债
        "eps",               # 每股收益
        "roe",               # 净资产收益率
    ]

    NEWS_ENTITIES = [
        "company",           # 涉及公司
        "event_type",        # 事件类型(并购/分红/财报发布/...)
        "impact",            # 影响评估
        "related_companies", # 关联公司
    ]

    def __init__(self):
        super().__init__(
            name="ExtractionAgent",
            description="财务指标与实体抽取"
        )

    def process(self, context: AgentContext) -> AgentResult:
        documents = context.parsed_data or []

        # 1. 抽取财务指标
        metrics = self._extract_metrics(documents)

        # 2. 抽取实体与事件
        entities = self._extract_entities(documents)

        # 3. 生成多角度查询
        queries = self._generate_queries(documents, metrics, entities)

        # 评估抽取质量
        metric_score = self._evaluate_extraction(metrics, entities)
        query_score = self._evaluate_queries(queries, documents)

        result_data = {
            "metrics": metrics,
            "entities": entities,
            "queries": queries,
            "_scores": {
                "extraction": metric_score,
                "query_rewrite": query_score,
            }
        }

        return AgentResult(
            success=True,
            message=f"抽取 {len(metrics)} 项指标, {len(entities)} 个实体, 生成 {len(queries)} 个查询",
            data=result_data,
            context_updates={
                "extracted_features": result_data,
                "intermediate_findings": [{"stage": "extraction", "metrics": list(metrics.keys()),
                                           "extraction_score": metric_score, "query_score": query_score}]
            }
        )

    def _evaluate_extraction(self, metrics: Dict, entities: List[Dict]) -> float:
        """评估指标和实体抽取的覆盖率"""
        metric_hit = sum(1 for m in self.FINANCIAL_METRICS if m in metrics)
        metric_rate = metric_hit / max(len(self.FINANCIAL_METRICS), 1)

        entity_hit = min(len(entities), len(self.NEWS_ENTITIES))
        entity_rate = entity_hit / max(len(self.NEWS_ENTITIES), 1)

        return 0.6 * metric_rate + 0.4 * entity_rate

    def _evaluate_queries(self, queries: List[str], documents: List[Dict]) -> float:
        """评估生成查询的多样性和覆盖率"""
        if not queries:
            return 0.0
        # 查询数量评分
        count_score = min(1.0, len(queries) / 3)
        # 多样性：基于长度和内容差异的简单评估
        lengths = [len(q) for q in queries]
        diversity = 1.0 if len(set(lengths)) > 1 else 0.6
        return 0.4 * count_score + 0.6 * diversity

    def _extract_metrics(self, documents: List[Dict]) -> Dict[str, Any]:
        """从文档中抽取财务指标"""
        # 实际实现: NLP/NER 或 LLM 调用
        pass
        return {}

    def _extract_entities(self, documents: List[Dict]) -> List[Dict]:
        """抽取实体与事件"""
        # 实际实现: NER 管道
        pass
        return []

    def _generate_queries(self, documents: List[Dict], metrics: Dict, entities: List[Dict]) -> List[str]:
        """生成多角度检索查询"""
        # 实际实现: 根据抽取结果构建多样化查询
        pass
        return []
