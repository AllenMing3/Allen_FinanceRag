"""
ExtractionAgent — AI/科技行业关键指标抽取 (Function Calling 版)

Agent 只负责决策和编排，所有重活委托给 tools:
- extract_financial_metrics: 业务指标抽取 (LLM-first)
- extract_entities: 实体与事件抽取 (LLM-first)
- generate_search_queries: 多角度检索查询生成 (LLM-first)
"""
from typing import Dict, Any, List

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult


class ExtractionAgent(BaseAgent):
    """
    Agent 2: AI/科技行业信息抽取

    从原始文档中抽取结构化业务数据。
    所有抽取逻辑下沉至 extraction_tools，Agent 只做编排 + 质量评估。
    """

    # AI 行业指标体系（用于质量评分）
    AI_METRICS = [
        # 财务
        "revenue",              # 营业收入
        "net_income",           # 净利润
        "gross_margin",         # 毛利率
        "rd_expense",           # 研发费用
        "arr",                  # 年度经常性收入
        # 算力
        "gpu_count",            # GPU/芯片数量
        "training_cluster_size", # 训练集群规模
        "inference_cost_per_token", # 推理成本
        # 模型
        "model_params",         # 模型参数量
        "context_window",       # 上下文窗口
        # 商业
        "api_calls",            # API调用量
        "customer_count",       # 客户数
    ]

    NEWS_ENTITIES = [
        "company",           # 涉及公司
        "event_type",        # 事件类型
        "impact",            # 影响评估
        "ai_models",         # AI 模型/产品
        "tech_terms",        # 技术术语
    ]

    def __init__(self):
        super().__init__(
            name="ExtractionAgent",
            description="财务指标与实体抽取 (Function Calling)"
        )

    def process(self, context: AgentContext) -> AgentResult:
        documents = context.parsed_data or []

        if not documents:
            return AgentResult(
                success=False,
                message="无文档可供抽取",
                data={"metrics": {}, "entities": {}, "queries": []},
            )

        # 合并所有文档文本
        combined_text = "\n\n".join(
            d.get("text", "") for d in documents if d.get("text")
        )

        if not combined_text:
            return AgentResult(
                success=False,
                message="文档文本为空",
                data={"metrics": {}, "entities": {}, "queries": []},
            )

        # ---- 1. 调用工具: 财务指标抽取 ----
        metrics = {}
        try:
            metrics = self.call_tool("extract_financial_metrics", text=combined_text)
        except RuntimeError as e:
            print(f"[ExtractionAgent] 指标抽取失败: {e}")

        # ---- 2. 调用工具: 实体与事件抽取 ----
        entities = {}
        try:
            entities = self.call_tool("extract_entities", text=combined_text)
        except RuntimeError as e:
            print(f"[ExtractionAgent] 实体抽取失败: {e}")

        # ---- 3. 调用工具: 生成多角度查询 ----
        queries = []
        try:
            queries = self.call_tool(
                "generate_search_queries",
                text=combined_text,
                metrics=metrics,
                entities=entities,
            )
        except RuntimeError as e:
            print(f"[ExtractionAgent] 查询生成失败: {e}")

        # ---- 4. 质量评估 ----
        metric_score = self._evaluate_extraction(metrics, entities)
        query_score = self._evaluate_queries(queries)

        result_data = {
            "metrics": metrics,
            "entities": entities,
            "queries": queries,
            "_scores": {
                "extraction": metric_score,
                "query_rewrite": query_score,
            },
            "_confidence": {
                "metrics": metrics.get("_confidence", "none"),
                "entities": entities.get("_confidence", "none"),
            },
        }

        # 统计指标数量 (排除内部字段)
        metric_count = sum(1 for k in metrics if not k.startswith("_"))
        entity_count = sum(
            1 for v in entities.values()
            if isinstance(v, list) and not str(v).startswith("_")
        )

        return AgentResult(
            success=True,
            message=f"抽取 {metric_count} 项指标, {entity_count} 类实体, 生成 {len(queries)} 个查询",
            data=result_data,
            context_updates={
                "extracted_features": result_data,
                "intermediate_findings": [{
                    "stage": "extraction",
                    "metrics": [k for k in metrics if not k.startswith("_")],
                    "extraction_score": metric_score,
                    "query_score": query_score,
                }]
            }
        )

    # ===================== 质量评分 =====================

    def _evaluate_extraction(self, metrics: Dict, entities: Dict) -> float:
        """评估指标和实体抽取的覆盖率 — AI 行业维度"""
        # 指标命中率 (AI 指标体系)
        metric_hit = sum(1 for m in self.AI_METRICS if m in metrics)
        metric_rate = metric_hit / max(len(self.AI_METRICS), 1)

        # 实体: 按有内容的类别计数
        entity_categories = ["companies", "persons", "ai_models", "chips_hardware",
                             "tech_terms", "financial_figures", "event",
                             "industries", "key_topics"]
        entity_hit = sum(
            1 for cat in entity_categories
            if entities.get(cat) and str(entities.get(cat)) not in ("[]", "{}")
        )
        entity_rate = min(entity_hit / 4.0, 1.0)

        return 0.6 * metric_rate + 0.4 * entity_rate

    def _evaluate_queries(self, queries: List[str]) -> float:
        """评估生成查询的多样性和覆盖率"""
        if not queries:
            return 0.0
        # 查询数量评分
        count_score = min(1.0, len(queries) / 3)
        # 多样性: 基于长度差异
        lengths = [len(q) for q in queries]
        diversity = 1.0 if len(set(lengths)) > 1 else 0.6
        return 0.4 * count_score + 0.6 * diversity
