"""
Query Planner — LLM 驱动的查询计划生成

在 QueryParser (纯规则) 之后，用一次 LLM call 将查询拆解为:
- 意图分类 (summary / comparison / timeline / deep_dive / factual)
- 子查询列表 (每个带 source 和 mode)
- 执行策略 (parallel / sequential)

设计原则:
- 简单: 一次 LLM call，JSON 输出
- 解耦: 不改现有 QueryParser / Retriever
- 可降级: LLM 调用失败时回退为单查询
"""
from dataclasses import dataclass, field
from typing import List, Optional

from financial_rag.llm import get_llm, get_caller, LLMCaller


# ===================== 计划类型 =====================

@dataclass
class SubQuery:
    """子查询"""
    query: str          # 子查询文本
    source: str = "all" # kb / news / graph / all
    mode: str = "hybrid" # local / global / hybrid / mix
    purpose: str = ""   # 这个子查询的目的

@dataclass
class QueryPlan:
    """查询计划"""
    original_query: str
    intent: str = "factual"          # summary / comparison / timeline / deep_dive / factual
    sub_queries: List[SubQuery] = field(default_factory=list)
    strategy: str = "parallel"       # parallel / sequential

    @property
    def is_simple(self) -> bool:
        """是否需要拆解 — 单子查询视为简单查询"""
        return len(self.sub_queries) <= 1


# ===================== Prompt =====================

_PLAN_SYSTEM = """你是一个查询规划器。根据用户的中文财经查询，生成一个 JSON 执行计划。

## 意图类型
- summary: 综合概览 (如 "AI行业最近怎么样")
- comparison: 对比分析 (如 "英伟达和华为芯片谁强")
- timeline: 时间线梳理 (如 "OpenAI融资历程")
- deep_dive: 深度分析 (如 "商汤科技的生成式AI业务前景")
- factual: 事实查询 (如 "茅台今天收盘价多少")

## 子查询来源
- kb: 知识库 (已导入的分析报告、文档)
- news: 新闻 (实时新闻抓取)
- graph: 知识图谱 (实体关系推理)
- all: 全部来源

## 子查询模式
- local: 精确属性查询
- global: 全局概览
- hybrid: 跨文档关系
- mix: 多路融合 (默认)

## 输出格式 (严格 JSON)
{
  "intent": "comparison",
  "strategy": "parallel",
  "sub_queries": [
    {"query": "英伟达AI芯片最新产品和性能参数", "source": "all", "mode": "local", "purpose": "获取英伟达芯片信息"},
    {"query": "华为昇腾AI芯片产品和技术参数", "source": "all", "mode": "local", "purpose": "获取华为芯片信息"},
    {"query": "英伟达和华为AI芯片竞争格局和市场对比", "source": "all", "mode": "hybrid", "purpose": "对比分析"}
  ]
}

## 规则
1. 简单事实查询 → 1 个子查询即可
2. 对比类 → 为每个对比方单独生成子查询 + 一个综合子查询
3. 复杂分析 → 2-4 个子查询，不要超过 5 个
4. 每个子查询要具体、可检索，避免泛泛的问题
5. graph 来源适合实体关系推理 (如 "A 和 B 有什么关系")
"""


# ===================== Planner =====================

class QueryPlanner:
    """
    查询计划器 — 一次 LLM call 生成子查询计划

    使用:
    >>> planner = QueryPlanner()
    >>> plan = planner.plan("英伟达和华为的AI芯片竞争格局")
    >>> plan.intent
    'comparison'
    >>> len(plan.sub_queries)
    3
    """

    def __init__(self, caller: Optional[LLMCaller] = None):
        if caller is None:
            caller = get_caller(get_llm())
        self._caller = caller

    def plan(self, query: str) -> QueryPlan:
        """生成查询计划，LLM 失败时降级为单查询"""
        result = self._caller.call_json(
            messages=query,
            system=_PLAN_SYSTEM,
            max_json_retries=1,
            use_cache=True,
        )

        if not result or not isinstance(result, dict):
            return self._fallback(query)

        try:
            return self._parse_plan(query, result)
        except Exception:
            return self._fallback(query)

    def _parse_plan(self, query: str, raw: dict) -> QueryPlan:
        """解析 LLM 返回的 JSON 为 QueryPlan"""
        sub_queries = []
        for sq in raw.get("sub_queries", []):
            if isinstance(sq, dict) and sq.get("query"):
                sub_queries.append(SubQuery(
                    query=sq["query"],
                    source=sq.get("source", "all"),
                    mode=sq.get("mode", "mix"),
                    purpose=sq.get("purpose", ""),
                ))

        return QueryPlan(
            original_query=query,
            intent=raw.get("intent", "factual"),
            sub_queries=sub_queries,
            strategy=raw.get("strategy", "parallel"),
        )

    def _fallback(self, query: str) -> QueryPlan:
        """降级: 单子查询，全来源"""
        return QueryPlan(
            original_query=query,
            intent="factual",
            sub_queries=[SubQuery(query=query, source="all", mode="mix", purpose="直接检索")],
            strategy="parallel",
        )
