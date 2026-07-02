"""
PipelineScheduler — 端到端 Pipeline 中央调度器

五次阶段:
  获取 → 索引(RAG) → 加工(Agent/FC) → 输出(SlotFiller) → 进化(Scoring)

用法:
    from financial_rag.core.pipeline import PipelineScheduler, PipelineConfig
    scheduler = PipelineScheduler(orchestrator, retriever, registry, llm)
    result = scheduler.run("茅台2024年营收增长了多少？")
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import time
import logging

from financial_rag.core.orchestrator import AgentOrchestrator, ExecutionResult
from financial_rag.core.base import AgentContext

if TYPE_CHECKING:
    from financial_rag.tools import FunctionRegistry, ToolCallSession, ToolExecutor
    from financial_rag.templates import SlottedTemplate
    from financial_rag.slot_filler import SlotFiller
    from financial_rag.core.scorer import PipelineScoreCard
    from financial_rag.guard.reflector import HallucinationGuard
    from financial_rag.retrievers import HybridRetriever
    from financial_rag.core.agent_router import RoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline 调度配置"""

    enable_data_fetch: bool = True  # 阶段1: 是否获取实时数据(RSS/Tushare)
    enable_index: bool = True  # 阶段2: 是否建索引
    enable_agent_analysis: bool = True  # 阶段3: 是否跑Multi-Agent
    enable_slot_output: bool = True  # 阶段4: 是否槽位填充
    enable_scoring: bool = True  # 阶段5: 是否全链路打分
    verbose: bool = True


@dataclass
class PipelineResult:
    """Pipeline 执行结果 — 包含各阶段完整产物"""

    query: str = ""
    success: bool = False

    # 阶段1: 获取
    fetched_data: List[Dict] = field(default_factory=list)
    fetch_result: Optional[Dict] = None
    fetch_elapsed_ms: float = 0.0

    # 阶段2: 索引
    indexed_docs: int = 0
    retrieved_items: List[Dict] = field(default_factory=list)
    index_elapsed_ms: float = 0.0

    # 阶段3: 加工
    tool_call_stats: Any = None  # ToolCallStats
    agent_exec_result: Optional[ExecutionResult] = None
    process_elapsed_ms: float = 0.0

    # 阶段4: 输出
    final_output: str = ""
    fill_stats: Any = None  # FillStats
    output_elapsed_ms: float = 0.0

    # 阶段5: 进化
    scorecard: Any = None  # PipelineScoreCard
    hallucination_check: Optional[Dict] = None

    # 汇总
    total_elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class PipelineScheduler:
    """
    端到端 Pipeline 中央调度器

    接管 main.py 中散落的编排逻辑，实现五次阶段:
      获取 → 索引(RAG) → 加工(Agent/FC) → 输出(SlotFiller) → 进化(Scoring)

    用法:
        from financial_rag.core.pipeline import PipelineScheduler, PipelineConfig
        scheduler = PipelineScheduler(orchestrator, retriever, registry, llm)
        result = scheduler.run("茅台2024年营收增长了多少？")
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        retriever: Any = None,  # HybridRetriever
        registry: Any = None,  # FunctionRegistry
        executor: Any = None,  # ToolExecutor
        llm: Any = None,  # DashScopeLLM
        filler: Any = None,  # SlotFiller
        config: Optional[PipelineConfig] = None,
        data_orchestrator: Any = None,  # DataOrchestrator (多 Pool 模式)
    ):
        self.orchestrator = orchestrator
        self.retriever = retriever
        self.registry = registry
        self.executor = executor
        self.llm = llm
        self.filler = filler
        self.config = config or PipelineConfig()
        self.data_orchestrator = data_orchestrator
        self._logger = logging.getLogger(f"{__name__}.PipelineScheduler")

    def run(
        self,
        query: str,
        template: Any = None,  # SlottedTemplate
        max_fetch_news: int = 10,
        max_retrieve: int = 5,
    ) -> PipelineResult:
        """
        执行完整 Pipeline：获取 → 索引 → 加工 → 输出 → 进化

        Args:
            query: 用户自然语言查询
            template: 槽位模板，默认 QUICK_QA
            max_fetch_news: 最多获取几条新闻
            max_retrieve: 检索返回条数
        """
        result = PipelineResult(query=query)
        t_start = time.time()

        try:
            # ====== 阶段1: 获取 ======
            result = self._phase_fetch(query, result, max_fetch_news)

            # ====== 阶段2: 索引 (RAG) ======
            result = self._phase_index(query, result, max_retrieve)

            # ====== 阶段3: 加工 (Agent/FC) ======
            result = self._phase_process(query, result)

            # ====== 阶段4: 输出 (SlotFiller) ======
            result = self._phase_output(query, result, template)

            # ====== 阶段5: 进化 (Scoring + Reflection) ======
            result = self._phase_evolve(result)

            result.success = True
        except Exception as e:
            result.errors.append(str(e))
            self._logger.error(f"Pipeline 执行失败: {e}")
            result.success = False

        result.total_elapsed_ms = (time.time() - t_start) * 1000

        if self.config.verbose:
            print(
                f"\n[Pipeline] 完成: {'OK' if result.success else 'FAIL'}"
                f" | 总耗时 {result.total_elapsed_ms:.0f}ms"
                f" | 获取={result.fetch_elapsed_ms:.0f}ms"
                f" | 索引={result.index_elapsed_ms:.0f}ms"
                f" | 加工={result.process_elapsed_ms:.0f}ms"
                f" | 输出={result.output_elapsed_ms:.0f}ms"
            )

        return result

    # ==================== 阶段1: 获取 ====================

    def _phase_fetch(
        self, query: str, result: PipelineResult, max_news: int
    ) -> PipelineResult:
        """通过 Function Calling 自动选择数据工具获取数据"""
        t0 = time.time()

        if not self.config.enable_data_fetch or not self.registry or not self.llm:
            if self.config.verbose:
                print("[Pipeline] 阶段1: 获取 — 跳过 (未启用)")
            return result

        if self.config.verbose:
            print("[Pipeline] 阶段1: 获取 — 通过 Function Calling 选择数据源...")

        try:
            # 创建一个简短的 FC 会话，只负责拉起数据
            from financial_rag.tools import ToolCallSession

            session = ToolCallSession(
                llm=self.llm,
                registry=self.registry,
                executor=self.executor,
                max_rounds=2,
                verbose=False,
                system_prompt=(
                    "你是数据获取助手。根据用户问题，调用合适的数据工具获取信息。"
                    "只获取数据，不要分析，不要总结，不要截断。"
                ),
            )
            stats = session.run(
                query=f"获取以下查询需要的所有数据，保留完整内容: {query}",
            )

            result.tool_call_stats = stats

            # 提取获取到的原始数据，归一化为标准文档格式
            for call in stats.calls:
                if not call.success:
                    continue
                data = call.result
                if not isinstance(data, dict):
                    continue
                result.fetch_result = data
                # Normalize: tools may return data under different keys
                items = data.get("items") or data.get("results") or []
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    # Ensure standard fields exist (title, content, source, publish_time, url)
                    doc = {
                        "title": item.get("title", ""),
                        "content": item.get("content", "") or item.get("text", ""),
                        "source": item.get("source", call.name),
                        "publish_time": item.get("publish_time", ""),
                        "url": item.get("url", ""),
                    }
                    # Keep extra fields for downstream use
                    for k, v in item.items():
                        if k not in doc:
                            doc[k] = v
                    if doc["title"] or doc["content"]:
                        result.fetched_data.append(doc)

            if self.config.verbose:
                print(
                    f"  [获取] {len(stats.calls)} 次工具调用, "
                    f"获得 {len(result.fetched_data)} 条数据"
                )
        except Exception as e:
            self._logger.warning(f"阶段1 获取失败: {e}")
            result.errors.append(f"获取阶段: {e}")

        result.fetch_elapsed_ms = (time.time() - t0) * 1000
        return result

    # ==================== 阶段2: 索引 ====================

    def _phase_index(
        self, query: str, result: PipelineResult, max_retrieve: int
    ) -> PipelineResult:
        """将获取到的数据入 RAG 索引，并检索相关内容
        
        支持两种模式:
        - 单 Retriever 模式 (默认): 所有文档入同一个 Retriever
        - 多 Pool 模式 (data_orchestrator): 按文档类型路由到不同 Pool
        """
        t0 = time.time()

        has_indexer = self.retriever or self.data_orchestrator
        if not self.config.enable_index or not has_indexer:
            if self.config.verbose:
                print("[Pipeline] 阶段2: 索引 — 跳过 (未启用)")
            return result

        if self.config.verbose:
            mode = "多Pool" if self.data_orchestrator else "单Retriever"
            print(f"[Pipeline] 阶段2: 索引 — 数据入RAG库 + 混合检索 ({mode})...")

        try:
            # 将阶段1获取的数据转为文档
            docs = []
            if result.fetched_data:
                for item in result.fetched_data:
                    title = item.get("title", "")
                    content = item.get("content", "")
                    text = f"{title}\n{content}" if title else content
                    meta = {
                        "source": item.get("source", "rss"),
                        "publish_time": item.get("publish_time", ""),
                        "url": item.get("url", ""),
                    }
                    if text.strip():
                        docs.append({"text": text, "meta": meta})

            # 索引: 多 Pool 模式 vs 单 Retriever 模式
            if self.data_orchestrator:
                if docs:
                    ingest_stats = self.data_orchestrator.ingest(docs)
                    result.indexed_docs = ingest_stats.total_docs - ingest_stats.rejected
                    if self.config.verbose:
                        print(f"  [索引] {ingest_stats.summary()}")
                items = self.data_orchestrator.search(query, top_k=max_retrieve)
            else:
                if docs:
                    if self.retriever.documents:
                        self.retriever.add(docs)
                    else:
                        self.retriever.index(docs)
                    result.indexed_docs = len(docs)
                    if self.config.verbose:
                        total = len(self.retriever.documents)
                        print(f"  [索引] 新增 {len(docs)} 篇文档 (知识库共 {total} 篇)")
                items = self.retriever.search(query, top_k=max_retrieve) if self.retriever else []

            result.retrieved_items = items
            if self.config.verbose:
                print(f"  [检索] 命中 {len(items)} 条")
        except Exception as e:
            self._logger.warning(f"阶段2 索引失败: {e}")
            result.errors.append(f"索引阶段: {e}")

        result.index_elapsed_ms = (time.time() - t0) * 1000
        return result

    # ==================== 阶段3: 加工 ====================

    def _phase_process(self, query: str, result: PipelineResult) -> PipelineResult:
        """Multi-Agent 分析 + Function Calling 深度处理
    
        使用 AgentRouter 识别查询意图，动态选择 Agent 执行链:
        - kline:        AnalysisAgent → ScoringAgent
        - event_impact: AnalysisAgent → ScoringAgent
        - report:       IngestionAgent → AnalysisAgent → ScoringAgent
        - news:         IngestionAgent → AnalysisAgent → ScoringAgent
        - general:      IngestionAgent → AnalysisAgent → ScoringAgent
        """
        t0 = time.time()
    
        if not self.config.enable_agent_analysis:
            if self.config.verbose:
                print("[Pipeline] 阶段3: 加工 — 跳过 (未启用)")
            return result
    
        if self.config.verbose:
            print("[Pipeline] 阶段3: 加工 — Multi-Agent 分析...")
    
        try:
            # ---- Agent 路由 ----
            routing = self._route_query(query, result)
    
            if self.config.verbose:
                print(f"  [路由] {routing}")
    
            # ---- 动态设置执行链 ----
            orch = self.orchestrator
            chain = routing.agent_chain
            # 只使用已注册的 Agent
            valid_chain = [n for n in chain if n in orch.agents]
            if not valid_chain:
                # 降级: 使用所有已注册 Agent
                valid_chain = list(orch.pipeline) or list(orch.agents.keys())
            orch.set_pipeline(valid_chain)
    
            # ---- 构建上下文 ----
            context_text = query
            agent_context = None
            if result.retrieved_items:
                chunks = []
                for r in result.retrieved_items[:5]:
                    text = r.get("text", "")
                    if text:
                        chunks.append(text)
                if chunks:
                    context_text = (
                        f"查询: {query}\n\n参考数据:\n" + "\n---\n".join(chunks)
                    )
                # 将结构化检索结果 + 路由元数据 + 时间指标传入 AgentContext
                agent_context = AgentContext(
                    raw_input=context_text,
                    metadata={
                        "retrieved_items": result.retrieved_items,
                        "fetched_data": result.fetched_data,
                        "intent": routing.intent,
                        "fetch_elapsed_ms": result.fetch_elapsed_ms,
                        "index_elapsed_ms": result.index_elapsed_ms,
                        **routing.metadata,
                    },
                )
            else:
                # 即使没有检索结果，也传路由元数据 + 时间指标
                agent_context = AgentContext(
                    raw_input=query,
                    metadata={
                        "intent": routing.intent,
                        "fetch_elapsed_ms": result.fetch_elapsed_ms,
                        "index_elapsed_ms": result.index_elapsed_ms,
                        **routing.metadata,
                    },
                )
    
            agent_result = orch.execute(
                context_text, context=agent_context
            )
            result.agent_exec_result = agent_result
    
            if self.config.verbose:
                ok_count = sum(
                    1 for r in agent_result.agent_results if r.success
                )
                print(
                    f"  [加工] {ok_count}/{len(agent_result.agent_results)} Agent 成功"
                )
        except Exception as e:
            self._logger.warning(f"阶段3 加工失败: {e}")
            result.errors.append(f"加工阶段: {e}")
    
        result.process_elapsed_ms = (time.time() - t0) * 1000
        return result
    
    def _route_query(self, query: str, result: PipelineResult) -> "RoutingDecision":
        """使用 AgentRouter 识别意图并返回路由决策"""
        from financial_rag.core.agent_router import RoutingDecision, create_agent_router
    
        router = getattr(self.orchestrator, "agent_router", None)
        if router is None:
            router = create_agent_router()
    
        # 传递已有的 fetched_data 元数据辅助路由
        context_meta = {}
        if result.fetched_data:
            # 检查是否有日期信息可辅助 event_impact 判断
            for item in result.fetched_data[:3]:
                pt = item.get("publish_time", "")
                if pt:
                    context_meta["fetched_date"] = pt[:10]
                    break
    
        return router.route(query, context=context_meta)
    

    # ==================== 阶段4: 输出 ====================

    def _phase_output(
        self, query: str, result: PipelineResult, template: Any = None
    ) -> PipelineResult:
        """槽位填充输出 — 仅在 Phase 3 未生成有效输出时运行"""
        t0 = time.time()

        if not self.config.enable_slot_output or not self.filler:
            if self.config.verbose:
                print("[Pipeline] 阶段4: 输出 — 跳过 (未启用)")
            return result

        # Phase 3 Agent 已生成有效输出时，不再用槽位覆盖
        if result.final_output and len(result.final_output.strip()) > 50:
            if self.config.verbose:
                print(f"[Pipeline] 阶段4: 输出 — 跳过 (Agent 已生成 {len(result.final_output)} 字输出)")
            return result

        if self.config.verbose:
            print("[Pipeline] 阶段4: 输出 — 槽位填充...")

        try:
            from financial_rag.templates import QUICK_QA_TEMPLATE

            tmpl = template or QUICK_QA_TEMPLATE

            # 构建上下文（检索结果 + Agent 分析 + 完整文章）
            context_docs = []
            # 优先使用 Agent 分析结果（Phase 3 产出）
            if result.agent_exec_result:
                for ar in result.agent_exec_result.agent_results:
                    if ar.success and ar.data:
                        if isinstance(ar.data, dict):
                            # Prefer rendered markdown > message > skip dict repr
                            agent_text = ar.data.get("markdown", "") or ar.message or ""
                        elif isinstance(ar.data, str):
                            agent_text = ar.data
                        else:
                            agent_text = str(ar.data)
                        if agent_text.strip():
                            context_docs.append(f"【{ar.agent_name}分析】\n{agent_text}")
                if result.agent_exec_result.final_output:
                    context_docs.append(
                        f"【综合分析】\n{result.agent_exec_result.final_output}"
                    )
            # 检索结果
            if result.retrieved_items:
                for r in result.retrieved_items:
                    text = r.get("text", "")
                    if text:
                        context_docs.append(text)
            # 原始获取数据
            if result.fetched_data:
                for item in result.fetched_data[:5]:
                    title = item.get("title", "")
                    content = item.get("content", "")
                    full_text = f"【{title}】\n{content}" if title else content
                    if full_text.strip():
                        context_docs.append(full_text)

            if not context_docs:
                context_docs = [f"查询: {query} (暂无外部数据)"]

            fill_stats = self.filler.fill(tmpl, query=query, context_docs=context_docs)
            result.fill_stats = fill_stats
            result.final_output = self.filler.render(tmpl, fill_stats)

            if self.config.verbose:
                print(
                    f"  [输出] {fill_stats.filled_slots}/{fill_stats.total_slots} 槽位"
                    f" | TTFT avg={fill_stats.avg_ttft_ms:.0f}ms"
                )
        except Exception as e:
            self._logger.warning(f"阶段4 输出失败: {e}")
            result.errors.append(f"输出阶段: {e}")
            # 兜底输出
            if not result.final_output:
                result.final_output = query

        result.output_elapsed_ms = (time.time() - t0) * 1000
        return result

    # ==================== 阶段5: 进化 ====================

    def _phase_evolve(self, result: PipelineResult) -> PipelineResult:
        """全链路打分 + 防幻觉校验"""
        if not self.config.enable_scoring:
            return result

        if self.config.verbose:
            print("[Pipeline] 阶段5: 进化 — 全链路打分 + 防幻觉...")

        try:
            from financial_rag.core.scorer import PipelineScoreCard
            from financial_rag.guard.reflector import HallucinationGuard

            card = PipelineScoreCard(query=result.query)
            result.scorecard = card

            # 记录各阶段评分
            if result.fetched_data:
                card.record(
                    "fetch",
                    "数据获取",
                    0.9 if len(result.fetched_data) > 0 else 0.3,
                    elapsed_ms=result.fetch_elapsed_ms,
                )
            if result.retrieved_items:
                card.record(
                    "index",
                    "RAG检索",
                    0.85,
                    elapsed_ms=result.index_elapsed_ms,
                    details={"hit_count": len(result.retrieved_items)},
                )
            if result.agent_exec_result:
                agent_ok = sum(
                    1 for r in result.agent_exec_result.agent_results if r.success
                )
                agent_total = len(result.agent_exec_result.agent_results)
                card.record(
                    "process",
                    "Multi-Agent加工",
                    0.8 if agent_total > 0 else 0,
                    elapsed_ms=result.process_elapsed_ms,
                    details={"agents_ok": f"{agent_ok}/{agent_total}"},
                )
            if result.fill_stats:
                fill_rate = result.fill_stats.filled_slots / max(
                    result.fill_stats.total_slots, 1
                )
                card.record(
                    "output",
                    "槽位输出",
                    fill_rate,
                    elapsed_ms=result.output_elapsed_ms,
                    details={"template": result.fill_stats.template_name},
                )

            # 防幻觉校验
            if result.final_output:
                guard = HallucinationGuard()
                raw_items = result.retrieved_items or []
                check = guard.check(result.final_output, raw_items)
                result.hallucination_check = check
                card.record(
                    "hallucination",
                    "防幻觉校验",
                    check.get("overall_score", 1.0),
                    details={"risk": check.get("risk", "low")},
                )
                # 将评分卡片追加到最终输出 — 用户可见
                report = check.get("report", "")
                if report:
                    result.final_output = result.final_output + "\n" + report

        except Exception as e:
            self._logger.warning(f"阶段5 进化失败: {e}")
            result.errors.append(f"进化阶段: {e}")

        return result


# ===================== 工厂函数 =====================


def create_pipeline_scheduler(
    orchestrator: AgentOrchestrator,
    retriever: Any = None,
    registry: Any = None,
    executor: Any = None,
    llm: Any = None,
    filler: Any = None,
    config: Optional[PipelineConfig] = None,
    data_orchestrator: Any = None,
) -> PipelineScheduler:
    """快速创建 Pipeline 调度器"""
    return PipelineScheduler(
        orchestrator=orchestrator,
        retriever=retriever,
        registry=registry,
        executor=executor,
        llm=llm,
        filler=filler,
        config=config,
        data_orchestrator=data_orchestrator,
    )
