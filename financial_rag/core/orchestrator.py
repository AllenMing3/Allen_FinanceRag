"""
AgentOrchestrator — 多 Agent 协调调度引擎

职责:
1. 注册和管理多个 Agent
2. 决定 Agent 执行顺序
3. 处理 Agent 之间的数据传递
4. 支持 SEQUENTIAL / PARALLEL / CONDITIONAL 三种模式
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from financial_rag.core.base import (
    BaseAgent,
    AgentContext,
    AgentResult,
    ExecutionMode,
)

if TYPE_CHECKING:
    from financial_rag.core.protocol import MessageBus
    from financial_rag.llm.model_router import ModelRouter

logger = logging.getLogger(__name__)


# ===================== 协调器配置 =====================

@dataclass
class CoordinatorConfig:
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel_agents: int = 3
    enable_retry: bool = True
    max_retries: int = 2
    timeout_seconds: float = 300.0
    verbose: bool = True


@dataclass
class ExecutionResult:
    """协调器执行结果"""
    success: bool
    agent_results: List[AgentResult] = field(default_factory=list)
    final_output: Any = None
    execution_time: float = 0.0
    execution_log: List[Dict] = field(default_factory=list)

    def get(self, agent_name: str) -> Optional[AgentResult]:
        for r in self.agent_results:
            if r.agent_name == agent_name:
                return r
        return None


# ===================== 协调器 =====================

class AgentOrchestrator:
    """
    Coordinate — 多 Agent 协调调度引擎

    职责:
    1. 注册和管理多个 Agent
    2. 决定 Agent 执行顺序
    3. 处理 Agent 之间的数据传递
    4. 支持 SEQUENTIAL / PARALLEL / CONDITIONAL 三种模式
    """

    def __init__(
        self,
        config: Optional[CoordinatorConfig] = None,
        model_router: Optional["ModelRouter"] = None,
    ):
        self.config = config or CoordinatorConfig()
        self.agents: Dict[str, BaseAgent] = {}
        self.pipeline: List[str] = []
        self.context: Optional[AgentContext] = None
        self.history: List[Dict] = []
        # MessageBus 集成
        self.message_bus: Optional["MessageBus"] = None
        self.use_message_bus: bool = False
        # ModelRouter 集成 — 统一的模型路由器
        self.model_router: Optional["ModelRouter"] = model_router

    def register(self, agent: BaseAgent, position: Optional[int] = None):
        """注册 Agent 并排入 pipeline

        如果 Agent 有 model_router 属性且 orchestrator 有统一 router，
        自动注入（若 Agent 尚未设置自己的 router）。
        """
        # 自动注入 ModelRouter
        if self.model_router is not None:
            if hasattr(agent, "model_router") and agent.model_router is None:
                agent.model_router = self.model_router
                if self.config.verbose:
                    logger.debug(f"[Coordinate] 注入 ModelRouter 到 {agent.name}")

        self.agents[agent.name] = agent
        if position is not None:
            self.pipeline.insert(position, agent.name)
        else:
            self.pipeline.append(agent.name)
        if self.config.verbose:
            print(f"[Coordinate] 注册 Agent: {agent.name} ({agent.description})")
        return self

    def register_all(self, *agents: BaseAgent):
        for a in agents:
            self.register(a)
        return self

    def set_pipeline(self, names: List[str]):
        """显式设置执行顺序"""
        for n in names:
            if n not in self.agents:
                raise ValueError(f"Agent 未注册: {n}")
        self.pipeline = names
        return self

    # -------------------- 主入口 --------------------

    def execute(
        self, raw_input: str, context: Optional[AgentContext] = None
    ) -> ExecutionResult:
        t0 = time.time()
        self.context = context or AgentContext(raw_input=raw_input)
        if raw_input:
            self.context.raw_input = raw_input

        log = []
        results = []

        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"[Coordinate] 执行模式: {self.config.execution_mode.value}")
            print(f"[Coordinate] Pipeline: {' → '.join(self.pipeline)}")
            print(f"{'=' * 60}\n")

        try:
            if self.config.execution_mode == ExecutionMode.SEQUENTIAL:
                results = self._run_sequential(log)
            elif self.config.execution_mode == ExecutionMode.PARALLEL:
                results = self._run_parallel(log)
            elif self.config.execution_mode == ExecutionMode.CONDITIONAL:
                results = self._run_conditional(log)

            ok = all(r.success for r in results)
        except Exception as e:
            log.append({"event": "error", "error": str(e)})
            ok = False

        elapsed = time.time() - t0
        result = ExecutionResult(
            success=ok,
            agent_results=results,
            final_output=self.context.final_answer,
            execution_time=elapsed,
            execution_log=log,
        )
        self.history.append({"time": t0, "result": result})
        return result

    # -------------------- 三种执行模式 --------------------

    def _run_sequential(self, log: List) -> List[AgentResult]:
        results = []
        for name in self.pipeline:
            agent = self.agents.get(name)
            if not agent or not agent.can_handle(self.context):
                continue
            r = self._run_one(agent, log)
            results.append(r)
            self._apply_updates(r)
            if not r.success:
                if not self.config.enable_retry:
                    break
                # Even with retry, log warning — downstream agents may get degraded context
                logger.warning(f"[Orchestrator] {name} failed; continuing chain with potentially degraded context")
        return results

    def _run_parallel(self, log: List) -> List[AgentResult]:
        results = []
        futures = {}
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_agents) as ex:
            for name in self.pipeline:
                agent = self.agents.get(name)
                if agent and agent.can_handle(self.context):
                    futures[ex.submit(self._run_one, agent, log)] = agent
            for f in futures:
                try:
                    r = f.result(timeout=self.config.timeout_seconds)
                    results.append(r)
                    self._apply_updates(r)
                except Exception as e:
                    results.append(
                        AgentResult(
                            success=False,
                            message=str(e),
                            agent_name=futures[f].name,
                        )
                    )
        return results

    def _run_conditional(self, log: List) -> List[AgentResult]:
        results = []
        for name in self.pipeline:
            agent = self.agents.get(name)
            if not agent or not agent.can_handle(self.context):
                continue
            r = self._run_one(agent, log)
            results.append(r)
            self._apply_updates(r)
            if not r.success:
                logger.warning(f"[Orchestrator] {name} failed in conditional chain; continuing")
        return results

    # -------------------- 内部方法 --------------------

    def _run_one(self, agent: BaseAgent, log: List) -> AgentResult:
        if self.config.verbose:
            print(f"[Coordinate] ▶ {agent.name} — {agent.description}")

        entry = {"agent": agent.name, "start": time.time()}
        log.append(entry)

        result = None
        for retry in range(self.config.max_retries + 1):
            try:
                result = agent.run(self.context)
                break
            except Exception:
                if retry < self.config.max_retries:
                    time.sleep(0.1)
                else:
                    result = AgentResult(
                        success=False,
                        message=f"重试{self.config.max_retries}次后失败",
                        agent_name=agent.name,
                    )

        entry.update({"end": time.time(), "ok": result.success})
        if self.config.verbose:
            status = "OK" if result.success else f"FAIL: {result.message}"
            elapsed_ms = (entry["end"] - entry["start"]) * 1000
            print(f"[Coordinate] {'✓' if result.success else '✗'} {agent.name} — {status} ({elapsed_ms:.0f}ms)")
        return result

    def _apply_updates(self, result: AgentResult):
        """应用 AgentResult 的 context_updates 到共享上下文，同时可选地发布到 MessageBus。

        dict 类型属性 (如 metadata) 采用 merge 而非 replace，
        防止后续 Agent 覆盖前序 Agent 写入的字段。
        """
        if result.context_updates:
            for k, v in result.context_updates.items():
                if hasattr(self.context, k):
                    current = getattr(self.context, k)
                    # dict 属性做 merge，不做 wholesale replace
                    if isinstance(current, dict) and isinstance(v, dict):
                        current.update(v)
                    # list 属性做 extend（如 intermediate_findings 累积各 Agent 结果）
                    elif isinstance(current, list) and isinstance(v, list):
                        setattr(self.context, k, list(current) + list(v))
                    else:
                        setattr(self.context, k, v)
                else:
                    self.context.metadata[k] = v

        # 当启用 MessageBus 时，同步发布消息
        if self.use_message_bus and self.message_bus is not None:
            from financial_rag.core.protocol import MessageAdapter

            # 获取当前总线上已有的消息 ID 作为 parent
            parent_ids = (
                list(self.message_bus._messages.keys())
                if self.message_bus._messages
                else []
            )

            msgs = MessageAdapter.from_agent_result(result, parent_msg_ids=parent_ids)
            for msg in msgs:
                self.message_bus.publish(msg)

            if self.config.verbose:
                print(
                    f"[MessageBus] 发布 {len(msgs)} 条消息 (Agent: {result.agent_name})"
                )

    def get_data_lineage(self) -> List[Dict]:
        """
        获取完整的数据链路追溯信息。

        要求 use_message_bus=True 且 message_bus 已初始化。

        Returns:
            每条消息的摘要列表，包含 sender、receiver、msg_type、payload_keys、parent_msg_ids。
            如果 MessageBus 未启用，返回空列表。
        """
        if not self.use_message_bus or self.message_bus is None:
            return []

        lineage = []
        # 按时间戳排序所有消息
        sorted_msgs = sorted(
            self.message_bus._messages.values(),
            key=lambda m: m.timestamp,
        )
        for msg in sorted_msgs:
            lineage.append(
                {
                    "msg_id": msg.msg_id,
                    "sender": msg.sender,
                    "receiver": msg.receiver,
                    "msg_type": msg.msg_type,
                    "payload_keys": list(msg.payload.keys()),
                    "parent_msg_ids": msg.parent_msg_ids,
                    "timestamp": msg.timestamp,
                }
            )
        return lineage

    def reset(self):
        self.context = None
        for a in self.agents.values():
            a.reset()
