"""
架构一: Coordinate — 多 Agent 协调调度器

核心设计:
- 注册多个专业化 Agent，按 pipeline 顺序/并行/条件执行
- 共享上下文 AgentContext 在 Agent 之间流转
- 支持失败重试、超时控制、执行追踪

与业务完全脱钩 — 通过接口约束而非具体实现
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


# ===================== 共享上下文 =====================

@dataclass
class AgentContext:
    """在 Multi-Agent 之间流转的共享数据容器"""
    raw_input: str = ""
    parsed_data: Any = None
    extracted_features: Dict = field(default_factory=dict)
    intermediate_findings: List[Dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "raw_input": self.raw_input[:200],
            "has_parsed_data": self.parsed_data is not None,
            "features_count": len(self.extracted_features),
            "findings_count": len(self.intermediate_findings),
            "metadata_keys": list(self.metadata.keys()),
        }


@dataclass
class AgentResult:
    """Agent 执行后的标准响应"""
    success: bool
    message: str = ""
    data: Any = None
    agent_name: str = ""
    execution_time: float = 0.0
    context_updates: Dict[str, Any] = field(default_factory=dict)


# ===================== Agent 基类 =====================

class BaseAgent(ABC):
    """Agent 抽象基类 — 与业务完全解耦"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE

    @abstractmethod
    def process(self, context: AgentContext) -> AgentResult:
        """子类实现：处理上下文的业务逻辑"""
        pass

    def can_handle(self, context: AgentContext) -> bool:
        """判断当前 Agent 是否能处理该上下文（用于条件执行）"""
        return True

    def run(self, context: AgentContext) -> AgentResult:
        """包装方法：状态管理 + 计时"""
        self.status = AgentStatus.RUNNING
        t0 = time.time()
        try:
            result = self.process(context)
            result.execution_time = time.time() - t0
            result.agent_name = self.name
            self.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
            return result
        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(
                success=False,
                message=str(e),
                agent_name=self.name,
                execution_time=time.time() - t0,
            )

    def reset(self):
        """重置 Agent 状态"""
        self.status = AgentStatus.IDLE


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

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()
        self.agents: Dict[str, BaseAgent] = {}
        self.pipeline: List[str] = []
        self.context: Optional[AgentContext] = None
        self.history: List[Dict] = []

    def register(self, agent: BaseAgent, position: Optional[int] = None):
        """注册 Agent 并排入 pipeline"""
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

    def execute(self, raw_input: str, context: Optional[AgentContext] = None) -> ExecutionResult:
        t0 = time.time()
        self.context = context or AgentContext(raw_input=raw_input)
        if raw_input:
            self.context.raw_input = raw_input

        log = []
        results = []

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[Coordinate] 执行模式: {self.config.execution_mode.value}")
            print(f"[Coordinate] Pipeline: {' → '.join(self.pipeline)}")
            print(f"{'='*60}\n")

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
            if not r.success and not self.config.enable_retry:
                break
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
                    results.append(AgentResult(success=False, message=str(e), agent_name=futures[f].name))
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
                    time.sleep(1)
                else:
                    result = AgentResult(success=False, message=f"重试{self.config.max_retries}次后失败", agent_name=agent.name)

        entry.update({"end": time.time(), "ok": result.success})
        return result

    def _apply_updates(self, result: AgentResult):
        if result.context_updates:
            for k, v in result.context_updates.items():
                if hasattr(self.context, k):
                    setattr(self.context, k, v)
                else:
                    self.context.metadata[k] = v

    def reset(self):
        self.context = None
        for a in self.agents.values():
            a.reset()
