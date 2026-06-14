"""
Agent 基础抽象 — BaseAgent, AgentContext, AgentResult, AgentStatus, ExecutionMode

与业务完全解耦，仅定义 Agent 的接口和数据容器。
所有 Agent 子类只需 `from financial_rag.core.base import BaseAgent, AgentContext, AgentResult`。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
import time

if TYPE_CHECKING:
    from financial_rag.tools.core import FunctionRegistry, ToolExecutor, ToolCallRequest


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
        self._registry: Optional["FunctionRegistry"] = None
        self._executor: Optional["ToolExecutor"] = None

    def bind_tools(self, registry: "FunctionRegistry", executor: "ToolExecutor" = None):
        """注入 Function Calling 能力 — 让 Agent 可以调用已注册的工具"""
        self._registry = registry
        if executor is None:
            from financial_rag.tools.core import ToolExecutor
            self._executor = ToolExecutor(registry)
        else:
            self._executor = executor

    def call_tool(self, name: str, **kwargs) -> Any:
        """调用已注册的工具，返回结果或抛出异常"""
        if not self._registry:
            raise RuntimeError(
                f"{self.name}: 未绑定 FunctionRegistry，无法调用工具 '{name}'。"
                f"请先调用 bind_tools() 注入能力。"
            )
        from financial_rag.tools.core import ToolCallRequest
        request = ToolCallRequest(
            id=f"agent_{self.name}_{time.time():.0f}",
            name=name,
            arguments=kwargs,
        )
        result = self._executor.execute(request)
        if not result.success:
            raise RuntimeError(f"工具 '{name}' 执行失败: {result.error}")
        return result.result

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
