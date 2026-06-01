"""
Agent 基类 - 所有智能体的基础
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time


class AgentStatus(Enum):
    """Agent 状态"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentContext:
    """Agent 上下文 - 在多 Agent 之间共享的数据"""
    # 原始输入
    raw_input: str = ""
    
    # 解析后的数据
    parsed_data: Any = None
    
    # 分析结果
    analysis_results: Dict = field(default_factory=dict)
    
    # 中间结论
    intermediate_findings: List[str] = field(default_factory=list)
    
    # 最终答案
    final_answer: str = ""
    
    # 元数据
    metadata: Dict = field(default_factory=dict)
    
    def add_finding(self, finding: str):
        """添加中间发现"""
        self.intermediate_findings.append(finding)
    
    def to_dict(self) -> Dict:
        return {
            "raw_input": self.raw_input,
            "analysis_results": self.analysis_results,
            "intermediate_findings": self.intermediate_findings,
            "final_answer": self.final_answer,
            "metadata": self.metadata
        }


@dataclass
class AgentResponse:
    """Agent 响应"""
    success: bool
    data: Any = None
    message: str = ""
    agent_name: str = ""
    execution_time: float = 0.0
    context_updates: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.agent_name:
            self.agent_name = "UnknownAgent"


class BaseAgent(ABC):
    """
    Agent 基类
    
    所有专用 Agent 都应继承此类，实现 process 方法
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description or f"{name} Agent"
        self.status = AgentStatus.IDLE
        self.tools: Dict[str, Callable] = {}
        self.execution_history: List[Dict] = []
    
    def register_tool(self, tool_name: str, tool_func: Callable):
        """注册工具"""
        self.tools[tool_name] = tool_func
        return self
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """使用工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具不存在: {tool_name}")
        
        tool = self.tools[tool_name]
        result = tool(**kwargs)
        
        # 记录工具使用
        self.execution_history.append({
            "tool": tool_name,
            "input": kwargs,
            "output": result,
            "timestamp": time.time()
        })
        
        return result
    
    @abstractmethod
    def process(self, context: AgentContext) -> AgentResponse:
        """
        处理任务 - 子类必须实现
        
        Args:
            context: 共享上下文
            
        Returns:
            AgentResponse: 处理结果
        """
        pass
    
    def run(self, context: AgentContext) -> AgentResponse:
        """
        运行 Agent（包装方法，包含状态管理和计时）
        """
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            # 执行实际处理
            response = self.process(context)
            response.agent_name = self.name
            response.execution_time = time.time() - start_time
            
            # 更新状态
            self.status = AgentStatus.COMPLETED if response.success else AgentStatus.FAILED
            
            return response
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResponse(
                success=False,
                message=f"执行失败: {str(e)}",
                agent_name=self.name,
                execution_time=time.time() - start_time
            )
    
    def can_handle(self, context: AgentContext) -> bool:
        """
        判断是否能处理当前上下文
        子类可重写此方法实现智能路由
        """
        return True
    
    def get_status(self) -> AgentStatus:
        """获取当前状态"""
        return self.status
    
    def reset(self):
        """重置状态"""
        self.status = AgentStatus.IDLE
        self.execution_history = []
    
    def __repr__(self):
        return f"{self.name}(status={self.status.value})"
