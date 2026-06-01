"""
Agent 协调器 - 管理和协调多个 Agent 的执行
"""
from typing import List, Dict, Optional, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base_agent import BaseAgent, AgentContext, AgentResponse, AgentStatus


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"          # 并行执行
    CONDITIONAL = "conditional"    # 条件执行


@dataclass
class OrchestratorConfig:
    """协调器配置"""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel_agents: int = 3
    enable_retry: bool = True
    max_retries: int = 2
    timeout_seconds: float = 300.0
    verbose: bool = True


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    agent_results: List[AgentResponse] = field(default_factory=list)
    final_output: Any = None
    execution_time: float = 0.0
    execution_log: List[Dict] = field(default_factory=list)
    
    def get_agent_result(self, agent_name: str) -> Optional[AgentResponse]:
        """获取特定 Agent 的结果"""
        for result in self.agent_results:
            if result.agent_name == agent_name:
                return result
        return None


class AgentOrchestrator:
    """
    Agent 协调器
    
    职责：
    1. 注册和管理多个 Agent
    2. 决定 Agent 执行顺序
    3. 处理 Agent 之间的数据传递
    4. 支持顺序、并行、条件执行
    5. 错误处理和重试
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_pipeline: List[str] = []  # Agent 执行顺序
        self.context: Optional[AgentContext] = None
        self.execution_history: List[Dict] = []
    
    def register_agent(self, agent: BaseAgent, position: Optional[int] = None):
        """
        注册 Agent
        
        Args:
            agent: Agent 实例
            position: 在 pipeline 中的位置，None 表示添加到末尾
        """
        self.agents[agent.name] = agent
        
        if position is not None:
            self.execution_pipeline.insert(position, agent.name)
        else:
            self.execution_pipeline.append(agent.name)
        
        if self.config.verbose:
            print(f"[Orchestrator] 注册 Agent: {agent.name}")
        
        return self
    
    def register_agents(self, *agents: BaseAgent):
        """批量注册 Agent"""
        for agent in agents:
            self.register_agent(agent)
        return self
    
    def set_pipeline(self, pipeline: List[str]):
        """设置执行管道（覆盖默认顺序）"""
        # 验证所有 Agent 已注册
        for agent_name in pipeline:
            if agent_name not in self.agents:
                raise ValueError(f"Agent 未注册: {agent_name}")
        
        self.execution_pipeline = pipeline
        return self
    
    def execute(self, raw_input: str, context: Optional[AgentContext] = None) -> ExecutionResult:
        """
        执行完整的 Agent 管道
        
        Args:
            raw_input: 原始输入（如日志文件路径）
            context: 可选的初始上下文
        
        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        
        # 初始化上下文
        self.context = context or AgentContext(raw_input=raw_input)
        if raw_input:
            self.context.raw_input = raw_input
        
        # 执行日志
        execution_log = []
        agent_results = []
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[Orchestrator] 开始执行 - 模式: {self.config.execution_mode.value}")
            print(f"{'='*60}\n")
        
        try:
            if self.config.execution_mode == ExecutionMode.SEQUENTIAL:
                agent_results = self._execute_sequential(execution_log)
            elif self.config.execution_mode == ExecutionMode.PARALLEL:
                agent_results = self._execute_parallel(execution_log)
            elif self.config.execution_mode == ExecutionMode.CONDITIONAL:
                agent_results = self._execute_conditional(execution_log)
            
            success = all(r.success for r in agent_results)
            
        except Exception as e:
            execution_log.append({
                "event": "execution_error",
                "error": str(e),
                "timestamp": time.time()
            })
            success = False
        
        execution_time = time.time() - start_time
        
        # 构建最终结果
        result = ExecutionResult(
            success=success,
            agent_results=agent_results,
            final_output=self.context.final_answer if self.context else None,
            execution_time=execution_time,
            execution_log=execution_log
        )
        
        # 保存执行历史
        self.execution_history.append({
            "timestamp": start_time,
            "result": result,
            "context": self.context.to_dict() if self.context else {}
        })
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[Orchestrator] 执行完成 - 成功: {success}, 耗时: {execution_time:.2f}s")
            print(f"{'='*60}\n")
        
        return result
    
    def _execute_sequential(self, execution_log: List[Dict]) -> List[AgentResponse]:
        """顺序执行所有 Agent"""
        results = []
        
        for agent_name in self.execution_pipeline:
            agent = self.agents.get(agent_name)
            if not agent:
                continue
            
            # 检查 Agent 是否能处理当前上下文
            if not agent.can_handle(self.context):
                if self.config.verbose:
                    print(f"[Orchestrator] 跳过 {agent.name} - 无法处理当前上下文")
                continue
            
            # 执行 Agent
            result = self._execute_agent(agent, execution_log)
            results.append(result)
            
            # 更新上下文
            self._update_context(result)
            
            # 如果失败且不允许继续，则中断
            if not result.success and not self.config.enable_retry:
                break
        
        return results
    
    def _execute_parallel(self, execution_log: List[Dict]) -> List[AgentResponse]:
        """并行执行 Agent（适用于独立的 Agent）"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_agents) as executor:
            # 提交所有任务
            future_to_agent = {}
            for agent_name in self.execution_pipeline:
                agent = self.agents.get(agent_name)
                if agent and agent.can_handle(self.context):
                    future = executor.submit(self._execute_agent, agent, execution_log)
                    future_to_agent[future] = agent
            
            # 收集结果
            for future in future_to_agent:
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                    self._update_context(result)
                except Exception as e:
                    agent = future_to_agent[future]
                    results.append(AgentResponse(
                        success=False,
                        message=f"执行超时或出错: {str(e)}",
                        agent_name=agent.name
                    ))
        
        return results
    
    def _execute_conditional(self, execution_log: List[Dict]) -> List[AgentResponse]:
        """条件执行（根据前一个 Agent 的结果决定下一个）"""
        results = []
        
        for agent_name in self.execution_pipeline:
            agent = self.agents.get(agent_name)
            if not agent:
                continue
            
            # 条件判断
            if not self._should_execute(agent, results):
                if self.config.verbose:
                    print(f"[Orchestrator] 条件跳过 {agent.name}")
                continue
            
            # 执行
            result = self._execute_agent(agent, execution_log)
            results.append(result)
            self._update_context(result)
        
        return results
    
    def _execute_agent(self, agent: BaseAgent, execution_log: List[Dict]) -> AgentResponse:
        """执行单个 Agent"""
        if self.config.verbose:
            print(f"\n[Orchestrator] 执行 Agent: {agent.name}")
            print(f"  描述: {agent.description}")
            print(f"  状态: {agent.status.value}")
        
        # 记录开始
        log_entry = {
            "agent": agent.name,
            "start_time": time.time(),
            "status": "running"
        }
        execution_log.append(log_entry)
        
        # 执行（带重试）
        result = None
        retries = 0
        
        while retries <= self.config.max_retries:
            try:
                result = agent.run(self.context)
                break
            except Exception as e:
                retries += 1
                if retries > self.config.max_retries:
                    result = AgentResponse(
                        success=False,
                        message=f"执行失败（重试{retries}次）: {str(e)}",
                        agent_name=agent.name
                    )
                else:
                    if self.config.verbose:
                        print(f"  重试 {retries}/{self.config.max_retries}...")
                    time.sleep(1)  # 简单延迟
        
        # 记录完成
        log_entry.update({
            "end_time": time.time(),
            "status": "completed" if result.success else "failed",
            "success": result.success,
            "message": result.message
        })
        
        if self.config.verbose:
            status_icon = "✓" if result.success else "✗"
            print(f"  结果: {status_icon} {result.message}")
            print(f"  耗时: {result.execution_time:.2f}s")
        
        return result
    
    def _update_context(self, result: AgentResponse):
        """根据 Agent 结果更新上下文"""
        if result.context_updates:
            for key, value in result.context_updates.items():
                if hasattr(self.context, key):
                    setattr(self.context, key, value)
                else:
                    self.context.metadata[key] = value
    
    def _should_execute(self, agent: BaseAgent, previous_results: List[AgentResponse]) -> bool:
        """判断是否应该执行该 Agent（条件执行模式）"""
        # 默认使用 Agent 自己的 can_handle 判断
        return agent.can_handle(self.context)
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """获取指定 Agent"""
        return self.agents.get(name)
    
    def get_execution_summary(self) -> Dict:
        """获取执行摘要"""
        return {
            "total_executions": len(self.execution_history),
            "registered_agents": list(self.agents.keys()),
            "pipeline": self.execution_pipeline,
            "config": {
                "execution_mode": self.config.execution_mode.value,
                "max_parallel": self.config.max_parallel_agents
            }
        }
    
    def reset(self):
        """重置状态"""
        self.context = None
        for agent in self.agents.values():
            agent.reset()
        if self.config.verbose:
            print("[Orchestrator] 已重置所有状态")


# 便捷函数
def create_default_orchestrator() -> AgentOrchestrator:
    """创建默认的协调器（包含所有 Agent）"""
    from .parser_agent import LogParserAgent
    from .analyst_agent import LogAnalystAgent
    from .solution_agent import SolutionAgent
    from .report_agent import ReportAgent
    
    orchestrator = AgentOrchestrator()
    
    # 注册所有 Agent（按执行顺序）
    orchestrator.register_agents(
        LogParserAgent(),
        LogAnalystAgent(),
        SolutionAgent(),
        ReportAgent()
    )
    
    return orchestrator


def quick_analyze_with_agents(log_input: str, verbose: bool = True) -> ExecutionResult:
    """
    使用 Multi-Agent 快速分析日志
    
    Args:
        log_input: 日志文件路径或日志内容
        verbose: 是否显示详细输出
    
    Returns:
        ExecutionResult: 执行结果
    """
    orchestrator = create_default_orchestrator()
    orchestrator.config.verbose = verbose
    
    return orchestrator.execute(log_input)
