"""
工厂函数 — create_orchestrator, create_hybrid_retriever, setup_environment

集中管理所有组件的创建逻辑，避免散落在各处。
"""
import os
import logging
from typing import Any, Optional

from financial_rag.core.base import ExecutionMode
from financial_rag.core.orchestrator import AgentOrchestrator, CoordinatorConfig
from financial_rag.core.agent_router import AgentRouter, create_agent_router

logger = logging.getLogger(__name__)


def create_orchestrator(retriever=None, llm=None) -> AgentOrchestrator:
    """创建 4-Agent 编排器 + AgentRouter

    注册 4 个 Agent:
    - CoordinatorAgent: 智能调度 (意图分类 + Agent 链选择)
    - IngestionAgent:   数据摄取 (财报/新闻)
    - AnalysisAgent:    统一分析 (指标抽取 + K线 + 事件 + 报告)
    - ScoringAgent:     全链路评分 + 防幻觉

    AgentRouter 决定每次查询走哪条 Agent 链，
    Orchestrator 负责执行链中的 Agent 并管理数据流。

    Args:
        retriever: 可选的 HybridRetriever (注入搜索工具)
        llm: 可选的 DashScopeLLM (注入抽取工具)
    """
    from financial_rag.agents.coordinator_agent import CoordinatorAgent
    from financial_rag.agents.ingestion_agent import IngestionAgent
    from financial_rag.agents.analysis_agent import AnalysisAgent
    from financial_rag.agents.scoring_agent import ScoringAgent
    from financial_rag.tools import create_financial_registry, ToolExecutor

    # 创建能力注册中心 (含检索/计算/抽取/新闻/分析/事件影响/评分/调度 全部工具)
    registry = create_financial_registry(retriever=retriever, llm=llm)
    executor = ToolExecutor(registry)

    # 创建 Agent 并绑定工具能力
    agents = [
        CoordinatorAgent(),
        IngestionAgent(),
        AnalysisAgent(),
        ScoringAgent(),
    ]
    for agent in agents:
        agent.bind_tools(registry, executor)

    orch = AgentOrchestrator(
        CoordinatorConfig(
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_retries=1,
        )
    )
    # 注册所有 Agent（不设置默认 pipeline，由 AgentRouter 动态决定）
    for agent in agents:
        orch.register(agent)

    # 附加 AgentRouter 到 orchestrator（供 Pipeline 使用）
    orch.agent_router = create_agent_router()

    return orch


def create_hybrid_retriever() -> Any:
    """创建混合检索引擎 — 实际实现在 retrievers/factory.py"""
    from financial_rag.retrievers.factory import create_hybrid_retriever as _create
    return _create()


def setup_environment() -> bool:
    """初始化环境，返回是否有 API Key"""
    from financial_rag.config import config as _cfg

    for d in [_cfg.data_dir, _cfg.kb_dir, _cfg.output_dir]:
        os.makedirs(d, exist_ok=True)

    has_key = bool(_cfg.llm.api_key)
    if not has_key:
        logger.warning("未设置 DASHSCOPE_API_KEY，使用纯本地模式")
    else:
        logger.debug(f"DashScope API 已配置，模型: {_cfg.llm.model}")
    return has_key
