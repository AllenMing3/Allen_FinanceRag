"""
智能模型路由系统 — 按任务复杂度自动选择 LLM 模型

特性:
- 基于预算约束自动降级
- 失败自动回退到更轻量模型
- 统计各模型调用次数/cost
- 支持手动指定某类任务固定用某模型
"""
import os
import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from .dashscope_client import DashScopeLLM

logger = logging.getLogger(__name__)


# ===================== 模型分层 =====================

class ModelTier(Enum):
    """模型分层 — 从轻到重"""
    LIGHT = "light"          # 简单任务: qwen-turbo
    STANDARD = "standard"    # 常规任务: qwen-plus
    HEAVY = "heavy"          # 复杂任务: qwen-max
    ULTRA = "ultra"          # 超复杂: qwen3-235b


# 分层到模型名的映射
TIER_MODEL_MAP: Dict[ModelTier, str] = {
    ModelTier.LIGHT: "qwen-turbo",
    ModelTier.STANDARD: "qwen-plus",
    ModelTier.HEAVY: "qwen-max",
    ModelTier.ULTRA: "qwen3-235b",
}

# 分层回退顺序: 当前层失败后降级到下一层
_TIER_FALLBACK_ORDER: List[ModelTier] = [
    ModelTier.ULTRA,
    ModelTier.HEAVY,
    ModelTier.STANDARD,
    ModelTier.LIGHT,
]

# 单次调用预估成本（元/1K tokens，基于阿里云定价）
_MODEL_COST_PER_1K: Dict[str, float] = {
    "qwen-turbo": 0.0003,
    "qwen-plus": 0.0008,
    "qwen-max": 0.002,
    "qwen3-235b": 0.004,
}


# ===================== 任务复杂度 =====================

class TaskComplexity(Enum):
    """任务复杂度分级"""
    TRIVIAL = "trivial"       # 纯文本解析、json格式化
    SIMPLE = "simple"         # 简单抽取、关键词识别
    MODERATE = "moderate"     # 指标计算、分类
    COMPLEX = "complex"       # 多步推理、趋势分析
    EXPERT = "expert"         # 综合分析、报告生成


# 任务复杂度 → 推荐模型分层
COMPLEXITY_TO_TIER: Dict[TaskComplexity, ModelTier] = {
    TaskComplexity.TRIVIAL: ModelTier.LIGHT,
    TaskComplexity.SIMPLE: ModelTier.STANDARD,
    TaskComplexity.MODERATE: ModelTier.STANDARD,
    TaskComplexity.COMPLEX: ModelTier.HEAVY,
    TaskComplexity.EXPERT: ModelTier.ULTRA,
}

# 任务类型 → 复杂度 的映射
TASK_COMPLEXITY_MAP: Dict[str, TaskComplexity] = {
    "ingestion": TaskComplexity.SIMPLE,
    "extraction": TaskComplexity.MODERATE,
    "analysis": TaskComplexity.COMPLEX,
    "forecast": TaskComplexity.COMPLEX,
    "report": TaskComplexity.EXPERT,
    "metadata": TaskComplexity.TRIVIAL,
    "summarize": TaskComplexity.MODERATE,
    "rerank": TaskComplexity.SIMPLE,
    # 通用默认
    "standard": TaskComplexity.MODERATE,
}

# Agent 名称 → 任务类型（从 Agent 名推断）
_AGENT_TO_TASK: Dict[str, str] = {
    "IngestionAgent": "ingestion",
    "ExtractionAgent": "extraction",
    "AnalysisAgent": "analysis",
    "ForecastAgent": "forecast",
    "ReportAgent": "report",
}


# ===================== 预算控制 =====================

@dataclass
class BudgetConfig:
    """预算配置"""
    max_cost_per_call: float = 0.1       # 单次调用最大金额（元）
    max_monthly_cost: float = 100.0      # 月度预算（元）
    prefer_cheap: bool = False           # 优先使用便宜模型
    enable_auto_degrade: bool = True     # 超预算时自动降级

    # 内部状态
    _monthly_spent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def can_use_tier(self, tier: ModelTier) -> bool:
        """判断当前预算下是否可以使用该分层模型"""
        if self._monthly_spent >= self.max_monthly_cost:
            return tier == ModelTier.LIGHT  # 超预算只能用最便宜的

        if not self.enable_auto_degrade:
            return True

        if self.prefer_cheap:
            # 倾向便宜模型：只能使用 STANDARD 及以下
            return tier in (ModelTier.LIGHT, ModelTier.STANDARD)

        return True

    def record_cost(self, model: str, tokens: int, cost: float = None):
        """记录一次调用成本"""
        if cost is None:
            cost_per_1k = _MODEL_COST_PER_1K.get(model, 0.001)
            cost = (tokens / 1000.0) * cost_per_1k

        with self._lock:
            self._monthly_spent += cost

    @property
    def monthly_spent(self) -> float:
        with self._lock:
            return self._monthly_spent

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.max_monthly_cost - self.monthly_spent)

    def reset_monthly(self):
        """重置月度预算（每月初调用）"""
        with self._lock:
            self._monthly_spent = 0.0


# ===================== 统计 =====================

@dataclass
class ModelRouterStats:
    """模型路由统计"""
    total_calls: int = 0
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    tokens_by_model: Dict[str, int] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    fallbacks: int = 0
    degradations: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_call(self, model: str, tokens: int = 0, cost: float = 0.0):
        with self._lock:
            self.total_calls += 1
            self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
            self.tokens_by_model[model] = self.tokens_by_model.get(model, 0) + tokens
            self.cost_by_model[model] = round(
                self.cost_by_model.get(model, 0.0) + cost, 6
            )

    def record_fallback(self):
        with self._lock:
            self.fallbacks += 1

    def record_degradation(self):
        with self._lock:
            self.degradations += 1

    def summary(self) -> Dict:
        with self._lock:
            total_cost = sum(self.cost_by_model.values())
            total_tokens = sum(self.tokens_by_model.values())
            return {
                "total_calls": self.total_calls,
                "calls_by_model": dict(self.calls_by_model),
                "tokens_by_model": dict(self.tokens_by_model),
                "total_tokens": total_tokens,
                "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model.items()},
                "total_cost": round(total_cost, 6),
                "fallbacks": self.fallbacks,
                "degradations": self.degradations,
            }


# ===================== 模型路由器 =====================

class ModelRouter:
    """智能模型路由 — 按任务复杂度自动选择模型

    使用示例::

        router = ModelRouter(api_key="xxx")
        # 自动选择
        llm = router.get_llm_for_agent("AnalysisAgent")
        response = llm.chat("分析财报...")

        # 手动指定
        router.override("AnalysisAgent", "qwen-max")
        # 移除覆盖
        router.remove_override("AnalysisAgent")

        # 查看统计
        print(router.stats.summary())
    """

    def __init__(
        self,
        api_key: str = None,
        budget_config: BudgetConfig = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
        self.budget = budget_config or BudgetConfig()
        self.stats = ModelRouterStats()

        # 懒加载缓存: model_name → DashScopeLLM
        self._clients: Dict[str, DashScopeLLM] = {}
        self._clients_lock = threading.Lock()

        # 手动覆盖: agent_name → model_name
        self._overrides: Dict[str, str] = {}
        self._overrides_lock = threading.Lock()

        # 任务类型覆盖: task_type → model_name
        self._task_overrides: Dict[str, str] = {}

    # ---- 获取 LLM ----

    def get_llm(
        self,
        task_type: str = "standard",
        complexity: TaskComplexity = None,
    ) -> DashScopeLLM:
        """根据任务类型获取合适的 LLM 客户端

        Args:
            task_type: 任务类型标识 (如 "analysis", "ingestion", 等)
            complexity: 显式指定复杂度（优先级高于 task_type 推断）

        Returns:
            DashScopeLLM 实例
        """
        # 1. 检查任务类型覆盖
        if task_type in self._task_overrides:
            model_name = self._task_overrides[task_type]
            logger.debug(f"[ModelRouter] 任务覆盖: {task_type} → {model_name}")
            return self._get_or_create_client(model_name)

        # 2. 确定复杂度
        if complexity is None:
            complexity = TASK_COMPLEXITY_MAP.get(
                task_type, TaskComplexity.MODERATE
            )

        # 3. 复杂度 → 模型分层
        tier = COMPLEXITY_TO_TIER[complexity]

        # 4. 预算检查 — 可能降级
        original_tier = tier
        while not self.budget.can_use_tier(tier):
            # 降级到下一层
            tier = self._degrade_tier(tier)
            if tier == original_tier:
                # 无法降级（已经是 LIGHT），使用 LIGHT
                break
        if tier != original_tier:
            logger.warning(
                f"[ModelRouter] 预算降级: {original_tier.value} → {tier.value}"
            )
            self.stats.record_degradation()

        # 5. 获取模型名
        model_name = TIER_MODEL_MAP[tier]

        logger.debug(
            f"[ModelRouter] 路由: task={task_type}, "
            f"complexity={complexity.value}, tier={tier.value}, model={model_name}"
        )
        return self._get_or_create_client(model_name)

    def get_llm_for_agent(self, agent_name: str) -> DashScopeLLM:
        """为特定 Agent 获取 LLM — 通过 agent 名推断复杂度

        Args:
            agent_name: Agent 名称 (如 "AnalysisAgent")

        Returns:
            DashScopeLLM 实例
        """
        # 1. 检查 Agent 级覆盖
        with self._overrides_lock:
            if agent_name in self._overrides:
                model_name = self._overrides[agent_name]
                logger.debug(f"[ModelRouter] Agent覆盖: {agent_name} → {model_name}")
                return self._get_or_create_client(model_name)

        # 2. 推断任务类型
        task_type = _AGENT_TO_TASK.get(agent_name, "standard")

        return self.get_llm(task_type=task_type)

    # ---- 覆盖管理 ----

    def override(self, agent_name: str, model: str):
        """强制某 Agent 使用指定模型

        Args:
            agent_name: Agent 名称
            model: 模型名 (如 "qwen-max")
        """
        with self._overrides_lock:
            self._overrides[agent_name] = model
            logger.info(f"[ModelRouter] 覆盖: {agent_name} → {model}")

    def remove_override(self, agent_name: str):
        """移除 Agent 的模型覆盖"""
        with self._overrides_lock:
            self._overrides.pop(agent_name, None)
            logger.info(f"[ModelRouter] 移除覆盖: {agent_name}")

    def override_task(self, task_type: str, model: str):
        """强制某类任务使用指定模型

        Args:
            task_type: 任务类型 (如 "analysis")
            model: 模型名
        """
        self._task_overrides[task_type] = model
        logger.info(f"[ModelRouter] 任务覆盖: {task_type} → {model}")

    def remove_task_override(self, task_type: str):
        """移除任务类型覆盖"""
        self._task_overrides.pop(task_type, None)

    # ---- 回退 ----

    def fallback(self, from_model: str, error: Exception) -> DashScopeLLM:
        """失败时回退到更轻量模型

        当某个模型调用失败时，自动尝试使用更轻量的模型。
        回退链: qwen3-235b → qwen-max → qwen-plus → qwen-turbo

        Args:
            from_model: 当前失败的模型
            error: 异常信息

        Returns:
            回退后的 DashScopeLLM 实例

        Raises:
            RuntimeError: 如果已经是 qwen-turbo，无法继续回退
        """
        # 找到当前模型对应的 tier
        current_tier = None
        for tier, model_name in TIER_MODEL_MAP.items():
            if model_name == from_model or from_model in (model_name,):
                current_tier = tier
                break

        if current_tier is None:
            # 无法识别模型，尝试用最轻的
            logger.warning(
                f"[ModelRouter] 无法识别模型 {from_model}，回退到 qwen-turbo"
            )
            self.stats.record_fallback()
            return self._get_or_create_client("qwen-turbo")

        # 找到下一个更轻的 tier
        fallback_tier = self._degrade_tier(current_tier)
        fallback_model = TIER_MODEL_MAP[fallback_tier]

        if fallback_tier == current_tier:
            # 已经是 LIGHT，无法回退
            raise RuntimeError(
                f"[ModelRouter] 模型 {from_model} 调用失败且已是最终回退层，"
                f"无法继续回退。原始错误: {error}"
            )

        logger.warning(
            f"[ModelRouter] 回退: {from_model} → {fallback_model} "
            f"(原因: {error})"
        )
        self.stats.record_fallback()
        return self._get_or_create_client(fallback_model)

    # ---- 统计 ----

    def record_usage(self, model: str, tokens: int, cost: float = None):
        """记录一次模型调用"""
        if cost is None:
            cost_per_1k = _MODEL_COST_PER_1K.get(model, 0.001)
            cost = (tokens / 1000.0) * cost_per_1k

        self.stats.record_call(model, tokens, cost)
        self.budget.record_cost(model, tokens, cost)

    def get_stats(self) -> Dict:
        """获取路由统计摘要"""
        return self.stats.summary()

    def get_budget_status(self) -> Dict:
        """获取预算状态"""
        return {
            "monthly_spent": round(self.budget.monthly_spent, 6),
            "monthly_budget": self.budget.max_monthly_cost,
            "remaining": round(self.budget.budget_remaining, 6),
            "prefer_cheap": self.budget.prefer_cheap,
            "auto_degrade": self.budget.enable_auto_degrade,
        }

    # ---- 内部方法 ----

    def _get_or_create_client(self, model_name: str) -> DashScopeLLM:
        """懒加载获取或创建 LLM 客户端"""
        if model_name not in self._clients:
            with self._clients_lock:
                # 双重检查
                if model_name not in self._clients:
                    logger.info(f"[ModelRouter] 创建客户端: {model_name}")
                    self._clients[model_name] = DashScopeLLM(
                        api_key=self.api_key,
                        model=model_name,
                    )
        return self._clients[model_name]

    @staticmethod
    def _degrade_tier(tier: ModelTier) -> ModelTier:
        """降级到下一层更轻量的模型"""
        if tier == ModelTier.LIGHT:
            return ModelTier.LIGHT
        idx = _TIER_FALLBACK_ORDER.index(tier)
        if idx < len(_TIER_FALLBACK_ORDER) - 1:
            return _TIER_FALLBACK_ORDER[idx + 1]
        return ModelTier.LIGHT
