"""
架构三: Reflection — ReAct 反思循环 + 多维评分 + 六层防幻觉

核心设计:
1. ReAct 循环 = Think → Retrieve → Act → Observe → Judge
2. 多维度置信度评分: 检索轮次、结果数量、相关度、一致性
3. 六层递进式防幻觉校验
4. 与业务脱钩: 只定义评分维度和检查层级，不绑定领域

两大组件:
- ReflectionLoop: ReAct 反思 + 多轮检索 + 置信度评估
- HallucinationGuard: 六层防幻觉中间件
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


# ===================== ReAct 循环 =====================

class ActionType(Enum):
    RETRIEVE = "retrieve"           # 检索知识库
    REFINE_QUERY = "refine_query"   # 优化查询
    ANALYZE = "analyze"             # 分析数据
    SYNTHESIZE = "synthesize"       # 综合答案
    FINISH = "finish"               # 完成
    RETRY = "retry"                 # 重试


@dataclass
class ThoughtStep:
    """ReAct 循环中每一步的思考记录"""
    step: int
    thought: str                     # 当前思考
    action: ActionType               # 决定动作
    action_input: str                # 动作输入
    observation: str = ""            # 动作结果观察
    reflection: str = ""             # 对结果的反思

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "thought": self.thought,
            "action": self.action.value,
            "action_input": self.action_input,
            "observation": self.observation,
            "reflection": self.reflection,
        }


@dataclass
class ReflectionState:
    """反思状态 — 维护多维度评分"""
    task: str = ""
    current_query: str = ""
    retrieved_contexts: List[Dict] = field(default_factory=list)
    synthesis: str = ""
    # 多维度评分
    confidence: float = 0.0          # 综合置信度
    retrieval_quality: float = 0.0   # 检索质量分
    consistency_score: float = 0.0   # 一致性分
    completeness_score: float = 0.0  # 完整性分
    citation_score: float = 0.0      # 引用准确度分
    # 控制
    should_continue: bool = True
    max_steps_reached: bool = False


@dataclass
class ReflectionConfig:
    max_retrievals: int = 3           # 最多检索次数
    retrieval_top_k: int = 5          # 每次检索数量
    max_steps: int = 6                # 最大循环步数
    min_confidence: float = 0.6       # 最低置信度阈值
    min_relevance: float = 0.3        # 最低相关度
    enable_self_reflection: bool = True


class ReflectionLoop(ABC):
    """
    Reflection — ReAct 反思循环引擎

    流程: Think → Act → Observe → Judge → (loop/stop)

    子类实现:
    - _do_retrieve(): 具体检索逻辑
    - _do_synthesize(): 答案综合逻辑
    - _assess_confidence(): 自定义置信度评估
    """

    def __init__(self, config: Optional[ReflectionConfig] = None):
        self.config = config or ReflectionConfig()
        self.state: Optional[ReflectionState] = None
        self.thought_history: List[ThoughtStep] = []

    # ========== ReAct 主循环 ==========

    def run(self, task: str, context: Optional[Dict] = None) -> Dict:
        """运行 Reflection 循环，返回最终答案 + 多维评分"""
        self.state = ReflectionState(task=task)
        self.thought_history = []

        step = 0
        while self.state.should_continue and step < self.config.max_steps:
            step += 1

            # 1. Think — 分析当前状态
            thought = self._think(step)

            # 2. Act — 执行动作
            observation = self._act(thought)

            # 3. 记录
            thought.observation = observation
            self.thought_history.append(thought)

            # 4. Judge — 反思 + 多维评分
            should_continue, reason = self._judge(thought)
            self.state.should_continue = should_continue

        # 最终综合
        final = self._synthesize_final()
        return {
            "answer": final,
            "confidence": self.state.confidence,
            "scores": {
                "retrieval_quality": self.state.retrieval_quality,
                "consistency": self.state.consistency_score,
                "completeness": self.state.completeness_score,
                "citation": self.state.citation_score,
            },
            "steps": len(self.thought_history),
            "thought_chain": [t.to_dict() for t in self.thought_history],
        }

    def _think(self, step: int) -> ThoughtStep:
        """Think 阶段: 决定下一步动作"""
        if self.state.synthesis and self.state.confidence >= self.config.min_confidence:
            return ThoughtStep(step=step, thought="信息充足，结束循环",
                               action=ActionType.FINISH, action_input="done")

        if not self.state.retrieved_contexts or step <= len(self.state.retrieved_contexts) + 1:
            return ThoughtStep(step=step, thought="需要检索更多信息",
                               action=ActionType.RETRIEVE, action_input=self._next_query())

        return ThoughtStep(step=step, thought="综合已有信息",
                           action=ActionType.SYNTHESIZE, action_input="synthesize")

    def _act(self, thought: ThoughtStep) -> str:
        """Act 阶段: 执行动作"""
        if thought.action == ActionType.RETRIEVE:
            return self._do_retrieve(thought.action_input)
        elif thought.action == ActionType.SYNTHESIZE:
            return self._do_synthesize()
        elif thought.action == ActionType.FINISH:
            return self.state.synthesis
        return "unknown_action"

    def _judge(self, thought: ThoughtStep) -> Tuple[bool, str]:
        """Judge 阶段: 多维评分 + 停止决策"""
        if thought.action == ActionType.SYNTHESIZE and self.state.synthesis:
            if self.state.confidence >= self.config.min_confidence:
                return False, f"置信度{self.state.confidence:.2f}达标，停止"
        if thought.step >= self.config.max_steps:
            self.state.max_steps_reached = True
            return False, "达到最大步数"
        if len(self.state.retrieved_contexts) >= self.config.max_retrievals:
            return False, "检索次数达上限"
        return True, "继续"

    # ========== 子类必须实现 ==========

    @abstractmethod
    def _do_retrieve(self, query: str) -> str:
        """执行一次检索"""
        pass

    def _do_synthesize(self) -> str:
        """综合检索结果为答案"""
        self.state.synthesis = "（子类实现综合逻辑）"
        self.state.confidence = self._assess_confidence()
        return self.state.synthesis

    def _assess_confidence(self) -> float:
        """多维度置信度评估（子类可重写）"""
        n = sum(len(ctx) for ctx in self.state.retrieved_contexts)
        confidence = 0.3 + min(0.2, n * 0.05) + min(0.3, 0.1 * len(self.state.retrieved_contexts))
        return min(0.95, confidence)

    def _next_query(self) -> str:
        """生成下一轮查询（子类可重写）"""
        return self.state.task

    def _synthesize_final(self) -> str:
        """最终综合"""
        if not self.state.synthesis:
            self._do_synthesize()
        return self.state.synthesis or "无法生成回答"


# ===================== 六层防幻觉中间件 =====================

class HallucinationGuard:
    """
    六层递进式防幻觉校验

    L1: 来源验证 — 每个断言必须能追溯到检索源
    L2: 一致性检查 — 回答内部不矛盾
    L3: 事实核查 — 关键数字/日期与来源一致
    L4: 完整性检查 — 是否遗漏来源中的关键信息
    L5: 引用准确性 — 是否有明确的引用标记
    L6: 综合评分 — 加权汇总输出总分

    所有维度与领域无关 — 纯文本逻辑检查
    """

    LAYER_WEIGHTS = {
        "L1_source_verification": 0.25,
        "L2_consistency":          0.15,
        "L3_fact_check":           0.20,
        "L4_completeness":         0.15,
        "L5_citation_accuracy":    0.15,
        "L6_overall":              0.10,
    }

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def check(self, answer: str, sources: List[Dict]) -> Dict:
        """执行六层全量检查"""
        checks = {}
        checks["L1_source_verification"] = self._l1_source_verification(answer, sources)
        checks["L2_consistency"] = self._l2_consistency(answer)
        checks["L3_fact_check"] = self._l3_fact_check(answer, sources)
        checks["L4_completeness"] = self._l4_completeness(answer, sources)
        checks["L5_citation_accuracy"] = self._l5_citation_accuracy(answer, sources)
        checks["L6_overall"] = self._l6_compute_overall(checks)

        passed = checks["L6_overall"]["score"] >= self.threshold
        return {
            "passed": passed,
            "overall_score": checks["L6_overall"]["score"],
            "risk": self._risk_level(checks["L6_overall"]["score"], checks),
            "checks": checks,
            "unverified": self._collect_unverified(checks),
        }

    def precheck(self, answer: str, sources: List[Dict]) -> Dict:
        """快速预检（仅 L1 + L3）"""
        l1 = self._l1_source_verification(answer, sources)
        l3 = self._l3_fact_check(answer, sources)
        score = l1["score"] * 0.6 + l3["score"] * 0.4
        return {"quick_score": score, "warning": score < 0.5}

    # ---------- L1: 来源验证 ----------
    def _l1_source_verification(self, answer: str, sources: List[Dict]) -> Dict:
        if not sources:
            return {"score": 0.0, "passed": False, "unverified": [answer[:100]]}
        sentences = [s.strip() for s in re.split(r'[。！？\n]', answer) if len(s.strip()) > 10]
        all_text = " ".join(s.get("text", "") for s in sources).lower()

        verified, unverified = 0, []
        for sent in sentences:
            words = set(re.findall(r'\w+', sent.lower()))
            if len(words) < 3:
                continue
            if sum(1 for w in words if w in all_text) / len(words) >= 0.3:
                verified += 1
            else:
                unverified.append(sent[:80])
        total = max(verified + len(unverified), 1)
        return {"score": verified / total, "passed": verified / total >= 0.6,
                "verified": verified, "unverified": unverified}

    # ---------- L2: 一致性 ----------
    def _l2_consistency(self, answer: str) -> Dict:
        pairs = [("增长", "下降"), ("盈利", "亏损"), ("增加", "减少"),
                 ("上升", "下跌"), ("利好", "利空")]
        sentences = re.split(r'[。！？\n]', answer)
        contradictions = []
        for a, b in pairs:
            if any(a in s for s in sentences) and any(b in s for s in sentences):
                contradictions.append(f"同时提及'{a}'和'{b}'")
        score = 1.0 - min(0.5, len(contradictions) * 0.1)
        return {"score": score, "passed": score >= 0.8, "contradictions": contradictions}

    # ---------- L3: 事实核查 ----------
    def _l3_fact_check(self, answer: str, sources: List[Dict]) -> Dict:
        if not sources:
            return {"score": 0.0, "passed": False}
        numbers = re.findall(r'\b\d{2,}(?:\.\d+)?[%％]?\b', answer)
        all_text = " ".join(s.get("text", "") for s in sources)
        verified = sum(1 for n in numbers if n in all_text)
        total = max(len(numbers), 1)
        return {"score": verified / total, "passed": verified / total >= 0.5,
                "facts_total": total, "facts_verified": verified}

    # ---------- L4: 完整性 ----------
    def _l4_completeness(self, answer: str, sources: List[Dict]) -> Dict:
        if not sources or len(sources) < 2:
            return {"score": 0.8, "passed": True}
        covered = 0
        for src in sources[:5]:
            text = src.get("text", "")
            first_sent = re.split(r'[。！？\n]', text)[0][:50]
            words = set(re.findall(r'\w+', first_sent.lower()))
            if words and sum(1 for w in words if w in answer.lower()) / len(words) >= 0.3:
                covered += 1
        score = covered / len(sources[:5]) if sources else 1.0
        return {"score": score, "passed": score >= 0.5}

    # ---------- L5: 引用准确性 ----------
    def _l5_citation_accuracy(self, answer: str, sources: List[Dict]) -> Dict:
        if not sources:
            return {"score": 0.0, "passed": False}
        has_ref = bool(re.findall(r'\[(?:ref-)?\d+\]', answer))
        has_marker = any(m in answer for m in ["来源", "引用", "参考", "根据", "数据显示"])
        score = 0.9 if has_ref else (0.7 if has_marker else 0.3)
        return {"score": score, "passed": score >= 0.5, "has_citations": has_ref or has_marker}

    # ---------- L6: 综合 ----------
    def _l6_compute_overall(self, checks: Dict) -> Dict:
        total = 0.0
        details = {}
        for layer, w in self.LAYER_WEIGHTS.items():
            if layer in checks:
                score = checks[layer].get("score", 0)
                total += score * w
                details[layer] = score
        return {"score": total, "passed": total >= self.threshold, "details": details}

    # ---------- 辅助 ----------
    def _risk_level(self, score: float, checks: Dict) -> str:
        uv = sum(len(checks.get(k, {}).get("unverified", [])) for k in checks)
        if score >= 0.8 and uv == 0:
            return "low"
        elif score >= 0.6 and uv <= 1:
            return "medium"
        return "high"

    def _collect_unverified(self, checks: Dict) -> List[str]:
        result = []
        for k in checks:
            result.extend(checks[k].get("unverified", []))
        return list(set(result))
