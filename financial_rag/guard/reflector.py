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


# ===================== 四层防幻觉中间件 =====================

def _tokenize_zh(text: str) -> List[str]:
    """jieba 分词，无 jieba 时回退到按字分"""
    try:
        import jieba
        return [w for w in jieba.cut(text) if len(w.strip()) > 1]
    except ImportError:
        return [c for c in text if c.strip()]


# 常见停用词（功能词、标点、虚词）
_STOP_WORDS = frozenset(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 "
    "没有 看 好 自己 这 他 她 它 们 那 被 从 以 对 而 与 及 或 但 还 其 之 "
    "为 于 把 向 让 给 用 过 能 可 应 将 所 如 果 因 此 等 且 已 又 再 更 最".split()
)

# 数字+单位 正则（匹配 "50.3亿元"、"20%"、"2024年" 等）
_NUM_UNIT_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*([%％万亿元亿万美元美元美刀港元港币人民币块个台套条家万人人次份月年日季])?',
    re.UNICODE,
)


class HallucinationGuard:
    """
    四层防幻觉校验 — 每层做实，分数透明

    L1: 来源锚定 — jieba 分词 + token 重叠率，逐句追溯到检索源
    L2: 数值一致 — 提取 answer 中的「数字+单位」对，交叉比对 source
    L3: 引用完整 — [N] 标记存在且 N 对应有效来源
    L4: 结构规范 — 输出是否包含预期段落（摘要/要点/风险等）

    接口向后兼容: check() / precheck() / format_report()
    """

    LAYER_WEIGHTS = {
        "L1_source_grounding": 0.35,
        "L2_numerical_fidelity": 0.25,
        "L3_citation_integrity": 0.20,
        "L4_structure_compliance": 0.20,
    }

    # 来源锚定阈值：句子 token 与某 source 重叠 >= 此值视为"锚定"
    GROUNDING_THRESHOLD = 0.15

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    # ================== 公共接口 ==================

    def check(self, answer: str, sources: List[Dict]) -> Dict:
        """执行四层全量检查"""
        checks = {}
        checks["L1_source_grounding"] = self._l1_source_grounding(answer, sources)
        checks["L2_numerical_fidelity"] = self._l2_numerical_fidelity(answer, sources)
        checks["L3_citation_integrity"] = self._l3_citation_integrity(answer, sources)
        checks["L4_structure_compliance"] = self._l4_structure_compliance(answer)
        checks["overall"] = self._compute_overall(checks)

        overall_score = checks["overall"]["score"]
        passed = overall_score >= self.threshold
        return {
            "passed": passed,
            "overall_score": overall_score,
            "risk": self._risk_level(overall_score, checks),
            "checks": checks,
            "unverified": checks.get("L1_source_grounding", {}).get("unanchored", []),
            "report": self.format_report(overall_score, checks),
        }

    def precheck(self, answer: str, sources: List[Dict]) -> Dict:
        """快速预检（仅 L1 + L2）"""
        l1 = self._l1_source_grounding(answer, sources)
        l2 = self._l2_numerical_fidelity(answer, sources)
        score = l1["score"] * 0.6 + l2["score"] * 0.4
        return {"quick_score": score, "warning": score < 0.5}

    # ================== L1: 来源锚定 ==================

    def _l1_source_grounding(self, answer: str, sources: List[Dict]) -> Dict:
        """每个声明是否能追溯到某篇 source"""
        if not sources:
            return {"score": 0.0, "passed": False,
                    "anchored": 0, "total": 0, "unanchored": []}

        # 预处理: 对每篇 source 做 jieba 分词
        source_tokens_list = []
        for s in sources:
            text = s.get("text", "")
            tokens = set(_tokenize_zh(text)) - _STOP_WORDS
            source_tokens_list.append(tokens)

        # 逐句检查
        sentences = [s.strip() for s in re.split(r'[。！？\n]', answer)
                     if len(s.strip()) > 8]
        if not sentences:
            return {"score": 1.0, "passed": True,
                    "anchored": 0, "total": 0, "unanchored": []}

        anchored, unanchored = 0, []
        for sent in sentences:
            sent_tokens = set(_tokenize_zh(sent)) - _STOP_WORDS
            if len(sent_tokens) < 2:
                anchored += 1  # 太短的跳过
                continue

            # 找最佳匹配的 source
            best_overlap = 0.0
            for src_tokens in source_tokens_list:
                if not src_tokens:
                    continue
                overlap = len(sent_tokens & src_tokens) / len(sent_tokens)
                best_overlap = max(best_overlap, overlap)

            if best_overlap >= self.GROUNDING_THRESHOLD:
                anchored += 1
            else:
                unanchored.append(sent[:80])

        total = len(sentences)
        ratio = anchored / total if total > 0 else 1.0
        return {
            "score": ratio,
            "passed": ratio >= 0.6,
            "anchored": anchored,
            "total": total,
            "unanchored": unanchored,
        }

    # ================== L2: 数值一致性 ==================

    def _l2_numerical_fidelity(self, answer: str, sources: List[Dict]) -> Dict:
        """answer 中的数字是否能在 source 中找到"""
        if not sources:
            # 没有 source 时，answer 里有数字就是问题
            ans_nums = self._extract_numbers(answer)
            return {"score": 0.0 if ans_nums else 1.0,
                    "passed": not bool(ans_nums),
                    "verified": 0, "total": len(ans_nums), "unmatched": ans_nums}

        all_source_text = " ".join(s.get("text", "") for s in sources)
        src_nums = set(self._extract_numbers(all_source_text))
        ans_nums = self._extract_numbers(answer)

        if not ans_nums:
            return {"score": 1.0, "passed": True,
                    "verified": 0, "total": 0, "unmatched": []}

        verified = sum(1 for n in ans_nums if n in src_nums)
        unmatched = [n for n in ans_nums if n not in src_nums]
        ratio = verified / len(ans_nums)
        return {
            "score": ratio,
            "passed": ratio >= 0.5,
            "verified": verified,
            "total": len(ans_nums),
            "unmatched": unmatched[:5],
        }

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """提取数字（含单位），返回标准化字符串列表"""
        matches = _NUM_UNIT_RE.findall(text)
        results = []
        for num_str, unit in matches:
            # 标准化: 去掉末尾的 .0
            try:
                num = float(num_str)
                normalized = str(int(num)) if num == int(num) else num_str
            except ValueError:
                normalized = num_str
            results.append(f"{normalized}{unit}")
        return results

    # ================== L3: 引用完整性 ==================

    def _l3_citation_integrity(self, answer: str, sources: List[Dict]) -> Dict:
        """检查 [N] 引用标记是否存在且对应有效来源"""
        # 找所有 [N] 标记
        citations = re.findall(r'\[(\d+)\]', answer)
        if not citations:
            # 没有 [N] 标记 — 检查是否有文字引用
            has_text_ref = any(m in answer for m in
                               ["来源", "引用", "参考", "根据", "数据显示",
                                "据报", "资料", "报告"])
            score = 0.6 if has_text_ref else 0.2
            return {"score": score, "passed": score >= 0.5,
                    "citations_found": 0, "valid": 0, "invalid": 0,
                    "has_text_reference": has_text_ref}

        n_sources = len(sources)
        valid, invalid = 0, 0
        for c in citations:
            idx = int(c)
            if 1 <= idx <= n_sources:
                valid += 1
            else:
                invalid += 1

        total = valid + invalid
        ratio = valid / total if total > 0 else 0.0
        # 有正确引用给基础分，invalid 扣分
        score = ratio * 0.9 + (0.1 if total >= 2 else 0.0)
        score = min(1.0, score)
        return {
            "score": score,
            "passed": score >= 0.5,
            "citations_found": total,
            "valid": valid,
            "invalid": invalid,
        }

    # ================== L4: 结构规范性 ==================

    def _l4_structure_compliance(self, answer: str) -> Dict:
        """输出是否包含预期的结构段落"""
        # 期望的结构标记（Markdown 标题 或 常见关键词段）
        expected_sections = [
            (r'(?:^|\n)#+\s*(?:摘要|概述|总结|Summary|Overview)', "摘要/概述"),
            (r'(?:^|\n)#+\s*(?:要点|关键|发现|Key|Finding)', "要点/发现"),
            (r'(?:^|\n)#+\s*(?:分析|Analysis|详情)', "分析/详情"),
            (r'(?:^|\n)#+\s*(?:风险|注意|提示|Risk|Warning|Caution)', "风险/提示"),
        ]

        found = []
        missing = []
        for pattern, name in expected_sections:
            if re.search(pattern, answer, re.IGNORECASE):
                found.append(name)
            else:
                missing.append(name)

        # 至少有 2 个结构段落算合格
        ratio = len(found) / len(expected_sections)
        score = min(1.0, ratio + 0.2) if len(found) >= 2 else ratio
        return {
            "score": score,
            "passed": score >= 0.5,
            "found_sections": found,
            "missing_sections": missing,
        }

    # ================== 综合 + 格式化 ==================

    def _compute_overall(self, checks: Dict) -> Dict:
        total = 0.0
        for layer, w in self.LAYER_WEIGHTS.items():
            if layer in checks:
                total += checks[layer].get("score", 0) * w
        return {"score": total, "passed": total >= self.threshold}

    def _risk_level(self, score: float, checks: Dict) -> str:
        unanchored = len(checks.get("L1_source_grounding", {}).get("unanchored", []))
        if score >= 0.8 and unanchored == 0:
            return "low"
        elif score >= 0.6 and unanchored <= 2:
            return "medium"
        return "high"

    def format_report(self, overall_score: float, checks: Dict) -> str:
        """生成用户可见的评分卡片 — 嵌入最终输出"""
        # 等级
        if overall_score >= 0.9:
            grade = "A"
        elif overall_score >= 0.8:
            grade = "B"
        elif overall_score >= 0.65:
            grade = "C"
        elif overall_score >= 0.5:
            grade = "D"
        else:
            grade = "F"

        lines = [f"\n---\n📊 **质量评分**: {grade} ({overall_score:.0%})"]

        # L1
        l1 = checks.get("L1_source_grounding", {})
        l1_score = l1.get("score", 0)
        anchored = l1.get("anchored", 0)
        total = l1.get("total", 0)
        lines.append(f"├ 来源锚定: {l1_score:.0%} — {anchored}/{total} 句有来源支撑")

        # L2
        l2 = checks.get("L2_numerical_fidelity", {})
        l2_score = l2.get("score", 0)
        verified = l2.get("verified", 0)
        l2_total = l2.get("total", 0)
        lines.append(f"├ 数值一致: {l2_score:.0%} — {verified}/{l2_total} 个数字与来源匹配")

        # L3
        l3 = checks.get("L3_citation_integrity", {})
        l3_score = l3.get("score", 0)
        valid = l3.get("valid", 0)
        cit_total = l3.get("citations_found", 0)
        if cit_total > 0:
            lines.append(f"├ 引用完整: {l3_score:.0%} — {valid}/{cit_total} 个引用标记正确")
        elif l3.get("has_text_reference"):
            lines.append(f"├ 引用完整: {l3_score:.0%} — 有文字引用（无 [N] 标记）")
        else:
            lines.append(f"├ 引用完整: {l3_score:.0%} — 无引用标记")

        # L4
        l4 = checks.get("L4_structure_compliance", {})
        l4_score = l4.get("score", 0)
        found = l4.get("found_sections", [])
        lines.append(f"└ 结构规范: {l4_score:.0%} — 包含 {len(found)}/4 个结构段落")

        # 警告
        warnings = []
        unanchored = l1.get("unanchored", [])
        if unanchored:
            for s in unanchored[:3]:
                warnings.append(f"无来源: \"{s}\"")
        unmatched = l2.get("unmatched", [])
        if unmatched:
            warnings.append(f"数值无来源: {', '.join(unmatched[:3])}")
        invalid = l3.get("invalid", 0)
        if invalid:
            warnings.append(f"{invalid} 个引用标记指向不存在的来源")
        missing = l4.get("missing_sections", [])
        if missing:
            warnings.append(f"缺少段落: {', '.join(missing)}")

        if warnings:
            lines.append("\n⚠️ **警告**:")
            for w in warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)
