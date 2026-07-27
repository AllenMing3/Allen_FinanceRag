"""
架构三: HallucinationGuard — 六层防幻觉校验

核心设计:
1. 六层递进式防幻觉: 规则层(L1-L4) + LLM层(L5-L6)
2. 与业务脱钩: 只定义评分维度和检查层级，不绑定领域
3. 透明评分: 每层得分 + 理由嵌入最终输出

各层实现已拆分到独立模块:
- L1-L4 (规则层): guard/rule_layers.py
- L5 (LLM质疑): guard/llm_critique.py
- L6 (LLM协助): guard/llm_assist.py
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


# ===================== 六层防幻觉中间件 =====================
# 各层实现已拆分到独立模块:
# - L1-L4 (规则层): guard/rule_layers.py
# - L5 (LLM质疑): guard/llm_critique.py
# - L6 (LLM协助): guard/llm_assist.py

from .rule_layers import (
    l1_source_grounding,
    l2_numerical_fidelity,
    l3_citation_integrity,
    l4_structure_compliance,
)
from .llm_critique import llm_critique
from .llm_assist import llm_assist


class HallucinationGuard:
    """
    六层防幻觉校验 — 规则层(L1-L4) + LLM层(L5-L6)，分数透明

    L1: 来源锚定 — jieba 分词 + token 重叠率，逐句追溯到检索源
    L2: 数值一致 — 提取 answer 中的「数字+单位」对，交叉比对 source
    L3: 引用完整 — [N] 标记存在且 N 对应有效来源
    L4: 结构规范 — 输出是否包含预期段落（摘要/要点/风险等）
    L5: LLM质疑 — LLM 审查规则层结果，找出规则盲区（假阳性/假阴性）
    L6: LLM协助 — LLM 独立逐句扫描，发现规则抓不到的幻觉

    接口向后兼容: check() / precheck() / format_report()
    """

    LAYER_WEIGHTS = {
        "L1_source_grounding": 0.25,
        "L2_numerical_fidelity": 0.15,
        "L3_citation_integrity": 0.10,
        "L4_structure_compliance": 0.10,
        "L5_llm_critique": 0.20,
        "L6_llm_assist": 0.20,
    }

    # 来源锚定阈值：句子 token 与某 source 重叠 >= 此值视为"锚定"
    GROUNDING_THRESHOLD = 0.15

    # LLM 层触发阈值：规则层总分低于此值时触发 L5+L6
    LLM_TRIGGER_THRESHOLD = 0.85

    def __init__(self, threshold: float = 0.6, llm=None):
        self.threshold = threshold
        self._llm = llm

    # ================== 公共接口 ==================

    def check(self, answer: str, sources: List[Dict], mode: str = "rag") -> Dict:
        """执行六层全量检查（L1-L4 规则层 + L5-L6 LLM层）

        Args:
            mode: "rag" (RAG 查询) 或 "analysis" (深度分析，宽松引用+结构)
        """
        checks = {}
        checks["L1_source_grounding"] = l1_source_grounding(answer, sources)
        checks["L2_numerical_fidelity"] = l2_numerical_fidelity(answer, sources)
        checks["L3_citation_integrity"] = l3_citation_integrity(answer, sources, mode=mode)
        checks["L4_structure_compliance"] = l4_structure_compliance(answer, mode=mode)

        # L5+L6: LLM 层 — 条件触发，跳过时写入显式标记（禁止静默省略）
        rule_score = self._compute_rule_score(checks)
        if self._llm is not None and rule_score < self.LLM_TRIGGER_THRESHOLD:
            logger.info(f"[Guard] L5/L6 触发 (rule_score={rule_score:.2f} < {self.LLM_TRIGGER_THRESHOLD})")
            checks["L5_llm_critique"] = llm_critique(answer, sources, checks, self._llm)
            checks["L6_llm_assist"] = llm_assist(answer, sources, self._llm)
            logger.info(f"[Guard] L5={checks['L5_llm_critique'].get('score', 0):.2f}, L6={checks['L6_llm_assist'].get('score', 0):.2f}")
        else:
            if self._llm is None:
                reason = "LLM 未注入，LLM 层无法执行"
                logger.debug("L5/L6 skipped: no LLM injected")
            else:
                reason = f"规则层得分 {rule_score:.0%} ≥ {self.LLM_TRIGGER_THRESHOLD:.0%}，LLM 层未触发"
                logger.debug(f"L5/L6 skipped: rule score {rule_score:.2f} >= {self.LLM_TRIGGER_THRESHOLD}")
            checks["L5_llm_critique"] = {"score": 0.0, "skipped": True, "skip_reason": reason}
            checks["L6_llm_assist"] = {"score": 0.0, "skipped": True, "skip_reason": reason}

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
            "llm_layers_active": not checks.get("L5_llm_critique", {}).get("skipped", True),
        }

    def _compute_rule_score(self, checks: Dict) -> float:
        """仅计算 L1-L4 规则层的加权总分（用于 LLM 层触发判断）"""
        rule_weights = {
            "L1_source_grounding": 0.35,
            "L2_numerical_fidelity": 0.25,
            "L3_citation_integrity": 0.20,
            "L4_structure_compliance": 0.20,
        }
        total = 0.0
        for layer, w in rule_weights.items():
            if layer in checks:
                total += checks[layer].get("score", 0) * w
        return total

    def precheck(self, answer: str, sources: List[Dict]) -> Dict:
        """快速预检（仅 L1 + L2）"""
        l1 = l1_source_grounding(answer, sources)
        l2 = l2_numerical_fidelity(answer, sources)
        score = l1["score"] * 0.6 + l2["score"] * 0.4
        return {"quick_score": score, "warning": score < 0.5}

    # ================== 综合 + 格式化 ==================

    def _compute_overall(self, checks: Dict) -> Dict:
        # 归一化权重：只对实际执行的层求加权平均，跳过的层不参与计算
        active_weights = {
            k: w for k, w in self.LAYER_WEIGHTS.items()
            if k in checks and not checks[k].get("skipped")
        }
        total_weight = sum(active_weights.values())
        if total_weight == 0:
            return {"score": 0.0, "passed": False}
        total = 0.0
        for layer, w in active_weights.items():
            total += checks[layer].get("score", 0) * (w / total_weight)
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

        # L5 LLM质疑
        l5 = checks.get("L5_llm_critique")
        if l5:
            if l5.get("skipped"):
                lines.append(f"├ LLM质疑: 未执行 — {l5.get('skip_reason', '未知原因')}")
            else:
                l5_score = l5.get("score", 0)
                fp = l5.get("false_positives", [])
                fn = l5.get("false_negatives", [])
                lines.append(f"├ LLM质疑: {l5_score:.0%} — 发现 {len(fp)} 条假阳性, {len(fn)} 条假阴性")
                for item in (fp + fn)[:3]:
                    lines.append(f"  - {item}")

        # L6 LLM协助
        l6 = checks.get("L6_llm_assist")
        if l6:
            if l6.get("skipped"):
                lines.append(f"└ LLM协助: 未执行 — {l6.get('skip_reason', '未知原因')}")
            else:
                l6_score = l6.get("score", 0)
                unsupported = l6.get("unsupported_claims", [])
                fabricated = l6.get("fabricated_claims", [])
                total_claims = l6.get("total_claims", 0)
                supported = l6.get("supported_claims", 0)
                lines.append(f"└ LLM协助: {l6_score:.0%} — {supported}/{total_claims} 句有语义支撑")
                for item in fabricated[:3]:
                    lines.append(f"  - 编造: \"{item}\"")
                for item in unsupported[:3]:
                    lines.append(f"  - 无据: \"{item}\"")

        return "\n".join(lines)
