"""
架构三: HallucinationGuard — L1-L4 规则层防幻觉校验

核心设计:
1. 四层确定性规则校验: 零 LLM 依赖、零成本、毫秒级，供运行时每请求使用
2. 与业务脱钩: 只定义评分维度和检查层级，不绑定领域
3. 透明评分: 每层得分 + 理由嵌入最终输出

各层实现已拆分到独立模块:
- L1-L4 (规则层): guard/rule_layers.py

深度语义评判（LLM-as-judge）不在运行时路径，由离线 RAGAS faithfulness
评估承担: financial_rag/eval/ragas_judge.py（旧 L5/L6 LLM 层已淘汰删除）。
"""
from typing import List, Dict

# ===================== 规则层防幻觉中间件 =====================
# 各层实现已拆分到独立模块:
# - L1-L4 (规则层): guard/rule_layers.py

from .rule_layers import (
    l1_source_grounding,
    l2_numerical_fidelity,
    l3_citation_integrity,
    l4_structure_compliance,
)


class HallucinationGuard:
    """
    L1-L4 规则层防幻觉校验 — 确定性、零 LLM 依赖，分数透明

    L1: 来源锚定 — jieba 分词 + token 重叠率，逐句追溯到检索源
    L2: 数值一致 — 提取 answer 中的「数字+单位」对，交叉比对 source
    L3: 引用完整 — [N] 标记存在且 N 对应有效来源
    L4: 结构规范 — 输出是否包含预期段落（摘要/要点/风险等）

    接口向后兼容: check() / precheck() / format_report()
    """

    # 权重 = 原六层权重中 L1-L4 部分按原比例归一（L5/L6 已淘汰）
    LAYER_WEIGHTS = {
        "L1_source_grounding": 0.417,
        "L2_numerical_fidelity": 0.25,
        "L3_citation_integrity": 0.167,
        "L4_structure_compliance": 0.167,
    }

    # 来源锚定阈值：与 rule_layers.GROUNDING_THRESHOLD 保持一致（0.30，实测分离间隙）
    GROUNDING_THRESHOLD = 0.30

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    # ================== 公共接口 ==================

    def check(self, answer: str, sources: List[Dict], mode: str = "rag") -> Dict:
        """执行 L1-L4 规则层全量检查（确定性，无 LLM 调用）

        Args:
            mode: "rag" (RAG 查询) 或 "analysis" (深度分析，宽松引用+结构)
        """
        checks = {}
        checks["L1_source_grounding"] = l1_source_grounding(answer, sources)
        checks["L2_numerical_fidelity"] = l2_numerical_fidelity(answer, sources)
        checks["L3_citation_integrity"] = l3_citation_integrity(answer, sources, mode=mode)
        checks["L4_structure_compliance"] = l4_structure_compliance(answer, mode=mode)

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

        return "\n".join(lines)
