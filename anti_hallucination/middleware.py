"""
六层防幻觉中间件

层级设计:
  L1: 来源验证 - 每个断言必须有来源
  L2: 一致性检查 - 回答内部不自相矛盾
  L3: 事实核查 - 关键数字/日期与来源一致
  L4: 完整性检查 - 是否遗漏关键信息
  L5: 引用准确性 - 引用内容是否真实匹配
  L6: 综合评分 - 加权汇总
"""
from typing import Dict, Any, List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class HallucinationMiddleware:
    """
    六层防幻觉中间件

    每层返回 (passed, score, details)
    通过 weighted sum 计算总分
    """

    # 各层权重
    LAYER_WEIGHTS = {
        "L1_source_verification": 0.25,
        "L2_consistency": 0.15,
        "L3_fact_check": 0.20,
        "L4_completeness": 0.15,
        "L5_citation_accuracy": 0.15,
        "L6_overall": 0.10,
    }

    def __init__(self):
        self.threshold = 0.6  # 总阈值

    def full_check(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        执行全部六层检查

        Args:
            answer: 生成的回答
            sources: 检索到的来源

        Returns:
            完整检查结果
        """
        checks = {}

        # L1: 来源验证
        checks["L1_source_verification"] = self._check_source_verification(answer, sources)

        # L2: 一致性检查
        checks["L2_consistency"] = self._check_consistency(answer)

        # L3: 事实核查
        checks["L3_fact_check"] = self._check_facts(answer, sources)

        # L4: 完整性检查
        checks["L4_completeness"] = self._check_completeness(answer, sources)

        # L5: 引用准确性
        checks["L5_citation_accuracy"] = self._check_citation_accuracy(answer, sources)

        # L6: 综合评分
        overall = self._compute_overall(checks)
        checks["L6_overall"] = overall

        # 汇总
        passed = overall["score"] >= self.threshold

        unverified = []
        for check_name, check in checks.items():
            if check.get("unverified"):
                unverified.extend(check["unverified"])

        logger.info(
            f"防幻觉检查: passed={passed}, "
            f"score={overall['score']:.2f}, "
            f"unverified={len(unverified)}条"
        )

        return {
            "passed": passed,
            "overall_score": overall["score"],
            "checks": checks,
            "unverified_claims": list(set(unverified)),
        }

    def precheck(self, answer: str, sources: List[Dict]) -> Dict:
        """快速预检（仅 L1 + L3）"""
        l1 = self._check_source_verification(answer, sources)
        l3 = self._check_facts(answer, sources)
        quick_score = l1["score"] * 0.6 + l3["score"] * 0.4
        return {
            "quick_score": quick_score,
            "warning": quick_score < 0.5,
            "L1": l1,
            "L3": l3,
        }

    # ===== L1: 来源验证 =====
    def _check_source_verification(self, answer: str, sources: List[Dict]) -> Dict:
        """
        验证每个关键断言是否能在 sources 中找到支撑
        """
        if not sources:
            return {"score": 0.0, "passed": False, "unverified": [answer[:100]]}

        # 提取答案中的关键句子
        sentences = re.split(r'[。！？\n]', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        # 合并所有来源文本
        all_source_text = " ".join(s.get("text", "") for s in sources)

        verified = 0
        unverified = []

        for sent in sentences:
            # 简单检查：句子中的关键词是否出现在来源中
            words = set(re.findall(r'\w+', sent.lower()))
            if len(words) < 3:
                continue

            # 检查至少30%的关键词在来源中出现
            match_count = sum(1 for w in words if w in all_source_text.lower())
            if match_count / len(words) >= 0.3:
                verified += 1
            else:
                unverified.append(sent[:80])

        total = max(verified + len(unverified), 1)
        score = verified / total

        return {
            "score": score,
            "passed": score >= 0.6,
            "verified_count": verified,
            "unverified_count": len(unverified),
            "unverified": unverified,
        }

    # ===== L2: 一致性检查 =====
    def _check_consistency(self, answer: str) -> Dict:
        """
        检查回答内部是否自相矛盾
        """
        # 检查常见的矛盾模式
        contradictions = []

        patterns = [
            (r'成功', r'失败'),
            (r'正常', r'异常'),
            (r'增加', r'减少'),
            (r'启动', r'停止'),
        ]

        sentences = re.split(r'[。！？\n]', answer)
        for pat_a, pat_b in patterns:
            has_a = any(re.search(pat_a, s) for s in sentences)
            has_b = any(re.search(pat_b, s) for s in sentences)
            if has_a and has_b:
                # 同时出现可能矛盾，但不一定
                contradictions.append(f"同时提及'{pat_a}'和'{pat_b}'")

        score = 1.0 - min(0.5, len(contradictions) * 0.1)

        return {
            "score": score,
            "passed": score >= 0.8,
            "contradictions": contradictions,
        }

    # ===== L3: 事实核查 =====
    def _check_facts(self, answer: str, sources: List[Dict]) -> Dict:
        """
        检查回答中的关键数字、日期是否与来源一致
        """
        if not sources:
            return {"score": 0.0, "passed": False, "unverified": []}

        # 提取回答中的数字和日期
        numbers = re.findall(r'\b\d{2,}\b', answer)
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', answer)

        all_source_text = " ".join(s.get("text", "") for s in sources)

        # 检查数字是否在来源中出现
        verified_numbers = sum(1 for n in numbers if n in all_source_text)
        verified_dates = sum(1 for d in dates if d in all_source_text)

        total_facts = len(numbers) + len(dates)
        if total_facts == 0:
            score = 1.0  # 无事实数据，默认通过
        else:
            score = (verified_numbers + verified_dates) / total_facts

        return {
            "score": score,
            "passed": score >= 0.5,
            "facts_total": total_facts,
            "facts_verified": verified_numbers + verified_dates,
        }

    # ===== L4: 完整性检查 =====
    def _check_completeness(self, answer: str, sources: List[Dict]) -> Dict:
        """
        检查是否遗漏了来源中的关键信息
        """
        if not sources or len(sources) < 2:
            return {"score": 0.8, "passed": True}

        # 检查来源中的关键信息是否被回答覆盖
        source_key_info = []
        for src in sources[:5]:
            text = src.get("text", "")
            # 提取每个来源的第一句
            first_sent = re.split(r'[。！？\n]', text)[0]
            if len(first_sent) > 5:
                source_key_info.append(first_sent[:50])

        covered = 0
        for info in source_key_info:
            words = set(re.findall(r'\w+', info.lower()))
            if words:
                match = sum(1 for w in words if w in answer.lower())
                if match / len(words) >= 0.3:
                    covered += 1

        score = covered / len(source_key_info) if source_key_info else 1.0

        return {
            "score": score,
            "passed": score >= 0.5,
            "sources_covered": covered,
            "sources_total": len(source_key_info),
        }

    # ===== L5: 引用准确性 =====
    def _check_citation_accuracy(self, answer: str, sources: List[Dict]) -> Dict:
        """
        检查回答中是否准确引用了来源
        """
        if not sources:
            return {"score": 0.0, "passed": False}

        # 检查是否有引用标记
        has_citations = bool(re.findall(r'\[(?:ref-)?\d+\]', answer))
        has_source_markers = any(
            marker in answer for marker in ["来源", "引用", "参考", "根据"]
        )

        if has_citations:
            score = 0.9
        elif has_source_markers:
            score = 0.7
        elif len(sources) > 0:
            score = 0.3  # 有来源但没引用
        else:
            score = 0.0

        return {
            "score": score,
            "passed": score >= 0.5,
            "has_citations": has_citations,
            "has_source_markers": has_source_markers,
        }

    # ===== L6: 综合评分 =====
    def _compute_overall(self, checks: Dict) -> Dict:
        """加权汇总"""
        total = 0.0
        details = {}

        for layer, weight in self.LAYER_WEIGHTS.items():
            if layer in checks:
                score = checks[layer].get("score", 0)
                total += score * weight
                details[layer] = score

        return {
            "score": total,
            "passed": total >= self.threshold,
            "layer_details": details,
        }
