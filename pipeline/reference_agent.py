"""
Reference Agent - 引用验证 + 防幻觉校验

功能:
1. 验证回答中每个断言是否可追溯到检索结果
2. 六层防幻觉校验
3. 生成带引用的可信回答
"""
from typing import Dict, Any, List, Optional
import logging

from anti_hallucination.middleware import HallucinationMiddleware

logger = logging.getLogger(__name__)


class ReferenceAgent:
    """
    引用验证 Agent

    职责:
    1. 逐句验证是否有来源支撑
    2. 标注无来源的断言
    3. 生成 citation map
    4. 计算 hallucination risk
    """

    def __init__(self):
        self.middleware = HallucinationMiddleware()

    def verify(
        self,
        answer: str,
        sources: List[Dict],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        验证回答

        Args:
            answer: AnalyzerAgent 生成的回答
            sources: 检索到的源文档
            context: 上下文

        Returns:
            {
                "verified_answer": 验证后的回答,
                "citations": 引用列表,
                "hallucination_risk": "low/medium/high",
                "confidence": 置信度,
                "passed": 是否通过验证,
                "checks": 各层检查结果
            }
        """
        context = context or {}

        # 1. 通过六层防幻觉中间件
        check_result = self.middleware.full_check(answer, sources)

        # 2. 生成引用
        citations = self._build_citations(answer, sources)

        # 3. 标注无来源断言
        verified_answer = self._annotate(answer, check_result)

        # 4. 计算风险等级
        risk = self._assess_risk(check_result)

        confidence = check_result.get("overall_score", 0.0)
        passed = check_result.get("passed", False) and confidence >= 0.6

        logger.info(
            f"ReferenceAgent: passed={passed}, risk={risk}, "
            f"confidence={confidence:.2f}"
        )

        return {
            "verified_answer": verified_answer,
            "citations": citations,
            "hallucination_risk": risk,
            "confidence": confidence,
            "passed": passed,
            "checks": check_result,
        }

    def _build_citations(self, answer: str, sources: List[Dict]) -> List[Dict]:
        """构建引用列表"""
        citations = []
        for i, src in enumerate(sources[:5], 1):
            citations.append({
                "id": f"ref-{i}",
                "text": src.get("text", "")[:100],
                "score": src.get("score", 0),
                "source": src.get("metadata", {}).get("source", "unknown"),
            })
        return citations

    def _annotate(self, answer: str, check_result: Dict) -> str:
        """标注无来源的断言"""
        unverified = check_result.get("unverified_claims", [])
        if not unverified:
            return answer

        # 在回答末尾附加警告
        warning = "\n\n---\n[!] 以下断言未经充分验证:\n"
        for claim in unverified[:3]:
            warning += f"- {claim}\n"

        return answer + warning

    def _assess_risk(self, check_result: Dict) -> str:
        """评估风险等级"""
        score = check_result.get("overall_score", 0)
        unverified = len(check_result.get("unverified_claims", []))

        if score >= 0.8 and unverified == 0:
            return "low"
        elif score >= 0.6 and unverified <= 1:
            return "medium"
        else:
            return "high"
