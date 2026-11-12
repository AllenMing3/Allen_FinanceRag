"""
guard/service.py — ReliabilityService 防幻觉统一入口（路线图方向二）

病根：HallucinationGuard.check() 曾散落 6 个调用点，每处自拼 sources、自定 mode、
自决触发 → 同一输出走不同入口分数不可复现；空输入时整层评分无痕消失。

本模块只收敛接线，Guard 引擎（reflector / rule_layers）一行不动：
- normalize_sources: raw 检索项 / {"text": ...} / 字符串 → 统一 [{"text": ...}]
- check: 空 answer / 空 sources → 显式 skipped 结构（不再 return {} 或 None）；
  正常 → guard.check 原始返回 + skipped/skip_reason 键
- mode 显式传参（"rag" / "analysis"），无隐式默认分歧

skipped 契约（序列化后前端显示「未校验」）：
    {"skipped": True, "skip_reason": "empty_answer"|"no_sources",
     "passed": False, "overall_score": 0.0, "risk": "unknown",
     "checks": {}, "unverified": [], "report": ""}
"""
import logging
from typing import Dict, List, Optional

from .reflector import HallucinationGuard

logger = logging.getLogger(__name__)

SKIP_EMPTY_ANSWER = "empty_answer"
SKIP_NO_SOURCES = "no_sources"

# 人读原因（报告/日志用；前端有自己的映射）
SKIP_REASON_LABELS = {
    SKIP_EMPTY_ANSWER: "无输出文本",
    SKIP_NO_SOURCES: "无检索来源",
}


def normalize_sources(sources: Optional[List]) -> List[Dict]:
    """把各调用点形态不一的 sources 归一为 Guard 契约 [{"text": ...}]。

    兼容：raw 检索项（含 meta/score 等额外键）、{"text": ...} 字典、纯字符串。
    空文本 / 纯空白 / 无 text 键的项一律过滤。
    """
    normalized: List[Dict] = []
    for item in sources or []:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text", "")
        else:
            continue
        if text and text.strip():
            normalized.append({"text": text})
    return normalized


class ReliabilityService:
    """防幻觉统一入口：source 归一 + mode 透传 + check + 显式 skipped。

    所有调用点（scoring_tools / analysis_router / kb_router / pipeline /
    conversation_tools / core.router CLI）必须经由此类，禁止直接实例化
    HallucinationGuard —— 同一输出走任何入口分数都可复现。
    """

    def __init__(self, threshold: float = 0.6):
        self._guard = HallucinationGuard(threshold=threshold)

    def check(self, answer: str, sources: Optional[List], mode: str = "rag") -> Dict:
        """执行 L1-L4 校验；空输入返回显式 skipped，绝不静默消失。

        Args:
            answer: 待校验的最终输出文本
            sources: 证据列表（形态见 normalize_sources）
            mode: "rag"（期望 [N] 引用 + Markdown）/ "analysis"（期望【】段落）
        """
        if not (answer or "").strip():
            return self._skipped(SKIP_EMPTY_ANSWER)
        normalized = normalize_sources(sources)
        if not normalized:
            return self._skipped(SKIP_NO_SOURCES)
        result = self._guard.check(answer, normalized, mode=mode)
        result["skipped"] = False
        result["skip_reason"] = ""
        return result

    @staticmethod
    def _skipped(reason: str) -> Dict:
        logger.info(f"[Reliability] 防幻觉校验跳过: {SKIP_REASON_LABELS.get(reason, reason)}")
        return {
            "skipped": True,
            "skip_reason": reason,
            "passed": False,
            "overall_score": 0.0,
            "risk": "unknown",
            "checks": {},
            "unverified": [],
            "report": "",
        }