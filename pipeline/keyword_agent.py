"""
Agent 2: 关键词 Agent - 提取关键词和意图
"""
from typing import Dict, Any, List, Optional
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class KeywordAgent:
    """
    关键词 Agent - 提取关键词和识别意图

    功能:
    1. 提取技术关键词(错误码、组件名、异常类型)
    2. 意图识别(error/performance/config/unknown)
    3. 生成结构化查询
    """

    # 技术关键词模式
    TECH_PATTERNS = {
        "error_code": re.compile(r'\b([A-Z]{2,6}[_-]?\d{3,6})\b'),       # ERROR_001, E1001
        "exception": re.compile(r'\b(\w+(?:Error|Exception|Fault|Failure))\b'),
        "component": re.compile(r'\b([a-z]+(?:Service|Manager|Handler|Controller|Module))\b'),
        "endpoint": re.compile(r'(?:GET|POST|PUT|DELETE)\s+([/\w]+)'),
        "ip_address": re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'),
        "port": re.compile(r'\bport\s*[:=]?\s*(\d+)\b', re.IGNORECASE),
        "database": re.compile(r'\b(mysql|postgres|mongodb|redis|elasticsearch|kafka|rabbitmq)\b', re.IGNORECASE),
    }

    # 意图关键词
    INTENT_PATTERNS = {
        "error": ["error", "exception", "failed", "failure", "crash", "panic", "fatal",
                  "错误", "异常", "失败", "崩溃"],
        "performance": ["timeout", "slow", "latency", "memory", "cpu", "disk", "throughput",
                       "超时", "缓慢", "内存", "性能"],
        "config": ["config", "configuration", "setting", "parameter", "env",
                  "配置", "参数", "设置"],
        "connection": ["connect", "connection", "network", "socket", "refused",
                      "连接", "网络"],
        "security": ["auth", "permission", "denied", "unauthorized", "forbidden",
                    "认证", "权限", "拒绝"],
    }

    def extract(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        提取关键词和意图

        Args:
            text: 清洗后的文本
            context: 额外上下文

        Returns:
            {
                "keywords": [...],      # 所有关键词
                "tech_keywords": {...},  # 按类型分组的技术关键词
                "intent": "...",         # 主要意图
                "intent_scores": {...},  # 意图分数
                "query_suggestions": [...],  # 建议的检索查询
            }
        """
        context = context or {}

        # 1. 提取技术关键词
        tech_keywords = self._extract_tech_keywords(text)

        # 2. 意图识别
        intent, intent_scores = self._detect_intent(text)

        # 3. 生成查询建议
        query_suggestions = self._generate_queries(text, tech_keywords, intent)

        # 4. 汇总所有关键词
        all_keywords = []
        for k_type, k_list in tech_keywords.items():
            all_keywords.extend(k_list)
        all_keywords = list(set(all_keywords))  # 去重

        logger.info(
            f"KeywordAgent: 提取{len(all_keywords)}个关键词, "
            f"意图={intent}({intent_scores.get(intent, 0):.2f})"
        )

        return {
            "keywords": all_keywords,
            "tech_keywords": tech_keywords,
            "intent": intent,
            "intent_scores": intent_scores,
            "query_suggestions": query_suggestions,
        }

    def _extract_tech_keywords(self, text: str) -> Dict[str, List[str]]:
        """提取技术关键词"""
        result = {}
        for name, pattern in self.TECH_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                result[name] = list(set(matches))
        return result

    def _detect_intent(self, text: str) -> tuple[str, Dict[str, float]]:
        """意图识别"""
        text_lower = text.lower()
        scores = {}

        for intent, keywords in self.INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = min(1.0, score / len(keywords) * 5)  # 归一化

        if not scores:
            return "general", {"general": 0.5}

        # 取最高分意图
        primary_intent = max(scores, key=scores.get)
        return primary_intent, scores

    def _generate_queries(
        self,
        text: str,
        tech_keywords: Dict[str, List[str]],
        intent: str
    ) -> List[str]:
        """生成检索查询建议"""
        queries = []

        # 基础查询：前200字符摘要
        summary = text[:200]
        queries.append(summary)

        # 错误码查询
        if tech_keywords.get("error_code"):
            codes = " ".join(tech_keywords["error_code"][:3])
            queries.append(f"错误码: {codes}")

        # 异常类型查询
        if tech_keywords.get("exception"):
            exceptions = " ".join(tech_keywords["exception"][:3])
            queries.append(f"异常类型: {exceptions}")

        # 组件相关查询
        if tech_keywords.get("component"):
            components = " ".join(tech_keywords["component"][:3])
            queries.append(f"相关组件: {components}")

        # 意图专项查询
        intent_queries = {
            "error": f"错误原因和解决方案",
            "performance": f"性能优化和资源配置",
            "config": f"配置参数和最佳实践",
            "connection": f"网络连接和通信问题",
            "security": f"安全策略和权限配置",
        }

        if intent in intent_queries:
            queries.append(intent_queries[intent])

        return queries[:5]  # 最多5条
