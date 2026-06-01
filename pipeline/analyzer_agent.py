"""
Agent 3: 分析 Agent - 检索 + 推理 + 生成答案

核心能力:
1. Hybrid RAG (BM25 + Vector + RRF 融合)
2. Agentic 多轮检索
3. 防幻觉预处理
"""
from typing import Dict, Any, List, Optional
import logging

from anti_hallucination.hybrid_retriever import HybridRetriever
from anti_hallucination.middleware import HallucinationMiddleware

logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """
    分析 Agent - 核心推理引擎

    功能:
    1. Hybrid 检索 (BM25 + Vector + RRF)
    2. 多轮 Agentic 检索
    3. 答案合成
    """

    def __init__(self):
        self.retriever = HybridRetriever()
        self.hallucination_check = HallucinationMiddleware()

    def analyze(
        self,
        cleaned_text: str,
        keywords: List[str],
        intent: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        分析主流程

        Args:
            cleaned_text: 清洗后文本
            keywords: 关键词列表
            intent: 意图
            context: 上下文

        Returns:
            分析结果
        """
        context = context or {}

        # 1. 构建查询
        queries = self._build_queries(cleaned_text, keywords, intent)

        # 2. Hybrid 检索（多轮）
        all_sources = []
        retrieval_count = 0
        for query in queries[:4]:  # 最多4轮检索
            results = self.retriever.retrieve(
                query=query,
                top_k=10,
            )
            if results:
                all_sources.extend(results)
                retrieval_count += 1

        # 3. 去重排序
        all_sources = self._deduplicate_sources(all_sources)

        # 4. 合成答案
        answer = self._synthesize(cleaned_text, all_sources, intent, keywords)

        # 5. 防幻觉预检
        precheck = self.hallucination_check.precheck(answer, all_sources)

        return {
            "success": True,
            "answer": answer,
            "sources": all_sources,
            "retrieval_count": retrieval_count,
            "precheck": precheck,
        }

    def _build_queries(self, text: str, keywords: List[str], intent: str) -> List[str]:
        """构建多角度查询"""
        queries = [text[:300]]  # 原始文本截断

        # 关键词组合查询
        if keywords:
            queries.append(" ".join(keywords[:5]))

        # 意图专项查询
        intent_map = {
            "error": f"错误: {text[:200]} 原因 解决方案",
            "performance": f"性能: {text[:200]} 优化 配置",
            "config": f"配置: {text[:200]} 参数 设置",
            "connection": f"连接: {text[:200]} 网络 超时",
            "security": f"安全: {text[:200]} 权限 认证",
        }
        if intent in intent_map:
            queries.append(intent_map[intent])

        return queries

    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """去重并保留最高分的"""
        seen = {}
        for s in sources:
            text_hash = hash(s.get("text", "")[:100])
            if text_hash not in seen or s.get("score", 0) > seen[text_hash].get("score", 0):
                seen[text_hash] = s

        return sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)[:10]

    def _synthesize(
        self,
        text: str,
        sources: List[Dict],
        intent: str,
        keywords: List[str]
    ) -> str:
        """合成答案"""
        if not sources:
            return self._no_sources_answer(text, keywords, intent)

        parts = []

        # 标题
        intent_labels = {
            "error": "问题分析",
            "performance": "性能分析",
            "config": "配置分析",
            "connection": "连接分析",
            "security": "安全分析",
            "general": "综合分析",
        }
        parts.append(f"## {intent_labels.get(intent, '分析结果')}")

        # 核心发现
        top_sources = sources[:3]
        parts.append("\n### 核心发现")
        for i, s in enumerate(top_sources, 1):
            snippet = s.get("text", "")[:200]
            score = s.get("score", 0)
            parts.append(f"\n{i}. (相关度: {score:.2f}) {snippet}")

        # 如果有关键词
        if keywords:
            parts.append(f"\n### 关键信息\n{', '.join(keywords[:10])}")

        return "\n".join(parts)

    def _no_sources_answer(self, text: str, keywords: List[str], intent: str) -> str:
        """知识库无匹配时的回答"""
        return (
            f"## 分析结果\n\n"
            f"知识库中未找到相关记录。\n\n"
            f"### 检测到的关键信息\n{', '.join(keywords) if keywords else '无'}\n\n"
            f"### 输入摘要\n{text[:300]}...\n\n"
            f"建议: 扩充知识库或提供更多上下文。"
        )
