"""
RAG 查询引擎 - 日志分析和知识库问答的核心引擎
"""
from typing import List, Dict, Optional, Any, Iterator
from dataclasses import dataclass
from enum import Enum

from llama_index.core import PromptTemplate
from llama_index.core.response.schema import Response

from config import config
from knowledge_base import KnowledgeBase, LogKnowledgeBase


class QueryType(Enum):
    """查询类型"""
    ANALYSIS = "analysis"        # 日志分析
    TROUBLESHOOT = "troubleshoot" # 故障排查
    EXPLANATION = "explanation"  # 解释说明
    SUMMARY = "summary"          # 摘要总结
    GENERAL = "general"          # 一般问答


@dataclass
class QueryResult:
    """查询结果"""
    answer: str
    sources: List[Dict]
    query_type: QueryType
    confidence: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class RAGEngine:
    """RAG 引擎 - 日志分析和知识库问答"""
    
    # 查询模板
    ANALYSIS_TEMPLATE = """你是一个专业的日志分析专家。请基于以下检索到的上下文信息，分析日志内容并回答问题。

## 上下文信息
{context_str}

## 用户问题
{query_str}

## 分析要求
1. 识别日志中的关键信息和异常模式
2. 解释错误原因（如果有错误）
3. 提供具体的解决方案或建议
4. 如果信息不足，请明确说明

请用中文回答，结构清晰，包含：
- 问题分析
- 根因（如适用）
- 解决方案/建议
"""

    TROUBLESHOOT_TEMPLATE = """你是一个专业的系统故障排查专家。请基于以下检索到的上下文信息，帮助排查问题。

## 上下文信息
{context_str}

## 用户问题
{query_str}

## 排查步骤
1. 分析日志中的错误信息和警告
2. 识别可能的原因
3. 提供逐步排查建议
4. 给出修复方案

请用中文回答，按优先级列出排查步骤和解决方案。
"""

    EXPLANATION_TEMPLATE = """你是一个技术文档专家。请基于以下检索到的上下文信息，解释相关概念或日志内容。

## 上下文信息
{context_str}

## 用户问题
{query_str}

## 解释要求
1. 用通俗易懂的语言解释
2. 提供相关背景知识
3. 如有代码或配置示例，请一并说明

请用中文回答，确保解释清晰易懂。
"""

    SUMMARY_TEMPLATE = """请基于以下检索到的上下文信息，对日志内容进行摘要总结。

## 上下文信息
{context_str}

## 用户要求
{query_str}

## 摘要要求
1. 总结主要事件和趋势
2. 突出关键问题和异常
3. 统计信息（如错误数量、频率等）

请用中文回答，结构清晰，重点突出。
"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.query_engine = None
        self._init_query_engine()
    
    def _init_query_engine(self):
        """初始化查询引擎"""
        if self.kb.index is None:
            raise ValueError("知识库索引未构建")
        
        self.query_engine = self.kb.get_query_engine()
    
    def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        query_lower = query.lower()
        
        # 故障排查关键词
        troubleshoot_keywords = ['怎么解决', '如何修复', '报错', '故障', '排查', 'troubleshoot', 'fix', 'error', 'solve']
        if any(kw in query_lower for kw in troubleshoot_keywords):
            return QueryType.TROUBLESHOOT
        
        # 分析关键词
        analysis_keywords = ['分析', '为什么', '原因', 'analyze', 'why', 'reason', 'cause']
        if any(kw in query_lower for kw in analysis_keywords):
            return QueryType.ANALYSIS
        
        # 解释关键词
        explanation_keywords = ['什么是', '解释', '说明', '什么是', 'what is', 'explain', 'meaning']
        if any(kw in query_lower for kw in explanation_keywords):
            return QueryType.EXPLANATION
        
        # 摘要关键词
        summary_keywords = ['总结', '摘要', '概况', 'summary', 'overview', '统计']
        if any(kw in query_lower for kw in summary_keywords):
            return QueryType.SUMMARY
        
        return QueryType.GENERAL
    
    def _get_prompt_template(self, query_type: QueryType) -> PromptTemplate:
        """获取对应查询类型的提示模板"""
        templates = {
            QueryType.ANALYSIS: self.ANALYSIS_TEMPLATE,
            QueryType.TROUBLESHOOT: self.TROUBLESHOOT_TEMPLATE,
            QueryType.EXPLANATION: self.EXPLANATION_TEMPLATE,
            QueryType.SUMMARY: self.SUMMARY_TEMPLATE,
            QueryType.GENERAL: self.ANALYSIS_TEMPLATE,
        }
        return PromptTemplate(templates.get(query_type, self.ANALYSIS_TEMPLATE))
    
    def query(self, question: str, query_type: Optional[QueryType] = None) -> QueryResult:
        """
        执行查询
        
        Args:
            question: 用户问题
            query_type: 查询类型（可选，自动检测）
        
        Returns:
            QueryResult: 查询结果
        """
        # 检测查询类型
        if query_type is None:
            query_type = self._detect_query_type(question)
        
        print(f"查询类型: {query_type.value}")
        
        # 更新查询引擎的提示模板
        prompt_template = self._get_prompt_template(query_type)
        self.query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})
        
        # 执行查询
        response = self.query_engine.query(question)
        
        # 提取来源信息
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                sources.append({
                    "text": node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text,
                    "score": getattr(node, 'score', 0),
                    "metadata": node.node.metadata
                })
        
        # 计算置信度（基于来源分数）
        confidence = 0.0
        if sources:
            scores = [s['score'] for s in sources if s['score'] is not None]
            if scores:
                confidence = sum(scores) / len(scores)
        
        return QueryResult(
            answer=str(response),
            sources=sources,
            query_type=query_type,
            confidence=confidence
        )
    
    def chat(self, message: str, chat_history: Optional[List[Dict]] = None) -> QueryResult:
        """
        聊天模式（支持上下文）
        
        Args:
            message: 用户消息
            chat_history: 聊天历史
        
        Returns:
            QueryResult: 查询结果
        """
        chat_engine = self.kb.get_chat_engine()
        
        # 如果有历史记录，先加载
        if chat_history:
            for item in chat_history:
                # 这里简化处理，实际应该使用 chat_engine 的内存功能
                pass
        
        response = chat_engine.chat(message)
        
        return QueryResult(
            answer=str(response),
            sources=[],  # 聊天模式可能不返回来源
            query_type=QueryType.GENERAL,
            confidence=0.8
        )
    
    def analyze_logs(self, focus: Optional[str] = None) -> QueryResult:
        """
        自动分析日志
        
        Args:
            focus: 分析重点（如特定服务、错误类型等）
        
        Returns:
            QueryResult: 分析结果
        """
        if focus:
            question = f"请分析日志中的 {focus} 相关问题，包括错误原因和解决方案"
        else:
            question = "请全面分析这些日志，识别所有错误、警告和异常，并提供解决方案"
        
        return self.query(question, QueryType.ANALYSIS)
    
    def troubleshoot_error(self, error_message: str) -> QueryResult:
        """
        针对特定错误进行故障排查
        
        Args:
            error_message: 错误信息
        
        Returns:
            QueryResult: 排查结果
        """
        question = f"日志中出现错误: {error_message}。请分析原因并提供解决方案。"
        return self.query(question, QueryType.TROUBLESHOOT)
    
    def summarize_logs(self, time_range: Optional[str] = None) -> QueryResult:
        """
        总结日志
        
        Args:
            time_range: 时间范围描述
        
        Returns:
            QueryResult: 总结结果
        """
        if time_range:
            question = f"请总结 {time_range} 的日志情况，包括主要事件、错误统计和趋势"
        else:
            question = "请总结这些日志的整体情况，包括主要事件、错误统计和关键问题"
        
        return self.query(question, QueryType.SUMMARY)
    
    def batch_query(self, questions: List[str]) -> List[QueryResult]:
        """批量查询"""
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results


class LogAnalyzer:
    """日志分析器 - 高级日志分析功能"""
    
    def __init__(self, log_kb: LogKnowledgeBase):
        self.log_kb = log_kb
        self.rag_engine = RAGEngine(log_kb)
    
    def get_overview(self) -> Dict:
        """获取日志概览"""
        stats = self.log_kb.get_stats()
        error_analysis = self.log_kb.get_error_analysis()
        
        return {
            "statistics": stats,
            "error_summary": error_analysis
        }
    
    def find_similar_errors(self, error_text: str, top_k: int = 5) -> List[Dict]:
        """查找相似的错误"""
        return self.log_kb.search_similar(error_text, top_k=top_k)
    
    def generate_report(self) -> str:
        """生成日志分析报告"""
        # 获取概览
        overview = self.get_overview()
        
        # 使用 RAG 生成详细分析
        summary_result = self.rag_engine.summarize_logs()
        analysis_result = self.rag_engine.analyze_logs()
        
        report = f"""# 日志分析报告

## 1. 基本信息
- 文档总数: {overview['statistics']['total_documents']}
- 索引状态: {'已构建' if overview['statistics']['index_built'] else '未构建'}

## 2. 错误摘要
{overview['error_summary']}

## 3. 日志总结
{summary_result.answer}

## 4. 详细分析
{analysis_result.answer}

---
报告生成时间: {__import__('datetime').datetime.now().isoformat()}
"""
        return report


# 便捷函数
def create_rag_engine(knowledge_base: KnowledgeBase) -> RAGEngine:
    """创建 RAG 引擎"""
    return RAGEngine(knowledge_base)


def quick_analyze_logs(log_entries: List[Any], question: str) -> QueryResult:
    """快速分析日志的便捷函数"""
    from knowledge_base import create_knowledge_base_from_logs
    
    # 创建知识库
    kb = create_knowledge_base_from_logs(log_entries)
    
    # 创建引擎并查询
    engine = RAGEngine(kb)
    return engine.query(question)
