"""
日志分析 Agent - 负责深度分析日志内容
"""
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentContext, AgentResponse
from knowledge_base import LogKnowledgeBase
from rag_engine import RAGEngine


class LogAnalystAgent(BaseAgent):
    """
    日志分析 Agent
    
    职责：
    1. 分析日志模式和趋势
    2. 识别异常和错误根因
    3. 关联多个日志事件
    4. 提供诊断结论
    """
    
    def __init__(self):
        super().__init__(
            name="LogAnalystAgent",
            description="深度分析日志内容，识别问题和根因"
        )
        self.knowledge_base: LogKnowledgeBase = None
        self.rag_engine: RAGEngine = None
        self.analysis_results: Dict = {}
    
    def process(self, context: AgentContext) -> AgentResponse:
        """
        分析日志
        
        期望 context.parsed_data 包含已解析的日志条目
        """
        if not context.parsed_data:
            return AgentResponse(
                success=False,
                message="没有可分析的日志数据，请先调用 LogParserAgent"
            )
        
        # 构建知识库（如果还没有）
        if not self.knowledge_base:
            self._build_knowledge_base(context)
        
        # 执行多维度分析
        analyses = []
        
        # 1. 错误分析
        error_analysis = self._analyze_errors()
        if error_analysis:
            analyses.append(error_analysis)
            context.add_finding(f"发现 {error_analysis.get('error_count', 0)} 类错误")
        
        # 2. 模式分析
        pattern_analysis = self._analyze_patterns()
        if pattern_analysis:
            analyses.append(pattern_analysis)
        
        # 3. 趋势分析
        trend_analysis = self._analyze_trends()
        if trend_analysis:
            analyses.append(trend_analysis)
        
        # 4. RAG 深度分析
        rag_analysis = self._rag_analysis(context)
        if rag_analysis:
            analyses.append(rag_analysis)
        
        # 整合分析结果
        self.analysis_results = {
            "analyses": analyses,
            "summary": self._generate_summary(analyses),
            "recommendations": self._generate_recommendations(analyses)
        }
        
        # 更新上下文
        context.analysis_results = self.analysis_results
        
        return AgentResponse(
            success=True,
            data=self.analysis_results,
            message=f"完成 {len(analyses)} 项分析，发现 {len(self.analysis_results.get('recommendations', []))} 个建议",
            context_updates={
                "analysis_results": self.analysis_results
            }
        )
    
    def _build_knowledge_base(self, context: AgentContext):
        """构建知识库"""
        from log_parser import LogEntry
        
        self.knowledge_base = LogKnowledgeBase()
        
        # 过滤有效条目
        entries = [e for e in context.parsed_data if isinstance(e, LogEntry)]
        
        if entries:
            self.knowledge_base.add_logs(entries).build_index()
            self.rag_engine = RAGEngine(self.knowledge_base)
    
    def _analyze_errors(self) -> Dict:
        """分析错误"""
        if not self.knowledge_base:
            return {}
        
        errors = self.knowledge_base.get_error_analysis()
        
        # 使用 RAG 深入分析
        if self.rag_engine:
            result = self.rag_engine.analyze_logs()
            return {
                "type": "error_analysis",
                "error_count": len(self.knowledge_base.log_entries),
                "error_summary": errors,
                "detailed_analysis": result.answer,
                "sources": [s['text'][:200] for s in result.sources[:3]]
            }
        
        return {
            "type": "error_analysis",
            "error_summary": errors
        }
    
    def _analyze_patterns(self) -> Dict:
        """分析模式"""
        if not self.knowledge_base or not self.knowledge_base.log_entries:
            return {}
        
        entries = self.knowledge_base.log_entries
        
        # 简单的模式识别
        patterns = {
            "repeated_errors": [],
            "error_sequences": [],
            "time_clusters": []
        }
        
        # 找出重复的错误消息
        error_msgs = {}
        for entry in entries:
            if entry.level in ['ERROR', 'CRITICAL']:
                msg = entry.message[:100]  # 取前100字符作为key
                error_msgs[msg] = error_msgs.get(msg, 0) + 1
        
        # 找出重复3次以上的错误
        for msg, count in error_msgs.items():
            if count >= 3:
                patterns["repeated_errors"].append({
                    "message": msg,
                    "count": count
                })
        
        return {
            "type": "pattern_analysis",
            "patterns": patterns,
            "findings": [
                f"发现 {len(patterns['repeated_errors'])} 个重复错误模式"
            ] if patterns["repeated_errors"] else []
        }
    
    def _analyze_trends(self) -> Dict:
        """分析趋势"""
        if not self.knowledge_base or not self.knowledge_base.log_entries:
            return {}
        
        entries = self.knowledge_base.log_entries
        
        # 按时间分组统计（简化版）
        time_groups = {}
        for entry in entries:
            if entry.timestamp:
                hour = entry.timestamp.strftime("%Y-%m-%d %H:00")
                if hour not in time_groups:
                    time_groups[hour] = {"total": 0, "errors": 0}
                time_groups[hour]["total"] += 1
                if entry.level in ['ERROR', 'CRITICAL']:
                    time_groups[hour]["errors"] += 1
        
        # 找出错误高峰期
        peak_hours = sorted(
            time_groups.items(),
            key=lambda x: x[1]["errors"],
            reverse=True
        )[:3]
        
        return {
            "type": "trend_analysis",
            "time_distribution": time_groups,
            "peak_error_hours": [
                {"hour": h, "errors": d["errors"]} 
                for h, d in peak_hours
            ],
            "findings": [
                f"错误高峰期: {peak_hours[0][0]} ({peak_hours[0][1]['errors']} 个错误)"
            ] if peak_hours else []
        }
    
    def _rag_analysis(self, context: AgentContext) -> Dict:
        """使用 RAG 进行深度分析"""
        if not self.rag_engine:
            return {}
        
        # 针对不同方面进行查询
        queries = [
            "日志中有哪些关键问题需要关注？",
            "系统性能状况如何？",
            "有哪些潜在的风险？"
        ]
        
        findings = []
        for query in queries:
            try:
                result = self.rag_engine.query(query)
                if result.confidence > 0.5:
                    findings.append({
                        "query": query,
                        "finding": result.answer,
                        "confidence": result.confidence
                    })
            except:
                pass
        
        return {
            "type": "rag_analysis",
            "findings": findings
        }
    
    def _generate_summary(self, analyses: List[Dict]) -> str:
        """生成分析摘要"""
        parts = []
        
        for analysis in analyses:
            if "findings" in analysis and analysis["findings"]:
                parts.extend(analysis["findings"])
        
        return "\n".join(parts) if parts else "未发现明显问题"
    
    def _generate_recommendations(self, analyses: List[Dict]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for analysis in analyses:
            if analysis.get("type") == "error_analysis":
                if analysis.get("error_count", 0) > 0:
                    recommendations.append("建议优先处理错误日志，检查系统稳定性")
            
            elif analysis.get("type") == "pattern_analysis":
                patterns = analysis.get("patterns", {})
                if patterns.get("repeated_errors"):
                    recommendations.append("发现重复错误模式，建议检查相关代码或配置")
            
            elif analysis.get("type") == "trend_analysis":
                peak_hours = analysis.get("peak_error_hours", [])
                if peak_hours:
                    recommendations.append(f"注意错误高峰期: {peak_hours[0]['hour']}，建议排查该时段的系统负载")
        
        return recommendations
    
    def can_handle(self, context: AgentContext) -> bool:
        """需要已解析的日志数据"""
        return context.parsed_data is not None
    
    def query_knowledge_base(self, question: str) -> str:
        """查询知识库"""
        if not self.rag_engine:
            return "知识库未初始化"
        
        result = self.rag_engine.query(question)
        return result.answer
