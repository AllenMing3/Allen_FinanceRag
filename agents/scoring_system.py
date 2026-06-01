"""
Agentic RAG 评分系统

解决评分标准上升太快的问题，提供渐进式评分方案
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from .agentic_base_agent import ThoughtStep, ActionType, RetrievalContext


class ScoreType(Enum):
    """评分类型"""
    RETRIEVAL_QUALITY = "retrieval_quality"     # 检索质量
    ANSWER_QUALITY = "answer_quality"           # 回答质量
    PROCESS_QUALITY = "process_quality"         # 过程质量
    OVERALL_CONFIDENCE = "overall_confidence"   # 总体置信度


@dataclass
class ScoreResult:
    """评分结果"""
    score_type: ScoreType
    score: float
    weight: float
    explanation: str
    details: Dict[str, Any]


class ScoringSystem:
    """
    完整的评分系统
    
    解决评分标准上升太快的问题：
    1. 多维度评分，避免单一指标
    2. 渐进式权重调整
    3. 可配置的阈值
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # 默认权重配置
        self.weights = config.get('weights', {
            ScoreType.RETRIEVAL_QUALITY: 0.3,
            ScoreType.ANSWER_QUALITY: 0.4,
            ScoreType.PROCESS_QUALITY: 0.2,
            ScoreType.OVERALL_CONFIDENCE: 0.1
        })
        
        # 阈值配置
        self.thresholds = config.get('thresholds', {
            'high_confidence': 0.7,
            'medium_confidence': 0.5,
            'low_confidence': 0.3
        })
        
        # 性能监控
        self.score_history: List[Dict] = []
    
    def score_retrieval_quality(self, retrieved_contexts: List[RetrievalContext]) -> ScoreResult:
        """
        检索质量评分
        
        解决：避免仅靠检索次数和数量就获得高分
        """
        if not retrieved_contexts:
            return ScoreResult(
                score_type=ScoreType.RETRIEVAL_QUALITY,
                score=0.0,
                weight=self.weights[ScoreType.RETRIEVAL_QUALITY],
                explanation="无检索结果",
                details={"error": "no_results"}
            )
        
        score = 0.0
        details = {
            "retrieval_count": 0,
            "total_results": 0,
            "avg_relevance": 0.0,
            "unique_sources": 0,
            "diversity_score": 0.0
        }
        
        # 1. 检索次数（权重降低）
        retrieval_count = sum(ctx.retrieval_count for ctx in retrieved_contexts)
        details["retrieval_count"] = retrieval_count
        # 从 0.1 降到 0.05，避免次数过多就高分
        retrieval_score = min(0.2, retrieval_count * 0.05)
        score += retrieval_score
        
        # 2. 结果数量
        total_results = sum(len(ctx.results) for ctx in retrieved_contexts)
        details["total_results"] = total_results
        count_score = min(0.2, total_results * 0.02)
        score += count_score
        
        # 3. 相关度质量（权重提高）
        all_scores = []
        for ctx in retrieved_contexts:
            for result in ctx.results:
                if 'score' in result:
                    all_scores.append(result['score'])
        
        if all_scores:
            avg_relevance = sum(all_scores) / len(all_scores)
            details["avg_relevance"] = avg_relevance
            # 从 0.4 升到 0.5，更注重质量而非数量
            relevance_score = min(0.5, avg_relevance * 0.5)
            score += relevance_score
        
        # 4. 结果多样性
        sources = set()
        for ctx in retrieved_contexts:
            for result in ctx.results:
                source = result.get('metadata', {}).get('source', 'unknown')
                sources.add(source)
        
        unique_sources = len(sources)
        details["unique_sources"] = unique_sources
        diversity_score = min(0.1, unique_sources / 10.0)
        details["diversity_score"] = diversity_score
        score += diversity_score
        
        final_score = min(1.0, max(0.0, score))
        
        return ScoreResult(
            score_type=ScoreType.RETRIEVAL_QUALITY,
            score=final_score,
            weight=self.weights[ScoreType.RETRIEVAL_QUALITY],
            explanation=f"检索质量: {final_score:.2f} (检索{retrieval_count}次, {total_results}结果, 相关度{details.get('avg_relevance', 0):.2f})",
            details=details
        )
    
    def score_answer_quality(self, answer: str, question: str, 
                           retrieved_contexts: List[RetrievalContext]) -> ScoreResult:
        """
        回答质量评分
        
        基于回答与问题和检索结果的相关性
        """
        if not answer:
            return ScoreResult(
                score_type=ScoreType.ANSWER_QUALITY,
                score=0.0,
                weight=self.weights[ScoreType.ANSWER_QUALITY],
                explanation="无回答内容",
                details={"error": "no_answer"}
            )
        
        score = 0.0
        details = {
            "answer_length": len(answer),
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "actionability_score": 0.0
        }
        
        # 1. 相关性（回答是否针对问题）
        relevance_score = self._calculate_relevance(answer, question)
        details["relevance_score"] = relevance_score
        score += relevance_score * 0.3
        
        # 2. 完整性（是否包含关键要素）
        completeness_score = self._calculate_completeness(answer)
        details["completeness_score"] = completeness_score
        score += completeness_score * 0.3
        
        # 3. 准确性（基于检索结果）
        accuracy_score = self._calculate_accuracy(answer, retrieved_contexts)
        details["accuracy_score"] = accuracy_score
        score += accuracy_score * 0.2
        
        # 4. 可操作性
        actionability_score = self._calculate_actionability(answer)
        details["actionability_score"] = actionability_score
        score += actionability_score * 0.2
        
        final_score = min(1.0, max(0.0, score))
        
        return ScoreResult(
            score_type=ScoreType.ANSWER_QUALITY,
            score=final_score,
            weight=self.weights[ScoreType.ANSWER_QUALITY],
            explanation=f"回答质量: {final_score:.2f} (相关性{relevance_score:.2f}, 完整性{completeness_score:.2f})",
            details=details
        )
    
    def score_process_quality(self, steps: List[ThoughtStep]) -> ScoreResult:
        """
        过程质量评分
        
        评估 ReAct 循环的质量
        """
        if not steps:
            return ScoreResult(
                score_type=ScoreType.PROCESS_QUALITY,
                score=0.0,
                weight=self.weights[ScoreType.PROCESS_QUALITY],
                explanation="无执行步骤",
                details={"error": "no_steps"}
            )
        
        score = 0.0
        details = {
            "total_steps": len(steps),
            "retrieval_steps": 0,
            "efficiency_score": 0.0,
            "reflection_depth": 0.0
        }
        
        # 1. 步骤合理性
        logical_score = self._assess_step_logic(steps)
        score += logical_score * 0.4
        
        # 2. 检索效率
        retrieval_steps = [s for s in steps if s.action == ActionType.RETRIEVE]
        details["retrieval_steps"] = len(retrieval_steps)
        efficiency_score = self._assess_efficiency(steps)
        details["efficiency_score"] = efficiency_score
        score += efficiency_score * 0.3
        
        # 3. 反思深度
        reflection_score = self._assess_reflection(steps)
        details["reflection_depth"] = reflection_score
        score += reflection_score * 0.3
        
        final_score = min(1.0, max(0.0, score))
        
        return ScoreResult(
            score_type=ScoreType.PROCESS_QUALITY,
            score=final_score,
            weight=self.weights[ScoreType.PROCESS_QUALITY],
            explanation=f"过程质量: {final_score:.2f} ({len(steps)}步骤, {len(retrieval_steps)}次检索)",
            details=details
        )
    
    def calculate_overall_score(self, question: str, answer: str, 
                              retrieved_contexts: List[RetrievalContext],
                              steps: List[ThoughtStep]) -> Dict[str, Any]:
        """
        计算总体评分
        
        解决评分标准上升太快的问题：
        1. 多维度加权平均
        2. 避免单一指标主导
        3. 提供详细解释
        """
        # 计算各维度评分
        retrieval_score = self.score_retrieval_quality(retrieved_contexts)
        answer_score = self.score_answer_quality(answer, question, retrieved_contexts)
        process_score = self.score_process_quality(steps)
        
        # 加权平均
        weighted_scores = [
            retrieval_score.score * retrieval_score.weight,
            answer_score.score * answer_score.weight,
            process_score.score * process_score.weight
        ]
        
        overall_score = sum(weighted_scores)
        
        # 置信度评估（基于总体评分）
        confidence_level = self._calculate_confidence_level(overall_score)
        
        result = {
            "overall_score": overall_score,
            "confidence_level": confidence_level,
            "score_breakdown": {
                "retrieval_quality": retrieval_score.to_dict(),
                "answer_quality": answer_score.to_dict(),
                "process_quality": process_score.to_dict()
            },
            "recommendation": self._get_recommendation(overall_score, confidence_level)
        }
        
        # 记录历史
        self.score_history.append({
            "question": question,
            "overall_score": overall_score,
            "timestamp": "2024-01-01T00:00:00"  # 实际使用时应该用真实时间
        })
        
        return result
    
    def _calculate_relevance(self, answer: str, question: str) -> float:
        """计算回答与问题的相关性"""
        if not question or not answer:
            return 0.0
        
        # 简单的关键词匹配
        question_words = set(re.findall(r'\w+', question.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(answer_words))
        return min(1.0, overlap / len(question_words))
    
    def _calculate_completeness(self, answer: str) -> float:
        """计算回答的完整性"""
        # 检查是否包含关键要素
        key_elements = ['问题', '原因', '解决方案', '建议', '步骤']
        found_elements = sum(1 for elem in key_elements if elem in answer)
        
        return min(1.0, found_elements / len(key_elements))
    
    def _calculate_accuracy(self, answer: str, retrieved_contexts: List[RetrievalContext]) -> float:
        """基于检索结果评估准确性"""
        if not retrieved_contexts:
            return 0.5  # 无检索结果，中等分数
        
        # 简单的关键词匹配（实际应该用更复杂的方法）
        all_context_text = " ".join(
            result.get('text', '') 
            for ctx in retrieved_contexts 
            for result in ctx.results
        )
        
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set(re.findall(r'\w+', all_context_text.lower()))
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        return min(1.0, overlap / len(answer_words))
    
    def _calculate_actionability(self, answer: str) -> float:
        """计算回答的可操作性"""
        # 检查是否包含可执行内容
        actionable_indicators = [
            '步骤', '命令', '配置', '修改', '检查', '验证',
            '重启', '更新', '安装', '设置'
        ]
        
        found_indicators = sum(1 for indicator in actionable_indicators 
                              if indicator in answer)
        
        return min(1.0, found_indicators / len(actionable_indicators))
    
    def _assess_step_logic(self, steps: List[ThoughtStep]) -> float:
        """评估步骤逻辑合理性"""
        if len(steps) < 2:
            return 0.5
        
        # 检查是否有合理的思考-行动模式
        logical_transitions = 0
        for i in range(1, len(steps)):
            prev = steps[i-1]
            curr = steps[i]
            
            # 检查是否有合理的行动序列
            if (prev.action == ActionType.RETRIEVE and 
                curr.action in [ActionType.ANALYZE, ActionType.SYNTHESIZE]):
                logical_transitions += 1
        
        return min(1.0, logical_transitions / (len(steps) - 1))
    
    def _assess_efficiency(self, steps: List[ThoughtStep]) -> float:
        """评估检索效率"""
        retrieval_steps = [s for s in steps if s.action == ActionType.RETRIEVE]
        
        if len(retrieval_steps) == 0:
            return 0.5  # 没有检索，中等分数
        
        # 检索次数越少，效率越高（但要有足够信息）
        efficiency = max(0.0, 1.0 - (len(retrieval_steps) - 1) * 0.2)
        
        return efficiency
    
    def _assess_reflection(self, steps: List[ThoughtStep]) -> float:
        """评估反思深度"""
        reflection_count = sum(1 for step in steps if step.reflection)
        return min(1.0, reflection_count / len(steps) if steps else 0.0)
    
    def _calculate_confidence_level(self, overall_score: float) -> str:
        """基于总体评分计算置信度等级"""
        if overall_score >= self.thresholds['high_confidence']:
            return "high"
        elif overall_score >= self.thresholds['medium_confidence']:
            return "medium"
        else:
            return "low"
    
    def _get_recommendation(self, overall_score: float, confidence_level: str) -> str:
        """根据评分提供建议"""
        if confidence_level == "high":
            return "回答质量高，可以直接使用"
        elif confidence_level == "medium":
            return "回答质量中等，建议人工复核"
        else:
            return "回答质量低，需要重新检索或人工处理"


# 简化的评分系统（适合你现在的情况）
class SimpleScoringSystem:
    """
    简化版评分系统
    
    专门解决评分标准上升太快的问题
    """
    
    def __init__(self):
        # 调整权重，更注重质量而非数量
        self.weights = {
            'retrieval_count': 0.05,    # 降低检索次数权重
            'result_count': 0.02,       # 保持结果数量权重
            'relevance_quality': 0.5,   # 提高相关度权重
            'diversity': 0.1            # 增加多样性权重
        }
    
    def assess_confidence_v2(self, retrieved_contexts: List[RetrievalContext]) -> float:
        """
        改进的置信度评估
        
        解决评分标准上升太快的问题
        """
        if not retrieved_contexts:
            return 0.1  # 降低基础置信度
        
        confidence = 0.1  # 基础置信度降低
        
        # 1. 检索次数（权重降低）
        retrieval_count = sum(ctx.retrieval_count for ctx in retrieved_contexts)
        confidence += min(0.2, retrieval_count * self.weights['retrieval_count'])
        
        # 2. 结果数量
        total_results = sum(len(ctx.results) for ctx in retrieved_contexts)
        confidence += min(0.2, total_results * self.weights['result_count'])
        
        # 3. 结果质量（权重提高）
        all_scores = []
        for ctx in retrieved_contexts:
            for result in ctx.results:
                if 'score' in result:
                    all_scores.append(result['score'])
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            confidence += min(0.5, avg_score * self.weights['relevance_quality'])
        
        # 4. 结果多样性
        sources = set()
        for ctx in retrieved_contexts:
            for result in ctx.results:
                source = result.get('metadata', {}).get('source', 'unknown')
                sources.add(source)
        
        diversity_score = min(0.1, len(sources) / 10.0)
        confidence += diversity_score * self.weights['diversity']
        
        return min(0.95, max(0.1, confidence))


def demo_scoring_system():
    """演示评分系统"""
    print("\n" + "=" * 70)
    print("评分系统演示")
    print("=" * 70)
    
    # 创建评分系统
    scoring_system = ScoringSystem()
    
    # 模拟数据
    question = "日志中出现连接超时错误怎么办？"
    answer = "检查网络连接，查看防火墙设置，重启服务"
    
    # 模拟检索结果
    retrieved_contexts = [
        RetrievalContext(
            query="连接超时",
            results=[
                {"text": "网络连接超时可能是防火墙问题", "score": 0.8},
                {"text": "检查网络配置和DNS设置", "score": 0.7}
            ]
        )
    ]
    
    # 模拟步骤
    steps = [
        ThoughtStep(1, "检索错误信息", ActionType.RETRIEVE, "连接超时", 
                   "检索到2条相关信息", "结果相关")
    ]
    
    # 计算评分
    result = scoring_system.calculate_overall_score(question, answer, retrieved_contexts, steps)
    
    print(f"总体评分: {result['overall_score']:.2f}")
    print(f"置信度等级: {result['confidence_level']}")
    print(f"建议: {result['recommendation']}")
    
    # 展示详细评分
    for score_type, score_info in result['score_breakdown'].items():
        print(f"\n{score_type}:")
        print(f"  分数: {score_info['score']:.2f}")
        print(f"  解释: {score_info['explanation']}")


if __name__ == "__main__":
    demo_scoring_system()