"""
持续学习模块

功能:
1. 收集用户反馈
2. 记录查询-回答对
3. 自动更新知识库
4. 评分趋势监控
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """反馈记录"""
    query: str
    answer: str
    rating: int  # 1-5
    confidence: float
    hallucination_risk: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    user_comment: str = ""


@dataclass
class LearningStats:
    """学习统计"""
    total_queries: int = 0
    avg_rating: float = 0.0
    avg_confidence: float = 0.0
    high_risk_count: int = 0
    improvement_trend: float = 0.0


class ContinuousLearning:
    """
    持续学习引擎

    功能:
    1. 反馈收集与存储
    2. 评分趋势分析
    3. 自动知识库更新建议
    4. 弱项识别
    """

    def __init__(self, storage_path: str = "./data/learning"):
        self.storage_path = storage_path
        self.feedback_history: List[FeedbackRecord] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load_history()

    def record_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        confidence: float,
        hallucination_risk: str,
        tags: List[str] = None,
        user_comment: str = "",
    ):
        """记录用户反馈"""
        record = FeedbackRecord(
            query=query,
            answer=answer,
            rating=rating,
            confidence=confidence,
            hallucination_risk=hallucination_risk,
            tags=tags or [],
            user_comment=user_comment,
        )

        self.feedback_history.append(record)
        self._save_record(record)

        logger.info(f"反馈记录: rating={rating}, confidence={confidence:.2f}")

    def get_stats(self) -> LearningStats:
        """获取学习统计"""
        stats = LearningStats()

        if not self.feedback_history:
            return stats

        stats.total_queries = len(self.feedback_history)
        stats.avg_rating = sum(r.rating for r in self.feedback_history) / stats.total_queries
        stats.avg_confidence = sum(r.confidence for r in self.feedback_history) / stats.total_queries
        stats.high_risk_count = sum(1 for r in self.feedback_history if r.hallucination_risk == "high")

        # 趋势分析（最近10次 vs 总体）
        recent = self.feedback_history[-10:]
        if len(recent) >= 2 and len(self.feedback_history) >= 10:
            old_avg = sum(r.rating for r in self.feedback_history[:-10]) / max(len(self.feedback_history) - 10, 1)
            new_avg = sum(r.rating for r in recent) / len(recent)
            stats.improvement_trend = new_avg - old_avg

        return stats

    def get_weak_areas(self) -> List[str]:
        """识别弱项"""
        if not self.feedback_history:
            return ["数据不足，无法分析"]

        weak_areas = []

        # 低分查询分析
        low_rated = [r for r in self.feedback_history if r.rating <= 2]
        if len(low_rated) > len(self.feedback_history) * 0.3:
            weak_areas.append("整体评分偏低，需优化检索质量")

        # 高幻觉风险
        high_risk = [r for r in self.feedback_history if r.hallucination_risk == "high"]
        if len(high_risk) > len(self.feedback_history) * 0.2:
            weak_areas.append("幻觉风险偏高，需扩充知识库")

        # 标签分析
        tag_counter = {}
        for r in self.feedback_history:
            for tag in r.tags:
                tag_counter[tag] = tag_counter.get(tag, 0) + 1
                if r.rating <= 2:
                    tag_counter[f"{tag}_low"] = tag_counter.get(f"{tag}_low", 0) + 1

        for tag, count in tag_counter.items():
            if tag.endswith("_low") and count > 2:
                base_tag = tag.replace("_low", "")
                weak_areas.append(f"标签 [{base_tag}] 相关问题回答较差")

        return weak_areas or ["当前表现良好"]

    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        suggestions = []

        stats = self.get_stats()
        weak_areas = self.get_weak_areas()

        if stats.avg_rating < 3.0:
            suggestions.append("检索召回率可能不足，建议扩充知识库")
        if stats.avg_confidence < 0.5:
            suggestions.append("置信度偏低，建议调整检索参数(top_k, RRF权重)")
        if stats.high_risk_count > 0:
            suggestions.append(f"发现{stats.high_risk_count}次高幻觉风险，建议审查知识库质量")
        if stats.improvement_trend < -0.2:
            suggestions.append("评分呈下降趋势，需排查知识库是否过期")

        return suggestions or ["系统运行良好"]

    def _save_record(self, record: FeedbackRecord):
        """持久化单条记录"""
        filepath = os.path.join(self.storage_path, "feedback.jsonl")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "query": record.query,
                "rating": record.rating,
                "confidence": record.confidence,
                "hallucination_risk": record.hallucination_risk,
                "timestamp": record.timestamp,
                "tags": record.tags,
                "user_comment": record.user_comment,
            }, ensure_ascii=False) + "\n")

    def _load_history(self):
        """加载历史记录"""
        filepath = os.path.join(self.storage_path, "feedback.jsonl")
        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    record = FeedbackRecord(
                        query=data.get("query", ""),
                        answer="",  # 不存完整回答节省空间
                        rating=data.get("rating", 0),
                        confidence=data.get("confidence", 0),
                        hallucination_risk=data.get("hallucination_risk", "unknown"),
                        timestamp=data.get("timestamp", ""),
                        tags=data.get("tags", []),
                        user_comment=data.get("user_comment", ""),
                    )
                    self.feedback_history.append(record)
                except json.JSONDecodeError:
                    continue

        logger.info(f"已加载 {len(self.feedback_history)} 条历史反馈")
