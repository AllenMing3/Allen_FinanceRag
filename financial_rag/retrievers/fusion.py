"""
多策略融合排序

提供三种融合策略:
1. RRF (Reciprocal Rank Fusion) — 基于 rank 的融合，最鲁棒
2. Linear — 归一化分数加权求和
3. Weighted RRF — 结合 rank 和 score 的混合融合

还包含:
- FusionStats: 融合统计信息
- hybrid_fusion(): 统一入口，按策略名选择算法
"""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _doc_hash(text: str) -> str:
    """稳定的文档标识 — MD5 前 16 字符，避免 Python hash() 碰撞"""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class FusionStats:
    """融合统计"""
    strategy: str = ""
    total_candidates: int = 0
    fused_count: int = 0
    channel_counts: Dict[str, int] = field(default_factory=dict)
    consensus_count: int = 0        # 多通道都命中的文档数
    unique_docs: int = 0            # 去重后的文档总数


def hybrid_fusion(
    result_lists: List[tuple],
    top_k: int,
    rrf_k: int = 60,
    strategy: str = "rrf",
) -> tuple:
    """
    融合多通道检索结果 — 统一入口

    Args:
        result_lists: [(results, weight, channel_name), ...]
        top_k: 返回数量
        rrf_k: RRF 常数 (默认 60)
        strategy: "rrf" | "linear" | "weighted_rrf"

    Returns:
        (fused_results, FusionStats)
    """
    if strategy == "linear":
        return _linear_fusion(result_lists, top_k)
    elif strategy == "weighted_rrf":
        return _weighted_rrf_fusion(result_lists, top_k, rrf_k)
    else:
        return rrf_fusion(result_lists, top_k, rrf_k)


def rrf_fusion(
    result_lists: List[tuple],
    top_k: int,
    rrf_k: int = 60,
) -> tuple:
    """
    RRF 融合 — Reciprocal Rank Fusion

    公式: score(d) = sum(weight_i / (k + rank_i(d)))
    优点: 只依赖 rank，不受不同通道分数尺度差异影响
    """
    rrf_scores: Dict[int, float] = {}
    doc_map: Dict[int, Dict] = {}
    channel_ranks: Dict[str, Dict[int, int]] = {}
    channel_scores_map: Dict[str, Dict[int, float]] = {}
    doc_channels: Dict[int, int] = {}  # 每个文档被几个通道命中

    for results, weight, channel_name in result_lists:
        ranks = {}
        scores = {}
        for r in results:
            text = r.get("text", "")
            doc_idx = _doc_hash(text)
            rank = r.get("rank", 1)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + weight / (rrf_k + rank)
            ranks[doc_idx] = rank
            scores[doc_idx] = r.get("score", 0)
            doc_map[doc_idx] = r
            doc_channels[doc_idx] = doc_channels.get(doc_idx, 0) + 1
        channel_ranks[channel_name] = ranks
        channel_scores_map[channel_name] = scores

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for rank_i, (doc_idx, rrf_score) in enumerate(ranked):
        item = dict(doc_map[doc_idx])
        item["score"] = rrf_score
        item["rrf_score"] = rrf_score
        item["retriever"] = "hybrid"
        item["rank"] = rank_i + 1
        for ch_name in channel_ranks:
            item[f"{ch_name}_rank"] = channel_ranks[ch_name].get(doc_idx)
            item[f"{ch_name}_score"] = channel_scores_map[ch_name].get(doc_idx)
        result.append(item)

    stats = FusionStats(
        strategy="rrf",
        total_candidates=sum(len(r) for r, _, _ in result_lists),
        fused_count=len(result),
        channel_counts={name: len(r) for r, _, name in result_lists},
        consensus_count=sum(1 for c in doc_channels.values() if c > 1),
        unique_docs=len(doc_map),
    )
    return result, stats


def _linear_fusion(
    result_lists: List[tuple],
    top_k: int,
) -> tuple:
    """
    线性融合 — 归一化分数加权求和

    公式: score(d) = sum(weight_i * normalized_score_i(d))
    归一化: (score - min) / (max - min)
    注意: 不同通道分数尺度不同时效果不如 RRF
    """
    doc_scores: Dict[int, float] = {}
    doc_map: Dict[int, Dict] = {}
    doc_channels: Dict[int, int] = {}

    for results, weight, channel_name in result_lists:
        if not results:
            continue
        # 归一化
        scores_list = [r.get("score", 0) for r in results]
        min_s = min(scores_list)
        max_s = max(scores_list)
        score_range = max_s - min_s if max_s > min_s else 1.0

        for r in results:
            text = r.get("text", "")
            doc_idx = _doc_hash(text)
            norm_score = (r.get("score", 0) - min_s) / score_range
            doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + weight * norm_score
            doc_map[doc_idx] = r
            doc_channels[doc_idx] = doc_channels.get(doc_idx, 0) + 1

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for rank_i, (doc_idx, score) in enumerate(ranked):
        item = dict(doc_map[doc_idx])
        item["score"] = score
        item["retriever"] = "hybrid_linear"
        item["rank"] = rank_i + 1
        result.append(item)

    stats = FusionStats(
        strategy="linear",
        total_candidates=sum(len(r) for r, _, _ in result_lists),
        fused_count=len(result),
        channel_counts={name: len(r) for r, _, name in result_lists},
        consensus_count=sum(1 for c in doc_channels.values() if c > 1),
        unique_docs=len(doc_map),
    )
    return result, stats


def _weighted_rrf_fusion(
    result_lists: List[tuple],
    top_k: int,
    rrf_k: int = 60,
) -> tuple:
    """
    加权 RRF — 结合 rank 和原始分数的混合融合

    公式: score(d) = sum(weight_i * (alpha / (k + rank) + (1-alpha) * norm_score))
    alpha=0.7: 偏重 rank (更鲁棒)，alpha=0.3: 偏重 score (更精确)
    """
    alpha = 0.7
    doc_scores: Dict[int, float] = {}
    doc_map: Dict[int, Dict] = {}
    doc_channels: Dict[int, int] = {}

    for results, weight, channel_name in result_lists:
        if not results:
            continue
        scores_list = [r.get("score", 0) for r in results]
        min_s = min(scores_list)
        max_s = max(scores_list)
        score_range = max_s - min_s if max_s > min_s else 1.0

        for r in results:
            text = r.get("text", "")
            doc_idx = _doc_hash(text)
            rank = r.get("rank", 1)
            norm_score = (r.get("score", 0) - min_s) / score_range
            combined = alpha / (rrf_k + rank) + (1 - alpha) * norm_score
            doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + weight * combined
            doc_map[doc_idx] = r
            doc_channels[doc_idx] = doc_channels.get(doc_idx, 0) + 1

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for rank_i, (doc_idx, score) in enumerate(ranked):
        item = dict(doc_map[doc_idx])
        item["score"] = score
        item["retriever"] = "hybrid_wrrf"
        item["rank"] = rank_i + 1
        result.append(item)

    stats = FusionStats(
        strategy="weighted_rrf",
        total_candidates=sum(len(r) for r, _, _ in result_lists),
        fused_count=len(result),
        channel_counts={name: len(r) for r, _, name in result_lists},
        consensus_count=sum(1 for c in doc_channels.values() if c > 1),
        unique_docs=len(doc_map),
    )
    return result, stats
