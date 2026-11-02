"""
L1 检索层 IR 指标 — 纯函数、零外部依赖、离线确定。

这是"证明检索变好了吗"的量化核心：给定一个排序结果和相关度标注，
算出标准 IR 指标。所有函数不碰网络/LLM/随机，同输入必同输出。

输入约定:
- ranked: 按 rank 排序、按 doc_id 去重后的逻辑文档 id 序列 (List[str])
- rel_ids: 相关文档 id 集合 (grade >= 1)，用于二元指标
- grades:  {doc_id: grade} 分级相关度 (0-3)，用于 NDCG

所有指标 ∈ [0.0, 1.0]，越大越好；空输入返回 0.0（不抛异常）。
"""
import math
from typing import Dict, Iterable, List, Sequence, Set

# 报告/聚合用的固定指标名（@k 后缀在报告层按实际 k 标注）
METRIC_NAMES = ["recall", "precision", "mrr", "ndcg", "map"]


def relevant_ids(grades: Dict[str, int], min_grade: int = 1) -> Set[str]:
    """从分级标注中提取相关文档集合（grade >= min_grade）。"""
    return {d for d, g in grades.items() if g >= min_grade}


def recall_at_k(ranked: Sequence[str], rel_ids: Iterable[str], k: int) -> float:
    """召回率@k = top-k 命中的相关文档数 / 相关文档总数。"""
    rel = set(rel_ids)
    if not rel:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in rel)
    return hits / len(rel)


def precision_at_k(ranked: Sequence[str], rel_ids: Iterable[str], k: int) -> float:
    """精确率@k = top-k 中相关文档数 / k。"""
    if k <= 0:
        return 0.0
    rel = set(rel_ids)
    hits = sum(1 for d in ranked[:k] if d in rel)
    return hits / k


def mrr(ranked: Sequence[str], rel_ids: Iterable[str]) -> float:
    """平均倒数排名（单查询）= 首个相关文档的 1/rank。无命中返回 0。"""
    rel = set(rel_ids)
    for i, d in enumerate(ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], grades: Dict[str, int], k: int) -> float:
    """分级 NDCG@k。

    增益 gain = 2^g - 1；折扣 disc = log2(rank + 1)（rank 从 1 起）。
    DCG = sum(gain_i / disc_i)；IDCG 为理想排序（相关文档按 grade 降序置顶）的 DCG。
    返回 DCG / IDCG ∈ [0,1]。
    """
    if k <= 0:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        g = grades.get(d, 0)
        if g > 0:
            dcg += (2 ** g - 1) / math.log2(i + 2)  # i+2 = rank+1
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def map_at_k(ranked: Sequence[str], rel_ids: Iterable[str], k: int) -> float:
    """平均精度 AP@k = (sum of precision@每个命中位置) / min(相关数, k)。"""
    rel = set(rel_ids)
    if not rel:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            hits += 1
            sum_prec += hits / (i + 1)
    denom = min(len(rel), k)
    return sum_prec / denom if denom > 0 else 0.0


def evaluate_query(ranked: Sequence[str], grades: Dict[str, int], k: int) -> Dict[str, float]:
    """对单条查询的排序结果算全套指标，返回 {metric_name: score}。"""
    rel = relevant_ids(grades)
    return {
        "recall": recall_at_k(ranked, rel, k),
        "precision": precision_at_k(ranked, rel, k),
        "mrr": mrr(ranked, rel),
        "ndcg": ndcg_at_k(ranked, grades, k),
        "map": map_at_k(ranked, rel, k),
    }


def dedupe_ranked(items: Sequence[Dict], id_getter=None) -> List[str]:
    """从检索结果项列表提取按 doc_id 去重（保留最高 rank）的逻辑文档 id 序列。

    多 chunk 文档的多个 chunk 共享同一 meta.doc_id，指标按逻辑文档计，
    故去重保留首次出现（即最高排名）的那一个。
    """
    if id_getter is None:
        id_getter = lambda it: it.get("meta", {}).get("doc_id")
    seen: Set[str] = set()
    ordered: List[str] = []
    for it in items:
        did = id_getter(it)
        if did is None or did in seen:
            continue
        seen.add(did)
        ordered.append(did)
    return ordered
