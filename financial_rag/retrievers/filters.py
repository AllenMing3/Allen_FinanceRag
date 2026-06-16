"""
元数据过滤器 — Metadata Filter

支持的操作符:
- 精确匹配: {"source": "news"}
- 多值匹配: {"source": ["news", "rss"]}
- 范围过滤: {"date": {"gte": "2024-01-01", "lte": "2024-12-31"}}
- 取反:     {"$not": {"source": "spam"}}
- 存在性:   {"$exists": "stock_code"} / {"$not_exists": "stock_code"}
- 正则:     {"title": {"$regex": "AI|芯片"}}

Soft mode: 文档缺少字段时不过滤 (QueryParser 自动注入用)
"""
import re as _re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass
class FilterStats:
    """过滤统计"""
    before_count: int = 0
    after_count: int = 0
    filtered_out: int = 0
    conditions_checked: int = 0
    soft_skips: int = 0           # soft key 跳过次数

    @property
    def pass_rate(self) -> float:
        if self.before_count == 0:
            return 0.0
        return self.after_count / self.before_count


def apply_filters(
    candidates: List[Dict],
    filters: Dict,
    soft_keys: Set[str] = None,
    collect_stats: bool = False,
) -> List[Dict]:
    """
    按元数据过滤候选文档

    Args:
        candidates: 候选文档列表
        filters: 过滤条件 (支持精确/多值/范围/NOT/EXISTS/REGEX)
        soft_keys: 软过滤字段集合 — 文档缺少这些字段时不拒绝
        collect_stats: 是否收集统计 (默认 False)

    Returns:
        过滤后的文档列表
    """
    soft_keys = soft_keys or set()
    stats = FilterStats(before_count=len(candidates))
    result = []

    for item in candidates:
        meta = item.get("meta", item.get("metadata", {}))
        passed, checks, skips = _match_all(meta, filters, soft_keys)
        stats.conditions_checked += checks
        stats.soft_skips += skips
        if passed:
            result.append(item)

    stats.after_count = len(result)
    stats.filtered_out = stats.before_count - stats.after_count

    if collect_stats:
        # 附加统计到返回列表 (通过 module-level 变量，避免改签名)
        _last_stats["stats"] = stats

    return result


def get_last_stats() -> Optional[FilterStats]:
    """获取最近一次 apply_filters(collect_stats=True) 的统计"""
    return _last_stats.get("stats")


_last_stats: Dict = {}


def _match_all(meta: Dict, filters: Dict, soft_keys: Set[str]) -> tuple:
    """
    检查 meta 是否满足所有过滤条件

    Returns:
        (passed: bool, conditions_checked: int, soft_skips: int)
    """
    checks = 0
    skips = 0

    for key, condition in filters.items():
        # 特殊操作符
        if key == "$not":
            checks += 1
            # 取反: condition 本身是一个 filter dict
            neg_passed, nc, ns = _match_all(meta, condition, soft_keys)
            checks += nc
            skips += ns
            if neg_passed:
                return False, checks, skips  # NOT 匹配成功 → 拒绝
            continue

        if key == "$exists":
            checks += 1
            # condition 是字段名 (str 或 list[str])
            fields = [condition] if isinstance(condition, str) else condition
            if not all(meta.get(f) is not None for f in fields):
                return False, checks, skips
            continue

        if key == "$not_exists":
            checks += 1
            fields = [condition] if isinstance(condition, str) else condition
            if any(meta.get(f) is not None for f in fields):
                return False, checks, skips
            continue

        # 普通字段匹配
        value = meta.get(key)
        checks += 1

        # Soft key: 文档没有此字段时跳过
        if value is None and key in soft_keys:
            skips += 1
            continue

        if not _match_value(value, condition, key in soft_keys):
            return False, checks, skips

    return True, checks, skips


def _match_value(value, condition, is_soft: bool) -> bool:
    """匹配单个字段的值"""
    # 值为 None 且非 soft → 拒绝
    if value is None:
        return is_soft

    if isinstance(condition, list):
        # 多值匹配 (OR)
        return value in condition

    elif isinstance(condition, dict):
        # 操作符字典
        for op, operand in condition.items():
            if op == "$regex":
                if not _re.search(operand, str(value)):
                    return False
            elif op == "$gte":
                if value < operand:
                    return False
            elif op == "$lte":
                if value > operand:
                    return False
            elif op == "$gt":
                if value <= operand:
                    return False
            elif op == "$lt":
                if value >= operand:
                    return False
            elif op == "$ne":
                if value == operand:
                    return False
            elif op == "$in":
                if value not in operand:
                    return False
            elif op == "$nin":
                if value in operand:
                    return False
            # 未知操作符忽略
        return True

    else:
        # 精确匹配
        return value == condition


def build_query_filters(parsed) -> Dict:
    """
    从 QueryResult 构建过滤条件 (供 HybridRetriever 使用)

    Args:
        parsed: QueryResult 实例

    Returns:
        filter dict
    """
    filters = {}
    if hasattr(parsed, 'stock_code') and parsed.stock_code:
        filters["stock_code"] = parsed.stock_code
    if hasattr(parsed, 'date') and parsed.date:
        filters["date"] = parsed.date
    if hasattr(parsed, 'date_range') and parsed.date_range:
        filters.update(parsed.date_range)
    return filters
