"""
Agent 工具集 - 供各个 Agent 使用的工具函数
"""
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timedelta


def search_similar_errors(log_entries: List[Any], error_text: str, top_k: int = 5) -> List[Dict]:
    """
    搜索相似的错误日志
    
    Args:
        log_entries: 日志条目列表
        error_text: 要搜索的错误文本
        top_k: 返回数量
    
    Returns:
        相似的日志条目
    """
    from log_parser import LogEntry
    
    results = []
    error_lower = error_text.lower()
    
    for entry in log_entries:
        if isinstance(entry, LogEntry):
            # 计算相似度（简单实现）
            message_lower = entry.message.lower()
            similarity = 0
            
            # 关键词匹配
            error_keywords = set(re.findall(r'\w+', error_lower))
            message_keywords = set(re.findall(r'\w+', message_lower))
            
            if error_keywords and message_keywords:
                intersection = error_keywords & message_keywords
                union = error_keywords | message_keywords
                similarity = len(intersection) / len(union)
            
            # 子串匹配加分
            if error_lower in message_lower or message_lower in error_lower:
                similarity += 0.3
            
            if similarity > 0.1:  # 阈值
                results.append({
                    "entry": entry,
                    "similarity": min(similarity, 1.0),
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else None
                })
    
    # 按相似度排序
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results[:top_k]


def analyze_error_frequency(log_entries: List[Any], time_window_minutes: int = 60) -> Dict:
    """
    分析错误频率
    
    Args:
        log_entries: 日志条目列表
        time_window_minutes: 时间窗口（分钟）
    
    Returns:
        频率分析结果
    """
    from log_parser import LogEntry
    
    errors = [e for e in log_entries if isinstance(e, LogEntry) and e.level in ['ERROR', 'CRITICAL']]
    
    if not errors:
        return {"error_count": 0, "frequency": "N/A"}
    
    # 按时间分组
    time_groups = {}
    for entry in errors:
        if entry.timestamp:
            # 按时间窗口分组
            window_key = entry.timestamp.replace(
                minute=(entry.timestamp.minute // time_window_minutes) * time_window_minutes,
                second=0,
                microsecond=0
            )
            time_groups[window_key] = time_groups.get(window_key, 0) + 1
    
    # 计算频率
    if time_groups:
        avg_frequency = sum(time_groups.values()) / len(time_groups)
        max_frequency = max(time_groups.values())
        peak_time = max(time_groups.items(), key=lambda x: x[1])[0]
    else:
        avg_frequency = len(errors)
        max_frequency = len(errors)
        peak_time = None
    
    return {
        "error_count": len(errors),
        "time_window_minutes": time_window_minutes,
        "average_frequency": round(avg_frequency, 2),
        "max_frequency": max_frequency,
        "peak_time": peak_time.isoformat() if peak_time else None,
        "time_distribution": {k.isoformat(): v for k, v in sorted(time_groups.items())}
    }


def extract_error_patterns(log_entries: List[Any]) -> List[Dict]:
    """
    提取错误模式
    
    Args:
        log_entries: 日志条目列表
    
    Returns:
        错误模式列表
    """
    from log_parser import LogEntry
    from collections import defaultdict
    
    errors = [e for e in log_entries if isinstance(e, LogEntry) and e.level in ['ERROR', 'CRITICAL']]
    
    # 按消息分组
    pattern_groups = defaultdict(list)
    
    for entry in errors:
        # 提取错误模式（去除变量部分）
        pattern = re.sub(r'\d+', '<NUM>', entry.message)
        pattern = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', pattern)
        pattern = re.sub(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', pattern)
        
        pattern_groups[pattern].append(entry)
    
    # 找出重复模式
    patterns = []
    for pattern, entries in pattern_groups.items():
        if len(entries) >= 2:  # 至少出现2次
            patterns.append({
                "pattern": pattern,
                "count": len(entries),
                "sample_messages": [e.message for e in entries[:3]],
                "first_occurrence": min(e.timestamp for e in entries if e.timestamp).isoformat() if any(e.timestamp for e in entries) else None,
                "last_occurrence": max(e.timestamp for e in entries if e.timestamp).isoformat() if any(e.timestamp for e in entries) else None
            })
    
    # 按出现次数排序
    patterns.sort(key=lambda x: x["count"], reverse=True)
    
    return patterns


def correlate_events(log_entries: List[Any], time_delta_seconds: int = 60) -> List[Dict]:
    """
    关联相关事件
    
    找出在相近时间内发生的相关日志
    
    Args:
        log_entries: 日志条目列表
        time_delta_seconds: 时间差阈值（秒）
    
    Returns:
        事件关联结果
    """
    from log_parser import LogEntry
    
    # 只处理有时间戳的条目
    timed_entries = [(e, e.timestamp) for e in log_entries if isinstance(e, LogEntry) and e.timestamp]
    timed_entries.sort(key=lambda x: x[1])
    
    correlations = []
    
    for i, (entry1, ts1) in enumerate(timed_entries):
        related = []
        
        # 向前查找
        for entry2, ts2 in timed_entries[i+1:]:
            if (ts2 - ts1).total_seconds() <= time_delta_seconds:
                # 检查是否相关（同一来源或相似消息）
                if (entry1.source == entry2.source or 
                    any(word in entry2.message.lower() for word in entry1.message.lower().split())):
                    related.append(entry2)
            else:
                break
        
        if len(related) >= 2:  # 至少2个相关事件
            correlations.append({
                "trigger_event": entry1,
                "related_events": related,
                "time_span_seconds": (related[-1].timestamp - entry1.timestamp).total_seconds() if related else 0,
                "event_chain": [entry1.message] + [e.message for e in related]
            })
    
    return correlations


def generate_statistics(log_entries: List[Any]) -> Dict:
    """
    生成日志统计信息
    
    Args:
        log_entries: 日志条目列表
    
    Returns:
        统计信息
    """
    from log_parser import LogEntry
    
    stats = {
        "total_entries": len(log_entries),
        "by_level": {},
        "by_source": {},
        "by_hour": {},
        "error_rate": 0.0,
        "time_range": {"start": None, "end": None}
    }
    
    timestamps = []
    error_count = 0
    
    for entry in log_entries:
        if isinstance(entry, LogEntry):
            # 级别统计
            level = entry.level or "UNKNOWN"
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            # 错误计数
            if level in ['ERROR', 'CRITICAL', 'FATAL']:
                error_count += 1
            
            # 来源统计
            source = entry.source or "UNKNOWN"
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            
            # 时间统计
            if entry.timestamp:
                timestamps.append(entry.timestamp)
                hour = entry.timestamp.strftime("%Y-%m-%d %H:00")
                stats["by_hour"][hour] = stats["by_hour"].get(hour, 0) + 1
    
    # 计算错误率
    if stats["total_entries"] > 0:
        stats["error_rate"] = round(error_count / stats["total_entries"] * 100, 2)
    
    # 时间范围
    if timestamps:
        stats["time_range"]["start"] = min(timestamps).isoformat()
        stats["time_range"]["end"] = max(timestamps).isoformat()
    
    return stats


def suggest_monitoring_rules(log_entries: List[Any]) -> List[Dict]:
    """
    基于日志内容建议监控规则
    
    Args:
        log_entries: 日志条目列表
    
    Returns:
        监控规则建议
    """
    from log_parser import LogEntry
    
    suggestions = []
    
    # 分析错误模式
    patterns = extract_error_patterns(log_entries)
    
    for pattern in patterns[:5]:  # 前5个模式
        suggestions.append({
            "rule_name": f"监控: {pattern['pattern'][:50]}...",
            "condition": f"日志消息匹配: {pattern['pattern'][:100]}",
            "threshold": f"5分钟内出现 {max(1, pattern['count'] // 10)} 次",
            "severity": "high" if pattern['count'] > 10 else "medium",
            "action": "发送告警通知"
        })
    
    # 基于来源的建议
    stats = generate_statistics(log_entries)
    for source, count in stats.get("by_source", {}).items():
        if count > 100:  # 高频来源
            suggestions.append({
                "rule_name": f"监控 {source} 错误率",
                "condition": f"来源为 {source} 且级别为 ERROR",
                "threshold": "错误率超过 5%",
                "severity": "medium",
                "action": "检查服务健康状态"
            })
    
    return suggestions


# 工具注册表（供 Agent 使用）
AGENT_TOOLS = {
    "search_similar_errors": search_similar_errors,
    "analyze_error_frequency": analyze_error_frequency,
    "extract_error_patterns": extract_error_patterns,
    "correlate_events": correlate_events,
    "generate_statistics": generate_statistics,
    "suggest_monitoring_rules": suggest_monitoring_rules,
}


def get_tool(name: str) -> Optional[callable]:
    """获取工具函数"""
    return AGENT_TOOLS.get(name)


def list_tools() -> List[str]:
    """列出所有可用工具"""
    return list(AGENT_TOOLS.keys())
