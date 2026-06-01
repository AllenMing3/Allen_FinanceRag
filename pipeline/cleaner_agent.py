"""
Agent 1: 清洗 Agent - 去噪、去垃圾、标准化
"""
from typing import Dict, Any, List
import re
import logging

logger = logging.getLogger(__name__)


class CleanerAgent:
    """
    清洗 Agent - 负责清洗垃圾信息

    功能:
    1. 去除空白/换行噪音
    2. 过滤非日志格式内容
    3. 去除重复行
    4. 标准化时间戳格式
    5. 过滤低信息量行(纯数字/纯符号)
    """

    # 垃圾模式
    NOISE_PATTERNS = [
        re.compile(r'^\s*$'),                        # 空行
        re.compile(r'^[=\-*#]{3,}$'),                # 分隔线
        re.compile(r'^\d+$'),                         # 纯数字
        re.compile(r'^[.,;:!?]+$'),                   # 纯标点
        re.compile(r'^[a-zA-Z]{1,2}$'),              # 单字母
    ]

    # 日志格式正则
    LOG_PATTERNS = [
        # 标准日志格式: 2024-01-01 12:00:00 LEVEL message
        re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'),
        # 方括号格式: [2024-01-01 12:00:00]
        re.compile(r'\[.*?\d{4}-\d{2}-\d{2}.*?\]'),
        # syslog 格式
        re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'),
    ]

    def clean(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        清洗输入文本

        Args:
            text: 原始文本
            context: 额外上下文

        Returns:
            {"text": 清洗后文本, "removed": 移除行数, "stats": 统计信息}
        """
        context = context or {}
        original_lines = text.split('\n')
        removed_count = 0
        kept_lines = []
        stats = {"total": len(original_lines), "kept": 0, "noise": 0, "duplicate": 0}

        seen = set()

        for line in original_lines:
            stripped = line.strip()

            # 1. 去重
            if stripped in seen:
                removed_count += 1
                stats["duplicate"] += 1
                continue
            seen.add(stripped)

            # 2. 过滤噪声
            if self._is_noise(stripped):
                removed_count += 1
                stats["noise"] += 1
                continue

            # 3. 标准化
            cleaned = self._normalize(stripped)
            kept_lines.append(cleaned)

        stats["kept"] = len(kept_lines)
        stats["removed"] = removed_count

        result = "\n".join(kept_lines)

        logger.info(
            f"CleanerAgent: {len(original_lines)}行 -> {len(kept_lines)}行 "
            f"(去重{stats['duplicate']}, 噪声{stats['noise']})"
        )

        return {
            "text": result,
            "removed": removed_count,
            "stats": stats,
            "is_log": self._detect_log_format(kept_lines),
        }

    def _is_noise(self, line: str) -> bool:
        """判断是否为噪声行"""
        for pattern in self.NOISE_PATTERNS:
            if pattern.match(line):
                return True
        return False

    def _normalize(self, line: str) -> str:
        """标准化行"""
        # 统一空格
        line = re.sub(r'\s+', ' ', line)
        # 去除首尾空白
        line = line.strip()
        return line

    def _detect_log_format(self, lines: List[str]) -> str:
        """检测日志格式"""
        if not lines:
            return "unknown"

        for i, line in enumerate(lines[:20]):  # 检查前20行
            for pattern in self.LOG_PATTERNS:
                if pattern.search(line):
                    return "structured_log"

        return "plain_text"
