"""
日志解析模块 - 支持多种日志格式的解析和处理
"""
import json
import re
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config import config


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: Optional[datetime]
    level: Optional[str]  # ERROR, WARN, INFO, DEBUG 等
    source: Optional[str]  # 日志来源（服务名、IP等）
    message: str
    raw_line: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_document_text(self) -> str:
        """转换为文档文本格式"""
        parts = []
        if self.timestamp:
            parts.append(f"时间: {self.timestamp.isoformat()}")
        if self.level:
            parts.append(f"级别: {self.level}")
        if self.source:
            parts.append(f"来源: {self.source}")
        parts.append(f"消息: {self.message}")
        
        if self.metadata:
            for key, value in self.metadata.items():
                parts.append(f"{key}: {value}")
        
        return "\n".join(parts)


class LogParser(ABC):
    """日志解析器基类"""
    
    @abstractmethod
    def parse(self, line: str) -> Optional[LogEntry]:
        """解析单行日志"""
        pass
    
    @abstractmethod
    def can_parse(self, sample_lines: List[str]) -> bool:
        """判断是否能解析给定的日志样本"""
        pass


class NginxLogParser(LogParser):
    """Nginx 日志解析器"""
    
    # 标准 Nginx combined 格式
    PATTERN = re.compile(
        r'^(?P<ip>\S+)\s+'           # IP 地址
        r'-\s+'                       # 占位符
        r'(?P<user>\S+)\s+'           # 用户
        r'\[(?P<time>[^\]]+)\]\s+'    # 时间
        r'"(?P<request>[^"]*)"\s+'   # 请求
        r'(?P<status>\d+)\s+'         # 状态码
        r'(?P<bytes>\d+)\s+'          # 字节数
        r'"(?P<referer>[^"]*)"\s+'   # Referer
        r'"(?P<ua>[^"]*)"'           # User-Agent
    )
    
    def parse(self, line: str) -> Optional[LogEntry]:
        match = self.PATTERN.match(line)
        if not match:
            return None
        
        data = match.groupdict()
        
        # 解析时间
        timestamp = None
        try:
            timestamp = datetime.strptime(data['time'], '%d/%b/%Y:%H:%M:%S %z')
        except:
            pass
        
        # 判断日志级别（基于状态码）
        status = int(data['status'])
        if status >= 500:
            level = "ERROR"
        elif status >= 400:
            level = "WARN"
        else:
            level = "INFO"
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            source=data['ip'],
            message=f"{data['request']} -> {status}",
            raw_line=line,
            metadata={
                "status_code": status,
                "bytes_sent": int(data['bytes']),
                "referer": data['referer'],
                "user_agent": data['ua'][:200]  # 截断过长的 UA
            }
        )
    
    def can_parse(self, sample_lines: List[str]) -> bool:
        parsed_count = sum(1 for line in sample_lines if self.parse(line))
        return parsed_count / len(sample_lines) > 0.8 if sample_lines else False


class SyslogParser(LogParser):
    """系统日志解析器"""
    
    # Syslog 格式
    PATTERNS = [
        # RFC 3164 格式
        re.compile(
            r'^(?P<month>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+'
            r'(?P<time>\d{2}:\d{2}:\d{2})\s+'
            r'(?P<host>\S+)\s+'
            r'(?P<service>[^\[:]+)(\[(?P<pid>\d+)\])?:\s+'
            r'(?P<message>.*)$'
        ),
        # 带年份的格式
        re.compile(
            r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})\s+'
            r'(?P<level>[A-Z]+)\s+'
            r'(?P<message>.*)$'
        )
    ]
    
    def parse(self, line: str) -> Optional[LogEntry]:
        for pattern in self.PATTERNS:
            match = pattern.match(line)
            if match:
                data = match.groupdict()
                
                # 解析时间戳
                timestamp = None
                if 'timestamp' in data:
                    try:
                        timestamp = datetime.fromisoformat(data['timestamp'].replace(' ', 'T'))
                    except:
                        pass
                elif 'month' in data:
                    try:
                        year = datetime.now().year
                        time_str = f"{year} {data['month']} {data['day']} {data['time']}"
                        timestamp = datetime.strptime(time_str, '%Y %b %d %H:%M:%S')
                    except:
                        pass
                
                # 提取日志级别
                level = data.get('level', 'INFO')
                message = data.get('message', line)
                source = data.get('host') or data.get('service', 'unknown')
                
                metadata = {k: v for k, v in data.items() if k not in ['timestamp', 'level', 'message', 'host', 'service'] and v}
                
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    source=source,
                    message=message,
                    raw_line=line,
                    metadata=metadata
                )
        
        return None
    
    def can_parse(self, sample_lines: List[str]) -> bool:
        parsed_count = sum(1 for line in sample_lines if self.parse(line))
        return parsed_count / len(sample_lines) > 0.5 if sample_lines else False


class JSONLogParser(LogParser):
    """JSON 格式日志解析器"""
    
    def parse(self, line: str) -> Optional[LogEntry]:
        try:
            data = json.loads(line.strip())
            
            # 尝试提取常见字段
            timestamp = None
            level = None
            source = None
            message = ""
            
            # 时间戳字段
            for ts_field in ['timestamp', 'time', '@timestamp', 'ts', 'datetime']:
                if ts_field in data:
                    try:
                        ts_val = data[ts_field]
                        if isinstance(ts_val, (int, float)):
                            timestamp = datetime.fromtimestamp(ts_val)
                        else:
                            for fmt in config.log_parser.timestamp_formats:
                                try:
                                    timestamp = datetime.strptime(str(ts_val), fmt)
                                    break
                                except:
                                    continue
                        if timestamp:
                            break
                    except:
                        pass
            
            # 日志级别字段
            for level_field in ['level', 'severity', 'log_level', 'status']:
                if level_field in data:
                    level = str(data[level_field]).upper()
                    break
            
            # 消息字段
            for msg_field in ['message', 'msg', 'log', 'text', 'content']:
                if msg_field in data:
                    message = str(data[msg_field])
                    break
            
            # 来源字段
            for src_field in ['source', 'service', 'app', 'application', 'host', 'logger']:
                if src_field in data:
                    source = str(data[src_field])
                    break
            
            if not message:
                message = line[:500]  # 使用原始行的一部分
            
            # 其他字段放入 metadata
            metadata = {k: v for k, v in data.items() if k not in [
                'timestamp', 'time', '@timestamp', 'ts', 'datetime',
                'level', 'severity', 'log_level', 'status',
                'message', 'msg', 'log', 'text', 'content',
                'source', 'service', 'app', 'application', 'host', 'logger'
            ]}
            
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=source,
                message=message,
                raw_line=line,
                metadata=metadata
            )
            
        except json.JSONDecodeError:
            return None
    
    def can_parse(self, sample_lines: List[str]) -> bool:
        parsed_count = sum(1 for line in sample_lines if self.parse(line))
        return parsed_count / len(sample_lines) > 0.8 if sample_lines else False


class GenericLogParser(LogParser):
    """通用日志解析器（兜底方案）"""
    
    # 尝试匹配常见模式
    LEVEL_PATTERN = re.compile(r'\b(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\b', re.IGNORECASE)
    TIMESTAMP_PATTERN = re.compile(r'\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})\b')
    
    def parse(self, line: str) -> Optional[LogEntry]:
        # 提取日志级别
        level_match = self.LEVEL_PATTERN.search(line)
        level = level_match.group(1).upper() if level_match else None
        
        # 提取时间戳
        timestamp = None
        ts_match = self.TIMESTAMP_PATTERN.search(line)
        if ts_match:
            ts_str = ts_match.group(1)
            for fmt in config.log_parser.timestamp_formats:
                try:
                    timestamp = datetime.strptime(ts_str, fmt)
                    break
                except:
                    continue
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            source=None,
            message=line.strip(),
            raw_line=line
        )
    
    def can_parse(self, sample_lines: List[str]) -> bool:
        return True  # 通用解析器总是可以解析


class LogParserFactory:
    """日志解析器工厂"""
    
    PARSERS = [
        NginxLogParser(),
        JSONLogParser(),
        SyslogParser(),
    ]
    
    @classmethod
    def get_parser(cls, file_path: str, log_format: Optional[str] = None) -> LogParser:
        """获取合适的解析器"""
        
        # 如果指定了格式，使用对应解析器
        if log_format:
            format_map = {
                'nginx': NginxLogParser(),
                'syslog': SyslogParser(),
                'json': JSONLogParser(),
            }
            if log_format in format_map:
                return format_map[log_format]
        
        # 自动检测格式
        sample_lines = cls._read_sample(file_path)
        
        for parser in cls.PARSERS:
            if parser.can_parse(sample_lines):
                print(f"检测到日志格式: {parser.__class__.__name__}")
                return parser
        
        print("使用通用日志解析器")
        return GenericLogParser()
    
    @classmethod
    def _read_sample(cls, file_path: str, num_lines: int = 10) -> List[str]:
        """读取日志样本"""
        lines = []
        try:
            with open(file_path, 'r', encoding=config.log_parser.encoding, errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= num_lines:
                        break
                    line = line.strip()
                    if line:
                        lines.append(line)
        except Exception as e:
            print(f"读取日志样本失败: {e}")
        
        return lines


class LogProcessor:
    """日志处理器 - 批量处理日志文件"""
    
    def __init__(self, parser: LogParser):
        self.parser = parser
        self.entries: List[LogEntry] = []
    
    def process_file(self, file_path: str, progress_callback: Optional[Callable] = None) -> List[LogEntry]:
        """处理日志文件"""
        self.entries = []
        line_count = 0
        parsed_count = 0
        
        with open(file_path, 'r', encoding=config.log_parser.encoding, errors='ignore') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                
                if not line:
                    continue
                
                entry = self.parser.parse(line)
                if entry:
                    self.entries.append(entry)
                    parsed_count += 1
                
                # 进度回调
                if progress_callback and line_count % 1000 == 0:
                    progress_callback(line_count, parsed_count)
        
        print(f"处理完成: {line_count} 行, 成功解析 {parsed_count} 条")
        return self.entries
    
    def process_directory(self, dir_path: str, pattern: str = "*.log") -> Dict[str, List[LogEntry]]:
        """处理目录中的所有日志文件"""
        results = {}
        path = Path(dir_path)
        
        for log_file in path.glob(pattern):
            print(f"\n处理文件: {log_file}")
            parser = LogParserFactory.get_parser(str(log_file))
            self.parser = parser
            entries = self.process_file(str(log_file))
            results[str(log_file)] = entries
        
        return results
    
    def get_entries_by_level(self, level: str) -> List[LogEntry]:
        """按级别筛选日志条目"""
        return [e for e in self.entries if e.level and e.level.upper() == level.upper()]
    
    def get_error_entries(self) -> List[LogEntry]:
        """获取错误日志"""
        return [e for e in self.entries if e.level and e.level.upper() in ['ERROR', 'CRITICAL', 'FATAL']]
    
    def to_documents(self) -> List[str]:
        """转换为文档列表"""
        return [entry.to_document_text() for entry in self.entries]
    
    def get_statistics(self) -> Dict:
        """获取日志统计信息"""
        stats = {
            'total_entries': len(self.entries),
            'by_level': {},
            'by_source': {},
            'time_range': {'start': None, 'end': None}
        }
        
        timestamps = []
        for entry in self.entries:
            # 级别统计
            level = entry.level or 'UNKNOWN'
            stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
            
            # 来源统计
            source = entry.source or 'UNKNOWN'
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            # 时间范围
            if entry.timestamp:
                timestamps.append(entry.timestamp)
        
        if timestamps:
            stats['time_range']['start'] = min(timestamps).isoformat()
            stats['time_range']['end'] = max(timestamps).isoformat()
        
        return stats


# 便捷函数
def parse_log_file(file_path: str, log_format: Optional[str] = None) -> List[LogEntry]:
    """解析日志文件的便捷函数"""
    parser = LogParserFactory.get_parser(file_path, log_format)
    processor = LogProcessor(parser)
    return processor.process_file(file_path)
