"""
日志解析 Agent - 负责解析各种格式的日志
"""
from typing import List, Dict
from .base_agent import BaseAgent, AgentContext, AgentResponse
from log_parser import LogParserFactory, LogProcessor, LogEntry


class LogParserAgent(BaseAgent):
    """
    日志解析 Agent
    
    职责：
    1. 自动检测日志格式
    2. 解析日志文件
    3. 提取结构化信息
    4. 生成日志统计
    """
    
    def __init__(self):
        super().__init__(
            name="LogParserAgent",
            description="解析各种格式的日志文件，提取结构化信息"
        )
        self.parsed_entries: List[LogEntry] = []
        self.parser_info: Dict = {}
    
    def process(self, context: AgentContext) -> AgentResponse:
        """
        解析日志
        
        期望 context.raw_input 为日志文件路径或日志内容
        """
        input_data = context.raw_input
        
        # 判断输入类型
        if isinstance(input_data, str):
            if input_data.endswith('.log') or '/' in input_data or '\\' in input_data:
                # 文件路径
                return self._parse_file(input_data, context)
            else:
                # 直接是日志内容
                return self._parse_content(input_data, context)
        elif isinstance(input_data, list):
            # 已经是日志条目列表
            self.parsed_entries = input_data
            return self._build_response(context)
        else:
            return AgentResponse(
                success=False,
                message="不支持的输入类型，请提供日志文件路径或日志内容"
            )
    
    def _parse_file(self, file_path: str, context: AgentContext) -> AgentResponse:
        """解析日志文件"""
        import os
        
        if not os.path.exists(file_path):
            return AgentResponse(
                success=False,
                message=f"日志文件不存在: {file_path}"
            )
        
        # 自动检测格式并获取解析器
        parser = LogParserFactory.get_parser(file_path)
        self.parser_info = {
            "parser_type": parser.__class__.__name__,
            "file_path": file_path
        }
        
        # 解析文件
        processor = LogProcessor(parser)
        self.parsed_entries = processor.process_file(file_path)
        
        # 更新上下文
        context.parsed_data = self.parsed_entries
        
        return self._build_response(context)
    
    def _parse_content(self, content: str, context: AgentContext) -> AgentResponse:
        """解析日志内容（多行文本）"""
        lines = content.strip().split('\n')
        
        # 使用通用解析器
        from log_parser import GenericLogParser
        parser = GenericLogParser()
        
        self.parsed_entries = []
        for line in lines:
            if line.strip():
                entry = parser.parse(line)
                if entry:
                    self.parsed_entries.append(entry)
        
        self.parser_info = {
            "parser_type": "GenericLogParser",
            "line_count": len(lines)
        }
        
        context.parsed_data = self.parsed_entries
        
        return self._build_response(context)
    
    def _build_response(self, context: AgentContext) -> AgentResponse:
        """构建响应"""
        if not self.parsed_entries:
            return AgentResponse(
                success=False,
                message="未能解析到任何日志条目"
            )
        
        # 生成统计信息
        stats = self._generate_stats()
        
        # 添加到上下文发现
        context.add_finding(f"成功解析 {len(self.parsed_entries)} 条日志")
        context.add_finding(f"检测到 {stats.get('error_count', 0)} 个错误")
        
        return AgentResponse(
            success=True,
            data={
                "entries": self.parsed_entries,
                "stats": stats,
                "parser_info": self.parser_info
            },
            message=f"成功解析 {len(self.parsed_entries)} 条日志，发现 {stats.get('error_count', 0)} 个错误",
            context_updates={
                "parsed_data": self.parsed_entries,
                "log_stats": stats
            }
        )
    
    def _generate_stats(self) -> Dict:
        """生成统计信息"""
        stats = {
            "total": len(self.parsed_entries),
            "by_level": {},
            "by_source": {},
            "error_count": 0,
            "warning_count": 0
        }
        
        for entry in self.parsed_entries:
            # 级别统计
            level = entry.level or "UNKNOWN"
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            # 错误/警告计数
            if level in ["ERROR", "CRITICAL", "FATAL"]:
                stats["error_count"] += 1
            elif level in ["WARN", "WARNING"]:
                stats["warning_count"] += 1
            
            # 来源统计
            source = entry.source or "UNKNOWN"
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        return stats
    
    def can_handle(self, context: AgentContext) -> bool:
        """判断是否能处理 - 只要有原始输入就可以尝试解析"""
        return bool(context.raw_input)
    
    def get_entries_by_level(self, level: str) -> List[LogEntry]:
        """获取指定级别的日志条目"""
        return [e for e in self.parsed_entries if e.level and e.level.upper() == level.upper()]
    
    def get_error_entries(self) -> List[LogEntry]:
        """获取错误条目"""
        return [e for e in self.parsed_entries 
                if e.level and e.level.upper() in ["ERROR", "CRITICAL", "FATAL"]]
