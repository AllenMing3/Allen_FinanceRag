"""
配置文件 - 日志解析和知识库问答系统配置
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class LogParserConfig:
    """日志解析配置"""
    # 支持的日志格式
    supported_formats: List[str] = field(default_factory=lambda: [
        "nginx",
        "apache",
        "syslog",
        "json",
        "csv",
        "application",
        "custom"
    ])
    
    # 日志文件编码
    encoding: str = "utf-8"
    
    # 日志分块大小（行数）
    chunk_size: int = 1000
    
    # 时间戳格式列表
    timestamp_formats: List[str] = field(default_factory=lambda: [
        "%Y-%m-%d %H:%M:%S",
        "%d/%b/%Y:%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%b %d %H:%M:%S",
    ])
    
    # 自定义正则表达式模式（用于特定日志格式）
    custom_patterns: Dict[str, str] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """RAG 配置"""
    # 向量存储路径
    vector_store_path: str = "./storage/vector_store"
    
    # 嵌入模型
    embedding_model: str = "text-embedding-3-small"
    
    # LLM 模型
    llm_model: str = "gpt-4o-mini"
    
    # 检索 top-k
    similarity_top_k: int = 5
    
    # 响应模式
    response_mode: str = "compact"
    
    # 是否启用聊天模式
    enable_chat_mode: bool = True
    
    # 系统提示词
    system_prompt: str = """你是一个专业的日志分析助手。你的任务是：
1. 分析用户提供的日志内容
2. 根据知识库中的信息解释日志含义
3. 识别潜在的问题和异常
4. 提供解决方案或建议

请基于检索到的上下文信息回答，如果信息不足请明确说明。"""


@dataclass
class AppConfig:
    """应用配置"""
    # OpenAI API
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    
    # 日志配置
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # 子配置
    log_parser: LogParserConfig = field(default_factory=LogParserConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    
    # 数据目录
    data_dir: str = "./data"
    knowledge_base_dir: str = "./knowledge_base"
    output_dir: str = "./output"
    
    def __post_init__(self):
        # 确保目录存在
        for dir_path in [self.data_dir, self.knowledge_base_dir, self.output_dir, self.rag.vector_store_path]:
            os.makedirs(dir_path, exist_ok=True)


# 全局配置实例
config = AppConfig()
