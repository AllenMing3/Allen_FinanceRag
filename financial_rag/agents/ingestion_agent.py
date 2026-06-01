"""
IngestionAgent — 财报/经济新闻数据摄取

功能:
- 从文件、URL、API 获取原始财务数据
- 初步解析 PDF/HTML/TXT 格式
- 按季度/年度/公司组织数据结构
"""
import os
from typing import Dict, Any, List, Optional

from financial_rag.core.coordinator import BaseAgent, AgentContext, AgentResult


class IngestionAgent(BaseAgent):
    """
    Agent 1: 数据摄取

    负责将原始财报/新闻文本加载到系统中
    """

    def __init__(self):
        super().__init__(
            name="IngestionAgent",
            description="财报与新闻数据摄取"
        )

    def process(self, context: AgentContext) -> AgentResult:
        """摄取数据源"""
        raw_input = context.raw_input

        # 判断输入类型
        source_type = self._detect_source_type(raw_input)
        documents = []

        if source_type == "file":
            documents = self._ingest_file(raw_input)
        elif source_type == "text":
            documents = self._ingest_text(raw_input)
        elif source_type == "directory":
            documents = self._ingest_directory(raw_input)
        else:
            documents = [{"text": raw_input, "source": "direct_input"}]

        return AgentResult(
            success=len(documents) > 0,
            message=f"摄取 {len(documents)} 份文档",
            data=documents,
            context_updates={
                "parsed_data": documents,
                "metadata": {"source_type": source_type, "doc_count": len(documents)}
            }
        )

    def _detect_source_type(self, raw_input: str) -> str:
        if not raw_input:
            return "empty"
        if os.path.isfile(raw_input):
            return "file"
        if os.path.isdir(raw_input):
            return "directory"
        return "text"

    def _ingest_file(self, path: str) -> List[Dict]:
        """从文件摄取"""
        # 实际实现: PDF/HTML/TXT 解析
        # 这里 pass，保留接口
        pass
        return []

    def _ingest_text(self, text: str) -> List[Dict]:
        """从文本直接摄取"""
        # 实际实现: 分块、清洗、标准化
        pass
        return []

    def _ingest_directory(self, dir_path: str) -> List[Dict]:
        """从目录批量摄取"""
        # 实际实现: 遍历目录，按公司/季度归类
        pass
        return []
