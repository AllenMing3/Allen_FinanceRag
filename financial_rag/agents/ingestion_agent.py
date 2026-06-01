"""
IngestionAgent — 财报/经济新闻数据摄取

功能:
- 从文件、URL、API 获取原始财务数据
- 初步解析 PDF/HTML/TXT 格式
- 按季度/年度/公司组织数据结构
- 【打分】元数据解析完整度评分
"""
import os
from typing import Dict, Any, List, Optional

from financial_rag.core.coordinator import BaseAgent, AgentContext, AgentResult


class IngestionAgent(BaseAgent):
    """
    Agent 1: 数据摄取

    负责将原始财报/新闻文本加载到系统中
    同时评估 metadata 解析的质量
    """

    # 期望的元数据字段
    EXPECTED_METADATA_FIELDS = [
        "source", "company", "date", "fiscal_period",
        "currency", "doc_type", "text_length",
    ]

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

        # 评估 metadata 解析质量
        metadata_score, fields_found, fields_meta = self._evaluate_metadata(documents)

        return AgentResult(
            success=len(documents) > 0,
            message=f"摄取 {len(documents)} 份文档 (metadata覆盖率: {fields_found}/{len(self.EXPECTED_METADATA_FIELDS)})",
            data=documents,
            context_updates={
                "parsed_data": documents,
                "metadata": {
                    "source_type": source_type,
                    "doc_count": len(documents),
                    "metadata_score": metadata_score,
                    "metadata_fields_found": fields_found,
                    "metadata_fields_expected": len(self.EXPECTED_METADATA_FIELDS),
                    "metadata_detail": fields_meta,
                }
            }
        )

    # ===================== Metadata 评分 =====================

    def _evaluate_metadata(self, documents: List[Dict]) -> tuple:
        """
        评估元数据解析的完整性

        Returns:
            (score, fields_found, detail_dict)
        """
        if not documents:
            return 0.0, 0, {}

        # 取第一个文档的 metadata 字段检查
        first_doc = documents[0]
        meta = first_doc.get("metadata", {}) if isinstance(first_doc, dict) else {}

        # 合并文档自身字段
        all_fields = {}
        for field in self.EXPECTED_METADATA_FIELDS:
            if field in first_doc:
                all_fields[field] = bool(first_doc[field])
            elif field in meta:
                all_fields[field] = bool(meta[field])
            else:
                all_fields[field] = False

        fields_found = sum(1 for v in all_fields.values() if v)
        total = len(self.EXPECTED_METADATA_FIELDS)
        score = fields_found / max(total, 1)

        return score, fields_found, all_fields

    def evaluate_metadata(self, documents: List[Dict]) -> Dict:
        """公开 API: 获取 metadata 评分详情"""
        score, found, detail = self._evaluate_metadata(documents)
        return {
            "score": round(score, 3),
            "fields_found": found,
            "fields_expected": len(self.EXPECTED_METADATA_FIELDS),
            "fields_detail": detail,
        }

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
