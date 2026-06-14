"""
IngestionAgent — 财报/经济新闻数据摄取 (Function Calling 版)

Agent 负责数据加载和编排，抽取逻辑委托给 tools:
- extract_document_metadata: 元数据抽取 (LLM-first)
- detect_document_type: 文档类型检测 (纯关键词)

保留的文件加载能力:
- JSONL / TXT / PDF / 目录批量摄取
- 文本清洗
- 元数据评分
"""
import os
import re
import json
from typing import Dict, Any, List

from financial_rag.core.base import BaseAgent, AgentContext, AgentResult


class IngestionAgent(BaseAgent):
    """
    Agent 1: 数据摄取

    负责将原始财报/新闻文本加载到系统中。
    元数据抽取委托给 extract_document_metadata 工具，
    Agent 只做数据加载 + 质量评分。
    """

    # 期望的元数据字段
    EXPECTED_METADATA_FIELDS = [
        "source", "company", "date", "fiscal_period",
        "currency", "doc_type", "text_length",
    ]

    def __init__(self):
        super().__init__(
            name="IngestionAgent",
            description="财报与新闻数据摄取 (Function Calling)"
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
            documents = [{"text": raw_input, "metadata": {"source": "direct_input"}}]

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

        first_doc = documents[0]
        meta = first_doc.get("metadata", {}) if isinstance(first_doc, dict) else {}

        all_fields = {}
        for field_name in self.EXPECTED_METADATA_FIELDS:
            if field_name in first_doc:
                all_fields[field_name] = bool(first_doc[field_name])
            elif field_name in meta:
                all_fields[field_name] = bool(meta[field_name])
            else:
                all_fields[field_name] = False

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

    # ===================== 文件摄取 =====================

    def _ingest_file(self, path: str) -> List[Dict]:
        """从文件摄取 — 支持 JSONL / TXT / PDF"""
        documents = []
        ext = os.path.splitext(path)[1].lower()

        if ext == ".jsonl":
            documents = self._load_jsonl(path)
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            documents = self._ingest_text(text)
        elif ext == ".pdf":
            try:
                text = self._parse_pdf(path)
                documents = self._ingest_text(text)
            except Exception as e:
                documents = [{"text": f"[PDF未解析] {os.path.basename(path)}",
                              "metadata": {"source": path, "doc_type": "PDF文件", "error": str(e)}}]
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents = self._ingest_text(text)
            except UnicodeDecodeError:
                documents = [{"text": f"[二进制文件] {os.path.basename(path)}",
                              "metadata": {"source": path, "doc_type": "未知格式"}}]

        return documents

    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载 JSONL 文件，每行一个 JSON 对象"""
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if "text" not in doc:
                        continue
                    if "metadata" not in doc:
                        doc["metadata"] = {
                            k: v for k, v in doc.items()
                            if k in self.EXPECTED_METADATA_FIELDS
                        }
                    doc["metadata"]["text_length"] = len(doc.get("text", ""))
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"[IngestionAgent] JSONL 第{line_num}行解析失败: {e}")
        return documents

    def _parse_pdf(self, path: str) -> str:
        """解析 PDF 文件为纯文本"""
        try:
            import pymupdf  # PyMuPDF
            doc = pymupdf.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            raise ImportError("PDF 解析需要安装 PyMuPDF: pip install PyMuPDF")

    # ===================== 文本摄取 (核心 — 调用工具) =====================

    def _ingest_text(self, text: str) -> List[Dict]:
        """
        从纯文本摄取。

        处理流程：
        1. 清洗文本
        2. 调用 extract_document_metadata 工具提取元数据
        3. 调用 detect_document_type 工具补充文档类型
        """
        if not text or not text.strip():
            return []

        # 1. 清洗文本
        text = self._clean_text(text)

        # 2. 调用工具提取元数据
        metadata = {}
        try:
            metadata = self.call_tool("extract_document_metadata", text=text)
        except RuntimeError as e:
            print(f"[IngestionAgent] 元数据抽取失败: {e}")
            metadata = {"_confidence": "none"}

        # 3. 补充文档类型
        if not metadata.get("doc_type"):
            try:
                doc_type = self.call_tool("detect_document_type", text=text)
                metadata["doc_type"] = doc_type
            except RuntimeError:
                metadata["doc_type"] = "其他"

        # 4. 补充 text_length
        metadata["text_length"] = len(text)

        return [{"text": text, "metadata": metadata}]

    def _clean_text(self, text: str) -> str:
        """清洗文本：去重空白、统一换行、移除控制字符"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
        text = text.strip()
        return text

    # ===================== 目录批量摄取 =====================

    def _ingest_directory(self, dir_path: str) -> List[Dict]:
        """从目录批量摄取 — 遍历文件，按公司/季度归类"""
        documents = []
        supported_exts = {".jsonl", ".txt", ".pdf"}

        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_exts:
                    filepath = os.path.join(root, filename)
                    try:
                        docs = self._ingest_file(filepath)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"[IngestionAgent] 文件读取失败: {filepath} — {e}")

        return documents
