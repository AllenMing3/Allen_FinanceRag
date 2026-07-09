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
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

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
            documents = [{"text": raw_input, "meta": {"source": "direct_input"}}]

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
        meta = first_doc.get("meta", {}) if isinstance(first_doc, dict) else {}

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
        """从文件摄取 — 支持 JSONL / TXT / PDF / PNG/JPG"""
        documents = []
        ext = os.path.splitext(path)[1].lower()

        if ext == ".jsonl":
            documents = self._load_jsonl(path)
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            documents = self._ingest_text(text)
        elif ext == ".pdf":
            documents = self._ingest_pdf(path)
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            documents = self._ingest_image(path)
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents = self._ingest_text(text)
            except UnicodeDecodeError as e:
                logger.warning(f"[IngestionAgent] binary file skipped: {path} — {e}")
                documents = [{"text": f"[二进制文件] {os.path.basename(path)}",
                              "meta": {"source": path, "doc_type": "未知格式"}}]

        return documents

    def _ingest_pdf(self, path: str) -> List[Dict]:
        """解析 PDF 文件并走文本摄取流程（委托给 parse_pdf_file 工具）"""
        try:
            result = self.call_tool("parse_pdf_file", pdf_path=path)
            text = result.get("text", "")
            if not text or not text.strip():
                error = result.get("_error", "")
                warning = result.get("_warning", "")
                note = error or warning or "无文本内容"
                return [{"text": f"[PDF未解析] {os.path.basename(path)}: {note}",
                         "meta": {"source": path, "doc_type": "PDF文件", "parse_type": "pdf"}}]
            # PDF 解析成功，走标准文本摄取（清洗 + 分类 + 元数据抽取）
            docs = self._ingest_text(text)
            # 补充 PDF 元数据
            for doc in docs:
                doc.setdefault("meta", {})
                doc["meta"]["source"] = doc["meta"].get("source", path)
                doc["meta"]["parse_type"] = "pdf"
                doc["meta"]["original_file"] = os.path.basename(path)
                doc["meta"]["pdf_page_count"] = result.get("page_count", 0)
            return docs
        except RuntimeError as e:
            logger.error(f"[IngestionAgent] PDF tool call failed: {path} — {e}")
            return [{"text": f"[PDF未解析] {os.path.basename(path)}",
                     "meta": {"source": path, "doc_type": "PDF文件", "error": str(e)}}]

    def _ingest_image(self, path: str) -> List[Dict]:
        """用多模态模型解析图片并走文本摄取流程（委托给 describe_image_file 工具）"""
        try:
            result = self.call_tool("describe_image_file", image_path=path)
            description = result.get("description", "")
            if not description or not description.strip():
                error = result.get("_error", "无描述")
                return [{"text": f"[图片未解析] {os.path.basename(path)}: {error}",
                         "meta": {"source": path, "doc_type": "图片", "parse_type": "image"}}]

            # 图片描述走标准文本摄取（清洗 + 分类 + 元数据抽取）
            docs = self._ingest_text(description)
            # 补充图片元数据
            for doc in docs:
                doc.setdefault("meta", {})
                doc["meta"]["source"] = doc["meta"].get("source", path)
                doc["meta"]["parse_type"] = "image"
                doc["meta"]["original_file"] = os.path.basename(path)
                doc["meta"]["vision_model"] = result.get("vision_model", "")
            return docs
        except RuntimeError as e:
            logger.error(f"[IngestionAgent] Image tool call failed: {path} — {e}")
            return [{"text": f"[图片未解析] {os.path.basename(path)}",
                     "meta": {"source": path, "doc_type": "图片", "error": str(e)}}]

    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载 JSONL 文件，每行一个 JSON 对象"""
        documents = []
        skipped_lines = 0
        no_text_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if "text" not in doc:
                        no_text_lines += 1
                        continue
                    if "meta" not in doc:
                        doc["meta"] = {
                            k: v for k, v in doc.items()
                            if k in self.EXPECTED_METADATA_FIELDS
                        }
                    doc["meta"]["text_length"] = len(doc.get("text", ""))
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    logger.warning(f"[IngestionAgent] JSONL {path}:{line_num} parse error: {e}")

        if skipped_lines > 0:
            logger.warning(f"[IngestionAgent] JSONL {path}: {skipped_lines} lines skipped (JSON errors)")
        if no_text_lines > 0:
            logger.warning(f"[IngestionAgent] JSONL {path}: {no_text_lines} lines missing 'text' field")
        logger.info(f"[IngestionAgent] Loaded {len(documents)} docs from {path}")
        return documents

    # ===================== 文本摄取 (核心 — 调用工具) =====================

    def _ingest_text(self, text: str) -> List[Dict]:
        """
        从纯文本摄取。

        处理流程：
        1. 清洗文本 (TextPreprocessor)
        2. 相关性门控 (RelevanceGate)
        3. 文档分类 (DocTypeClassifier)
        4. 调用 extract_document_metadata 工具补充元数据
        5. LLM 补充 doc_type (仅当分类器置信度低时)
        """
        if not text or not text.strip():
            return []

        # 1. 清洗文本
        text = self._clean_text(text)

        # 2. 相关性门控
        from financial_rag.retrievers.preprocessor import RelevanceGate, DocTypeClassifier
        gate = RelevanceGate()
        passed, reason, kw_count = gate.check(text)
        if not passed:
            return [{"text": text, "meta": {
                "_rejected": True, "_reject_reason": reason,
                "_keyword_count": kw_count,
            }}]

        # 3. 文档分类 (快速，无 LLM)
        classifier = DocTypeClassifier()
        classify_result = classifier.classify(text)
        doc_type = classify_result["doc_type"]
        doc_type_confidence = classify_result["confidence"]

        # 4. 调用工具提取元数据
        metadata = {}
        try:
            metadata = self.call_tool("extract_document_metadata", text=text)
        except RuntimeError as e:
            logger.warning(f"[IngestionAgent] metadata extraction failed: {e}")
            metadata = {"_confidence": "none", "_extraction_error": str(e)}

        # 5. 补充 doc_type (分类器结果优先，LLM 仅当兑底)
        if not metadata.get("doc_type"):
            if doc_type != "other" and doc_type_confidence >= 0.5:
                metadata["doc_type"] = doc_type
            else:
                try:
                    llm_type = self.call_tool("detect_document_type", text=text)
                    metadata["doc_type"] = llm_type
                except RuntimeError:
                    metadata["doc_type"] = doc_type  # 分类器结果兑底

        # 补充分类信息
        metadata["text_length"] = len(text)
        metadata["_classify"] = classify_result
        metadata["_relevance_keywords"] = kw_count

        return [{"text": text, "meta": metadata}]

    def _clean_text(self, text: str) -> str:
        """清洗文本：委托给 TextPreprocessor"""
        from financial_rag.retrievers.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        return preprocessor.process(text)

    # ===================== 目录批量摄取 =====================

    def _ingest_directory(self, dir_path: str) -> List[Dict]:
        """从目录批量摄取 — 遍历文件，按公司/季度归类"""
        documents = []
        supported_exts = {".jsonl", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
        failed_files = 0

        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_exts:
                    filepath = os.path.join(root, filename)
                    try:
                        docs = self._ingest_file(filepath)
                        documents.extend(docs)
                    except Exception as e:
                        failed_files += 1
                        logger.error(f"[IngestionAgent] file read failed: {filepath} — {e}")

        if failed_files > 0:
            logger.warning(
                f"[IngestionAgent] directory {dir_path}: {failed_files} files failed to read"
            )
        logger.info(f"[IngestionAgent] directory {dir_path}: {len(documents)} docs loaded")
        return documents
