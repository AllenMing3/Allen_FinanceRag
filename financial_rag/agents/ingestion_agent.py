"""
IngestionAgent — 财报/经济新闻数据摄取

功能:
- 从文件、URL、API 获取原始财务数据
- 初步解析 PDF/HTML/TXT 格式
- 按季度/年度/公司组织数据结构
- 【打分】元数据解析完整度评分
"""
import os
import re
import json
from typing import Dict, Any, List, Optional

from financial_rag.config import config
from financial_rag.core.base import BaseAgent, AgentContext, AgentResult
from financial_rag.llm.dashscope_client import get_llm
from financial_rag.prompts import (
    METADATA_EXTRACTION_SYSTEM,
    METADATA_EXTRACTION_PROMPT,
)
from financial_rag.agents.utils import build_news_context


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
        self._llm = None

    def _get_llm(self):
        """懒加载 LLM 实例"""
        if self._llm is None:
            try:
                self._llm = get_llm(
                    api_key=config.llm.api_key,
                    model=config.llm.model,
                    temperature=0.0,
                )
            except (ImportError, ValueError):
                self._llm = None  # 无 API key 时回退到纯规则模式
        return self._llm

    def process(self, context: AgentContext) -> AgentResult:
        """摄取数据源"""
        raw_input = context.raw_input

        # 提取新闻元数据作为先验知识（辅助解析）
        self._news_context = context.metadata.get("news_context", [])

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
            # PDF 解析（可选依赖）
            try:
                text = self._parse_pdf(path)
                documents = self._ingest_text(text)
            except Exception as e:
                # 回退：标记为未解析
                documents = [{"text": f"[PDF未解析] {os.path.basename(path)}",
                              "metadata": {"source": path, "doc_type": "PDF文件", "error": str(e)}}]
        else:
            # 尝试作为文本读取
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
                    # 确保有 text 和 metadata 字段
                    if "text" not in doc:
                        continue
                    if "metadata" not in doc:
                        # 尝试从顶层字段推断 metadata
                        doc["metadata"] = {
                            k: v for k, v in doc.items()
                            if k in self.EXPECTED_METADATA_FIELDS
                        }
                    # 补充 text_length
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

    # ===================== 文本摄取（核心） =====================

    def _ingest_text(self, text: str) -> List[Dict]:
        """
        从纯文本直接摄取。

        处理流程：
        1. 分块 — 将长文本按段落分割
        2. 清洗 — 去除多余空白和特殊字符
        3. 元数据提取 — 使用 LLM 或正则自动提取元数据
        4. doc_type 自动检测 — 财报/公告/新闻/政策
        """
        if not text or not text.strip():
            return []

        # 1. 清洗文本
        text = self._clean_text(text)

        # 2. 自动提取元数据
        metadata = self._auto_extract_metadata(text)

        # 3. 自动检测文档类型
        if not metadata.get("doc_type"):
            metadata["doc_type"] = self._detect_doc_type(text)

        # 4. 补充 text_length
        metadata["text_length"] = len(text)

        document = {
            "text": text,
            "metadata": metadata,
        }

        return [document]

    def _clean_text(self, text: str) -> str:
        """清洗文本：去重空白、统一换行、移除控制字符"""
        # 移除多余空白行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行内多余空格
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # 移除零宽字符
        text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
        # 首尾去空白
        text = text.strip()
        return text

    def _auto_extract_metadata(self, text: str) -> Dict[str, str]:
        """
        自动提取元数据。

        策略：
        1. 先用正则快速提取（source、date、company 等）
        2. 如果正则覆盖率不足（<4 个字段），尝试 LLM 提取
        3. 如果 LLM 不可用，仅用正则结果
        """
        # 步骤1: 正则快速提取
        regex_meta = self._regex_extract_metadata(text)

        # 计算已提取的有效字段数
        filled_count = sum(1 for v in regex_meta.values() if v)
        expected_keys = ["source", "company", "date", "doc_type", "currency"]
        regex_filled = sum(1 for k in expected_keys if regex_meta.get(k))

        # 步骤2: 如果正则覆盖率不足且有 LLM，用 LLM 补充
        if regex_filled < 4:
            llm_meta = self._llm_extract_metadata(text)
            if llm_meta:
                # 合并：正则优先（正则通常更精确），LLM 补充空字段
                merged = {}
                for k in expected_keys:
                    merged[k] = regex_meta.get(k) or llm_meta.get(k, "")
                # 补充其他字段
                for k in ["fiscal_period"]:
                    merged[k] = regex_meta.get(k) or llm_meta.get(k, "")
                return merged

        # 补充 fiscal_period
        if not regex_meta.get("fiscal_period"):
            regex_meta["fiscal_period"] = self._regex_extract_fiscal_period(text)

        return regex_meta

    def _regex_extract_metadata(self, text: str) -> Dict[str, str]:
        """基于正则的元数据快速提取"""
        metadata = {
            "source": "",
            "company": "",
            "date": "",
            "fiscal_period": "",
            "currency": "CNY",
            "doc_type": "",
        }

        # 提取日期：支持多种格式
        date_patterns = [
            (r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
            (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        ]
        for pattern, formatter in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    metadata["date"] = formatter(match)
                except (ValueError, IndexError):
                    pass
                break

        # 提取公司名称：常见中文公司名模式
        company_patterns = [
            r'([\u4e00-\u9fa5]{2,6}(?:集团|公司|控股|股份|科技|银行|证券|基金|保险|信托|租赁)(?:有限公司|股份有限公司)?)',
            r'([\u4e00-\u9fa5]{2,4}(?:茅台|阿里|腾讯|百度|京东|字节|美团|小米|华为|宁德|比亚迪))',
        ]
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                company = match.group(1)
                # 避免误匹配"有限公司"等通用后缀
                if len(company) >= 4:
                    metadata["company"] = company
                    break

        # 提取来源/发布机构
        source_patterns = [
            (r'(上交所|深交所|港交所|纽交所|纳斯达克)', lambda m: f"{m.group(1)}公告"),
            (r'中国人民银行|央行', lambda m: "央行公告"),
            (r'证监会|银保监会|国家金融监督管理总局', lambda m: f"{m.group(0)}公告"),
            (r'财政部|发改委|商务部|工信部', lambda m: f"{m.group(0)}公告"),
            (r'据(?:多家)?财经媒体', lambda m: "财经媒体"),
            (r'据([\u4e00-\u9fa5]{2,6}(?:社|报|网|新闻))', lambda m: m.group(1)),
        ]
        for pattern, formatter in source_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["source"] = formatter(match)
                break

        # 提取币种
        if re.search(r'人民币|亿元|万元|CNY|RMB', text):
            metadata["currency"] = "CNY"
        elif re.search(r'美元|USD|亿美元|百万美元', text):
            metadata["currency"] = "USD"
        elif re.search(r'港元|HKD|亿港元', text):
            metadata["currency"] = "HKD"

        # 提取财报期间
        metadata["fiscal_period"] = self._regex_extract_fiscal_period(text)

        return metadata

    def _regex_extract_fiscal_period(self, text: str) -> str:
        """提取财报期间"""
        patterns = [
            r'(\d{4})\s*年\s*年度报告',
            r'(\d{4})\s*年\s*第?\s*(\d)\s*季度?报告',
            r'(\d{4})\s*年\s*半年报',
            r'(\d{4})\s*财年',
            r'(\d{4})\s*年度?业绩',
            r'(\d{4})\s*年年报',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return f"{groups[0]}年度"
                elif len(groups) == 2:
                    return f"{groups[0]}Q{groups[1]}"
        return ""

    def _llm_extract_metadata(self, text: str) -> Optional[Dict[str, str]]:
        """使用 LLM 提取元数据（备用方案）"""
        llm = self._get_llm()
        if llm is None:
            return None

        # 取前 8000 字符发送给 LLM，保留足够的原文信息
        prompt_text = text[:8000]
        user_prompt = METADATA_EXTRACTION_PROMPT.format(text=prompt_text)

        # 注入新闻元数据作为先验知识
        system_prompt = METADATA_EXTRACTION_SYSTEM
        news_ctx = build_news_context(self._news_context)
        if news_ctx:
            system_prompt += f"\n\n以下是近期相关新闻动态，可辅助判断文档的主题、公司和时间背景：\n{news_ctx}"

        try:
            response = llm.chat(
                messages=user_prompt,
                system=system_prompt,
                max_tokens=512,
                temperature=0.0,
            )
            content = response.content.strip()
            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[IngestionAgent] LLM 元数据提取失败: {e}")

        return None

    def _detect_doc_type(self, text: str) -> str:
        """
        自动检测文档类型。

        检测规则（按优先级）：
        1. 财报特征：包含"营业收入"、"净利润"、"每股收益"等密集财务指标
        2. 公告特征：包含"公告"、"通知"、"决定"等公文用语
        3. 政策特征：央行/证监会/财政部等监管机构发布
        4. 新闻特征：包含"报道"、"记者"、"据悉"等新闻用语
        5. 研究报告：包含"评级"、"目标价"、"买入/卖出"等
        """
        # 财报特征
        financial_keywords = ["营业收入", "净利润", "每股收益", "毛利率", "ROE", "经营活动现金流",
                              "资产负债表", "利润表", "现金流量表", "归属于上市公司股东", "基本每股收益"]
        if sum(1 for kw in financial_keywords if kw in text) >= 3:
            return "年报"

        # 季报特征
        if re.search(r'第?\s*[一二三1-3]\s*季度?报告|Q[1-3]\s*报告', text):
            return "季报"

        # 政策文件特征
        policy_keywords = ["下调", "上调", "存款准备金", "利率", "LPR", "货币政策",
                          "监管", "行政处罚", "暂行办法", "通知", "决定"]
        policy_sources = ["中国人民银行", "央行", "证监会", "银保监会", "财政部",
                         "发改委", "国家金融监督管理总局"]
        if any(kw in text for kw in policy_sources):
            return "政策文件"
        if sum(1 for kw in policy_keywords if kw in text) >= 2:
            return "政策文件"

        # 公告特征
        if re.search(r'(公告|通知|声明|决定)\s*(编号|第|如下)', text):
            return "公告"

        # 研究报告特征
        research_keywords = ["评级", "目标价", "买入", "卖出", "增持", "减持", "中性",
                            "研报", "研究报告", "盈利预测"]
        if sum(1 for kw in research_keywords if kw in text) >= 2:
            return "研究报告"

        # 新闻特征（兜底）
        news_keywords = ["报道", "记者", "据悉", "据透露", "消息人士", "分析人士",
                        "最新消息", "快讯", "独家"]
        if any(kw in text for kw in news_keywords) or len(text) > 100:
            return "新闻报道"

        return "其他"

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
