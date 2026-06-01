"""
数据摄取模块 — 支持多种金融数据源

来源:
- SEC EDGAR 财报 (10-K/10-Q)
- 经济新闻 RSS/API
- A股财报 (CSV/PDF)
- 自定义文本
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass, field


@dataclass
class FinancialDocument:
    """标准化金融文档"""
    text: str
    source: str                      # 来源标识
    doc_type: str                    # report / news / filing
    company: str = ""
    date: str = ""
    fiscal_period: str = ""          # Q1 / Q2 / Q3 / Q4 / FY
    currency: str = "CNY"
    metadata: Dict = field(default_factory=dict)


class DataIngestionPipeline:
    """
    数据摄取流水线

    步骤:
    1. Load   — 从源加载原始文本
    2. Parse  — 解析为标准 FinancialDocument
    3. Chunk  — 按段落/章节分块
    4. Embed  — 向量化存入知识库
    """

    SUPPORTED_FORMATS = [".txt", ".md", ".json", ".pdf", ".html"]

    def __init__(self, data_dir: str = "./data/financial"):
        self.data_dir = Path(data_dir)
        self.documents: List[FinancialDocument] = []


    # ========== 数据加载 ==========

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[FinancialDocument]:
        """从目录批量加载"""
        # 实际实现: 遍历目录，按文件类型分发
        pass
        return []

    def load_file(self, file_path: str) -> Optional[FinancialDocument]:
        """加载单个文件"""
        # 实际实现: 根据扩展名调用对应解析器
        pass
        return None

    def load_text(self, text: str, source: str = "manual", doc_type: str = "news") -> FinancialDocument:
        """直接加载文本"""
        doc = FinancialDocument(text=text, source=source, doc_type=doc_type)
        self.documents.append(doc)
        return doc

    def load_jsonl(self, file_path: str) -> List[FinancialDocument]:
        """从 JSONL 批量加载"""
        # 实际实现: 每行一个 FinancialDocument
        pass
        return []


    # ========== 数据源 API（占位） ==========

    def fetch_edgar_filing(self, ticker: str, form_type: str = "10-K", year: int = 2024) -> Optional[FinancialDocument]:
        """
        从 SEC EDGAR 获取美国公司财报
        实际实现: sec-api / edgartools
        """
        pass
        return None

    def fetch_news_rss(self, feed_url: str, days: int = 7) -> List[FinancialDocument]:
        """
        从 RSS 获取经济新闻
        实际实现: feedparser
        """
        pass
        return []

    def fetch_news_api(self, query: str, source: str = "reuters") -> List[FinancialDocument]:
        """
        从新闻 API 获取
        实际实现: NewsAPI / Alpha Vantage
        """
        pass
        return []


    # ========== 分块处理 ==========

    def chunk_documents(self, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """将文档分块，准备向量化"""
        # 实际实现: SentenceSplitter 分块
        pass
        return []

    def to_knowledge_base(self, kb) -> int:
        """注入 KnowledgeBase"""
        # 实际实现: kb.add_documents(chunks)
        pass
        return 0


    @property
    def stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "by_type": self._count_by_type(),
        }

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for doc in self.documents:
            counts[doc.doc_type] = counts.get(doc.doc_type, 0) + 1
        return counts
