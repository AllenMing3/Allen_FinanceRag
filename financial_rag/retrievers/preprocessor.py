"""
文本预处理器 — Text Preprocessor

三大能力:
1. 基础清洗 (TextPreprocessor) — HTML/URL/空白/控制字符/段落去重/样板去除
2. 相关性门控 (RelevanceGate) — 过滤完全无关的文档
3. 文档分类 (DocTypeClassifier) — 快速分类: news/financial_report/macro_data/query/other

用法:
    preprocessor = TextPreprocessor()
    clean_text = preprocessor.process(raw_text)

    gate = RelevanceGate()
    passed, reason, kw_count = gate.check(text)

    classifier = DocTypeClassifier()
    result = classifier.classify(text)
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple

from .dictionaries import (
    FINANCIAL_TERMS, INDUSTRY_TERMS, STOCK_MAP,
    DOC_TYPE_KEYWORDS, DOC_TYPE_PATTERNS,
)


# ===================== 清洗规则 (预编译) =====================

_HTML_TAG = re.compile(r'<[^>]+>')
_URL = re.compile(r'https?://\S+|www\.\S+')
_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_MULTI_NEWLINE = re.compile(r'\n{3,}')
_MULTI_SPACE = re.compile(r'[ \t]{2,}')
_MULTI_CJK_SPACE = re.compile(r'([，。！？；：])\s+')
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_ZERO_WIDTH = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff]')
_REPEATED_CHARS = re.compile(r'(.)\1{4,}')
_CHINESE_CHAR = re.compile(r'[\u4e00-\u9fff]')

# 样板行模式: 短行 (< 20字) + 常见样板词
_BOILERPLATE_PATTERN = re.compile(
    r'^(.{0,20}(?:版权|copyright|关注|微信|公众号|扫码|二维码|点击阅读原文|'
    r'免责声明|转载|来源|编辑|责编|版权与免责声明).{0,20})$',
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class CleanStats:
    """清洗统计"""
    original_len: int = 0
    cleaned_len: int = 0
    html_removed: int = 0
    urls_removed: int = 0
    control_removed: int = 0
    paragraphs_deduped: int = 0
    boilerplate_removed: int = 0

    @property
    def retention(self) -> float:
        if self.original_len == 0:
            return 0.0
        return self.cleaned_len / self.original_len

    @property
    def is_over_cleaned(self) -> bool:
        return 0 < self.retention < 0.3


# ===================== TextPreprocessor =====================

class TextPreprocessor:
    """
    文本清洗流水线 — 所有步骤可配置开关
    """

    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
        remove_control_chars: bool = True,
        remove_zero_width: bool = True,
        collapse_repeated: bool = True,
        collapse_whitespace: bool = True,
        strip: bool = True,
        dedup_paragraphs: bool = False,
        remove_boilerplate: bool = False,
    ):
        self._steps: List[Callable[[str, CleanStats], str]] = []
        if remove_html:
            self._steps.append(self._step_remove_html)
        if remove_urls:
            self._steps.append(self._step_remove_urls)
        if remove_emails:
            self._steps.append(self._step_remove_emails)
        if normalize_unicode:
            self._steps.append(self._step_normalize_unicode)
        if remove_control_chars:
            self._steps.append(self._step_remove_control)
        if remove_zero_width:
            self._steps.append(self._step_remove_zero_width)
        if collapse_repeated:
            self._steps.append(self._step_collapse_repeated)
        if remove_boilerplate:
            self._steps.append(self._step_remove_boilerplate)
        if collapse_whitespace:
            self._steps.append(self._step_collapse_whitespace)
        if dedup_paragraphs:
            self._steps.append(self._step_dedup_paragraphs)
        if strip:
            self._steps.append(self._step_strip)

    def process(self, text: str, collect_stats: bool = False) -> str:
        """清洗单条文本"""
        if not text:
            return text

        stats = CleanStats(original_len=len(text))
        for step in self._steps:
            text = step(text, stats)
        stats.cleaned_len = len(text)

        if collect_stats:
            self._last_stats = stats
        return text

    def process_batch(self, texts: List[str], collect_stats: bool = False) -> List[str]:
        """批量清洗"""
        return [self.process(t, collect_stats=collect_stats) for t in texts]

    def get_last_stats(self) -> Optional[CleanStats]:
        return getattr(self, '_last_stats', None)

    # ===================== 清洗步骤 =====================

    @staticmethod
    def _step_remove_html(text: str, stats: CleanStats) -> str:
        matches = _HTML_TAG.findall(text)
        stats.html_removed += len(matches)
        return _HTML_TAG.sub('', text)

    @staticmethod
    def _step_remove_urls(text: str, stats: CleanStats) -> str:
        matches = _URL.findall(text)
        stats.urls_removed += len(matches)
        return _URL.sub('', text)

    @staticmethod
    def _step_remove_emails(text: str, stats: CleanStats) -> str:
        return _EMAIL.sub('', text)

    @staticmethod
    def _step_normalize_unicode(text: str, stats: CleanStats) -> str:
        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def _step_remove_control(text: str, stats: CleanStats) -> str:
        before = len(text)
        text = _CONTROL_CHARS.sub('', text)
        stats.control_removed += before - len(text)
        return text

    @staticmethod
    def _step_remove_zero_width(text: str, stats: CleanStats) -> str:
        before = len(text)
        text = _ZERO_WIDTH.sub('', text)
        stats.control_removed += before - len(text)
        return text

    @staticmethod
    def _step_collapse_repeated(text: str, stats: CleanStats) -> str:
        return _REPEATED_CHARS.sub(r'\1\1\1', text)

    @staticmethod
    def _step_collapse_whitespace(text: str, stats: CleanStats) -> str:
        text = _MULTI_NEWLINE.sub('\n\n', text)
        text = _MULTI_SPACE.sub(' ', text)
        text = _MULTI_CJK_SPACE.sub(r'\1', text)
        return text

    @staticmethod
    def _step_strip(text: str, stats: CleanStats) -> str:
        return text.strip()

    @staticmethod
    def _step_dedup_paragraphs(text: str, stats: CleanStats) -> str:
        """段落级去重: 按 \\n\\n 分段，hash 去重"""
        paragraphs = text.split('\n\n')
        seen = set()
        unique = []
        for p in paragraphs:
            p_stripped = p.strip()
            if not p_stripped:
                unique.append(p)
                continue
            h = hash(p_stripped)
            if h in seen:
                stats.paragraphs_deduped += 1
                continue
            seen.add(h)
            unique.append(p)
        return '\n\n'.join(unique)

    @staticmethod
    def _step_remove_boilerplate(text: str, stats: CleanStats) -> str:
        """移除样板行: 版权声明、微信公众号推广、免责声明等"""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            if _BOILERPLATE_PATTERN.match(line.strip()):
                stats.boilerplate_removed += 1
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)


# ===================== RelevanceGate =====================

class RelevanceGate:
    """
    相关性门控 — 过滤完全无关的文档

    三层检查:
    1. 文本长度 (太短 = 无意义)
    2. 中文字符占比 (太低 = 垃圾/乱码)
    3. 领域关键词命中 (零命中 = 与金融无关)
    """

    def __init__(
        self,
        min_length: int = 20,
        min_chinese_ratio: float = 0.1,
        min_keywords: int = 1,
    ):
        self.min_length = min_length
        self.min_chinese_ratio = min_chinese_ratio
        self.min_keywords = min_keywords

        # 合并所有领域关键词
        self._all_keywords: set = set()
        self._all_keywords.update(FINANCIAL_TERMS)
        self._all_keywords.update(INDUSTRY_TERMS)
        self._all_keywords.update(STOCK_MAP.keys())

    def check(self, text: str) -> Tuple[bool, str, int]:
        """
        检查文档是否相关

        Returns:
            (passed, reason, keyword_count)
        """
        if not text:
            return False, "empty_text", 0

        # 1. 长度检查
        if len(text) < self.min_length:
            return False, f"too_short({len(text)}<{self.min_length})", 0

        # 2. 中文字符占比
        chinese_count = len(_CHINESE_CHAR.findall(text))
        ratio = chinese_count / max(len(text), 1)
        if ratio < self.min_chinese_ratio:
            return False, f"low_chinese_ratio({ratio:.2f}<{self.min_chinese_ratio})", 0

        # 3. 领域关键词命中
        keyword_count = sum(1 for kw in self._all_keywords if kw in text)
        if keyword_count < self.min_keywords:
            return False, f"no_domain_keywords({keyword_count}<{self.min_keywords})", keyword_count

        return True, "pass", keyword_count


# ===================== DocTypeClassifier =====================

class DocTypeClassifier:
    """
    文档类型分类器 — 快速 regex + dictionary 分类

    分类:
    - financial_report: 财报/年报/季报
    - news: 新闻报道
    - macro_data: 宏观数据/政策
    - query: 用户提问
    - other: 兜底
    """

    # 预编译正则
    _query_patterns = [
        re.compile(p) for p in DOC_TYPE_PATTERNS.get("query", [])
    ]

    def classify(self, text: str) -> Dict:
        """
        分类文档

        Returns:
            {"doc_type": str, "confidence": float, "matched_keywords": [str]}
        """
        if not text:
            return {"doc_type": "other", "confidence": 0.0, "matched_keywords": []}

        scores: Dict[str, List[str]] = {}

        # 1. 关键词匹配
        for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in text]
            if matched:
                scores[doc_type] = matched

        # 2. 正则匹配 (query 类型)
        for pattern in self._query_patterns:
            if pattern.search(text):
                scores.setdefault("query", []).append(pattern.pattern)
                break

        if not scores:
            return {"doc_type": "other", "confidence": 0.3, "matched_keywords": []}

        # 3. 选择得分最高的类型
        best_type = max(scores, key=lambda t: len(scores[t]))
        best_keywords = scores[best_type]

        # 置信度: 基于命中关键词数, 上限 1.0
        confidence = min(1.0, len(best_keywords) / 3.0)

        return {
            "doc_type": best_type,
            "confidence": round(confidence, 2),
            "matched_keywords": best_keywords[:5],  # 最多返回 5 个
        }
