"""
Metadata Schema + 正则抽取器

定义文档 metadata 的"身份证"（DocMetadata），
以及入库时用正则从文本中抽取 company / publish_date / sector 的工具函数。

设计原则:
- 不调 LLM，纯正则 + 词典匹配
- 所有 metadata 字段在此定义，_flatten_meta 按白名单保留
- 抽取失败不阻塞入库，字段留空即可

用法:
    from financial_rag.retrievers.metadata import extract_metadata, CHROMA_META_WHITELIST

    meta = extract_metadata(text, title="商汤科技2024年报")
    # → {"company": "商汤科技", "publish_date": "2024", "sector": "AI"}
"""
import re
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ===================== Schema 定义 =====================

@dataclass
class DocMetadata:
    """文档 metadata 标准字段

    入库时尽量打满，检索时按字段过滤/路由/溯源。
    """
    # === 必填（入库时系统自动分配）===
    doc_id: str = ""              # 唯一标识 (persistence.make_doc_id)
    title: str = ""               # 文档标题
    doc_type: str = ""            # annual_report / research / news / concept / other
    source: str = ""              # 来源: pdf / rss / manual / upload / news:xxx
    date_added: str = ""          # 入库日期 (YYYY-MM-DD)

    # === 过滤用（检索时 QueryParser 会查这些）===
    company: str = ""             # 关联公司（如"商汤科技"）
    sector: str = ""              # 行业（如"AI"、"半导体"）
    publish_date: str = ""        # 发布日期 (YYYY-MM-DD 或 YYYY-MM 或 YYYY)

    # === 溯源用（答案出处）===
    source_file: str = ""         # 原始文件路径
    url: str = ""                 # 原文链接

    # === 切分用（chunk 级）===
    chunk_id: int = 0
    chunk_count: int = 1
    source_id: str = ""           # 来源文档标识 (title hash)

    # === 记录用（不参与过滤）===
    text_length: int = 0

    def to_dict(self) -> Dict:
        """转为 dict（跳过空值和下划线前缀的内部字段）"""
        d = asdict(self)
        return {k: v for k, v in d.items() if v and not k.startswith("_")}


# Chroma 白名单 — _flatten_meta 只保留这些字段
CHROMA_META_WHITELIST: Set[str] = {
    "doc_id", "title", "doc_type", "source", "date_added",
    "company", "sector", "publish_date",
    "source_file", "url",
    "chunk_id", "chunk_count", "source_id",
    "text_length",
    # 兼容旧字段
    "keyword", "news_source", "fetched_at", "publish_time",
    "parse_type", "file",
}


# ===================== 公司名词典 =====================

# 从 stocks_extended.json + STOCK_MAP 加载公司关键词
_COMPANY_KEYWORDS: Dict[str, str] = {}  # {关键词: 规范公司名}


def _load_company_keywords():
    """加载公司名映射（惰性，只加载一次）"""
    global _COMPANY_KEYWORDS
    if _COMPANY_KEYWORDS:
        return

    # 1. 内置 STOCK_MAP
    try:
        from financial_rag.retrievers.dictionaries import STOCK_MAP
        for keyword, (ts_code, name) in STOCK_MAP.items():
            _COMPANY_KEYWORDS[keyword] = name
            # 全名也加进去
            if name != keyword:
                _COMPANY_KEYWORDS[name] = name
    except ImportError:
        pass

    # 2. 扩展词典 stocks_extended.json
    ext_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "dictionaries", "stocks_extended.json"
    ))
    if os.path.exists(ext_path):
        try:
            with open(ext_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for keyword, (code, name) in data.get("stocks", {}).items():
                _COMPANY_KEYWORDS[keyword] = name
                if name != keyword:
                    _COMPANY_KEYWORDS[name] = name
            # synonym_groups: 所有同义词指向第一个
            for group in data.get("synonym_groups", []):
                if len(group) >= 2:
                    canonical = group[0]
                    for syn in group[1:]:
                        _COMPANY_KEYWORDS[syn] = canonical
        except Exception as e:
            logger.warning(f"加载 stocks_extended.json 失败: {e}")

    logger.debug(f"公司词典加载完成: {len(_COMPANY_KEYWORDS)} 个关键词")


# 公司名后缀模式（正则兜底：匹配 "XX科技有限公司" 等）
_COMPANY_SUFFIX_RE = re.compile(
    r'([\u4e00-\u9fff]{2,8}(?:科技|集团|股份|控股|电子|信息|智能|数据|网络|通信|半导体|新能源))'
)


# ===================== 行业/板块词典 =====================

_SECTOR_KEYWORDS: Dict[str, str] = {
    # AI
    "AI": "AI", "人工智能": "AI", "大模型": "AI", "GPT": "AI",
    "算力": "AI", "GPU": "AI", "机器学习": "AI", "深度学习": "AI",
    "自然语言处理": "AI", "计算机视觉": "AI", "AIGC": "AI",
    # 半导体
    "芯片": "半导体", "半导体": "半导体", "光刻机": "半导体",
    "封装": "半导体", "晶圆": "半导体", "EDA": "半导体",
    # 新能源
    "新能源": "新能源", "光伏": "新能源", "风电": "新能源",
    "储能": "新能源", "锂电": "新能源", "氢能": "新能源",
    # 电动车
    "电动车": "电动车", "智能驾驶": "电动车", "自动驾驶": "电动车",
    "新能源汽车": "电动车",
    # 通信
    "5G": "通信", "6G": "通信", "通信": "通信", "基站": "通信",
    # 云计算
    "云计算": "云计算", "大数据": "云计算", "物联网": "云计算",
    "边缘计算": "云计算", "SaaS": "云计算",
    # 机器人
    "机器人": "机器人", "人形机器人": "机器人", "具身智能": "机器人",
    # 医药
    "医药": "医药", "创新药": "医药", "CXO": "医药", "医疗器械": "医药",
    # 消费
    "消费": "消费", "白酒": "消费", "食品": "消费", "家电": "消费",
    # 金融
    "银行": "金融", "保险": "金融", "证券": "金融", "基金": "金融",
    "金融科技": "金融",
}


# ===================== 日期正则 =====================

_DATE_PATTERNS = [
    # 2024年6月1日 / 2024年6月
    (re.compile(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]'), "full"),
    (re.compile(r'(\d{4})\s*年\s*(\d{1,2})\s*月'), "month"),
    # 报告期: "2024年第一季度" / "2024Q1" / "2024H1" (必须在 year 之前)
    (re.compile(r'(\d{4})\s*年?\s*第[一二三四]季度'), "quarter"),
    (re.compile(r'(\d{4})\s*[QH](\d)'), "quarter"),
    # 2024年 (最宽泛，放最后)
    (re.compile(r'(\d{4})\s*年'), "year"),
    # 2024-06-01 / 2024/06/01
    (re.compile(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})'), "full"),
    (re.compile(r'(\d{4})[-/](\d{1,2})'), "month"),
    # 20240601
    (re.compile(r'(\d{4})(\d{2})(\d{2})'), "full"),
]

# 季度映射
_QUARTER_MAP = {"一": "03", "二": "06", "三": "09", "四": "12", "1": "03", "2": "06", "3": "09", "4": "12"}


# ===================== 核心抽取函数 =====================

def extract_metadata(text: str, title: str = "", existing_meta: Optional[Dict] = None) -> Dict:
    """从文本中用正则抽取 metadata 字段

    Args:
        text: 文档正文（清洗后）
        title: 文档标题（可选，优先从标题抽取）
        existing_meta: 已有的 meta dict（不覆盖已有非空字段）

    Returns:
        抽取到的字段 dict（只包含非空结果）
    """
    existing = existing_meta or {}
    result = {}

    # 合并标题+正文前500字做抽取（长文只看开头）
    scan_text = (title + "\n" + text[:500]) if title else text[:500]

    # 1. 公司名
    if not existing.get("company"):
        company = _extract_company(scan_text)
        if company:
            result["company"] = company

    # 2. 发布日期
    if not existing.get("publish_date"):
        date = _extract_date(scan_text)
        if date:
            result["publish_date"] = date

    # 3. 行业/板块
    if not existing.get("sector"):
        sector = _extract_sector(scan_text)
        if sector:
            result["sector"] = sector

    return result


def _extract_company(text: str) -> str:
    """抽取公司名：词典优先，正则兜底"""
    _load_company_keywords()

    # 词典匹配（按关键词长度降序，避免短词误匹配）
    for keyword in sorted(_COMPANY_KEYWORDS.keys(), key=len, reverse=True):
        if keyword in text:
            return _COMPANY_KEYWORDS[keyword]

    # 正则兜底：匹配"XX科技/集团/股份"模式
    m = _COMPANY_SUFFIX_RE.search(text)
    if m:
        return m.group(1)

    return ""


def _extract_date(text: str) -> str:
    """抽取日期，返回 YYYY-MM-DD / YYYY-MM / YYYY 格式"""
    for pattern, level in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue

        if level == "full":
            y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
            return f"{y}-{mo}-{d}"
        elif level == "month":
            y, mo = m.group(1), m.group(2).zfill(2)
            return f"{y}-{mo}"
        elif level == "quarter":
            y = m.group(1)
            # 尝试从匹配文本中提取季度号
            q_match = re.search(r'第([一二三四])季度', m.group(0))
            if q_match:
                month = _QUARTER_MAP.get(q_match.group(1), "12")
            else:
                q_match2 = re.search(r'[QH](\d)', m.group(0))
                month = _QUARTER_MAP.get(q_match2.group(1), "12") if q_match2 else "12"
            return f"{y}-{month}"
        elif level == "year":
            return m.group(1)

    return ""


def _extract_sector(text: str) -> str:
    """抽取行业：按关键词匹配，返回出现最多的行业"""
    sector_counts: Dict[str, int] = {}
    text_lower = text.lower()

    for keyword, sector in _SECTOR_KEYWORDS.items():
        if keyword.lower() in text_lower:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if not sector_counts:
        return ""

    # 返回出现次数最多的行业
    return max(sector_counts, key=sector_counts.get)
