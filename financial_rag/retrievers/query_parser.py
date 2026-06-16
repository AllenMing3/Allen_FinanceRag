"""
Query Parser — 查询解析器

用正则 + 字典将用户查询拆解为结构化数据:
- 实体抽取: 股票代码/名称、日期/时间范围
- 关键词抽取: 领域关键词 + 权重（给 BM25 用）
- 查询类型: analysis / factual / comparison / other

不依赖 LLM，纯本地解析。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from financial_rag.retrievers.dictionaries import (
    FINANCIAL_TERMS, INDUSTRY_TERMS, ACTION_TERMS, STOP_WORDS,
    QUERY_TYPE_PATTERNS, STOCK_MAP,
)


# ===================== 结果类型 =====================

@dataclass
class QueryResult:
    """查询解析结果"""
    raw_query: str
    
    # 实体
    stock_code: str = ""           # "600519.SH"
    stock_name: str = ""           # "贵州茅台"
    stock_keyword: str = ""        # "茅台" (原始匹配词)
    date: str = ""                 # "2024-06-01" (绝对日期)
    date_range: Optional[Dict] = None  # {"gte": "2024-05-01", "lte": "2024-06-01"}
    
    # 关键词 (term, weight) — 给 BM25 用
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    
    # 查询类型
    query_type: str = "other"      # analysis / factual / comparison / other
    
    def get_weighted_terms(self) -> List[str]:
        """返回加权关键词列表（高权重词重复多次，提升 BM25 召回）"""
        terms = []
        for term, weight in self.keywords:
            # weight >= 2.0 的词重复 2 次，提升 BM25 匹配优先级
            repeat = min(3, max(1, int(weight)))
            terms.extend([term] * repeat)
        return terms
    
    def get_filters(self) -> Dict:
        """返回可用于 metadata 过滤的条件"""
        filters = {}
        if self.stock_code:
            filters["stock_code"] = self.stock_code
        if self.date:
            filters["date"] = self.date
        if self.date_range:
            filters.update(self.date_range)
        return filters


# ===================== 解析器 =====================

class QueryParser:
    """
    查询解析器 — 正则 + 字典，不依赖 LLM
    
    使用:
    >>> parser = QueryParser()
    >>> result = parser.parse("茅台最近一周走势怎么样")
    >>> result.stock_code
    '600519.SH'
    >>> result.keywords
    [('茅台', 3.0), ('走势', 1.0)]
    """
    
    def __init__(self, stock_map: Optional[Dict] = None):
        """
        Args:
            stock_map: 股票关键词映射 {keyword: (ts_code, name)}
                       默认使用 kline_tools.STOCK_MAP
        """
        self._stock_map = stock_map
    
    @property
    def stock_map(self) -> Dict:
        if self._stock_map is None:
            self._stock_map = STOCK_MAP
        return self._stock_map
    
    def parse(self, query: str) -> QueryResult:
        """解析查询，返回结构化 QueryResult"""
        result = QueryResult(raw_query=query)
        
        # 1. 实体抽取
        self._extract_stock(query, result)
        self._extract_date(query, result)
        
        # 2. 关键词抽取（带权重）
        self._extract_keywords(query, result)
        
        # 3. 查询类型分类
        self._classify_type(query, result)
        
        return result
    
    # ---- 实体抽取 ----
    
    def _extract_stock(self, query: str, result: QueryResult):
        """从查询中抽取股票实体"""
        # 优先: 字典匹配（"茅台" → 600519.SH）
        for keyword, (ts_code, name) in self.stock_map.items():
            if keyword in query:
                result.stock_code = ts_code
                result.stock_name = name
                result.stock_keyword = keyword
                return
        
        # 次优先: 6位数字代码
        m = re.search(r'(\d{6})', query)
        if m:
            code = m.group(1)
            if code.startswith("6"):
                result.stock_code = f"{code}.SH"
            elif code.startswith("0") or code.startswith("3"):
                result.stock_code = f"{code}.SZ"
            else:
                result.stock_code = code
    
    def _extract_date(self, query: str, result: QueryResult):
        """从查询中抽取日期/时间范围"""
        # 绝对日期: 2024-06-01, 2024年6月1日, 20240601
        for pat in [
            r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?',
            r'(\d{4})(\d{2})(\d{2})',
        ]:
            m = re.search(pat, query)
            if m:
                y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
                result.date = f"{y}-{mo}-{d}"
                return
        
        # 相对日期 → date_range
        today = datetime.now()
        
        relative_patterns = [
            (r'最近?一[周周]', 7),
            (r'最近?两[周周]', 14),
            (r'最近?半[月个]月?', 15),
            (r'[最]?近一[个月个]?月?', 30),
            (r'最近?三[月个]月?', 90),
            (r'[最]?近[一二两三四五七]天', 7),
            (r'上[周周]', 7),
            (r'上[月个]月?', 30),
        ]
        
        for pat, days in relative_patterns:
            if re.search(pat, query):
                start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
                result.date_range = {"gte": start_date, "lte": end_date}
                return
    
    # ---- 关键词抽取 ----
    
    def _extract_keywords(self, query: str, result: QueryResult):
        """抽取关键词并赋权重"""
        keywords = []
        query_lower = query.lower()
        
        # 高权重: 股票实体词（用户明确提到的标的）
        if result.stock_keyword:
            keywords.append((result.stock_keyword, 3.0))
        
        # 中高权重: 金融术语
        for term in FINANCIAL_TERMS:
            if term.lower() in query_lower:
                keywords.append((term, 2.0))
        
        # 中权重: 行业/主题词
        for term in INDUSTRY_TERMS:
            if term.lower() in query_lower:
                keywords.append((term, 1.5))
        
        # 低权重: 动作词（不影响检索，但记录）
        action_found = []
        for term in ACTION_TERMS:
            if term in query:
                action_found.append(term)
                keywords.append((term, 1.0))
        
        # 去重（stock_keyword 可能同时是行业词）
        seen = set()
        unique_keywords = []
        for term, weight in keywords:
            if term not in seen:
                seen.add(term)
                unique_keywords.append((term, weight))
        
        # 如果没提取到任何关键词，用分词兜底（过滤停用词）
        if not unique_keywords:
            tokens = self._fallback_tokenize(query)
            for token in tokens:
                if token not in STOP_WORDS and len(token) > 1:
                    unique_keywords.append((token, 1.0))
        
        result.keywords = unique_keywords
    
    def _fallback_tokenize(self, text: str) -> List[str]:
        """简单分词（无 jieba 时的回退）"""
        # 中文: 双字滑窗 + 完整段
        # 英文: 按单词
        raw = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+|\d+', text.lower())
        tokens = []
        for seg in raw:
            if re.match(r'^[\u4e00-\u9fff]+$', seg) and len(seg) > 1:
                for j in range(len(seg) - 1):
                    tokens.append(seg[j:j+2])
                if len(seg) <= 6:
                    tokens.append(seg)
            else:
                tokens.append(seg)
        return tokens
    
    # ---- 查询类型分类 ----
    
    def _classify_type(self, query: str, result: QueryResult):
        """基于关键词判断查询类型"""
        for qtype, patterns in QUERY_TYPE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, query):
                    result.query_type = qtype
                    return
        result.query_type = "other"
