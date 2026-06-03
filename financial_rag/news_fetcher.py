"""
新闻获取模块 — 基于 akshare 拉取财经新闻

功能:
- 个股新闻: 按股票代码获取相关新闻
- 财经快讯: 实时财经电报/快讯
- 关键词搜索: 按关键词搜索新闻
- 均注册为 FunctionRegistry Tool，LLM 可通过 Function Calling 调起

数据源: 东方财富、新浪财经 (通过 akshare)
MCP 替代方案: 等价于 china-stock-mcp 的 get_news_data，但直接集成，无需独立部署
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    ak = None  # type: ignore


# ===================== 数据结构 =====================

@dataclass
class NewsItem:
    """单条新闻"""
    title: str
    content: str = ""
    source: str = ""
    publish_time: str = ""
    url: str = ""
    sentiment: str = ""  # 正面/负面/中性

    def to_dict(self) -> Dict:
        """导出为字典，保留完整 content 供下游 RAG 分析。"""
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "publish_time": self.publish_time,
            "url": self.url,
            "sentiment": self.sentiment,
        }

    def to_document(self) -> Dict:
        """转为 IngestionAgent 兼容的文档格式"""
        return {
            "text": f"{self.title}\n{self.content}",
            "metadata": {
                "source": self.source,
                "date": self.publish_time[:10] if self.publish_time else "",
                "doc_type": "新闻报道",
                "url": self.url,
                "sentiment": self.sentiment,
            }
        }


@dataclass
class NewsResult:
    """新闻查询结果"""
    query: str
    items: List[NewsItem] = field(default_factory=list)
    total: int = 0
    elapsed_ms: float = 0
    source: str = "akshare"

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "total": self.total,
            "source": self.source,
            "elapsed_ms": round(self.elapsed_ms, 0),
            "items": [item.to_dict() for item in self.items],
        }

    def to_documents(self) -> List[Dict]:
        return [item.to_document() for item in self.items]


# ===================== 新闻获取函数 =====================

def fetch_stock_news(
    stock_code: str = "600519",
    max_news: int = 30,
) -> Dict:
    """获取指定股票的近期新闻。

    底层调用东方财富个股新闻接口，涵盖公告、研报、媒体报道等。

    Args:
        stock_code: 股票代码，如 '600519'(茅台)、'000858'(五粮液)
        max_news: 最大返回条数，默认 30
    """
    if not HAS_AKSHARE:
        return {"error": "请安装 akshare: pip install akshare", "items": []}

    import time
    t0 = time.time()

    try:
        # akshare 个股新闻接口
        df = ak.stock_news_em(symbol=stock_code)
        if df is None or df.empty:
            return {"query": f"个股新闻: {stock_code}", "total": 0, "items": [], "elapsed_ms": (time.time()-t0)*1000}

        items = []
        for _, row in df.head(max_news).iterrows():
            title = str(row.get("新闻标题", row.get("标题", row.get("title", ""))))
            content = str(row.get("新闻内容", row.get("内容", row.get("content", ""))))
            source = str(row.get("文章来源", row.get("来源", row.get("source", ""))))
            pub_time = str(row.get("发布时间", row.get("时间", row.get("publish_time", ""))))
            url = str(row.get("新闻链接", row.get("链接", row.get("url", ""))))
            items.append(NewsItem(
                title=title,
                content=content,
                source=source,
                publish_time=pub_time,
                url=url,
                sentiment=_guess_sentiment(title),
            ).to_dict())

        return {
            "query": f"个股新闻: {stock_code}",
            "total": len(items),
            "items": items,
            "elapsed_ms": (time.time() - t0) * 1000,
        }
    except Exception as e:
        logger.warning(f"获取个股新闻失败 [{stock_code}]: {e}")
        return {"query": f"个股新闻: {stock_code}", "total": 0, "items": [], "error": str(e)}


def fetch_financial_news(
    keyword: str = "",
    max_news: int = 20,
) -> Dict:
    """搜索财经新闻（按关键词或获取最新财经电报）。

    支持按关键词搜索，空关键词返回最新财经快讯。

    Args:
        keyword: 搜索关键词，如 '茅台'、'降准'、'新能源'，空字符串返回最新电报
        max_news: 最大返回条数，默认 20
    """
    if not HAS_AKSHARE:
        return {"error": "请安装 akshare: pip install akshare", "items": []}

    import time
    t0 = time.time()

    try:
        if keyword:
            # 按关键词搜索（东方财富）
            df = ak.stock_info_global_em()
            if df is not None and not df.empty:
                # 过滤标题含有关键词的
                title_col = next((c for c in df.columns if "标题" in c or "title" in str(c).lower()), None)
                if title_col:
                    df = df[df[title_col].astype(str).str.contains(keyword, na=False)]
                items = _df_to_news_items(df, max_news)
            else:
                items = []
        else:
            # 最新财经电报/快讯
            try:
                df = ak.stock_info_global_em()  # 全球财经快讯
                items = _df_to_news_items(df, max_news)
            except Exception:
                # 回退：尝试其他接口
                df = ak.stock_zh_a_alerts_cls()
                items = _df_to_news_items(df, max_news)

        return {
            "query": f"财经新闻: {keyword or '最新快讯'}",
            "total": len(items),
            "items": items,
            "elapsed_ms": (time.time() - t0) * 1000,
        }
    except Exception as e:
        logger.warning(f"获取财经新闻失败 [{keyword}]: {e}")
        return {"query": f"财经新闻: {keyword}", "total": 0, "items": [], "error": str(e)}


def fetch_announcements(
    stock_code: str = "600519",
    max_news: int = 20,
) -> Dict:
    """获取上市公司公告（财报、重大事项等）。

    Args:
        stock_code: 股票代码
        max_news: 最大返回条数
    """
    if not HAS_AKSHARE:
        return {"error": "请安装 akshare: pip install akshare", "items": []}

    import time
    t0 = time.time()

    try:
        # 东方财富个股公告
        df = ak.stock_notice_report(symbol=stock_code)
        items = _df_to_news_items(df, max_news) if df is not None and not df.empty else []
        return {
            "query": f"公司公告: {stock_code}",
            "total": len(items),
            "items": items,
            "elapsed_ms": (time.time() - t0) * 1000,
        }
    except Exception as e:
        logger.warning(f"获取公告失败 [{stock_code}]: {e}")
        return {"query": f"公司公告: {stock_code}", "total": 0, "items": [], "error": str(e)}


# ===================== 辅助函数 =====================

def _df_to_news_items(df, max_news: int) -> List[Dict]:
    """将 akshare DataFrame 转为 NewsItem dict 列表（兼容多种接口列名）"""
    items = []
    for _, row in df.head(max_news).iterrows():
        row_dict = row.to_dict()
        # 兼容不同接口的列名 (stock_news_em / stock_info_global_em / 其他)
        title = (
            row_dict.get("新闻标题")
            or row_dict.get("标题")
            or row_dict.get("title")
            or str(row_dict.get("content", row_dict.get("摘要", "")))[:50]
        )
        content = (
            row_dict.get("新闻内容")
            or row_dict.get("内容")
            or row_dict.get("摘要")
            or row_dict.get("content", "")
        )
        source = (
            row_dict.get("文章来源")
            or row_dict.get("来源")
            or row_dict.get("source", "财经快讯")
        )
        pub_time = (
            row_dict.get("发布时间")
            or row_dict.get("时间")
            or row_dict.get("publish_time", "")
        )
        url = (
            row_dict.get("新闻链接")
            or row_dict.get("链接")
            or row_dict.get("url", "")
        )
        items.append(NewsItem(
            title=str(title) if title else "",
            content=str(content) if content else "",
            source=str(source) if source else "",
            publish_time=str(pub_time) if pub_time else "",
            url=str(url) if url else "",
            sentiment=_guess_sentiment(str(title)),
        ).to_dict())
    return items


def _guess_sentiment(title: str) -> str:
    """基于标题关键词简单推断情绪"""
    positive_words = ["增长", "上升", "利好", "突破", "创新高", "盈利", "分红", "回购", "超预期"]
    negative_words = ["下跌", "亏损", "暴雷", "处罚", "违规", "退市", "减持", "下滑", "预警"]

    pos_count = sum(1 for w in positive_words if w in title)
    neg_count = sum(1 for w in negative_words if w in title)

    if pos_count > neg_count:
        return "正面"
    elif neg_count > pos_count:
        return "负面"
    return "中性"


# ===================== 便捷函数：一键获取练手数据 =====================

def get_sample_news_for_rag(
    stock_codes: List[str] = None,
    keywords: List[str] = None,
    max_per_source: int = 5,
) -> List[Dict]:
    """
    一键获取多只股票 + 多个关键词的最新新闻，返回 IngestionAgent 兼容格式。

    用途: 作为 RAG 系统练手/测试的实时数据源，替代手写 JSONL。

    Args:
        stock_codes: 股票代码列表，默认 ['600519'(茅台), '000858'(五粮液)]
        keywords: 关键词列表，默认 ['央行', '降准', '新能源']
        max_per_source: 每个来源最多取几条

    Returns:
        List[Dict]: 可直接传给 IngestionAgent 的文档列表
    """
    all_docs = []

    stock_codes = stock_codes or ["600519", "000858"]
    keywords = keywords or ["央行", "降准", "新能源"]

    for code in stock_codes:
        result = fetch_stock_news(code, max_news=max_per_source)
        for item in result.get("items", []):
            all_docs.append({
                "text": f"{item['title']}\n{item['content']}",
                "metadata": {
                    "source": item.get("source", ""),
                    "company": _code_to_name(code),
                    "date": item.get("publish_time", "")[:10],
                    "doc_type": "新闻报道",
                    "stock_code": code,
                }
            })

    for kw in keywords:
        result = fetch_financial_news(keyword=kw, max_news=max_per_source)
        for item in result.get("items", []):
            all_docs.append({
                "text": f"{item['title']}\n{item['content']}",
                "metadata": {
                    "source": item.get("source", ""),
                    "date": item.get("publish_time", "")[:10],
                    "doc_type": "新闻报道",
                    "keyword": kw,
                }
            })

    logger.info(f"[NewsFetcher] 获取 {len(all_docs)} 条新闻 ("
                f"{len(stock_codes)} 只股票 + {len(keywords)} 个关键词)")
    return all_docs


def _code_to_name(code: str) -> str:
    """股票代码 → 简称映射（常用股）"""
    mapping = {
        "600519": "贵州茅台", "000858": "五粮液", "000568": "泸州老窖",
        "002304": "洋河股份", "000799": "酒鬼酒", "600809": "山西汾酒",
        "601318": "中国平安", "600036": "招商银行", "000001": "平安银行",
        "601398": "工商银行", "600030": "中信证券", "300750": "宁德时代",
        "002594": "比亚迪", "601012": "隆基绿能", "000651": "格力电器",
        "000333": "美的集团", "600887": "伊利股份", "002415": "海康威视",
    }
    return mapping.get(code, code)
