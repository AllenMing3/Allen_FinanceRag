"""
RSS 新闻获取模块 — 国内直连 API + feedparser 备用

功能:
- 同花顺: 实时财经资讯（直连 10jqka API）
- 新浪财经: 国内财经新闻（直连 Sina Roll API）
- 东方财富: 按关键词搜索新闻（直连 Eastmoney Search API）

数据源: 全部使用国内直连 API，不依赖海外服务
"""
import json
import logging
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    feedparser = None  # type: ignore

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None  # type: ignore


# ===================== 通用 HTTP 配置 =====================

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_TIMEOUT = 10  # 秒


# ===================== 同花顺 (10jqka) 直连 API =====================

def fetch_ths_news(max_news: int = 30) -> List[Dict]:
    """
    同花顺财经资讯 API — 实时财经新闻

    API: https://news.10jqka.com.cn/tapp/news/push/stock/
    """
    if not HAS_HTTPX:
        logger.warning("请安装 httpx: pip install httpx")
        return []

    try:
        url = "https://news.10jqka.com.cn/tapp/news/push/stock/"
        params = {
            "page": "1",
            "tag": "",
            "track": "website",
            "pagesize": str(max_news),
        }
        headers = {
            **_HEADERS,
            "Referer": "https://news.10jqka.com.cn/",
        }

        resp = httpx.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"同花顺 API 返回 {resp.status_code}")
            return []

        data = resp.json()
        news_list = data.get("data", {}).get("list", [])

        items = []
        for item in news_list[:max_news]:
            title = item.get("title", "")
            content = item.get("digest", "") or title
            content = _strip_html(content)

            pub_time = item.get("ctime", "")
            # 同花顺返回格式: "2026-06-14 15:30:00"

            items.append({
                "title": title,
                "content": content[:500],
                "source": "同花顺",
                "publish_time": pub_time,
                "url": f"https://news.10jqka.com.cn/{item.get('id', '')}.shtml",
                "sentiment": _guess_sentiment(title),
            })

        logger.info(f"同花顺: {len(items)} 条")
        return items

    except Exception as e:
        logger.error(f"同花顺获取失败: {e}")
        return []


# ===================== 新浪财经直连 API =====================

def fetch_sina_finance(max_news: int = 30) -> List[Dict]:
    """
    新浪财经滚动新闻 API — 国内财经新闻

    API: https://feed.mix.sina.com.cn/api/roll/get
    频道: pageid=153 (财经), lid=2516 (全部)
    """
    if not HAS_HTTPX:
        logger.warning("请安装 httpx: pip install httpx")
        return []

    try:
        url = "https://feed.mix.sina.com.cn/api/roll/get"
        params = {
            "pageid": "153",
            "lid": "2516",
            "k": "",
            "num": max_news,
            "page": 1,
        }

        resp = httpx.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"新浪财经 API 返回 {resp.status_code}")
            return []

        data = resp.json()
        result_data = data.get("result", {}).get("data", [])

        items = []
        for item in result_data[:max_news]:
            title = item.get("title", "")
            content = item.get("intro", "") or item.get("summary", "")
            content = _strip_html(content)

            # 新浪返回 ctime 为 Unix 时间戳字符串
            pub_time = ""
            ctime = item.get("ctime", "")
            if ctime:
                try:
                    pub_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(int(ctime))
                    )
                except Exception:
                    pass

            items.append({
                "title": title,
                "content": content[:500],
                "source": "新浪财经",
                "publish_time": pub_time,
                "url": item.get("url", ""),
                "sentiment": _guess_sentiment(title),
            })

        logger.info(f"新浪财经: {len(items)} 条")
        return items

    except Exception as e:
        logger.error(f"新浪财经获取失败: {e}")
        return []


# ===================== 东方财富直连 API =====================

def fetch_eastmoney_search(keyword: str, max_news: int = 30) -> List[Dict]:
    """
    东方财富搜索 API — 按关键词搜索财经新闻

    API: https://search-api-web.eastmoney.com/search/jsonp
    使用 JSON param 格式发送请求，返回 JSONP 包装数据
    """
    if not HAS_HTTPX:
        logger.warning("请安装 httpx: pip install httpx")
        return []

    try:
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        # 东方财富搜索需要 JSON param 参数
        param_obj = {
            "uid": "",
            "keyword": keyword,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": max_news,
                    "preTag": "",
                    "postTag": "",
                }
            },
        }
        params = {
            "cb": "jQuery",
            "param": json.dumps(param_obj, ensure_ascii=False),
        }
        headers = {
            **_HEADERS,
            "Referer": "https://so.eastmoney.com/",
        }

        resp = httpx.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"东方财富 API 返回 {resp.status_code}")
            return []

        # 去掉 JSONP 包装: jQuery({ ... })
        text = resp.text
        json_start = text.find("(")
        json_end = text.rfind(")")
        if json_start >= 0 and json_end > json_start:
            text = text[json_start + 1:json_end]

        data = json.loads(text)
        # 结果在 result.cmsArticleWebOld 中
        result_list = data.get("result", {}).get("cmsArticleWebOld", [])

        items = []
        for item in result_list[:max_news]:
            title = item.get("title", "")
            title = _strip_html(title)
            content = item.get("content", "") or title
            content = _strip_html(content)

            pub_time = item.get("date", "")
            # 东方财富返回格式: "2024-01-15 10:30:00"

            items.append({
                "title": title,
                "content": content[:500],
                "source": "东方财富",
                "publish_time": pub_time,
                "url": item.get("url", ""),
                "sentiment": _guess_sentiment(title),
            })

        logger.info(f"东方财富搜索 '{keyword}': {len(items)} 条")
        return items

    except Exception as e:
        logger.error(f"东方财富搜索获取失败: {e}")
        return []


# ===================== 通用 RSS 获取（备用） =====================

def fetch_rss_news(
    feed_url: str,
    max_news: int = 30,
    source_name: str = "",
) -> List[Dict]:
    """
    通用 RSS feed 获取（仅作为备用，主流程使用直连 API）

    Args:
        feed_url: RSS feed URL
        max_news: 最大返回条数
        source_name: 来源名称（用于标记）
    """
    if not HAS_FEEDPARSER:
        logger.warning("请安装 feedparser: pip install feedparser")
        return []

    t0 = time.time()

    try:
        feed = feedparser.parse(feed_url)
        if not feed.entries:
            logger.warning(f"RSS feed 无内容: {feed_url}")
            return []

        items = []
        for entry in feed.entries[:max_news]:
            title = entry.get("title", "")
            content = entry.get("summary", "")
            content = _strip_html(content)

            link = entry.get("link", "")
            source = source_name or feed.get("feed", {}).get("title", "RSS")

            pub_time = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_time = time.strftime("%Y-%m-%d %H:%M:%S", entry.published_parsed)
                except Exception:
                    pass
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                try:
                    pub_time = time.strftime("%Y-%m-%d %H:%M:%S", entry.updated_parsed)
                except Exception:
                    pass

            items.append({
                "title": title,
                "content": content[:500],
                "source": source,
                "publish_time": pub_time,
                "url": link,
                "sentiment": _guess_sentiment(title),
            })

        elapsed = (time.time() - t0) * 1000
        logger.info(f"RSS 获取 {len(items)} 条 ({elapsed:.0f}ms): {source_name or feed_url}")
        return items

    except Exception as e:
        logger.error(f"RSS 获取失败 ({feed_url}): {e}")
        return []


# ===================== 搜索与聚合 =====================

def search_news(
    keyword: str,
    feeds: Optional[List[str]] = None,
    max_news: int = 30,
) -> Dict:
    """
    从多个国内数据源搜索新闻并按关键词过滤

    Args:
        keyword: 搜索关键词（支持逗号/顿号分隔多关键词）
        feeds: 未使用，保留兼容
        max_news: 每个源最大返回条数

    Returns:
        {
            "keyword": "AI人工智能",
            "total": 15,
            "items": [...],
            "elapsed_ms": 1200,
        }
    """
    # Mock 模式
    from financial_rag.config import is_mock_enabled
    if is_mock_enabled():
        from financial_rag.mock_data import mock_search_news
        return mock_search_news(keyword, max_news=max_news)

    t0 = time.time()
    all_items = []

    # 拆分关键词
    kw_parts = [
        kw.strip()
        for kw in keyword.replace("、", ",").replace("，", ",").split(",")
        if kw.strip()
    ]

    # 1. 东方财富搜索（直连 API，按关键词精准搜索）
    items = fetch_eastmoney_search(keyword, max_news=max_news)
    all_items.extend(items)

    # 2. 同花顺（直连 API，实时资讯）
    items = fetch_ths_news(max_news=max_news * 2)
    all_items.extend(items)

    # 3. 新浪财经（直连 API，滚动新闻）
    items = fetch_sina_finance(max_news=max_news)
    all_items.extend(items)

    # 按关键词过滤
    filtered = []
    for item in all_items:
        text = item.get("title", "") + " " + item.get("content", "")
        # 至少匹配一个关键词子词（长度 >= 2）
        if any(p in text for p in kw_parts if len(p) >= 2):
            filtered.append(item)
        elif not kw_parts:
            filtered.append(item)

    # 去重（按标题）
    seen_titles = set()
    deduped = []
    for item in filtered:
        title = item.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            deduped.append(item)

    deduped = deduped[:max_news]

    elapsed = (time.time() - t0) * 1000
    logger.info(f"新闻搜索 '{keyword}': 获取 {len(all_items)} 条, 过滤后 {len(deduped)} 条 ({elapsed:.0f}ms)")

    return {
        "keyword": keyword,
        "total": len(deduped),
        "items": deduped,
        "elapsed_ms": elapsed,
    }


def fetch_all_news(max_per_source: int = 20) -> List[Dict]:
    """
    获取所有数据源的最新新闻（不过滤）

    Returns:
        List of news item dicts
    """
    # Mock 模式
    from financial_rag.config import is_mock_enabled
    if is_mock_enabled():
        from financial_rag.mock_data import mock_fetch_all_news
        return mock_fetch_all_news(max_per_source=max_per_source)

    all_items = []

    # 同花顺（直连 API）
    items = fetch_ths_news(max_news=max_per_source)
    all_items.extend(items)

    # 新浪财经（直连 API）
    items = fetch_sina_finance(max_news=max_per_source)
    all_items.extend(items)

    # 东方财富（直连 API，取最新资讯）
    items = fetch_eastmoney_search("财经", max_news=max_per_source)
    all_items.extend(items)

    return all_items[:max_per_source * 3]


# ===================== 辅助函数 =====================

def _strip_html(text: str) -> str:
    """去除 HTML 标签"""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()


def _guess_sentiment(title: str) -> str:
    """基于标题关键词简单推断情绪"""
    positive_words = ["增长", "上升", "利好", "突破", "创新高", "盈利", "分红", "回购", "超预期", "上涨"]
    negative_words = ["下跌", "亏损", "暴雷", "处罚", "违规", "退市", "减持", "下滑", "预警", "暴跌"]

    pos_count = sum(1 for w in positive_words if w in title)
    neg_count = sum(1 for w in negative_words if w in title)

    if pos_count > neg_count:
        return "正面"
    elif neg_count > pos_count:
        return "负面"
    return "中性"
