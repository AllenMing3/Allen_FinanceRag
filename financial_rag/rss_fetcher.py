"""
RSS 新闻获取模块 — 基于 feedparser 拉取财经新闻 RSS

功能:
- 财联社电报: 实时财经快讯
- 新浪财经: 国内财经新闻
- 东方财富: 按关键词搜索新闻

数据源: RSSHub (开源 RSS 生成器) + 财联社直接 API
"""
import logging
import re
import time
from typing import Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    feedparser = None  # type: ignore


# ===================== RSS 源配置 =====================

# RSSHub 公共实例（可替换为自建实例）
RSSHUB_BASE = "https://rsshub.app"

# 默认 RSS 源
FEED_SOURCES = {
    "cls_telegraph": {
        "name": "财联社电报",
        "url": f"{RSSHUB_BASE}/cls/telegraph",
        "type": "rsshub",
    },
    "cls_depth": {
        "name": "财联社深度",
        "url": f"{RSSHUB_BASE}/cls/depth/1000",
        "type": "rsshub",
    },
    "sina_finance": {
        "name": "新浪财经",
        "url": f"{RSSHUB_BASE}/sina/finance",
        "type": "rsshub",
    },
}


# ===================== 核心获取函数 =====================

def fetch_rss_news(
    feed_url: str,
    max_news: int = 30,
    source_name: str = "",
) -> List[Dict]:
    """
    从 RSS feed 获取新闻

    Args:
        feed_url: RSS feed URL
        max_news: 最大返回条数
        source_name: 来源名称（用于标记）

    Returns:
        List of news item dicts: {title, content, source, publish_time, url, sentiment}
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
            # feedparser 的 summary 通常是内容摘要
            content = entry.get("summary", "")
            # 清理 HTML 标签
            content = _strip_html(content)

            link = entry.get("link", "")
            source = source_name or feed.get("feed", {}).get("title", "RSS")

            # 发布时间
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
                "content": content[:500],  # 限制内容长度
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


def fetch_cls_telegraph_api(max_news: int = 50) -> List[Dict]:
    """
    直接从财联社 API 获取电报快讯（feedparser 备用方案）

    财联社电报 API: https://www.cls.cn/nodeapi/updateTelegraphList
    """
    try:
        import httpx
        url = "https://www.cls.cn/nodeapi/updateTelegraphList"
        params = {
            "app": "CailianpressWeb",
            "os": "web",
            "sv": "8.4.6",
            "rn": max_news,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.cls.cn/telegraph",
        }

        resp = httpx.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"财联社 API 返回 {resp.status_code}")
            return []

        data = resp.json()
        roll_data = data.get("data", {}).get("roll_data", [])

        items = []
        for item in roll_data[:max_news]:
            title = item.get("title", "") or item.get("brief", "")[:50]
            content = item.get("content", "") or item.get("brief", "")
            content = _strip_html(content)

            pub_time = ""
            ctime = item.get("ctime", 0)
            if ctime:
                try:
                    pub_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ctime))
                except Exception:
                    pass

            items.append({
                "title": title,
                "content": content[:500],
                "source": "财联社",
                "publish_time": pub_time,
                "url": f"https://www.cls.cn/detail/{item.get('id', '')}",
                "sentiment": _guess_sentiment(title),
            })

        logger.info(f"财联社 API 获取 {len(items)} 条电报")
        return items

    except ImportError:
        logger.warning("请安装 httpx: pip install httpx")
        return []
    except Exception as e:
        logger.error(f"财联社 API 获取失败: {e}")
        return []


# ===================== 搜索与聚合 =====================

def search_news(
    keyword: str,
    feeds: Optional[List[str]] = None,
    max_news: int = 30,
) -> Dict:
    """
    从多个 RSS 源搜索新闻并按关键词过滤

    Args:
        keyword: 搜索关键词（支持逗号/顿号分隔多关键词）
        feeds: 自定义 feed URL 列表，默认使用内置源
        max_news: 每个源最大返回条数

    Returns:
        {
            "keyword": "AI人工智能",
            "total": 15,
            "items": [...],
            "elapsed_ms": 1200,
        }
    """
    t0 = time.time()
    all_items = []

    # 拆分关键词
    kw_parts = [
        kw.strip()
        for kw in keyword.replace("、", ",").replace("，", ",").split(",")
        if kw.strip()
    ]

    # 1. 尝试从东方财富 RSSHub 按关键词搜索（需 URL 编码）
    encoded_kw = quote(keyword)
    eastmoney_url = f"{RSSHUB_BASE}/eastmoney/search/{encoded_kw}"
    items = fetch_rss_news(eastmoney_url, max_news=max_news, source_name="东方财富")
    all_items.extend(items)

    # 2. 从财联社获取（优先直接 API，失败则用 RSS）
    cls_items = fetch_cls_telegraph_api(max_news=max_news * 2)
    if not cls_items:
        cls_items = fetch_rss_news(
            FEED_SOURCES["cls_telegraph"]["url"],
            max_news=max_news,
            source_name="财联社",
        )
    all_items.extend(cls_items)

    # 3. 从新浪财经获取
    items = fetch_rss_news(
        FEED_SOURCES["sina_finance"]["url"],
        max_news=max_news,
        source_name="新浪财经",
    )
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
    获取所有 RSS 源的最新新闻（不过滤）

    Returns:
        List of news item dicts
    """
    all_items = []

    # 财联社
    cls_items = fetch_cls_telegraph_api(max_news=max_per_source)
    if not cls_items:
        cls_items = fetch_rss_news(
            FEED_SOURCES["cls_telegraph"]["url"],
            max_news=max_per_source,
            source_name="财联社",
        )
    all_items.extend(cls_items)

    # 新浪财经
    items = fetch_rss_news(
        FEED_SOURCES["sina_finance"]["url"],
        max_news=max_per_source,
        source_name="新浪财经",
    )
    all_items.extend(items)

    # 财联社深度
    items = fetch_rss_news(
        FEED_SOURCES["cls_depth"]["url"],
        max_news=max_per_source,
        source_name="财联社深度",
    )
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
