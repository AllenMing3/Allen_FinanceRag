"""
新闻 MCP 客户端 — 封装 china-stock-mcp 的新闻/公告获取

当 config.mcp.enable_mcp=True 且配置了 china_stock_mcp_dir 时，
通过 MCP 协议调用第三方 china-stock-mcp 服务器获取新闻数据。
当 MCP 不可用时，自动回退到 feedparser RSS 新闻获取。

用法:
    client = NewsMCPClient()
    result = client.get_news("600519")      # MCP 优先，RSS 兆底
    result = client.get_financial_news("降准")  # 搜索财经新闻
"""
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NewsMCPClient:
    """
    新闻 MCP 客户端 — 封装 china-stock-mcp 的新闻工具
    
    支持:
    - get_news_data: 获取个股新闻 (MCP: get_news_data)
    - get_financial_news: 搜索财经新闻 (feedparser RSS 兆底)
    
    当 MCP 不可用时自动回退到 rss_fetcher.py 的 RSS 新闻获取。
    """

    def __init__(self):
        from financial_rag.config import config
        self._mcp_config = config.mcp
        self._mcp_available = self._check_mcp_available()

    def _check_mcp_available(self) -> bool:
        """检查 MCP 是否可用"""
        if not self._mcp_config.enable_mcp:
            return False
        if not self._mcp_config.china_stock_mcp_dir:
            logger.debug("MCP 未配置 china_stock_mcp_dir，使用 RSS 新闻获取")
            return False
        import os
        if not os.path.isdir(self._mcp_config.china_stock_mcp_dir):
            logger.warning(
                f"MCP 目录不存在: {self._mcp_config.china_stock_mcp_dir}，使用 RSS 新闻获取"
            )
            return False
        return True

    def _get_mcp_command(self) -> List[str]:
        """构建 MCP 服务器启动命令"""
        import os
        mcp_dir = self._mcp_config.china_stock_mcp_dir
        # china-stock-mcp 通常通过 python -m china_stock_mcp 启动
        # 或 python server.py
        server_py = os.path.join(mcp_dir, "server.py")
        if os.path.exists(server_py):
            return ["python", server_py]
        # 尝试 __main__.py 模式
        return ["python", "-m", "china_stock_mcp"]

    def _call_mcp_tool(self, tool_name: str, arguments: Dict) -> Optional[Dict]:
        """通过 MCP 调用工具，失败返回 None"""
        from financial_rag.mcp_client.client import call_mcp_sync

        try:
            result = call_mcp_sync(
                command=self._get_mcp_command(),
                tool_name=tool_name,
                arguments=arguments,
                cwd=self._mcp_config.china_stock_mcp_dir,
                timeout=self._mcp_config.timeout,
            )
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"raw_text": result}
            return result
        except Exception as e:
            logger.warning(f"MCP 调用失败 [{tool_name}]: {e}")
            return None

    @property
    def is_mcp_enabled(self) -> bool:
        return self._mcp_available

    # ===================== 公共 API =====================

    def get_news(self, stock_code: str = "600519", max_news: int = 30) -> Dict:
        """
        获取个股新闻。

        MCP 优先 (get_news_data)，失败回退到 RSS (search_news)。

        Args:
            stock_code: 股票代码
            max_news: 最大返回条数

        Returns:
            包含 items 列表的字典
        """
        # MCP 优先
        if self._mcp_available:
            result = self._call_mcp_tool(
                "get_news_data",
                {"symbol": stock_code, "output_format": "json"},
            )
            if result is not None:
                # 标准化输出格式
                return self._normalize_mcp_news_result(result, stock_code)

        # 回退: RSS 新闻获取
        from financial_rag.rss_fetcher import search_news
        result = search_news(keyword=stock_code, max_news=max_news)
        return {
            "query": f"个股新闻 (RSS): {stock_code}",
            "total": result.get("total", 0),
            "items": result.get("items", []),
            "source": "rss",
        }

    def get_financial_news(self, keyword: str = "", max_news: int = 20) -> Dict:
        """
        搜索财经新闻或获取最新快讯。

        目前 china-stock-mcp 主要提供个股新闻，
        通用财经新闻搜索使用 feedparser RSS 获取。

        Args:
            keyword: 搜索关键词
            max_news: 最大返回条数
        """
        # 通用财经新闻搜索使用 feedparser RSS
        from financial_rag.rss_fetcher import search_news
        return search_news(keyword=keyword, max_news=max_news)

    def get_announcements(self, stock_code: str = "600519", max_news: int = 20) -> Dict:
        """
        获取上市公司公告。

        Args:
            stock_code: 股票代码
            max_news: 最大返回条数
        """
        # 公告数据使用 RSS 搜索（按股票代码过滤）
        from financial_rag.rss_fetcher import search_news
        result = search_news(keyword=stock_code, max_news=max_news)
        return {
            "query": f"公告 (RSS): {stock_code}",
            "total": result.get("total", 0),
            "items": result.get("items", []),
            "source": "rss",
        }

    def list_mcp_tools(self) -> List[Dict]:
        """列出 MCP 服务器提供的所有工具 (调试用)"""
        if not self._mcp_available:
            return []

        import asyncio
        from financial_rag.mcp_client.client import MCPClient

        async def _list():
            async with MCPClient(
                command=self._get_mcp_command(),
                cwd=self._mcp_config.china_stock_mcp_dir,
                timeout=self._mcp_config.timeout,
            ) as client:
                return await client.list_tools()

        try:
            return asyncio.run(_list())
        except Exception as e:
            logger.warning(f"列出 MCP 工具失败: {e}")
            return []

    # ===================== 内部方法 =====================

    def _normalize_mcp_news_result(self, result: Dict, stock_code: str) -> Dict:
        """将 MCP 返回的新闻数据标准化为统一格式"""
        # MCP 返回格式可能是 markdown 文本或 JSON
        # 尝试解析 items
        items = result.get("items", [])

        if not items and "raw_text" in result:
            # MCP 返回了纯文本/markdown，尝试提取
            items = [{"title": "MCP 新闻", "content": result["raw_text"],
                      "source": "mcp", "publish_time": "", "url": ""}]

        if not items and isinstance(result, dict):
            # 可能是嵌套结构
            for key in ["data", "news", "results", "news_list"]:
                if key in result and isinstance(result[key], list):
                    items = result[key]
                    break

        return {
            "query": f"个股新闻 (MCP): {stock_code}",
            "total": len(items),
            "items": items,
            "source": "mcp",
        }
