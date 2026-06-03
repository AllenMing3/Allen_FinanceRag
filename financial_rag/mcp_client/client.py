"""
通用 MCP 客户端 — 通过 stdio 连接外部 MCP 服务器

使用 mcp SDK 的 ClientSession 通过 stdio transport 与 MCP 服务器通信。
支持:
- 列出服务器提供的工具
- 调用工具并获取结果
- 上下文管理器自动管理连接生命周期

用法:
    async with MCPClient(command=["python", "server.py"]) as client:
        tools = await client.list_tools()
        result = await client.call_tool("get_news_data", {"symbol": "000001"})
"""
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """
    通用 MCP 客户端 (stdio transport)

    通过子进程 stdio 与 MCP 服务器通信。

    Args:
        command: 启动 MCP 服务器的命令列表,
                 例如 ["python", "-m", "china_stock_mcp"]
                 或 ["node", "server.js"]
        cwd: 服务器进程的工作目录 (可选)
        timeout: 调用超时秒数
    """

    def __init__(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.command = command
        self.cwd = cwd
        self.timeout = timeout
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._cm = None  # context manager for stdio_client

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """建立与 MCP 服务器的连接"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self.command[0],
            args=self.command[1:],
            cwd=self.cwd,
        )

        self._cm = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._cm.__aenter__()
        self._session_cm = ClientSession(self._read_stream, self._write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        logger.info(f"MCP 连接已建立: {' '.join(self.command)}")

    async def disconnect(self):
        """关闭连接"""
        try:
            if self._session_cm:
                await self._session_cm.__aexit__(None, None, None)
            if self._cm:
                await self._cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"MCP 断开连接时出错: {e}")
        finally:
            self._session = None
            self._session_cm = None
            self._cm = None

    async def list_tools(self) -> List[Dict]:
        """列出服务器提供的所有工具"""
        if not self._session:
            raise RuntimeError("未连接到 MCP 服务器，请先调用 connect()")

        result = await asyncio.wait_for(
            self._session.list_tools(),
            timeout=self.timeout,
        )
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            })
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用 MCP 服务器的工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具返回的结果 (通常是文本内容列表)
        """
        if not self._session:
            raise RuntimeError("未连接到 MCP 服务器，请先调用 connect()")

        result = await asyncio.wait_for(
            self._session.call_tool(name, arguments),
            timeout=self.timeout,
        )

        # 提取文本内容
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
            else:
                texts.append(str(content))

        combined = "\n".join(texts)

        # 尝试解析为 JSON
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, ValueError):
            return combined

    @property
    def is_connected(self) -> bool:
        return self._session is not None


def call_mcp_sync(
    command: List[str],
    tool_name: str,
    arguments: Dict[str, Any],
    cwd: Optional[str] = None,
    timeout: float = 30.0,
) -> Any:
    """
    同步封装: 连接 → 调用 → 断开

    适用于在非 async 代码中调用 MCP 工具。

    Args:
        command: MCP 服务器启动命令
        tool_name: 工具名称
        arguments: 工具参数
        cwd: 工作目录
        timeout: 超时秒数

    Returns:
        工具返回结果
    """
    async def _run():
        async with MCPClient(command=command, cwd=cwd, timeout=timeout) as client:
            return await client.call_tool(tool_name, arguments)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已经在事件循环中，创建新线程
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run())
            return future.result(timeout=timeout + 5)
    else:
        return asyncio.run(_run())
