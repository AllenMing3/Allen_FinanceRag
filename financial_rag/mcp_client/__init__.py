"""
MCP Client 包 — 连接第三方 MCP 服务器 (如 china-stock-mcp)

提供:
- MCPClient: 通用 MCP 客户端，通过 stdio 连接外部 MCP 服务器
- NewsMCPClient: 封装 china-stock-mcp 的新闻相关工具调用
"""
from financial_rag.mcp_client.client import MCPClient
from financial_rag.mcp_client.news_client import NewsMCPClient

__all__ = ["MCPClient", "NewsMCPClient"]
