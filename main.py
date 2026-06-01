"""
Financial RAG — 财报/经济新闻智能分析系统（阿里百炼 DashScope）

用法:
    python main.py                     → 启动 Financial RAG
    python -m financial_rag.main demo  → 演示模式
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    """启动 Financial RAG"""
    from financial_rag.main import main as financial_main
    financial_main()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║      Financial RAG — 财报经济新闻智能分析         ║
║                                                  ║
║  模型: 阿里百炼 DashScope (全链路)                ║
║  • LLM:       qwen-plus / qwen-turbo / qwen-max  ║
║  • Embedding: text-embedding-v3                   ║
║  • Rerank:    gte-rerank                          ║
║                                                  ║
║  三大核心架构:                                    ║
║  • Coordinate  — 多 Agent 协调调度                ║
║  • Indexer     — 混合检索索引（BM25 + 向量 + RRF）║
║  • Reflection  — ReAct 反思 + 六层防幻觉           ║
║                                                  ║
║  用法: python -m financial_rag.main demo          ║
╚══════════════════════════════════════════════════╝
""")
    main()
