"""
主入口 — 支持两种模式:
  1. python main.py           → 启动 Financial RAG
  2. python multi_agent_demo.py → 启动 Multi-Agent 演示

原日志分析代码保留在 agents/ pipeline/ anti_hallucination/ 等目录。
Financial RAG 全新代码在 financial_rag/ 包中。
"""
import sys
from pathlib import Path

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).parent))


def run_financial_rag():
    """启动 Financial RAG 模式"""
    from financial_rag.main import main as financial_main
    financial_main()


def run_log_analysis():
    """启动日志分析模式（原有功能）"""
    import main as old_main
    # 旧版 main 入口兼容
    print("日志分析模式已迁移。请直接运行原有脚本。")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║         Financial RAG — 财报经济新闻智能分析      ║
║                                                  ║
║  三大核心架构:                                    ║
║  • Coordinate  — 多 Agent 协调调度                ║
║  • Indexer     — 多文本索引流水线                  ║
║  • Reflection  — ReAct 反思 + 六层防幻觉           ║
║                                                  ║
║  用法: pip install -r requirements.txt            ║
║        python -m financial_rag.main demo          ║
╚══════════════════════════════════════════════════╝
    """)
    run_financial_rag()
