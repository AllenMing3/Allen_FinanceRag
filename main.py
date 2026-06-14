"""Financial RAG — 启动入口"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    from financial_rag.web import main as web_main
    web_main()


if __name__ == "__main__":
    main()
