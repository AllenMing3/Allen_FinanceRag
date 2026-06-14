"""Financial RAG — CLI 入口 (开发用，用户请用 python main.py 启动 Web UI)"""
import argparse
import sys
import os

from financial_rag.core.router import CommandRouter
from financial_rag.core.factory import setup_environment


def main():
    parser = argparse.ArgumentParser(
        description="Financial RAG CLI",
    )

    sub = parser.add_subparsers(dest="command", help="命令")

    # web (default & primary)
    wp = sub.add_parser("web", help="启动 Web UI")
    wp.add_argument("--host", default="127.0.0.1")
    wp.add_argument("--port", type=int, default=8000)

    # pipeline
    plp = sub.add_parser("pipeline", help="端到端 Pipeline")
    plp.add_argument("query")
    plp.add_argument("-t", "--template", default="quick", choices=["quick", "fin", "news", "deep"])
    plp.add_argument("-v", "--verbose", action="store_true")

    # query
    qp = sub.add_parser("query", help="交互查询")
    qp.add_argument("-q", "--question")
    qp.add_argument("-i", "--interactive", action="store_true")

    # build
    bp = sub.add_parser("build", help="构建知识库")
    bp.add_argument("--dir")

    # news
    np = sub.add_parser("news", help="拉取新闻")
    np.add_argument("query")
    np.add_argument("-s", "--summarize", action="store_true")

    # kline
    kp = sub.add_parser("kline", help="K线分析")
    kp.add_argument("query")
    kp.add_argument("--days", type=int, default=30)
    kp.add_argument("-s", "--summarize", action="store_true")

    # analyze
    ap = sub.add_parser("analyze", help="Multi-Agent 分析")
    ap.add_argument("file")
    ap.add_argument("--parallel", action="store_true")

    # score
    sp = sub.add_parser("score", help="检索打分")
    sp.add_argument("query")
    sp.add_argument("-k", "--top_k", type=int, default=5)
    sp.add_argument("--json", dest="json")
    sp.add_argument("--local", action="store_true")

    # slot
    slp = sub.add_parser("slot", help="槽位填充")
    slp.add_argument("query")
    slp.add_argument("-t", "--template", default="quick", choices=["quick", "fin", "news", "deep"])
    slp.add_argument("-k", "--top_k", type=int, default=5)
    slp.add_argument("--no-freeform", dest="no_freeform", action="store_true")

    # toolcall
    tp = sub.add_parser("toolcall", help="Function Calling")
    tp.add_argument("query", nargs="?", default="")
    tp.add_argument("-l", "--list-tools", dest="list_tools", action="store_true")
    tp.add_argument("--multi-turn", dest="multi_turn", action="store_true")
    tp.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # 初始化环境
    if not setup_environment() and args.command not in ("score", "web", None):
        sys.exit(1)

    # web 命令单独处理
    if args.command == "web" or args.command is None:
        from financial_rag.web import main as web_main
        host = getattr(args, 'host', '127.0.0.1')
        port = getattr(args, 'port', 8000)
        os.environ["WEB_HOST"] = host
        os.environ["WEB_PORT"] = str(port)
        web_main()
        return

    # 交给 CommandRouter 处理
    router = CommandRouter()
    router.dispatch(args)


if __name__ == "__main__":
    main()
