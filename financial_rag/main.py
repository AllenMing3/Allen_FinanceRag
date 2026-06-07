"""
Financial RAG — 主入口

用法:
    python -m financial_rag.main pipeline "茅台2024年营收增长了多少？"    # 统一Pipeline
    python -m financial_rag.main query -i                                # 交互查询
    python -m financial_rag.main build --dir ./data/financial            # 构建知识库
    python -m financial_rag.main analyze ./data/financial/maotai.pdf     # Multi-Agent分析
    python -m financial_rag.main demo                                    # 演示
    python -m financial_rag.main score "茅台营收" -k 5                    # 检索打分
    python -m financial_rag.main slot "茅台财报" -t financial_report      # 槽位填充
    python -m financial_rag.main toolcall "茅台营收增长多少"              # Function Calling
    python -m financial_rag.main toolbar -l                              # 列出所有能力
    python -m financial_rag.main news "今天最大的AI新闻" -s               # 拉新闻
    python -m financial_rag.main kline "人工智能ETF" --days 30 -s         # ETF K线分析
    python -m financial_rag.main web                                      # 启动 Web UI

所有业务逻辑已迁至:
    - financial_rag.core.router.CommandRouter          (命令分发)
    - financial_rag.core.pipeline.PipelineScheduler    (Pipeline调度)
    - financial_rag.core.orchestrator.AgentOrchestrator (Agent编排)
    - financial_rag.agents.*                           (各Agent处理)
"""
import argparse
import sys
import os

from financial_rag.config import config
from financial_rag.core.router import CommandRouter
from financial_rag.core.factory import setup_environment
from financial_rag.templates import ALL_TEMPLATES


def main():
    parser = argparse.ArgumentParser(
        description="Financial RAG — 财报/经济新闻智能分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m financial_rag.main pipeline "茅台2024年营收增长了多少？"
  python -m financial_rag.main query -i
  python -m financial_rag.main query -q "茅台毛利率多少"
  python -m financial_rag.main build --dir ./my_reports
  python -m financial_rag.main analyze ./report.pdf
  python -m financial_rag.main demo
  python -m financial_rag.main score "茅台营收增长" -k 5
  python -m financial_rag.main slot "茅台财报" -t financial_report
  python -m financial_rag.main toolcall "茅台营收增长多少"
  python -m financial_rag.main toolcall -l
  python -m financial_rag.main news "今天最大的AI新闻" -s
  python -m financial_rag.main kline "人工智能ETF" --days 30 -s
        """
    )

    sub = parser.add_subparsers(dest="command", help="命令")

    # pipeline: 统一端到端入口
    plp = sub.add_parser("pipeline", help="统一Pipeline: 获取→索引→加工→输出→进化")
    plp.add_argument("query", help="用户查询")
    plp.add_argument("-t", "--template", default="quick",
                     choices=["quick", "fin", "news", "deep"],
                     help="输出模板 (默认 quick)")
    plp.add_argument("--max-fetch", type=int, default=10, help="最多获取几条数据")
    plp.add_argument("--max-retrieve", type=int, default=5, help="检索返回条数")
    plp.add_argument("-o", "--output", help="报告输出目录")
    plp.add_argument("-n", "--name", help="输出文件名")
    plp.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    # query
    qp = sub.add_parser("query", help="查询模式 (含全链路打分)")
    qp.add_argument("-q", "--question", help="查询问题")
    qp.add_argument("-i", "--interactive", action="store_true", help="交互模式")

    # build
    bp = sub.add_parser("build", help="构建知识库 (含检索打分)")
    bp.add_argument("--dir", help="财报/新闻目录")

    # analyze
    ap = sub.add_parser("analyze", help="Multi-Agent 分析")
    ap.add_argument("file", help="财报文件路径")
    ap.add_argument("--parallel", action="store_true", help="并行执行")
    ap.add_argument("--output", help="输出路径")

    # demo
    sub.add_parser("demo", help="演示模式")

    # score: 纯检索打分
    sp = sub.add_parser("score", help="仅检索链路打分测试")
    sp.add_argument("query", help="查询文本")
    sp.add_argument("-k", "--top-k", type=int, default=5, help="返回数量")
    sp.add_argument("--json", help="导出打分JSON文件路径")
    sp.add_argument("--local", action="store_true", help="纯本地模式")

    # slot: 槽位填充测试
    slp = sub.add_parser("slot", help="槽位填充 vs 自由生成 对比测试")
    slp.add_argument("query", help="查询文本")
    slp.add_argument("-t", "--template", default="financial_report",
                     choices=list(ALL_TEMPLATES.keys()),
                     help=f"模板名称 (默认 financial_report)")
    slp.add_argument("-k", "--top-k", type=int, default=5, help="检索数量")
    slp.add_argument("--no-freeform", action="store_true", help="跳过自由生成对照组")
    slp.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    # toolcall: Function Calling
    tlp = sub.add_parser("toolcall", help="Function Calling 能力注册中心测试")
    tlp.add_argument("query", help="查询文本")
    tlp.add_argument("--tool-choice", default="auto",
                     choices=["auto", "required", "none"],
                     help="tool_choice 策略")
    tlp.add_argument("--multi-turn", action="store_true", help="启用多轮调用")
    tlp.add_argument("--max-rounds", type=int, default=5, help="多轮最大轮次")
    tlp.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    tlp.add_argument("-l", "--list-tools", action="store_true", help="仅列出已注册能力")

    # news: 拉取新闻并保存为文档
    np = sub.add_parser("news", help="拉取财经新闻并保存为格式化文档")
    np.add_argument("query", help="搜索主题")
    np.add_argument("-o", "--output", help="输出目录")
    np.add_argument("-n", "--name", help="文件名")
    np.add_argument("-s", "--summarize", action="store_true", help="用 LLM 生成摘要")

    # kline: ETF K线分析
    kp = sub.add_parser("kline", help="拉取 ETF K线数据并保存为分析文档")
    kp.add_argument("query", help="ETF 主题关键词")
    kp.add_argument("-o", "--output", help="输出目录")
    kp.add_argument("-n", "--name", help="文件名")
    kp.add_argument("--days", type=int, default=30, help="回溯天数")
    kp.add_argument("--code", help="指定 ETF 代码")
    kp.add_argument("-s", "--summarize", action="store_true", help="用 LLM 生成技术分析")

    # web: 启动 Web UI
    wp = sub.add_parser("web", help="启动 Web UI (FastAPI)")
    wp.add_argument("--host", default="127.0.0.1", help="监听地址")
    wp.add_argument("--port", type=int, default=8000, help="监听端口")

    args = parser.parse_args()

    # 初始化环境
    if not setup_environment() and args.command not in ("demo", "score"):
        sys.exit(1)

    # web 命令单独处理
    if args.command == "web":
        from financial_rag.web import main as web_main
        import os
        os.environ["WEB_HOST"] = args.host
        os.environ["WEB_PORT"] = str(args.port)
        web_main()
        return

    # 交给 CommandRouter 处理
    router = CommandRouter()
    router.dispatch(args)


if __name__ == "__main__":
    main()
