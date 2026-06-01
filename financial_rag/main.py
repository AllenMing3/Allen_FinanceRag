"""
Financial RAG — 主入口（阿里百炼 DashScope）

用法:
    # 交互模式
    python -m financial_rag.main

    # 单次查询
    python -m financial_rag.main query "茅台2024年营收增长了多少？"

    # 构建知识库
    python -m financial_rag.main build --dir ./data/financial

    # 分析财报
    python -m financial_rag.main analyze ./data/financial/maotai_2024.pdf

    # 演示模式
    python -m financial_rag.main demo
"""
import argparse
import os
import sys

from financial_rag.config import config
from financial_rag.core.coordinator import (
    AgentOrchestrator, CoordinatorConfig, ExecutionMode
)
from financial_rag.core.indexer import (
    PipelineOrchestrator, PipelineConfig, PipelineStatus
)
from financial_rag.core.reflector import HallucinationGuard

from financial_rag.llm import get_llm, get_embedding, get_reranker
from financial_rag.retrievers import HybridRetriever

from financial_rag.agents.ingestion_agent import IngestionAgent
from financial_rag.agents.extraction_agent import ExtractionAgent
from financial_rag.agents.analysis_agent import AnalysisAgent
from financial_rag.agents.forecast_agent import ForecastAgent
from financial_rag.agents.report_agent import ReportAgent


# ===================== 工厂函数 =====================

def create_orchestrator() -> AgentOrchestrator:
    """创建完整的 Multi-Agent 协调器"""
    orch = AgentOrchestrator(CoordinatorConfig(
        execution_mode=ExecutionMode.SEQUENTIAL,
        verbose=config.coordinator.verbose,
        max_retries=config.coordinator.max_retries,
    ))
    orch.register_all(
        IngestionAgent(),
        ExtractionAgent(),
        AnalysisAgent(),
        ForecastAgent(),
        ReportAgent(),
    )
    return orch


def create_hybrid_retriever() -> HybridRetriever:
    """
    创建带阿里 Embedding + Rerank 的混合检索器

    全链路: BM25 → text-embedding-v3 → RRF → gte-rerank → Top-K
    """
    api_key = config.llm.api_key
    if not api_key:
        # 无 API Key → 纯本地模式（BM25 + Jaccard）
        print("[WARN] 未设置 DASHSCOPE_API_KEY，回退到纯本地检索")
        return HybridRetriever()

    return HybridRetriever(
        embedder=get_embedding(api_key=api_key),
        reranker=get_reranker(api_key=api_key),
    )


# ===================== 命令处理器 =====================

def setup_environment():
    """初始化环境"""
    for d in [config.data_dir, config.kb_dir, config.output_dir]:
        os.makedirs(d, exist_ok=True)

    has_key = bool(config.llm.api_key)
    if not has_key:
        print("[WARN] 未设置 DASHSCOPE_API_KEY，使用纯本地模式")
        print("       设置: export DASHSCOPE_API_KEY=sk-xxx")
        print("       获取: https://bailian.console.aliyun.com/\n")
    else:
        print(f"[INFO] DashScope API 已配置，模型: {config.llm.model}")
    return True


def cmd_query(args):
    """查询模式"""
    print("=" * 60)
    print("Financial RAG — 财报/经济新闻智能分析 (阿里百炼)")
    print("=" * 60)

    has_key = bool(config.llm.api_key)

    if args.interactive:
        print("\n交互模式，输入 'q' 退出\n")
        while True:
            try:
                q = input("输入问题: ").strip()
                if q.lower() in ('q', 'quit', 'exit'):
                    break
                if not q:
                    continue

                try:
                    if has_key:
                        # 全链路: LLM 生成 + 防幻觉校验
                        llm = get_llm(api_key=config.llm.api_key, model=config.llm.model)
                        response = llm.chat(
                            messages=f"你是财报分析专家。请用中文回答: {q}",
                            system="你是一个专业的金融分析师，回答必须准确、有依据。",
                        )
                        answer = response.content
                        guard = HallucinationGuard()
                        check = guard.check(answer, [])
                        print(f"\n[Qwen] {answer}\n")
                        print(f"[防幻觉] 评分: {check['overall_score']:.2f}, 风险: {check['risk']}")
                        print(f"[Token] {response.usage.get('total_tokens', 0)}")
                    else:
                        print(f"\n[模拟] 关于 {q} 的分析...\n")
                        print("[提示] 设置 DASHSCOPE_API_KEY 启用真实 LLM\n")
                except Exception as e:
                    print(f"\n[错误] API 调用失败: {e}")
                    print("[提示] 请检查 DASHSCOPE_API_KEY 是否正确\n")

            except KeyboardInterrupt:
                print("\n退出")
                break
    elif args.question:
        print(f"\n问题: {args.question}")
        print("分析中...")
        try:
            if has_key:
                llm = get_llm(api_key=config.llm.api_key, model=config.llm.model)
                response = llm.chat(args.question)
                print(f"\n{response.content}\n")
        except Exception as e:
            print(f"\n[错误] API 调用失败: {e}")


def cmd_build(args):
    """构建知识库（含 Embedding）"""
    print("=" * 60)
    print("构建金融知识库 (阿里百炼 Embedding)")
    print("=" * 60)
    dir_path = args.dir or config.data_dir

    if not os.path.isdir(dir_path):
        print(f"目录不存在: {dir_path}")
        return

    has_key = bool(config.llm.api_key)
    retriever = create_hybrid_retriever()

    # 模拟加载文档
    print(f"来源: {dir_path}")
    print(f"Embedding: {'text-embedding-v3' if has_key else '无 (纯本地)'}")
    print(f"Rerank:     {'gte-rerank' if has_key else '无 (纯本地)'}\n")

    # 构建示例文档
    sample_docs = [
        {"text": "贵州茅台2024年营收1738.52亿元，同比增长15.66%", "meta": {"source": "maotai_2024"}},
        {"text": "茅台2024年净利润862.28亿元，同比增长15.38%", "meta": {"source": "maotai_2024"}},
        {"text": "2024年茅台酒毛利率91.86%，ROE为34.19%", "meta": {"source": "maotai_2024"}},
        {"text": "2025年人民币汇率预计在7.0-7.3区间波动", "meta": {"source": "economic_outlook"}},
        {"text": "央行2025年一季度降准0.5个百分点，释放流动性约1万亿", "meta": {"source": "pboc_policy"}},
    ]
    retriever.index(sample_docs)
    print(f"已索引 {len(sample_docs)} 篇文档\n")

    # 测试检索
    test_qs = ["茅台营收多少", "汇率走势如何"]
    for q in test_qs:
        results = retriever.search(q, top_k=3)
        print(f"Q: {q}")
        for r in results:
            print(f"  [{r.get('retriever', '?')}] score={r.get('score', 0):.4f} | {r['text'][:60]}")
        print()


def cmd_analyze(args):
    """Multi-Agent 财报分析"""
    print("=" * 60)
    print("Multi-Agent 财报分析 (阿里百炼)")
    print("=" * 60)

    if not args.file:
        print("请指定财报文件")
        return

    orch = create_orchestrator()
    if args.parallel:
        orch.config.execution_mode = ExecutionMode.PARALLEL
        print("模式: 并行")
    print(f"Pipeline: {' -> '.join(orch.pipeline)}")
    print(f"文件: {args.file}")

    if config.llm.api_key:
        print(f"LLM: {config.llm.model} | Embedding: {config.llm.embedding_model} | Rerank: {config.llm.rerank_model}")
    print("=" * 60)

    result = orch.execute(args.file)
    print(f"\n完成: {result.success}, 耗时: {result.execution_time:.1f}s")
    for r in result.agent_results:
        icon = "OK" if r.success else "FAIL"
        print(f"  [{icon}] {r.agent_name}: {r.message}")


def cmd_demo():
    """演示模式 — 展示全链路"""
    print("=" * 60)
    print("Financial RAG — 演示 (阿里百炼全链路)")
    print("=" * 60)

    has_key = bool(config.llm.api_key)
    print(f"\n[配置]")
    print(f"  LLM:      {config.llm.model} {'(已配置)' if has_key else '(未配置)'}")
    print(f"  Embedding: {config.llm.embedding_model}")
    print(f"  Rerank:    {config.llm.rerank_model}")
    print(f"  Provider:  {config.llm.provider}")

    # 创建协调器
    orch = create_orchestrator()

    # 模拟财报数据
    sample = """
    贵州茅台 2024 年年度报告摘要:

    2024年实现营业收入 1,738.52 亿元，同比增长 15.66%；
    归属上市公司股东净利润 862.28 亿元，同比增长 15.38%；
    基本每股收益 68.64 元，同比增长 15.38%；
    毛利率 91.86%；
    ROE（净资产收益率）34.19%；
    经营活动现金流量净额 753.29 亿元。

    分产品看，茅台酒营收 1,465.33 亿元，系列酒营收 246.84 亿元。
    产能方面，2024年茅台酒基酒产量 5.63 万吨，系列酒基酒产量 4.81 万吨。

    2025年经营目标: 计划实现营业收入同比增长约 15%。
    """

    print(f"\n[输入数据]")
    print(sample.strip())
    print("\n" + "=" * 60)
    print("三大架构概览:")
    print("=" * 60)

    print(f"""
    1. Coordinate — 多 Agent 协调调度
       IngestionAgent → ExtractionAgent → AnalysisAgent
                       → ForecastAgent → ReportAgent

    2. Indexer — 多文本索引流水线 (阿里 Embedding + Rerank)
       BM25关键词 → text-embedding-v3 语义检索
                  → RRF 融合 → gte-rerank 精排 → Top-K

    3. Reflection — 多维评分 + 六层防幻觉
       ReAct循环: Think → Act → Judge
       六层校验: L1来源→L2一致性→L3事实→L4完整→L5引用→L6综合
    """)

    print(f"已注册 {len(orch.agents)} 个 Agent")
    print(f"Pipeline: {' -> '.join(orch.pipeline)}")
    print(f"\n全链路检索: BM25 → Embedding → RRF → Rerank → Top-K\n")

    # 如果有 API Key，跑一个真实检索 demo
    if has_key:
        try:
            retriever = create_hybrid_retriever()
            test_docs = [
                {"text": "贵州茅台2024年营业收入1738.52亿元，同比增长15.66%", "meta": {}},
                {"text": "茅台2024年归属净利润862.28亿元，同比增长15.38%", "meta": {}},
                {"text": "2024年茅台毛利率91.86%，ROE 34.19%", "meta": {}},
                {"text": "茅台酒营收1465.33亿元，系列酒营收246.84亿元", "meta": {}},
                {"text": "2024年茅台经营活动现金流量净额753.29亿元", "meta": {}},
            ]
            retriever.index(test_docs, precompute_embeddings=True)

            print("[全链路检索测试]")
            for q in ["茅台2024年盈利情况", "茅台各产品线营收"]:
                results = retriever.search(q, top_k=3, use_rerank=True)
                print(f"\n  Q: {q}")
                for r in results:
                    label = r.get('relevance_level', '?')
                    print(f"    [{r['retriever']}] relev={label} score={r.get('score', 0):.4f} | {r['text'][:60]}")
            print()
        except Exception as e:
            print(f"[注意] 检索demo失败: {e}\n")

    print("使用方法:")
    print("  python -m financial_rag.main query -i          # 交互查询")
    print("  python -m financial_rag.main analyze <文件>    # Multi-Agent分析")
    print("  python -m financial_rag.main build --dir <目录> # 构建知识库")


# ===================== main =====================

def main():
    parser = argparse.ArgumentParser(
        description="Financial RAG — 财报/经济新闻智能分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m financial_rag.main demo                          # 演示
  python -m financial_rag.main query -i                      # 交互查询
  python -m financial_rag.main query -q "茅台毛利率多少"      # 单次查询
  python -m financial_rag.main build --dir ./my_reports      # 建知识库
  python -m financial_rag.main analyze ./report.pdf            # Multi-Agent分析
  python -m financial_rag.main analyze ./report.pdf --parallel # 并行分析
        """
    )

    sub = parser.add_subparsers(dest="command", help="命令")

    # query
    qp = sub.add_parser("query", help="查询模式")
    qp.add_argument("-q", "--question", help="查询问题")
    qp.add_argument("-i", "--interactive", action="store_true", help="交互模式")

    # build
    bp = sub.add_parser("build", help="构建知识库")
    bp.add_argument("--dir", help="财报/新闻目录")

    # analyze
    ap = sub.add_parser("analyze", help="Multi-Agent 分析")
    ap.add_argument("file", help="财报文件路径")
    ap.add_argument("--parallel", action="store_true", help="并行执行")
    ap.add_argument("--output", help="输出路径")

    # demo
    sub.add_parser("demo", help="演示模式")

    args = parser.parse_args()

    if not setup_environment() and args.command != "demo":
        sys.exit(1)

    if args.command == "query":
        cmd_query(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "demo":
        cmd_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
