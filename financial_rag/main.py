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
import time

from financial_rag.config import config
from financial_rag.core.coordinator import (
    AgentOrchestrator, CoordinatorConfig, ExecutionMode
)
from financial_rag.core.indexer import (
    PipelineOrchestrator, PipelineConfig, PipelineStatus
)
from financial_rag.core.reflector import HallucinationGuard
from financial_rag.core.scorer import PipelineScoreCard

from financial_rag.llm import get_llm, get_embedding, get_reranker
from financial_rag.retrievers import HybridRetriever, jieba_tokenizer
from financial_rag.templates import (
    SlottedTemplate, FINANCIAL_REPORT_TEMPLATE, NEWS_BRIEF_TEMPLATE,
    QUICK_QA_TEMPLATE, DEEP_ANALYSIS_TEMPLATE,
    get_template, ALL_TEMPLATES,
)
from financial_rag.slot_filler import SlotFiller, FillStats, create_slot_filler
from financial_rag.tools import (
    FunctionRegistry, FunctionDef, ToolExecutor, ToolCallSession,
    ToolCallStats, ToolCallResult, CATEGORIES,
    create_financial_registry, create_tool_session,
)

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
    创建带阿里 Embedding + Rerank + Jieba 分词的混合检索器

    全链路: Jieba 分词 → BM25 → text-embedding-v3 → RRF → gte-rerank → Top-K
    """
    api_key = config.llm.api_key

    # 尝试加载 jieba 分词器
    tokenizer = None
    try:
        tokenizer = jieba_tokenizer()
    except ImportError:
        pass  # 回退到正则分词

    if not api_key:
        # 无 API Key → 纯本地模式（Jieba/正则 + BM25 + Jaccard）
        print("[WARN] 未设置 DASHSCOPE_API_KEY，回退到纯本地检索")
        return HybridRetriever(tokenizer=tokenizer)

    return HybridRetriever(
        embedder=get_embedding(api_key=api_key),
        reranker=get_reranker(api_key=api_key),
        tokenizer=tokenizer,
    )


# ===================== 打分展示工具 =====================

def show_scorecard(card: PipelineScoreCard, title: str = None):
    """打印评分卡详细信息"""
    if not card or not card.stages:
        print("\n[无评分数据]")
        return
    print()
    if title:
        print(title)
    print(card.summary())


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
        print("\n交互模式，输入 'q' 退出，输入 'score' 查看上次打分")
        print("模板: quick=快答, fin=财报, news=新闻, deep=深度分析 (默认 quick)\n")
        last_card = None
        current_template = QUICK_QA_TEMPLATE
        while True:
            try:
                q = input("输入问题: ").strip()
                if q.lower() in ('q', 'quit', 'exit'):
                    break
                if not q:
                    continue
                if q.lower() == 'score':
                    show_scorecard(last_card, "上次全链路打分卡:")
                    continue

                # 切换模板
                template_map = {
                    "quick": QUICK_QA_TEMPLATE, "fin": FINANCIAL_REPORT_TEMPLATE,
                    "news": NEWS_BRIEF_TEMPLATE, "deep": DEEP_ANALYSIS_TEMPLATE,
                }
                if q.lower() in template_map:
                    current_template = template_map[q.lower()]
                    print(f"[模板] 已切换到: {current_template.description}")
                    continue

                try:
                    card = PipelineScoreCard(query=q)
                    last_card = card

                    if has_key:
                        # 检索
                        retriever = create_hybrid_retriever()
                        sample_docs = [
                            {"text": "贵州茅台2024年营收1738.52亿元，同比增长15.66%", "meta": {"source": "maotai_2024"}},
                            {"text": "茅台2024年净利润862.28亿元，同比增长15.38%", "meta": {"source": "maotai_2024"}},
                            {"text": "2024年茅台酒毛利率91.86%，ROE为34.19%", "meta": {"source": "maotai_2024"}},
                            {"text": "茅台酒营收1465.33亿元，系列酒营收246.84亿元", "meta": {"source": "maotai_2024"}},
                            {"text": "2024年茅台经营活动现金流753.29亿元", "meta": {"source": "maotai_2024"}},
                            {"text": "2025年人民币汇率预计在7.0-7.3区间波动", "meta": {"source": "economic_outlook"}},
                            {"text": "央行2025年一季度降准0.5个百分点，释放流动性约1万亿", "meta": {"source": "pboc_policy"}},
                        ]
                        retriever.index(sample_docs)
                        results, ret_card = retriever.search_with_scores(q, top_k=3)
                        card.stages.extend(ret_card.stages)

                        # ---- 槽位填充替代自由生成 ----
                        llm = get_llm(api_key=config.llm.api_key, model=config.llm.model)
                        filler = create_slot_filler(llm=llm, scorecard=card, verbose=False)
                        context_docs = [r.get("text", "") for r in results[:3]]

                        t_fill = time.time()
                        fill_stats = filler.fill(current_template, query=q, context_docs=context_docs)
                        final_output = filler.render(current_template, fill_stats)
                        fill_elapsed = (time.time() - t_fill) * 1000

                        # 记录 LLM 生成汇总
                        card.record_llm(
                            score=fill_stats.filled_slots / max(fill_stats.total_slots, 1),
                            token_count=fill_stats.total_tokens,
                            model=config.llm.model,
                            elapsed_ms=fill_elapsed,
                        )

                        # 防幻觉校验（对最终渲染文本）
                        guard = HallucinationGuard()
                        check = guard.check(final_output, results)
                        card.record_hallucination(
                            overall_score=check['overall_score'],
                            layer_scores={k: v.get("score", 0) for k, v in check.get('checks', {}).items()},
                            risk=check['risk'],
                        )

                        # ---- 输出 ----
                        print(f"\n[模板] {current_template.description}")
                        print(f"[结果]")
                        print(final_output[:400])
                        if len(final_output) > 400:
                            print("...")

                        print(f"\n[性能] 总耗时 {fill_elapsed:.0f}ms, "
                              f"首Token avg={fill_stats.avg_ttft_ms:.0f}ms | "
                              f"槽位 {fill_stats.filled_slots}/{fill_stats.total_slots}"
                              f" (并行增益 {fill_stats.parallel_gain:.0%})")

                        show_scorecard(card)
                    else:
                        print(f"\n[模拟] 关于 {q} 的分析...\n")
                        print("[提示] 设置 DASHSCOPE_API_KEY 启用真实 LLM + 槽位填充\n")
                except Exception as e:
                    print(f"\n[错误] API 调用失败: {e}")
                    print("[提示] 请检查 DASHSCOPE_API_KEY 是否正确\n")

            except KeyboardInterrupt:
                print("\n退出")
                break
    elif args.question:
        print(f"\n问题: {args.question}")
        print("分析中 (槽位填充)...\n")
        try:
            if has_key:
                card = PipelineScoreCard(query=args.question)
                # 检索
                retriever = create_hybrid_retriever()
                sample_docs = [
                    {"text": "贵州茅台2024年营收1738.52亿元，同比增长15.66%", "meta": {"source": "maotai_2024"}},
                    {"text": "茅台2024年净利润862.28亿元，同比增长15.38%", "meta": {"source": "maotai_2024"}},
                    {"text": "2024年茅台酒毛利率91.86%，ROE为34.19%", "meta": {"source": "maotai_2024"}},
                ]
                retriever.index(sample_docs)
                results, ret_card = retriever.search_with_scores(args.question, top_k=3)
                card.stages.extend(ret_card.stages)
                # 槽位填充
                llm = get_llm(api_key=config.llm.api_key, model=config.llm.model)
                filler = create_slot_filler(llm=llm, scorecard=card, verbose=False)
                fill_stats = filler.fill(QUICK_QA_TEMPLATE, query=args.question,
                                         context_docs=[r.get("text", "") for r in results[:3]])
                final_output = filler.render(QUICK_QA_TEMPLATE, fill_stats)
                print(final_output)
                print(f"\n[性能] 槽位 {fill_stats.filled_slots}/{fill_stats.total_slots}, "
                      f"TTFT avg={fill_stats.avg_ttft_ms:.0f}ms")
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

    # 测试检索（带打分）
    test_qs = ["茅台营收多少", "汇率走势如何"]
    for q in test_qs:
        results, card = retriever.search_with_scores(q, top_k=3)
        print(f"Q: {q}")
        for r in results:
            print(f"  [{r.get('retriever', '?')}] score={r.get('score', 0):.4f} | {r['text'][:60]}")
        show_scorecard(card)
        print()


def cmd_analyze(args):
    """Multi-Agent 财报分析（带打分）"""
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

    # 创建打分卡
    card = PipelineScoreCard(query=os.path.basename(args.file))

    result = orch.execute(args.file)
    print(f"\n完成: {result.success}, 耗时: {result.execution_time:.1f}s")

    # 记录每个 Agent 的评分
    for i, r in enumerate(result.agent_results):
        icon = "OK" if r.success else "FAIL"
        # 计算 Agent 阶段评分
        agent_score = 0.0
        if r.success:
            if r.agent_name == "IngestionAgent":
                # 从 context_updates 中提取 metadata 评分
                meta = r.context_updates.get("metadata", {})
                agent_score = meta.get("metadata_score", 0.5)
                fields_found = meta.get("metadata_fields_found", 0)
                fields_expected = meta.get("metadata_fields_expected", 7)
                card.record_metadata(agent_score, fields_found, fields_expected,
                                     elapsed_ms=r.execution_time * 1000)
            elif r.agent_name == "ExtractionAgent":
                data = r.data or {}
                agent_score = data.get("_scores", {}).get("extraction", 0.5)
                card.record_keyword_extract(agent_score,
                    keyword_count=len(data.get("metrics", {})),
                    elapsed_ms=r.execution_time * 1000)
                query_score = data.get("_scores", {}).get("query_rewrite", 0.5)
                card.record_query_rewrite(query_score, result.context.raw_input if result.context else "",
                    rewritten_queries=data.get("queries", []),
                    elapsed_ms=0)
            else:
                agent_score = 0.5 + (0.3 if r.success else 0)
        print(f"  [{icon}] {r.agent_name}: {r.message} (评分: {agent_score:.2f})")

    show_scorecard(card, "📊 Multi-Agent 全链路打分卡:")


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

            print("[全链路检索测试 (含打分)]")
            for q in ["茅台2024年盈利情况", "茅台各产品线营收"]:
                results, card = retriever.search_with_scores(q, top_k=3, use_rerank=True)
                print(f"\n  Q: {q}")
                for r in results:
                    label = r.get('relevance_level', '?')
                    print(f"    [{r['retriever']}] relev={label} score={r.get('score', 0):.4f} | {r['text'][:60]}")
                show_scorecard(card)
            print()
        except Exception as e:
            print(f"[注意] 检索demo失败: {e}\n")

    print("使用方法:")
    print("  python -m financial_rag.main query -i          # 交互查询 (含打分)")
    print("  python -m financial_rag.main analyze <文件>    # Multi-Agent分析 (含打分)")
    print("  python -m financial_rag.main build --dir <目录> # 构建知识库 (含打分)")
    print("  python -m financial_rag.main score <查询文本>  # 仅跑检索打分")


def cmd_score(args):
    """仅运行检索打分（不调用 LLM）"""
    print("=" * 60)
    print("检索全链路打分测试")
    print("=" * 60)
    print(f"查询: {args.query}")

    if args.local:
        tokenizer = None
        try:
            tokenizer = jieba_tokenizer()
            print("分词器: Jieba (已加载金融词典)")
        except ImportError:
            print("分词器: 正则 (pip install jieba 启用中文分词)")
        retriever = HybridRetriever(tokenizer=tokenizer)
    else:
        retriever = create_hybrid_retriever()

    sample_docs = [
        {"text": "贵州茅台2024年营收1738.52亿元，同比增长15.66%", "meta": {"source": "maotai_2024"}},
        {"text": "茅台2024年净利润862.28亿元，同比增长15.38%", "meta": {"source": "maotai_2024"}},
        {"text": "2024年茅台酒毛利率91.86%，ROE为34.19%", "meta": {"source": "maotai_2024"}},
        {"text": "茅台酒营收1465.33亿元，系列酒营收246.84亿元", "meta": {"source": "maotai_2024"}},
        {"text": "2025年人民币汇率预计在7.0-7.3区间波动", "meta": {"source": "economic_outlook"}},
        {"text": "央行2025年一季度降准0.5个百分点", "meta": {"source": "pboc_policy"}},
        {"text": "2024年茅台经营活动现金流量净额753.29亿元", "meta": {"source": "maotai_2024"}},
    ]
    retriever.index(sample_docs)

    results, card = retriever.search_with_scores(args.query, top_k=args.top_k)

    print(f"\n检索结果 ({len(results)} 条):")
    for r in results:
        label = r.get('relevance_level', '?')
        print(f"  [{r['retriever']}] relev={label} score={r.get('score', 0):.4f} | {r['text'][:70]}")

    show_scorecard(card, "检索全链路打分卡:")

    if args.json:
        import json
        json_path = args.json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(card.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n评分详情已导出: {json_path}")


def cmd_slot(args):
    """槽位填充测试 — 对比自由生成 vs 槽位填充的首 Token 延迟"""
    has_key = bool(config.llm.api_key)
    if not has_key:
        print("[错误] 槽位填充需要 DASHSCOPE_API_KEY")
        return

    print("=" * 60)
    print("槽位填充测试 — 首 Token 延迟对比")
    print("=" * 60)

    template = get_template(args.template)
    if not template:
        print(f"未知模板: {args.template}")
        print(f"可选: {', '.join(ALL_TEMPLATES.keys())}")
        return

    print(f"模板: {template.name} — {template.description}")
    print(f"槽位: {len(template.slots)} 个, {len(template.phases)} 个阶段")
    print(f"查询: {args.query}")
    print("=" * 60)

    # 检索
    retriever = create_hybrid_retriever()
    sample_docs = [
        {"text": "贵州茅台2024年营收1738.52亿元，同比增长15.66%", "meta": {"source": "maotai_2024"}},
        {"text": "茅台2024年净利润862.28亿元，同比增长15.38%", "meta": {"source": "maotai_2024"}},
        {"text": "2024年茅台酒毛利率91.86%，ROE为34.19%", "meta": {"source": "maotai_2024"}},
        {"text": "茅台酒营收1465.33亿元，系列酒营收246.84亿元", "meta": {"source": "maotai_2024"}},
        {"text": "2024年茅台经营活动现金流753.29亿元", "meta": {"source": "maotai_2024"}},
        {"text": "2025年人民币汇率预计在7.0-7.3区间波动", "meta": {"source": "economic_outlook"}},
        {"text": "央行2025年一季度降准0.5个百分点", "meta": {"source": "pboc_policy"}},
    ]
    retriever.index(sample_docs)
    results, ret_card = retriever.search_with_scores(args.query, top_k=args.top_k)
    context_docs = [r.get("text", "") for r in results[:args.top_k]]

    # ---- 对比测试 ----
    llm = get_llm(api_key=config.llm.api_key, model=config.llm.model)
    card = PipelineScoreCard(query=args.query)

    if not args.no_freeform:
        # 传统自由生成（对照组）
        print("\n[对照组] 传统自由生成:")
        context_text = "\n".join(doc[:200] for doc in context_docs[:3])
        t_f = time.time()
        resp = llm.chat(
            messages=f"根据以下参考信息回答问题。\n参考:\n{context_text}\n\n问题: {args.query}",
            system="你是专业金融分析师，回答必须准确有依据。不确定请说明。",
            max_tokens=600,
        )
        free_elapsed = (time.time() - t_f) * 1000
        free_tokens = resp.usage.get('total_tokens', len(resp.content))
        print(f"  耗时: {free_elapsed:.0f}ms")
        print(f"  Tokens: {free_tokens}")
        print(f"  输出(前200字): {resp.content[:200]}{'...' if len(resp.content)>200 else ''}")

    # 槽位填充
    print(f"\n[实验组] 槽位填充:")
    filler = create_slot_filler(llm=llm, scorecard=card, verbose=args.verbose)

    t_fill = time.time()
    fill_stats = filler.fill(template, query=args.query, context_docs=context_docs)
    final_output = filler.render(template, fill_stats)
    fill_elapsed = (time.time() - t_fill) * 1000

    print(f"  总耗时: {fill_elapsed:.0f}ms")
    print(f"  总Tokens: {fill_stats.total_tokens}")
    print(f"  槽位: {fill_stats.filled_slots}/{fill_stats.total_slots} 个")
    print(f"  首Token: avg={fill_stats.avg_ttft_ms:.0f}ms, peak={fill_stats.peak_ttft_ms:.0f}ms")
    print(f"  并行增益: {fill_stats.parallel_gain:.0%}")

    if not args.no_freeform:
        print(f"\n  [对比] 槽位填充 vs 自由生成:")
        time_diff = free_elapsed - fill_elapsed
        direction = "更快" if time_diff > 0 else "更慢"
        print(f"    总耗时: {fill_elapsed:.0f}ms vs {free_elapsed:.0f}ms (槽位 {direction} {abs(time_diff):.0f}ms)")
        print(f"    Tokens:  {fill_stats.total_tokens} vs {free_tokens}")

    print(f"\n[槽位详情]")
    for key, r in fill_stats.slot_results.items():
        status = "OK" if r.filled else "FAIL"
        val_preview = r.value[:50] + "..." if len(r.value) > 50 else r.value
        print(f"  [{status}] {r.label:<8s} | TTFT={r.ttft_ms:5.0f}ms | {val_preview}")

    print(f"\n[渲染输出]")
    print(final_output[:400])
    if len(final_output) > 400:
        print("...")

    show_scorecard(card, "全链路打分卡 (含槽位评分):")


def cmd_toolcall(args):
    """Function Calling 模式测试"""
    has_key = bool(config.llm.api_key)
    if not has_key and not args.list_tools:
        print("[错误] Function Calling 需要 DASHSCOPE_API_KEY")
        return

    print("=" * 60)
    print("Function Calling — 能力注册中心测试")
    print("=" * 60)

    # 初始化检索器 + 注册中心
    retriever = create_hybrid_retriever()
    sample_docs = [
        {"text": "贵州茅台2024年营收1738.52亿元，同比增长15.66%", "meta": {"source": "maotai_2024"}},
        {"text": "茅台2024年净利润862.28亿元，同比增长15.38%", "meta": {"source": "maotai_2024"}},
        {"text": "2024年茅台酒毛利率91.86%，ROE为34.19%", "meta": {"source": "maotai_2024"}},
        {"text": "茅台酒营收1465.33亿元，系列酒营收246.84亿元", "meta": {"source": "maotai_2024"}},
        {"text": "2024年茅台经营活动现金流753.29亿元", "meta": {"source": "maotai_2024"}},
        {"text": "2025年人民币汇率预计在7.0-7.3区间波动", "meta": {"source": "economic_outlook"}},
        {"text": "央行2025年一季度降准0.5个百分点，释放流动性约1万亿", "meta": {"source": "pboc_policy"}},
    ]
    try:
        retriever.index(sample_docs)
    except Exception as e:
        if has_key:
            print(f"[WARN] 检索器索引失败: {e}")

    registry = create_financial_registry(retriever=retriever)
    print(registry)
    print()

    if args.list_tools:
        print(f"能力清单 ({len(registry)} 个):")
        for f in registry.functions.values():
            required = ", ".join(f.parameters.get("required", []))
            print(f"  [{f.category}] {f.name}")
            print(f"    {f.description[:80]}...")
            print(f"    参数: {required}")
        return

    llm = get_llm(api_key=config.llm.api_key, model=config.llm.model)
    card = PipelineScoreCard(query=args.query)

    print(f"问题: {args.query}")
    print(f"模式: {'多轮' if args.multi_turn else '单轮'} | "
          f"tool_choice: {args.tool_choice} | verbose: {args.verbose}")
    system = (
        "你是专业金融分析师。当需要具体数据时，必须调用提供的函数获取。"
        "不要捏造任何具体数字。如果函数返回了数据，基于数据给出准确分析。"
    )

    print("\n" + "=" * 60)
    print("执行 Function Calling 会话...")
    print("=" * 60)

    session = create_tool_session(
        llm=llm,
        retriever=retriever,
        registry=registry,
        system_prompt=system,
        max_rounds=args.max_rounds,
        verbose=args.verbose,
    )

    t_start = time.time()
    stats = session.run(args.query, scorecard=card)
    t_elapsed = (time.time() - t_start) * 1000

    # 输出结果
    print(f"\n[会话统计]")
    print(f"  轮次: {stats.rounds} 轮")
    print(f"  工具调用: {len(stats.calls)} 次 ({stats.succeeded} 成功, {stats.failed} 失败)")
    print(f"  使用的能力: {', '.join(stats.tools_used) or '(无)'}")
    print(f"  总耗时: {stats.total_elapsed_ms:.0f}ms")
    print(f"  Tokens: {stats.total_tokens}")

    print(f"\n[工具调用详情]")
    for c in stats.calls:
        icon = "OK" if c.success else "FAIL"
        result_preview = str(c.result)[:80] + "..." if len(str(c.result)) > 80 else str(c.result)
        print(f"  [{icon}] {c.name} ({c.elapsed_ms:.0f}ms) → {result_preview}")

    show_scorecard(card, "📊 Function Calling 全链路打分卡:")


# ===================== main =====================

def main():
    parser = argparse.ArgumentParser(
        description="Financial RAG — 财报/经济新闻智能分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m financial_rag.main demo                              # 演示
  python -m financial_rag.main query -i                          # 交互查询 (含槽位填充)
  python -m financial_rag.main query -q "茅台毛利率多少"          # 单次查询
  python -m financial_rag.main build --dir ./my_reports          # 建知识库
  python -m financial_rag.main analyze ./report.pdf              # Multi-Agent分析
  python -m financial_rag.main score "茅台营收增长" -k 5          # 仅检索打分
  python -m financial_rag.main slot "茅台财报" -t financial_report # 槽位填充对比
  python -m financial_rag.main toolcall "茅台营收增长多少"         # Function Calling
  python -m financial_rag.main toolcall "对比茅台和五粮液" -v     # 带日志
  python -m financial_rag.main toolcall -l                        # 列出所有能力
        """
    )

    sub = parser.add_subparsers(dest="command", help="命令")

    # query
    qp = sub.add_parser("query", help="查询模式 (含全链路打分)")
    qp.add_argument("-q", "--question", help="查询问题")
    qp.add_argument("-i", "--interactive", action="store_true", help="交互模式")

    # build
    bp = sub.add_parser("build", help="构建知识库 (含检索打分)")
    bp.add_argument("--dir", help="财报/新闻目录")

    # analyze
    ap = sub.add_parser("analyze", help="Multi-Agent 分析 (含全链路打分)")
    ap.add_argument("file", help="财报文件路径")
    ap.add_argument("--parallel", action="store_true", help="并行执行")
    ap.add_argument("--output", help="输出路径")

    # demo
    sub.add_parser("demo", help="演示模式 (含全链路打分)")

    # score: 纯检索打分
    sp = sub.add_parser("score", help="仅检索链路打分测试")
    sp.add_argument("query", help="查询文本")
    sp.add_argument("-k", "--top-k", type=int, default=5, help="返回数量 (默认5)")
    sp.add_argument("--json", help="导出打分JSON文件路径")
    sp.add_argument("--local", action="store_true", help="纯本地模式 (不使用 API)")

    # slot: 槽位填充测试
    slp = sub.add_parser("slot", help="槽位填充 vs 自由生成 对比测试")
    slp.add_argument("query", help="查询文本")
    slp.add_argument("-t", "--template", default="financial_report",
                     choices=list(ALL_TEMPLATES.keys()),
                     help=f"模板名称 (默认 financial_report)")
    slp.add_argument("-k", "--top-k", type=int, default=5, help="检索数量 (默认5)")
    slp.add_argument("--no-freeform", action="store_true", help="跳过自由生成对照组")
    slp.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    # toolcall: Function Calling 能力注册中心测试
    tlp = sub.add_parser("toolcall", help="Function Calling 能力注册中心测试")
    tlp.add_argument("query", help="查询文本")
    tlp.add_argument("--tool-choice", default="auto",
                     choices=["auto", "required", "none"],
                     help="tool_choice 策略 (默认 auto)")
    tlp.add_argument("--multi-turn", action="store_true",
                     help="启用多轮调用 (LLM 可多次选工具)")
    tlp.add_argument("--max-rounds", type=int, default=5,
                     help="多轮最大轮次 (默认 5)")
    tlp.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    tlp.add_argument("-l", "--list-tools", action="store_true",
                     help="仅列出已注册能力")

    args = parser.parse_args()

    if not setup_environment() and args.command not in ("demo", "score", "slot", "toolcall"):
        sys.exit(1)

    if args.command == "query":
        cmd_query(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "demo":
        cmd_demo()
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "slot":
        cmd_slot(args)
    elif args.command == "toolcall":
        cmd_toolcall(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
