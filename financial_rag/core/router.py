"""
CommandRouter — 命令路由器

接收 argparse 解析后的 args，分发到对应处理逻辑。
main.py 只负责 argparse，所有业务代码在此和 PipelineScheduler 中。
"""
import os
import time
import logging
from typing import Any

from financial_rag.core.base import ExecutionMode
from financial_rag.core.factory import create_orchestrator, create_hybrid_retriever
from financial_rag.core.pipeline import (
    PipelineConfig,
    PipelineScheduler,
    create_pipeline_scheduler,
)

logger = logging.getLogger(__name__)


class CommandRouter:
    """
    命令路由器 — 接收 argparse 解析后的 args，分发到对应处理逻辑。

    main.py 只负责 argparse，所有业务代码在此和 PipelineScheduler 中。
    """

    def __init__(self):
        from financial_rag.config import config as _cfg
        from financial_rag.llm import get_llm
        from financial_rag.tools import create_financial_registry, ToolExecutor, create_tool_session
        from financial_rag.slot_filler import create_slot_filler

        self._cfg = _cfg
        self._has_key = bool(_cfg.llm.api_key)
        self._llm = get_llm(api_key=_cfg.llm.api_key, model=_cfg.llm.model) if self._has_key else None
        self._retriever = create_hybrid_retriever()
        self._registry = create_financial_registry(retriever=self._retriever, llm=self._llm)
        self._executor = ToolExecutor(self._registry)
        self._filler = create_slot_filler(llm=self._llm, verbose=False) if self._llm else None
        self._orchestrator = create_orchestrator()

        self._scheduler = create_pipeline_scheduler(
            orchestrator=self._orchestrator,
            retriever=self._retriever,
            registry=self._registry,
            executor=self._executor,
            llm=self._llm,
            filler=self._filler,
            config=PipelineConfig(verbose=False),
        )

        # 内置样本数据（AI 行业）
        self._sample_docs = [
            {"text": "商汤科技2024年营收50.3亿元，同比增长36%，生成式AI业务收入占比达60%", "meta": {"source": "sensetime_2024"}},
            {"text": "日日新大模型API日均调用量突破2000万次，同比增长400%，企业客户数达5800家", "meta": {"source": "sensetime_2024"}},
            {"text": "训练集群规模达4万卡A100，算力利用率提升至85%，推理成本降至0.5元/百万token", "meta": {"source": "sensetime_2024"}},
            {"text": "英伟达发布Blackwell B200 GPU，单卡AI训练性能较H100提升4倍", "meta": {"source": "nvidia_2025"}},
            {"text": "智谱AI完成B+轮融资，估值超200亿元，GLM-5系列模型Q2发布", "meta": {"source": "zhipu_2025"}},
            {"text": "微软Azure部署10万张B200用于训练GPT-5，OpenAI表示推理成本将显著降低", "meta": {"source": "microsoft_2025"}},
            {"text": "谷歌宣布TPU v6将于Q4量产，直接对标Blackwell架构", "meta": {"source": "google_2025"}},
        ]

    # ---- 主入口 ----

    def dispatch(self, args) -> None:
        """根据 args.command 分发到对应处理器"""
        cmd = getattr(args, "command", None)
        method = getattr(self, f"_cmd_{cmd}", None) if cmd else None
        if method:
            method(args)
        else:
            print(f"[错误] 未知命令: {cmd}")

    # ===================== pipeline =====================

    def _cmd_pipeline(self, args) -> None:
        if not self._has_key:
            print("[错误] Pipeline 需要 DASHSCOPE_API_KEY"); return

        from financial_rag.templates import (
            QUICK_QA_TEMPLATE, FINANCIAL_REPORT_TEMPLATE,
            NEWS_BRIEF_TEMPLATE, DEEP_ANALYSIS_TEMPLATE,
        )

        print("=" * 60)
        print("Financial RAG — 统一 Pipeline")
        print("=" * 60)
        print(f"查询: {args.query}")
        print(f"模板: {args.template}")
        print(f"模式: {'详细' if getattr(args, 'verbose', False) else '静默'}")

        tmpl_map = {
            "quick": QUICK_QA_TEMPLATE, "fin": FINANCIAL_REPORT_TEMPLATE,
            "news": NEWS_BRIEF_TEMPLATE, "deep": DEEP_ANALYSIS_TEMPLATE,
        }
        template = tmpl_map.get(getattr(args, "template", "quick"), QUICK_QA_TEMPLATE)

        self._scheduler.config.verbose = getattr(args, "verbose", False)

        print(f"\nPipeline: 获取 → 索引(RAG) → 加工(Agent) → 输出(Slot) → 进化(Score)")
        print("=" * 60)

        result = self._scheduler.run(
            query=args.query, template=template,
            max_fetch_news=getattr(args, "max_fetch", None) or 10,
            max_retrieve=getattr(args, "max_retrieve", None) or 5,
        )

        print("\n" + "=" * 60)
        print("最终输出")
        print("=" * 60)
        print(result.final_output)

        print(f"\n[汇总] 获取={result.fetch_elapsed_ms:.0f}ms | "
              f"索引={result.index_elapsed_ms:.0f}ms | "
              f"加工={result.process_elapsed_ms:.0f}ms | "
              f"输出={result.output_elapsed_ms:.0f}ms | "
              f"总计={result.total_elapsed_ms:.0f}ms")

        if result.errors:
            print(f"[错误] {len(result.errors)} 个: {result.errors}")

        if result.scorecard:
            result.scorecard.print_summary("全链路打分卡:")

        output_dir = getattr(args, "output", None)
        if output_dir:
            self._save_output(result, args, output_dir)

    # ===================== query =====================

    def _cmd_query(self, args) -> None:
        from financial_rag.templates import (
            QUICK_QA_TEMPLATE, FINANCIAL_REPORT_TEMPLATE,
            NEWS_BRIEF_TEMPLATE, DEEP_ANALYSIS_TEMPLATE,
        )
        from financial_rag.core.scorer import PipelineScoreCard
        from financial_rag.core.reflector import HallucinationGuard

        print("=" * 60)
        print("Financial RAG — 财报/经济新闻智能分析 (阿里百炼)")
        print("=" * 60)

        if getattr(args, "interactive", False):
            print("\n交互模式，输入 'q' 退出，输入 'score' 查看上次打分")
            print("模板: quick=快答, fin=财报, news=新闻, deep=深度分析 (默认 quick)\n")
            last_card = None
            current_template = QUICK_QA_TEMPLATE
            tmpl_map = {
                "quick": QUICK_QA_TEMPLATE, "fin": FINANCIAL_REPORT_TEMPLATE,
                "news": NEWS_BRIEF_TEMPLATE, "deep": DEEP_ANALYSIS_TEMPLATE,
            }

            while True:
                try:
                    q = input("输入问题: ").strip()
                    if q.lower() in ('q', 'quit', 'exit'): break
                    if not q: continue
                    if q.lower() == 'score':
                        if last_card: last_card.print_summary("上次全链路打分卡:")
                        continue
                    if q.lower() in tmpl_map:
                        current_template = tmpl_map[q.lower()]
                        print(f"[模板] 已切换到: {current_template.description}")
                        continue

                    try:
                        card = PipelineScoreCard(query=q)
                        last_card = card

                        if self._has_key:
                            r = self._retriever
                            r.clear()
                            r.index(self._sample_docs)
                            results, ret_card = r.search_with_scores(q, top_k=3)
                            card.stages.extend(ret_card.stages)

                            filler = self._filler
                            context_docs = [r.get("text", "") for r in results[:3]]
                            t_fill = time.time()
                            fill_stats = filler.fill(current_template, query=q, context_docs=context_docs)
                            final_output = filler.render(current_template, fill_stats)
                            fill_elapsed = (time.time() - t_fill) * 1000

                            card.record_llm(
                                score=fill_stats.filled_slots / max(fill_stats.total_slots, 1),
                                token_count=fill_stats.total_tokens,
                                model=self._cfg.llm.model, elapsed_ms=fill_elapsed,
                            )

                            guard = HallucinationGuard()
                            check = guard.check(final_output, results)
                            card.record_hallucination(
                                overall_score=check['overall_score'],
                                layer_scores={k: v.get("score", 0) for k, v in check.get('checks', {}).items()},
                                risk=check['risk'],
                            )

                            print(f"\n[模板] {current_template.description}")
                            print(f"[结果]")
                            print(final_output[:400])
                            if len(final_output) > 400: print("...")
                            print(f"\n[性能] 总耗时 {fill_elapsed:.0f}ms, "
                                  f"首Token avg={fill_stats.avg_ttft_ms:.0f}ms | "
                                  f"槽位 {fill_stats.filled_slots}/{fill_stats.total_slots}"
                                  f" (并行增益 {fill_stats.parallel_gain:.0%})")

                            card.print_summary()
                        else:
                            print(f"\n[模拟] 关于 {q} 的分析...\n")
                            print("[提示] 设置 DASHSCOPE_API_KEY 启用真实 LLM + 槽位填充\n")
                    except Exception as e:
                        print(f"\n[错误] API 调用失败: {e}")
                        print("[提示] 请检查 DASHSCOPE_API_KEY 是否正确\n")
                except KeyboardInterrupt:
                    print("\n退出"); break

        elif getattr(args, "question", None):
            print(f"\n问题: {args.question}")
            print("分析中 (槽位填充)...\n")
            try:
                if self._has_key:
                    card = PipelineScoreCard(query=args.question)
                    r = self._retriever; r.clear(); r.index(self._sample_docs[:3])
                    results, ret_card = r.search_with_scores(args.question, top_k=3)
                    card.stages.extend(ret_card.stages)
                    filler = self._filler
                    fill_stats = filler.fill(QUICK_QA_TEMPLATE, query=args.question,
                                             context_docs=[r.get("text", "") for r in results[:3]])
                    final_output = filler.render(QUICK_QA_TEMPLATE, fill_stats)
                    print(final_output)
                    print(f"\n[性能] 槽位 {fill_stats.filled_slots}/{fill_stats.total_slots}, "
                          f"TTFT avg={fill_stats.avg_ttft_ms:.0f}ms")
            except Exception as e:
                print(f"\n[错误] API 调用失败: {e}")

    # ===================== build =====================

    def _cmd_build(self, args) -> None:
        from financial_rag.config import config as _cfg

        print("=" * 60)
        print("构建金融知识库 (阿里百炼 Embedding)")
        print("=" * 60)
        dir_path = getattr(args, "dir", None) or _cfg.data_dir

        if not os.path.isdir(dir_path):
            print(f"目录不存在: {dir_path}"); return

        r = self._retriever
        print(f"来源: {dir_path}")
        print(f"Embedding: {'text-embedding-v3' if self._has_key else '无 (纯本地)'}")
        print(f"Rerank:     {'qwen3-rerank' if self._has_key else '无 (纯本地)'}\n")

        r.index(self._sample_docs)
        print(f"已索引 {len(self._sample_docs)} 篇文档\n")

        for q in ["茅台营收多少", "汇率走势如何"]:
            results, card = r.search_with_scores(q, top_k=3)
            print(f"Q: {q}")
            for item in results:
                print(f"  [{item.get('retriever', '?')}] score={item.get('score', 0):.4f} | {item['text'][:60]}")
            card.print_summary()
            print()

    # ===================== analyze =====================

    def _cmd_analyze(self, args) -> None:
        from financial_rag.core.scorer import PipelineScoreCard

        print("=" * 60)
        print("Multi-Agent 财报分析 (阿里百炼)")
        print("=" * 60)

        if not getattr(args, "file", None):
            print("请指定财报文件"); return

        orch = self._orchestrator
        if getattr(args, "parallel", False):
            orch.config.execution_mode = ExecutionMode.PARALLEL
            print("模式: 并行")
        print(f"Pipeline: {' -> '.join(orch.pipeline)}")
        print(f"文件: {args.file}")
        if self._has_key:
            print(f"LLM: {self._cfg.llm.model} | Embedding: {self._cfg.llm.embedding_model} | Rerank: {self._cfg.llm.rerank_model}")
        print("=" * 60)

        card = PipelineScoreCard(query=os.path.basename(args.file))
        exec_result = orch.execute(args.file)
        print(f"\n完成: {exec_result.success}, 耗时: {exec_result.execution_time:.1f}s")

        for i, r in enumerate(exec_result.agent_results):
            icon = "OK" if r.success else "FAIL"
            agent_score = 0.0
            if r.success:
                if r.agent_name == "IngestionAgent":
                    meta = r.context_updates.get("metadata", {})
                    agent_score = meta.get("metadata_score", 0.5)
                    card.record_metadata(agent_score,
                        meta.get("metadata_fields_found", 0),
                        meta.get("metadata_fields_expected", 7),
                        elapsed_ms=r.execution_time * 1000)
                elif r.agent_name == "AnalysisAgent":
                    data = r.data or {}
                    agent_score = data.get("_scores", {}).get("extraction", 0.5)
                    card.record_keyword_extract(agent_score,
                        keyword_count=len(data.get("metrics", {})),
                        elapsed_ms=r.execution_time * 1000)
                else:
                    agent_score = 0.5 + (0.3 if r.success else 0)
            print(f"  [{icon}] {r.agent_name}: {r.message} (评分: {agent_score:.2f})")

        card.print_summary("Multi-Agent 全链路打分卡:")

    # ===================== score =====================

    def _cmd_score(self, args) -> None:
        from financial_rag.retrievers import HybridRetriever, jieba_tokenizer

        print("=" * 60)
        print("检索全链路打分测试")
        print("=" * 60)
        print(f"查询: {args.query}")

        if getattr(args, "local", False):
            tokenizer = None
            try:
                tokenizer = jieba_tokenizer()
                print("分词器: Jieba (已加载金融词典)")
            except ImportError:
                print("分词器: 正则 (pip install jieba 启用中文分词)")
            r = HybridRetriever(tokenizer=tokenizer)
        else:
            r = self._retriever

        r.index(self._sample_docs)
        results, card = r.search_with_scores(args.query, top_k=getattr(args, "top_k", 5))

        print(f"\n检索结果 ({len(results)} 条):")
        for item in results:
            print(f"  [{item['retriever']}] relev={item.get('relevance_level', '?')} "
                  f"score={item.get('score', 0):.4f} | {item['text'][:70]}")

        card.print_summary("检索全链路打分卡:")

        json_path = getattr(args, "json", None)
        if json_path:
            import json as _json
            with open(json_path, 'w', encoding='utf-8') as f:
                _json.dump(card.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n评分详情已导出: {json_path}")

    # ===================== slot =====================

    def _cmd_slot(self, args) -> None:
        if not self._has_key:
            print("[错误] 槽位填充需要 DASHSCOPE_API_KEY"); return

        from financial_rag.templates import get_template, ALL_TEMPLATES
        from financial_rag.core.scorer import PipelineScoreCard

        print("=" * 60)
        print("槽位填充测试 — 首 Token 延迟对比")
        print("=" * 60)

        template = get_template(args.template)
        if not template:
            print(f"未知模板: {args.template}")
            print(f"可选: {', '.join(ALL_TEMPLATES.keys())}"); return

        print(f"模板: {template.name} — {template.description}")
        print(f"槽位: {len(template.slots)} 个, {len(template.phases)} 个阶段")
        print(f"查询: {args.query}")
        print("=" * 60)

        r = self._retriever; r.clear(); r.index(self._sample_docs)
        results, ret_card = r.search_with_scores(args.query, top_k=getattr(args, "top_k", 5))
        context_docs = [r.get("text", "") for r in results[:getattr(args, "top_k", 5)]]

        llm = self._llm
        card = PipelineScoreCard(query=args.query)

        free_tokens = 0
        if not getattr(args, "no_freeform", False):
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

        print(f"\n[实验组] 槽位填充:")
        filler = self._filler
        t_fill = time.time()
        fill_stats = filler.fill(template, query=args.query, context_docs=context_docs)
        final_output = filler.render(template, fill_stats)
        fill_elapsed = (time.time() - t_fill) * 1000

        print(f"  总耗时: {fill_elapsed:.0f}ms")
        print(f"  总Tokens: {fill_stats.total_tokens}")
        print(f"  槽位: {fill_stats.filled_slots}/{fill_stats.total_slots} 个")
        print(f"  首Token: avg={fill_stats.avg_ttft_ms:.0f}ms, peak={fill_stats.peak_ttft_ms:.0f}ms")
        print(f"  并行增益: {fill_stats.parallel_gain:.0%}")

        if not getattr(args, "no_freeform", False):
            time_diff = free_elapsed - fill_elapsed
            direction = "更快" if time_diff > 0 else "更慢"
            print(f"\n  [对比] 槽位填充 vs 自由生成:")
            print(f"    总耗时: {fill_elapsed:.0f}ms vs {free_elapsed:.0f}ms (槽位 {direction} {abs(time_diff):.0f}ms)")
            print(f"    Tokens:  {fill_stats.total_tokens} vs {free_tokens}")

        print(f"\n[槽位详情]")
        for key, sr in fill_stats.slot_results.items():
            status = "OK" if sr.filled else "FAIL"
            val_preview = sr.value[:50] + "..." if len(sr.value) > 50 else sr.value
            print(f"  [{status}] {sr.label:<8s} | TTFT={sr.ttft_ms:5.0f}ms | {val_preview}")

        print(f"\n[渲染输出]")
        print(final_output[:400])
        if len(final_output) > 400: print("...")

        card.print_summary("全链路打分卡 (含槽位评分):")

    # ===================== toolcall =====================

    def _cmd_toolcall(self, args) -> None:
        from financial_rag.tools import create_tool_session
        from financial_rag.core.scorer import PipelineScoreCard

        if not self._has_key and not getattr(args, "list_tools", False):
            print("[错误] Function Calling 需要 DASHSCOPE_API_KEY"); return

        print("=" * 60)
        print("Function Calling — 能力注册中心测试")
        print("=" * 60)

        r = self._retriever
        try: r.index(self._sample_docs)
        except Exception as e:
            if self._has_key: print(f"[WARN] 检索器索引失败: {e}")

        print(self._registry); print()

        if getattr(args, "list_tools", False):
            print(f"能力清单 ({len(self._registry)} 个):")
            for f in self._registry.functions.values():
                required = ", ".join(f.parameters.get("required", []))
                print(f"  [{f.category}] {f.name}")
                print(f"    {f.description[:80]}...")
                print(f"    参数: {required}")
            return

        card = PipelineScoreCard(query=args.query)

        print(f"问题: {args.query}")
        print(f"模式: {'多轮' if getattr(args, 'multi_turn', False) else '单轮'} | "
              f"tool_choice: {getattr(args, 'tool_choice', 'auto')} | "
              f"verbose: {getattr(args, 'verbose', False)}")

        system = ("你是专业金融分析师。当需要具体数据时，必须调用提供的函数获取。"
                  "不要捏造任何具体数字。如果函数返回了数据，基于数据给出准确分析。")

        print("\n" + "=" * 60)
        print("执行 Function Calling 会话...")
        print("=" * 60)

        session = create_tool_session(
            llm=self._llm, retriever=r, registry=self._registry,
            system_prompt=system,
            max_rounds=getattr(args, "max_rounds", 5),
            verbose=getattr(args, "verbose", False),
        )

        t_start = time.time()
        stats = session.run(args.query, scorecard=card)
        t_elapsed = (time.time() - t_start) * 1000

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

        card.print_summary("Function Calling 全链路打分卡:")

    # ===================== news =====================

    def _cmd_news(self, args) -> None:
        """新闻抓取 — 纯路由：传参给 run_news_pipeline，只负责展示结果"""
        from financial_rag.tools.news_tools import run_news_pipeline

        query = args.query
        output_dir = getattr(args, "output", None) or self._cfg.output_dir
        do_summarize = getattr(args, "summarize", False)

        print("=" * 60)
        print(f"新闻抓取: {query}")
        print("=" * 60)

        data = run_news_pipeline(
            llm=self._llm if self._has_key else None,
            query=query,
            output_dir=output_dir,
            filename=getattr(args, "name", None) or "",
            summarize=do_summarize,
        )

        print(f"关键词: {data['main_keyword']}")
        print(f"获取到 {data['total_found']} 条新闻")
        print(f"[OK] 已保存到: {data['filepath']}"
              + (" + AI摘要" if data.get("has_summary") else ""))

    # ===================== kline =====================

    def _cmd_kline(self, args) -> None:
        """ETF K线 — 纯路由：传参给 run_kline_pipeline，只负责展示结果"""
        from financial_rag.tools.kline_tools import run_kline_pipeline

        query = args.query
        output_dir = getattr(args, "output", None) or self._cfg.output_dir
        do_summarize = getattr(args, "summarize", False)

        print("=" * 60)
        print(f"ETF K线分析: {query}")
        print("=" * 60)

        data = run_kline_pipeline(
            llm=self._llm if self._has_key else None,
            query=query,
            days=getattr(args, "days", 30),
            etf_code=getattr(args, "code", None) or "",
            output_dir=output_dir,
            filename=getattr(args, "name", None) or "",
            summarize=do_summarize,
        )

        if "error" in data:
            print(f"[!] {data['error']}"); return

        stats = data.get("stats", {})
        print(f"ETF: {data['etf_code']} {data['etf_name']}")
        print(f"最新价: {data.get('latest_price', '-')}, 涨跌: {data.get('change_pct', '-')}%")
        print(f"回溯: {data['lookback_days']}天, 数据点: {data['data_points']}")
        print(f"统计: 收盘={stats.get('latest_close')} | "
              f"区间涨跌={stats.get('period_change_pct')}% | "
              f"最高={stats.get('period_high')} | 最低={stats.get('period_low')} | "
              f"MA5={stats.get('ma5')} | MA10={stats.get('ma10')}")
        print(f"[OK] 已保存到: {data['filepath']}"
              + (" + AI分析" if data.get("has_analysis") else ""))

        alts = data.get("alternatives", [])
        if alts:
            print(f"其他相关 ETF ({len(alts)} 只):")
            for e in alts:
                print(f"  {e['code']}  {e['name']}")

    # ===================== 辅助 =====================

    def _save_output(self, result, args, output_dir: str) -> None:
        """保存 Pipeline 结果到文件"""
        from datetime import datetime as _dt
        os.makedirs(output_dir, exist_ok=True)
        filename = getattr(args, "name", None) or f"{_dt.now().strftime('%Y-%m-%d_%H%M%S')}_分析.md"
        filepath = os.path.join(output_dir, filename)
        lines = [
            f"# 分析报告", "",
            f"> 查询: {result.query}",
            f"> 时间: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 模型: {self._cfg.llm.model}", "",
            "---", "",
            result.final_output,
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n[OK] 报告已保存: {filepath}")
