"""
RAGAS 摸底评估脚本 — 对 FinRAG 知识库 RAG 链路做端到端质量评估

使用方式:
    # 1. 安装依赖
    pip install ragas datasets openai

    # 2. 运行评估（需要 DASHSCOPE_API_KEY）
    python experiments/ragas_eval.py

    # 3. 可选：指定自定义测试集
    python experiments/ragas_eval.py --testset my_queries.json

评估维度 (RAGAS 四大核心指标):
    - faithfulness:       回答是否基于检索到的 context（对标 L1-L2）
    - answer_relevancy:   回答是否切题
    - context_precision:  相关文档是否排在前面（验证 RRF+Rerank）
    - context_recall:     是否召回了所有相关文档（验证 chunker+top_k）

输出:
    - 终端打印各指标得分
    - 保存详细结果到 experiments/ragas_results.json
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict

# 确保项目根目录在 path 中
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ragas_eval")


# ===================== 默认测试集 =====================
# 覆盖 KB 中常见主题：AI 行业动态、公司新闻、技术概念、市场分析
DEFAULT_TEST_QUERIES = [
    "商汤科技2025年上半年营收是多少",
    "商汤的AI大模型业务收入占比多少",
    "商汤科技金融AI领域有哪些合作",
    "商汤的日日新大模型迭代到第几版了",
    "商汤科技什么时候能盈利",
    "商汤算力中心规模有多大",
]


# ===================== 系统初始化 =====================

def init_system():
    """初始化 FinRAG 检索器 + SlotFiller（复用 app_state 逻辑）"""
    from financial_rag.config import config as cfg
    from financial_rag.llm import get_llm
    from financial_rag.core.factory import create_hybrid_retriever
    from financial_rag.slot_filler import create_slot_filler
    from financial_rag.services.persistence import load_kb, INDEX_PATH

    if not cfg.llm.api_key:
        logger.error("需要 DASHSCOPE_API_KEY，请在 .env 中配置")
        sys.exit(1)

    llm = get_llm(api_key=cfg.llm.api_key, model=cfg.llm.model)
    retriever = create_hybrid_retriever()
    filler = create_slot_filler(llm=llm, verbose=False)

    # 加载 KB 并构建索引
    kb_docs = load_kb()
    if not kb_docs:
        logger.error("知识库为空，请先通过 Web UI 或 CLI 导入数据")
        sys.exit(1)

    logger.info(f"加载 KB: {len(kb_docs)} 篇文档")

    # 尝试加载持久化索引
    if os.path.exists(INDEX_PATH):
        try:
            retriever.load_index(INDEX_PATH)
            if len(retriever.documents) == len(kb_docs):
                logger.info(f"从磁盘加载索引 ({len(retriever.documents)} docs)")
            else:
                raise ValueError("doc count mismatch")
        except Exception:
            retriever.clear()
            retriever.index(kb_docs, precompute_embeddings=True)
            retriever.save_index(INDEX_PATH)
            logger.info("重建索引完成")
    else:
        retriever.index(kb_docs, precompute_embeddings=True)
        retriever.save_index(INDEX_PATH)
        logger.info("首次构建索引完成")

    return retriever, filler, kb_docs


# ===================== RAG 链路执行 =====================

def run_rag_query(retriever, filler, query: str, top_k: int = 5) -> Dict:
    """对单条 query 执行完整 RAG 链路，返回 RAGAS 所需字段"""
    from financial_rag.templates import QUICK_QA_TEMPLATE

    # 1. 检索
    results, scorecard = retriever.search_with_scores(query, top_k=top_k)
    contexts = [item.get("text", "") for item in results[:top_k]]

    # 2. 生成回答
    fill_stats = filler.fill(QUICK_QA_TEMPLATE, query=query, context_docs=contexts)
    answer = filler.render(QUICK_QA_TEMPLATE, fill_stats)

    # 3. 内部评分（对比参考）
    internal_score = scorecard.overall_score() if scorecard else 0.0

    return {
        "question": query,
        "contexts": contexts,
        "answer": answer,
        "internal_score": round(internal_score, 3),
        "retrieval_count": len(results),
        "elapsed_ms": round(scorecard.total_elapsed(), 1) if scorecard else 0,
    }


# ===================== RAGAS 评估 =====================

def setup_ragas_llm():
    """配置 RAGAS 使用 DashScope OpenAI 兼容接口作为 Judge LLM"""
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    # DashScope OpenAI 兼容端点
    llm = ChatOpenAI(
        model="qwen-plus",
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        max_tokens=2048,
    )
    return LangchainLLMWrapper(llm)


def setup_ragas_embeddings():
    """配置 RAGAS 使用 DashScope Embedding"""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas_evaluation(samples: List[Dict], with_ground_truth: bool = False) -> Dict:
    """运行 RAGAS 评估"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # 构建 RAGAS Dataset
    data = {
        "question": [s["question"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "answer": [s["answer"] for s in samples],
    }
    if with_ground_truth:
        data["ground_truth"] = [s.get("ground_truth", "") for s in samples]

    dataset = Dataset.from_dict(data)

    # 选择指标（context_precision / context_recall 都需要 ground_truth）
    metrics = [faithfulness, answer_relevancy]
    if with_ground_truth:
        metrics.extend([context_precision, context_recall])

    # 配置 Judge LLM + Embeddings
    judge_llm = setup_ragas_llm()
    judge_embeddings = setup_ragas_embeddings()

    logger.info(f"开始 RAGAS 评估: {len(samples)} 条样本, 指标: {[m.name for m in metrics]}")
    t0 = time.time()

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    elapsed = time.time() - t0
    logger.info(f"RAGAS 评估完成, 耗时 {elapsed:.1f}s")

    return result


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="RAGAS 摸底评估")
    parser.add_argument("--testset", type=str, help="自定义测试集 JSON 文件路径")
    parser.add_argument("--top-k", type=int, default=5, help="检索 top_k (默认 5)")
    parser.add_argument("--skip-ragas", action="store_true", help="只跑 RAG 链路，不跑 RAGAS 评估")
    args = parser.parse_args()

    # 加载测试集
    if args.testset:
        with open(args.testset, "r", encoding="utf-8") as f:
            queries = json.load(f)
        if isinstance(queries[0], dict):
            # 带 ground_truth 的格式: [{"question": "...", "ground_truth": "..."}]
            test_queries = [q["question"] for q in queries]
            ground_truths = {q["question"]: q.get("ground_truth", "") for q in queries}
        else:
            test_queries = queries
            ground_truths = {}
    else:
        test_queries = DEFAULT_TEST_QUERIES
        ground_truths = {}

    logger.info(f"测试集: {len(test_queries)} 条查询")

    # 初始化系统
    logger.info("初始化 FinRAG 系统...")
    retriever, filler, kb_docs = init_system()

    # 逐条执行 RAG 链路
    samples = []
    for i, query in enumerate(test_queries):
        logger.info(f"[{i+1}/{len(test_queries)}] {query}")
        try:
            sample = run_rag_query(retriever, filler, query, top_k=args.top_k)
            if query in ground_truths:
                sample["ground_truth"] = ground_truths[query]
            samples.append(sample)
            logger.info(
                f"  → 检索 {sample['retrieval_count']} 条, "
                f"内部评分 {sample['internal_score']:.2f}, "
                f"耗时 {sample['elapsed_ms']:.0f}ms"
            )
        except Exception as e:
            logger.error(f"  → 失败: {e}")
            continue

    if not samples:
        logger.error("所有查询都失败了，无法评估")
        return

    # 打印 RAG 链路摘要
    print("\n" + "=" * 70)
    print("  RAG 链路执行摘要")
    print("=" * 70)
    for s in samples:
        print(f"  Q: {s['question'][:40]}")
        print(f"     contexts: {len(s['contexts'])} 条, "
              f"answer: {len(s['answer'])} 字, "
              f"内部评分: {s['internal_score']:.2f}")
    print()

    if args.skip_ragas:
        logger.info("--skip-ragas 模式，跳过 RAGAS 评估")
    else:
        # 运行 RAGAS
        try:
            result = run_ragas_evaluation(
                samples,
                with_ground_truth=bool(ground_truths),
            )

            # 打印结果
            print("\n" + "=" * 70)
            print("  RAGAS 评估结果")
            print("=" * 70)
            scores = result.scores if hasattr(result, 'scores') else {}
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                print(df.to_string(index=False))
                print()

            # 汇总
            if hasattr(result, '__getitem__'):
                for metric_name in ["faithfulness", "answer_relevancy",
                                    "context_precision", "context_recall"]:
                    try:
                        score = result[metric_name]
                        print(f"  {metric_name:<22s}: {score:.3f}")
                    except (KeyError, TypeError):
                        pass

        except ImportError as e:
            logger.error(
                f"RAGAS 依赖未安装: {e}\n"
                "请运行: pip install ragas datasets langchain-openai"
            )
        except Exception as e:
            logger.error(f"RAGAS 评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存详细结果
    output_path = _project_root / "experiments" / "ragas_results.json"
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kb_doc_count": len(kb_docs),
        "top_k": args.top_k,
        "samples": samples,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
