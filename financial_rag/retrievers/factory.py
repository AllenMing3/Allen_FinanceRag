"""
检索器装配工厂 — 系统检索能力的唯一组装入口

打开这个文件，你就能看到检索系统由哪些零件组成、怎么配置。
改配置 → 改这里
看有什么 → 看这里
某个能力没生效 → 来这里检查是不是没接

组件清单:
- TextChunker:    文档切分（<500字不切，长文按段落边界切）
- QueryParser:    查询解析（正则+词典，抽取关键词/日期/同义词扩展）
- BM25Engine:     关键词检索
- VectorEngine:   语义检索（DashScope Embedding + ChromaDB）
- Reranker:       精排（qwen3-rerank）
- EmbeddingCache: 向量缓存（避免重复调 API）
- 质量门控:       rerank_score < threshold 时拦截低质量结果
"""
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_hybrid_retriever() -> Any:
    """创建混合检索引擎（BM25 + Vector + Rerank + Chunker + 质量门控）

    这是检索系统的唯一装配入口。
    所有检索相关的零件在这里组装，配置在这里调整。
    """
    from financial_rag.llm import get_embedding, get_reranker
    from financial_rag.retrievers.hybrid_engine import HybridRetriever, jieba_tokenizer
    from financial_rag.retrievers.chunker import TextChunker
    from financial_rag.retrievers.query_parser import QueryParser
    from financial_rag.config import config as _cfg

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    # ---- 分词器 ----
    tokenizer = None
    try:
        tokenizer = jieba_tokenizer()
    except ImportError:
        pass

    # ---- 切分器: <500字不切，长文按段落边界切 ----
    chunker = TextChunker(
        chunk_size=1500,
        chunk_overlap=100,
        min_chunk_size=80,
        skip_threshold=500,
    )

    # ---- 查询解析器 ----
    parser = QueryParser()

    # ---- Chroma 持久化目录 ----
    chroma_dir = os.path.join(_cfg.kb_dir, "chroma")
    os.makedirs(chroma_dir, exist_ok=True)

    # ---- BM25 FTS5 数据库 ----
    bm25_db_path = os.path.join(_cfg.kb_dir, "bm25_index.db")

    # ---- 组装 ----
    if not api_key:
        logger.warning("未设置 DASHSCOPE_API_KEY，回退给纯本地检索（BM25 + Jaccard）")
        return HybridRetriever(
            tokenizer=tokenizer,
            chunker=chunker,
            parser=parser,
            chroma_persist_dir=chroma_dir,
            bm25_db_path=bm25_db_path,
        )

    return HybridRetriever(
        embedder=get_embedding(api_key=api_key),
        reranker=get_reranker(api_key=api_key),
        tokenizer=tokenizer,
        chunker=chunker,
        parser=parser,
        chroma_persist_dir=chroma_dir,
        bm25_db_path=bm25_db_path,
    )
