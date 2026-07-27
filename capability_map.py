"""
能力地图 — 一键查看"我想改X → 去哪个文件"

用法: python capability_map.py
"""

# 能力域 → (tool名列表, 实现文件, 说明)
CAPABILITY_MAP = {
    "知识库管理": {
        "tools": ["kb_load", "kb_save", "kb_add", "kb_remove", "kb_stats"],
        "入口": "financial_rag/tools/kb_tools.py",
        "实现": "financial_rag/services/persistence.py",
        "说明": "KB文档的增删改查、去重、版本管理。换存储格式只改 persistence.py",
    },
    "文档切分": {
        "tools": ["chunk_document", "chunk_documents_batch"],
        "入口": "financial_rag/tools/chunk_tools.py  ← 策略配置在这",
        "实现": "financial_rag/retrievers/chunker.py",
        "说明": "长文切分为chunks。策略: default(500字) / paragraph(1500字) / none(不切)",
    },
    "文本清洗 + 门控": {
        "tools": ["(未注册，P1待收编)"],
        "入口": "financial_rag/api/ingest_router.py → _preprocess_docs()",
        "实现": "financial_rag/retrievers/preprocessor.py",
        "说明": "HTML去除、样板行删除、相关性门控、最低字数。改门控标准去 ingest_router 第36行",
    },
    "Metadata 治理": {
        "tools": ["(内部基础设施，不注册为 tool)"],
        "入口": "financial_rag/retrievers/metadata.py  ← Schema定义 + 正则抽取器",
        "实现": "入库: api/ingest_router.py _preprocess_docs() 调用 extract_metadata()",
        "子模块": {
            "Schema定义": "retrievers/metadata.py → DocMetadata dataclass",
            "正则抽取": "retrievers/metadata.py → extract_metadata() (company/date/sector)",
            "Chroma白名单": "retrievers/metadata.py → CHROMA_META_WHITELIST",
            "过滤消费": "retrievers/filters.py + query_parser.py get_filters()",
        },
        "说明": "改metadata字段 → metadata.py; 改抽取规则 → metadata.py; 改过滤逻辑 → filters.py",
    },
    "检索（混合）": {
        "tools": ["search_financial_data"],
        "入口": "financial_rag/tools/core.py → _make_search_tool()",
        "实现": "financial_rag/retrievers/retriever.py  ← 调度层（BM25+向量+RRF+Rerank）",
        "子模块": {
            "BM25关键词": "retrievers/bm25_engine.py",
            "向量语义": "retrievers/vector_engine.py",
            "RRF融合": "retrievers/fusion.py  ← 权重在这 (bm25=0.3, vector=0.7)",
            "Rerank精排": "DashScope qwen3-rerank API",
            "Metadata过滤": "retrievers/filters.py",
            "查询解析": "retrievers/query_parser.py  ← 同义词/概念扩展在这",
            "Embedding缓存": "retrievers/embedding_cache.py",
        },
        "说明": "改检索行为 → retriever.py; 改权重 → fusion.py; 改过滤 → filters.py",
    },
    "知识图谱": {
        "tools": ["query_knowledge_graph", "get_graph_stats"],
        "入口": "financial_rag/tools/graph_tools.py",
        "实现": "financial_rag/retrievers/lightrag_adapter.py",
        "说明": "LightRAG图谱查询。仅PDF/图片内容走图谱",
    },
    "新闻/数据获取": {
        "tools": ["fetch_stock_news", "fetch_financial_news", "fetch_announcements",
                  "fetch_news_report", "fetch_date_events", "fetch_kline_context"],
        "入口": "financial_rag/tools/news_tools.py + kline_tools.py + event_impact_tools.py",
        "实现": "financial_rag/rss_fetcher.py (新闻) + tushare_client.py (K线)",
        "说明": "外部数据源。新闻=同花顺/新浪/东方财富，K线=Tushare",
    },
    "深度分析": {
        "tools": ["analyze_news_deep", "analyze_topic_deep"],
        "入口": "financial_rag/tools/analysis_tools.py",
        "实现": "financial_rag/services/analysis.py",
        "说明": "新闻解读 + 话题调研。Agent链: AnalysisAgent → ScoringAgent",
    },
    "信息抽取": {
        "tools": ["extract_financial_metrics", "extract_entities",
                  "extract_document_metadata", "detect_document_type", "generate_search_queries"],
        "入口": "financial_rag/tools/extraction_tools.py",
        "实现": "同上（LLM驱动）",
        "说明": "从文本中抽取指标/实体/元数据",
    },
    "K线技术分析": {
        "tools": ["analyze_kline", "generate_kline_analysis", "fetch_kline_report"],
        "入口": "financial_rag/tools/kline_tools.py",
        "实现": "同上 + tushare_client.py",
        "说明": "K线数据获取 + 技术指标计算 + LLM分析",
    },
    "事件影响评估": {
        "tools": ["fetch_date_events", "assess_event_impact"],
        "入口": "financial_rag/tools/event_impact_tools.py",
        "实现": "同上（LLM驱动）",
        "说明": "日期事件 → 利好/利空判断",
    },
    "报告生成": {
        "tools": ["synthesize_report"],
        "入口": "financial_rag/tools/report_tools.py",
        "实现": "同上（LLM驱动）",
        "说明": "将分析结果合成为Markdown报告",
    },
    "防幻觉校验": {
        "tools": ["check_hallucination"],
        "入口": "financial_rag/tools/scoring_tools.py",
        "实现": "financial_rag/guard/reflector.py  ← 6层校验逻辑在这",
        "子模块": {
            "L1-L4规则层": "guard/rule_layers.py",
            "L5 LLM Critique": "guard/llm_critique.py",
            "L6 LLM Assist": "guard/llm_assist.py",
        },
        "说明": "改校验规则 → rule_layers.py; 改LLM审计 → llm_critique.py",
    },
    "评分系统（冻结）": {
        "tools": ["evaluate_pipeline_quality", "generate_score_report"],
        "入口": "financial_rag/tools/scoring_tools.py",
        "实现": "financial_rag/core/scorer.py + ingestion_scorer.py",
        "说明": "⚠️ 当前冻结，不改动",
    },
    "文档解析（PDF/图片）": {
        "tools": ["parse_pdf_file", "describe_image_file"],
        "入口": "financial_rag/tools/document_parse_tools.py",
        "实现": "PyMuPDF(PDF) + qwen-vl-plus(图片)",
        "说明": "非文本文件 → 文本。PDF用PyMuPDF本地解析，图片走多模态API",
    },
    "LLM客户端": {
        "tools": ["(基础设施，不注册为tool)"],
        "入口": "financial_rag/llm/dashscope_client.py",
        "实现": "同上 + llm/model_router.py + llm/caller.py",
        "说明": "改模型/参数 → config.py; 改调用逻辑 → dashscope_client.py",
    },
    "Agent编排": {
        "tools": ["classify_query_intent", "select_agent_chain"],
        "入口": "financial_rag/agents/ 目录",
        "实现": {
            "调度Agent": "agents/coordinator_agent.py",
            "分析Agent": "agents/analysis_agent.py  ← 最重的一个",
            "评分Agent": "agents/scoring_agent.py",
            "摄取Agent": "agents/ingestion_agent.py",
            "编排器": "core/orchestrator.py",
            "工厂": "core/factory.py  ← Agent创建和接线",
        },
        "说明": "改Agent流程 → 对应agent文件; 改调度顺序 → orchestrator.py",
    },
    "前端": {
        "tools": ["(无)"],
        "入口": "financial_rag/static/",
        "实现": {
            "HTML": "static/index.html",
            "JS模块": "static/modules/ (app.js, query.js, analyze.js, overview.js...)",
            "CSS": "static/styles/ (base.css, query.css, analyze.css...)",
        },
        "说明": "改页面 → index.html; 改交互 → modules/对应.js; 改样式 → styles/对应.css",
    },
    "API接口": {
        "tools": ["(无)"],
        "入口": "financial_rag/api/",
        "实现": {
            "KB管理": "api/kb_router.py",
            "数据导入": "api/ingest_router.py  ← 入库门控也在这",
            "深度分析": "api/analysis_router.py",
            "智能查询": "api/query_router.py",
            "共享状态": "api/app_state.py  ← 启动初始化在这",
        },
        "说明": "改接口 → 对应router; 改启动逻辑 → app_state.py",
    },
    "领域字典": {
        "tools": ["(无)"],
        "入口": "financial_rag/retrievers/dictionaries.py + dictionary_registry.py",
        "数据": "data/dictionaries/*.json  ← 加词改这里，不用改代码",
        "说明": "同义词/股票映射/行业术语。加词 → 改JSON文件即可热扩展",
    },
}


def print_map():
    """打印能力地图"""
    print("=" * 70)
    print("  FinRAG 能力地图 — 我想改X → 去哪个文件")
    print("=" * 70)

    for domain, info in CAPABILITY_MAP.items():
        tools_str = ", ".join(info["tools"][:4])
        if len(info["tools"]) > 4:
            tools_str += f" ... (+{len(info['tools'])-4})"

        print(f"\n{'─' * 70}")
        print(f"  【{domain}】")
        print(f"  Tools: {tools_str}")

        if isinstance(info.get("入口"), str):
            print(f"  入口:  {info['入口']}")
        if isinstance(info.get("实现"), str):
            print(f"  实现:  {info['实现']}")
        elif isinstance(info.get("实现"), dict):
            for k, v in info["实现"].items():
                print(f"         {k}: {v}")

        if "子模块" in info:
            for k, v in info["子模块"].items():
                print(f"         {k}: {v}")

        print(f"  >> {info['说明']}")

    print(f"\n{'=' * 70}")
    print(f"  总计: {sum(len(v['tools']) for v in CAPABILITY_MAP.values())} 个能力")
    print(f"  运行: python capability_map.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    print_map()
