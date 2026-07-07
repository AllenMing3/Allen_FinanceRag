"""
LightRAG 实验脚本 — 完全解耦，不影响现有系统

测试: qwen-plus 实体关系抽取 + 知识图谱查询
用法: python experiments/lightrag_experiment.py
"""
import asyncio
import os
import sys
import json
import logging

# LightRAG SDK
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, Tokenizer

# ============================================================
# 配置
# ============================================================
# 优先从 .env 文件读取，避免系统环境变量中的旧 key
API_KEY = ""
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    for line in open(env_path, encoding='utf-8'):
        if line.startswith("DASHSCOPE_API_KEY="):
            API_KEY = line.split("=", 1)[1].strip()
            break

if not API_KEY:
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

if not API_KEY:
    print("ERROR: 未找到 DASHSCOPE_API_KEY")
    sys.exit(1)

# 强制设置 dashscope 全局 key，避免系统环境变量中的旧 key 干扰
import dashscope
dashscope.api_key = API_KEY
os.environ["DASHSCOPE_API_KEY"] = API_KEY

LLM_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v3"
WORKING_DIR = os.path.join(os.path.dirname(__file__), "lightrag_demo", "rag_storage")

# ============================================================
# 自定义 Tokenizer (避免 tiktoken 下载被墙)
# ============================================================
class _SimpleInner:
    """Inner tokenizer — 用字符索引作为 token，decode 用索引还原文本"""
    _last_text: str = ""

    def encode(self, content, **kwargs):
        self._last_text = content
        # 每个字符一个 token，索引就是字符位置
        return list(range(len(content)))

    def decode(self, tokens):
        if not tokens or not self._last_text:
            return ""
        # tokens 是字符索引列表，用 min/max 还原子串
        start = min(tokens)
        end = max(tokens) + 1
        return self._last_text[start:end]

class SimpleTokenizer(Tokenizer):
    """继承 LightRAG 的 Tokenizer 基类，避免 tiktoken 下载"""
    def __init__(self):
        super().__init__(model_name="custom", tokenizer=_SimpleInner())

# ============================================================
# LLM / Embedding 适配函数
# ============================================================
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """使用原生 dashscope SDK 调用 qwen-plus"""
    import dashscope
    from dashscope import Generation

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    # 移除 LightRAG 传入的不兼容参数
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("token_tracker", None)

    # 处理 response_format
    result_format = "message"
    if kwargs.pop("response_format", None):
        result_format = "json_object"

    resp = Generation.call(
        model=LLM_MODEL,
        messages=messages,
        api_key=API_KEY,
        result_format=result_format,
        max_tokens=kwargs.get("max_tokens", 4096),
        temperature=kwargs.get("temperature", 0.7),
    )

    if resp.status_code != 200:
        raise Exception(f"DashScope API error: {resp.code} - {resp.message}")

    return resp.output.choices[0].message.content

async def embedding_func(texts):
    """使用原生 dashscope SDK 调用 text-embedding-v3"""
    import dashscope
    from dashscope import TextEmbedding

    resp = TextEmbedding.call(
        model=EMBEDDING_MODEL,
        input=texts,
        api_key=API_KEY,
        dimension=1024,
    )

    if resp.status_code != 200:
        raise Exception(f"DashScope Embedding error: {resp.code} - {resp.message}")

    import numpy as np
    embeddings = [item["embedding"] for item in resp.output["embeddings"]]
    return np.array(embeddings, dtype=np.float32)

# ============================================================
# 测试数据 — 从现有 news_metadata.json 取几条真实新闻
# ============================================================
SAMPLE_TEXTS = [
    # 新闻 1: 商汤科技
    """商汤科技发布2024年全年业绩报告。报告显示，商汤科技2024年营收达到50.2亿元人民币，
    同比增长36%。其中，生成式AI业务营收同比增长超过200%，成为公司最大收入来源。
    商汤科技CEO徐立表示，日日新大模型体系已服务超过500家企业客户，覆盖金融、医疗、
    教育等多个行业。公司预计2025年将继续保持高速增长。""",

    # 新闻 2: 英伟达
    """英伟达在GTC 2025大会上发布了新一代Blackwell Ultra GPU架构。该架构采用台积电
    4NP工艺制造，单芯片集成2080亿个晶体管。英伟达CEO黄仁勋表示，Blackwell Ultra
    相比上一代Hopper，AI推理性能提升4倍，训练性能提升2.5倍。首批客户包括微软、
    谷歌和Meta，预计2025年下半年开始大规模出货。""",

    # 新闻 3: OpenAI
    """OpenAI宣布完成新一轮66亿美元融资，估值达到1570亿美元。此轮融资由软银领投，
    微软跟投。OpenAI CEO Sam Altman表示，资金将用于推进AGI研究和基础设施建设。
    同时，OpenAI正在与商汤科技探讨在亚洲市场的合作可能性。此外，OpenAI的GPT-5
    模型预计将在2025年第三季度发布，据称在推理能力上有重大突破。""",

    # 新闻 4: 华为
    """华为发布昇腾910C AI芯片，性能对标英伟达H100。昇腾910C采用7nm工艺，
    FP16算力达到640 TFLOPS。华为轮值董事长徐直军表示，昇腾生态已汇聚超过
    200万开发者，支持国内90%以上的大模型训练。面对美国芯片出口限制，
    华为正加速构建自主可控的AI算力基础设施。""",

    # 新闻 5: 政策
    """中国国务院发布《新一代人工智能发展规划2025年实施方案》，明确提出到2025年底，
    AI核心产业规模突破5000亿元。方案重点支持大模型研发、AI芯片自主化、数据要素
    市场化三大方向。工信部同步宣布设立1000亿元AI产业引导基金，重点扶持商汤科技、
    华为、百度等头部企业的技术研发。""",
]

# ============================================================
# 主流程
# ============================================================
async def main():
    os.makedirs(WORKING_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 60)
    print(f"  LightRAG 实验")
    print(f"  LLM: {LLM_MODEL} | Embedding: {EMBEDDING_MODEL}")
    print(f"  Storage: {WORKING_DIR}")
    print("=" * 60)

    # 1. 初始化 LightRAG
    print("\n[1/4] 初始化 LightRAG...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            func=embedding_func,
            max_token_size=8000,
        ),
        embedding_batch_num=5,
        entity_extract_max_gleaning=1,
        chunk_token_size=300,
        chunk_overlap_token_size=50,
        tokenizer=SimpleTokenizer(),
    )
    await rag.initialize_storages()

    # 2. 插入文档（触发实体-关系抽取）
    print(f"\n[2/4] 插入 {len(SAMPLE_TEXTS)} 段新闻文本...")
    print("    (实体-关系抽取中，需要调用 LLM，请耐心等待...)")
    await rag.ainsert(SAMPLE_TEXTS)
    print("    ✓ 插入完成")

    # 3. 查看知识图谱
    print("\n[3/4] 知识图谱概览:")
    try:
        labels = await rag.get_graph_labels()
        print(f"    实体/标签数量: {len(labels) if labels else 0}")
        if labels:
            for i, label in enumerate(labels[:15]):
                print(f"      [{i+1}] {label}")
            if len(labels) > 15:
                print(f"      ... 共 {len(labels)} 个")
    except Exception as e:
        print(f"    (获取标签失败: {e})")

    # 导出图谱数据
    try:
        export = await rag.aexport_data(output_path=os.path.join(WORKING_DIR, "export.json"))
        print(f"\n    导出数据统计:")
        for key, val in export.items():
            count = len(val) if isinstance(val, (list, dict)) else val
            print(f"      {key}: {count}")
    except Exception as e:
        print(f"    (导出失败: {e})")

    # 4. 查询测试
    queries = [
        ("商汤科技的营收增长了多少？", "local"),
        ("AI芯片领域有哪些主要玩家和竞争关系？", "global"),
        ("OpenAI和商汤科技有什么联系？", "hybrid"),
        ("中国AI政策对哪些公司有利好？", "mix"),
    ]

    print(f"\n[4/4] 查询测试 ({len(queries)} 个问题):")
    for q, mode in queries:
        print(f"\n  ❓ [{mode}] {q}")
        try:
            result = await rag.aquery(q, param=QueryParam(mode=mode))
            # 截取前500字
            text = result if isinstance(result, str) else str(result)
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"  💡 {text}")
        except Exception as e:
            print(f"  ❌ 查询失败: {e}")

    # 清理
    await rag.finalize_storages()
    print("\n" + "=" * 60)
    print("  实验完成！数据保存在 experiments/lightrag_demo/rag_storage/")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
