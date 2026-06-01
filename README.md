# Financial RAG — 财报/经济新闻智能分析系统

基于阿里百炼 DashScope 的智能金融分析系统，支持财报解析、经济新闻检索和 Multi-Agent 协同分析。

## 功能特性

- **全链路阿里百炼**：LLM (Qwen) + Embedding (text-embedding-v3) + Rerank (gte-rerank)
- **三大核心架构**：Coordinate（多 Agent 协调）、Indexer（混合检索）、Reflection（反思防幻觉）
- **混合检索**：BM25 关键词 + 向量语义检索 → RRF 融合 → gte-rerank 精排
- **Multi-Agent 流水线**：Ingestion → Extraction → Analysis → Forecast → Report
- **六层防幻觉**：来源验证 → 一致性 → 事实性 → 完整性 → 引用 → 综合评分
- **多模式降级**：无 API Key 自动回退纯本地检索模式

## 项目结构

```
llamaindex/
├── main.py                       # 项目入口
├── README.md
├── requirements.txt              # 依赖列表
├── .env.example                  # 环境变量示例
├── data/
│   ├── financial/                # 财报/经济数据
│   └── knowledge_base/           # 知识库文档
└── financial_rag/                # 核心包
    ├── main.py                   # CLI 主入口
    ├── config.py                 # 配置（阿里百炼）
    ├── agents/                   # Multi-Agent 子模块
    ├── core/                     # 三大架构核心
    │   ├── coordinator.py        # Coordinate 多 Agent 协调
    │   ├── indexer.py            # Indexer 混合检索流水线
    │   └── reflector.py          # Reflection 反思防幻觉
    ├── llm/
    │   └── dashscope_client.py   # 阿里百炼客户端封装
    ├── retrievers/               # 混合检索器（BM25 + Embedding + Rerank）
    └── ingestion/                # 文档导入处理
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填写阿里百炼 API Key：

```bash
cp .env.example .env
# 编辑 .env，填入你的 Key：
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

> API Key 获取地址：https://bailian.console.aliyun.com/

> 没有 Key 也可以运行，系统会自动回退到纯本地模式（BM25 + Jaccard）。

### 3. 演示模式

```bash
python -m financial_rag.main demo
```

### 4. 交互式查询

```bash
python -m financial_rag.main query -i
```

### 5. 单次查询

```bash
python -m financial_rag.main query -q "茅台2024年营收增长了多少？"
```

### 6. 构建知识库

```bash
python -m financial_rag.main build --dir ./data/financial
```

### 7. Multi-Agent 财报分析

```bash
# 串行分析
python -m financial_rag.main analyze ./data/financial/maotai_2024.pdf

# 并行分析
python -m financial_rag.main analyze ./data/financial/maotai_2024.pdf --parallel
```

## 使用示例

### 基础查询

```python
from financial_rag.llm import get_llm
from financial_rag.config import config

llm = get_llm(api_key=config.llm.api_key, model="qwen-plus")
response = llm.chat("茅台2024年毛利率是多少？")
print(response.content)
```

### 混合检索

```python
from financial_rag.retrievers import HybridRetriever
from financial_rag.llm import get_embedding, get_reranker
from financial_rag.config import config

retriever = HybridRetriever(
    embedder=get_embedding(api_key=config.llm.api_key),
    reranker=get_reranker(api_key=config.llm.api_key),
)

docs = [
    {"text": "茅台2024年营收1738.52亿元，同比增长15.66%", "meta": {}},
    {"text": "茅台2024年净利润862.28亿元，同比增长15.38%", "meta": {}},
]
retriever.index(docs)

results = retriever.search("茅台盈利情况", top_k=3, use_rerank=True)
for r in results:
    print(f"[{r['retriever']}] score={r['score']:.4f} | {r['text']}")
```

### Multi-Agent 分析

```python
from financial_rag.main import create_orchestrator

orch = create_orchestrator()
result = orch.execute("./data/financial/report.pdf")

for r in result.agent_results:
    print(f"[{'OK' if r.success else 'FAIL'}] {r.agent_name}: {r.message}")
```

## 模型配置

| 组件 | 模型 | 说明 |
|------|------|------|
| LLM | qwen-plus / qwen-turbo / qwen-max | 阿里百炼 Qwen 系列 |
| Embedding | text-embedding-v3 | 1024 维向量 |
| Rerank | gte-rerank | 重排序精排 |

## 检索模式

系统支持三种模式自动切换：

| 模式 | 条件 | 链路 |
|------|------|------|
| 纯本地 | 无 API Key | BM25 + Jaccard → RRF |
| 带 Embedding | 仅配 API Key | BM25 + 向量检索 → RRF |
| 全链路 | API Key 可用 | BM25 + Embedding → RRF → gte-rerank |

## 命令行帮助

```bash
# 查看所有命令
python -m financial_rag.main --help

# 查看具体命令帮助
python -m financial_rag.main query --help
python -m financial_rag.main build --help
python -m financial_rag.main analyze --help
```

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key | 否（无 Key 自动降级本地模式） |

## 许可证

MIT License
