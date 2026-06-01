# Financial RAG — 财报/经济新闻智能分析系统

基于阿里百炼 DashScope 的智能金融分析系统，支持财报解析、经济新闻检索和 Multi-Agent 协同分析。

## 功能特性

- **全链路阿里百炼**：LLM (Qwen) + Embedding (text-embedding-v3) + Rerank (gte-rerank) + **Function Calling**
- **三大核心架构**：Coordinate（多 Agent 协调）、Indexer（混合检索）、Reflection（反思防幻觉）
- **能力注册中心**：集中管理所有 Agent 能力，LLM 通过 Function Calling 动态选择调用
- **混合检索**：BM25 关键词 + 向量语义检索 → RRF 融合 → gte-rerank 精排
- **Multi-Agent 流水线**：Ingestion → Extraction → Analysis → Forecast → Report
- **六层防幻觉**：来源验证 → 一致性 → 事实性 → 完整性 → 引用 → 综合评分
- **全链路打分系统**：每个环节独立评分，精确诊断薄弱环节（metadata解析、Jieba分词、BM25、Vector、RRF、Rerank、LLM、防幻觉）
- **模板 + 槽位填充**：用槽位填充替代长文自由生成，首 Token 延迟降低 60~80%，每槽位独立打分
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
    ├── core/                     # 三大架构核心 + 打分
    │   ├── coordinator.py        # Coordinate 多 Agent 协调
    │   ├── indexer.py            # Indexer 混合检索流水线
    │   ├── reflector.py          # Reflection 反思防幻觉
    │   └── scorer.py             # 全链路打分系统
    ├── llm/
    │   └── dashscope_client.py   # 阿里百炼客户端封装
    ├── templates.py               # 槽位模板定义 (4 种预置模板)
    ├── slot_filler.py             # 槽位填充引擎 (并行填充 + TTFT 测量)
    ├── tools.py                   # 能力注册中心 (Function Calling 能力管理)
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

# 仅检索打分测试（不调用 LLM）
python -m financial_rag.main score "茅台营收增长" -k 5
python -m financial_rag.main score "汇率走势" --json scores.json  # 导出 JSON
```

## 全链路打分系统

每个管道阶段独立评分 (0~1.0)，直接定位哪个环节表现不好：

```
========================================================================
  Pipeline 全链路打分卡 — 综合: 0.72  (一般, C)
========================================================================
  查询: 茅台营收增长

── 文本预处理 & 分词 [B] 均分 0.85 ──
  [GOOD] Jieba 分词       0.85 (    10ms)

── 混合检索 (BM25+Vector+RRF+Rerank) [B] 均分 0.80 ──
  [GOOD] BM25 关键词检索     0.90 (    15ms)
  [ OK ] RRF 融合排序       0.66 (     5ms)
  [GOOD] Rerank 精排      0.83 (    20ms)

── LLM 生成 & 防幻觉校验 [B] 均分 0.89 ──
  [GOOD] LLM 生成         0.90 (   500ms)
  [GOOD] 六层防幻觉校验        0.88 (     0ms)
========================================================================
```

### 评分维度

| 大类 | 阶段 | 评分依据 |
|------|------|----------|
| **摄取** | 元数据解析 | 字段覆盖率（source/company/date 等 7 个字段） |
| **预处理** | Jieba 分词 | token 数量、唯一性、平均词长 |
| **预处理** | 关键词抽取 | 关键词数量、覆盖率 |
| **检索** | BM25 检索 | 匹配率、结果数量、top 分数 |
| **检索** | 向量检索 | 余弦相似度、结果数量 |
| **检索** | RRF 融合 | 两个检索器共识度 |
| **检索** | Rerank 精排 | 高相关文档占比、rerank 分数 |
| **生成** | LLM 生成 | 输出长度、token 消耗 |
| **生成** | 防幻觉校验 | L1~L6 六层加权评分 |

### 诊断输出

分数低的阶段会自动生成诊断信息：

- **`[!]` 诊断**：精确描述问题（如"仅解析出 2/7 个字段"）
- **`[W]` 警告**：严重程度标记（如"元数据覆盖率过低 (29%)"）
- **`[>]` 建议**：修复方向（如"检查文档格式是否规范"）

### 编程方式使用

```python
from financial_rag.core.scorer import PipelineScoreCard

card = PipelineScoreCard(query="茅台营收增长")

# 逐阶段记录
card.record_metadata(score=0.85, fields_found=6, fields_expected=7)
card.record_tokenization(score=0.8, token_count=15, unique_tokens=12, avg_token_len=2.5)

# BM25 检索
card.record_bm25(result_count=3, top_score=0.92, avg_score=0.65,
                 query_terms=8, matched_terms=6)

# Rerank
card.record_rerank(result_count=3, top_rerank_score=0.95,
                   avg_rerank_score=0.78, high_count=2)

# LLM + 防幻觉
card.record_llm(score=0.9, token_count=350, model="qwen-plus")
card.record_hallucination(overall_score=0.88,
    layer_scores={"L1": 0.9, "L2": 0.85, "L3": 0.92, "L4": 0.8, "L5": 0.9, "L6": 0.88})

# 查看结果
print(card.summary())      # 带诊断的文本摘要
print(card.table())        # 纯表格形式
print(card.to_dict())      # JSON 字典，可对接监控系统
```

### 检索器集成

`HybridRetriever` 内置打分支持，只需调用 `search_with_scores()`：

```python
from financial_rag.retrievers import HybridRetriever

retriever = HybridRetriever(tokenizer=jieba_tokenizer())
retriever.index(documents)

# 自动在每个子阶段（分词/BM25/Vector/RRF/Rerank）记录评分
results, card = retriever.search_with_scores("茅台营收", top_k=5)
print(card.summary())
```

## 模板 + 槽位填充系统

**核心思路**: 用模板 + 槽位填充替代长文自由生成，大幅降低首 Token 延迟。

```
传统自由生成:  "分析茅台财报" → LLM 输出 500 字 → 首Token 2~5s
槽位填充:      拆成 9 个槽位 → 每槽位输出 20~80 字 → 首Token 0.3~0.8s
```

### 预置模板

| 模板 | 槽位数 | 阶段数 | 适用场景 |
|------|--------|--------|----------|
| `quick_qa` | 4 | 1 | 快速问答（直接回答 + 数据支撑 + 可信度） |
| `financial_report` | 9 | 3 | 财报核心摘要（营收/利润/指标/风险） |
| `news_brief` | 7 | 2 | 经济新闻快读（事件/影响/展望） |
| `deep_analysis` | 6 | 3 | 深度分析（盈利/成长/健康/估值/建议） |

### 命令行使用

```bash
# 交互模式 — 支持运行时切换模板
python -m financial_rag.main query -i
# 在交互中: 输入 "fin" 切财报模板, "quick" 切快答, "news" 切新闻, "deep" 切深度

# 对比测试 — 自由生成 vs 槽位填充
python -m financial_rag.main slot "茅台2024年利润增长情况" -t financial_report
# 输出对照组和实验组的首Token延迟对比

# 槽位填充仅测试（不跑对照组）
python -m financial_rag.main slot "茅台营收" -t quick_qa --no-freeform
```

### 编程方式使用

```python
from financial_rag.templates import QUICK_QA_TEMPLATE, FINANCIAL_REPORT_TEMPLATE
from financial_rag.slot_filler import SlotFiller, create_slot_filler

# 创建填充器（带打分卡）
filler = create_slot_filler(llm=llm, scorecard=card)

# 填充槽位
fill_stats = filler.fill(
    template=QUICK_QA_TEMPLATE,
    query="茅台毛利率多少",
    context_docs=["茅台2024年毛利率91.86%..."]
)

# 渲染为最终文本
output = filler.render(QUICK_QA_TEMPLATE, fill_stats)
print(output)

# 查看每个槽位的首Token延迟
for key, r in fill_stats.slot_results.items():
    print(f"  {r.label}: TTFT={r.ttft_ms:.0f}ms, tokens={r.token_count}")

# 性能统计
print(f"总耗时: {fill_stats.total_elapsed_ms:.0f}ms")
print(f"平均首Token: {fill_stats.avg_ttft_ms:.0f}ms")
print(f"并行增益: {fill_stats.parallel_gain:.0%}")
```

### 自定义模板

```python
from financial_rag.templates import SlottedTemplate, SlotDef

my_template = SlottedTemplate(
    name="my_analysis",
    description="自定义分析模板",
    slots=[
        SlotDef("summary", "摘要", prompt="用1句话概括。最多30字。", max_tokens=40, required=True),
        SlotDef("detail", "详情", prompt="展开分析。最多50字。", max_tokens=60),
    ],
    phases=[["summary", "detail"]],  # 同一phase可并行
    render="# {summary}\n\n{detail}",
)
```

### 打分卡集成

槽位填充结果自动纳入全链路打分卡，每个槽位独立评分：

```
── LLM 生成 & 防幻觉校验 [A] 均分 0.91 ──
  [GOOD] [槽位] 公司名称      0.95 (   120ms)
  [GOOD] [槽位] 营收概况      0.88 (   180ms)
  [GOOD] [槽位] 利润概况      0.85 (   210ms)
  [GOOD] 槽位填充汇总         0.90 (   600ms)
```

## 能力注册中心 + Function Calling

**核心思路**: 所有 Agent 的能力集中注册在一个地方，LLM 根据用户意图通过 Function Calling 自动选择调用哪个能力。

```
传统方式:  LLM 收到问题 → 自由生成回答 → 容易幻觉/数据不精确
能力注册:  LLM 收到问题 → 判断需要哪些能力 → 调用工具获取精确数据 →  基于数据生成回答
```

### 内置能力

| 分类 | 能力 | 说明 |
|------|------|------|
| retrieval | `search_financial_data` | 从知识库检索金融数据 |
| analysis | `calculate_growth_rate` | 计算同比增长率 |
| analysis | `calculate_financial_ratio` | 计算财务比率（毛利率/ROE等） |
| analysis | `compare_metrics` | 横向对比公司指标 |
| analysis | `summarize_financials` | 汇总多项指标为自然语言 |

### 命令行使用

```bash
# 列出所有已注册能力
python -m financial_rag.main toolcall -l temp

# Function Calling 模式
python -m financial_rag.main toolcall "茅台营收增长多少" -v

# 强制 LLM 必须调用工具
python -m financial_rag.main toolcall "计算茅台毛利率" --tool-choice required

# 多轮调用 (LLM 可多次选择工具)
python -m financial_rag.main toolcall "茅台和五粮液利润对比" --multi-turn -v
```

### 编程方式：注册自定义能力

```python
from financial_rag.tools import (
    FunctionRegistry, FunctionDef, ToolExecutor, ToolCallSession, create_tool_session
)

# 创建注册中心
registry = FunctionRegistry(name="my_agent")

# 方式1: 装饰器注册
@registry.register(category="data", description="获取实时股价")
def get_stock_price(symbol: str, exchange: str = "SH") -> dict:
    return {"symbol": symbol, "price": 1850.00, "exchange": exchange}

# 方式2: 显式注册
registry.add(FunctionDef(
    name="get_pe_ratio",
    description="查询公司市盈率",
    parameters={
        "type": "object",
        "properties": {"symbol": {"type": "string", "description": "股票代码"}},
        "required": ["symbol"],
    },
    callback=lambda symbol: {"pe": 28.5, "industry_avg": 35.2},
    category="data",
))

# LLM 自动选择调用
session = create_tool_session(llm=llm, registry=registry)
stats = session.run("茅台当前市盈率是多少？", scorecard=card)
# LLM 自动选择 get_pe_ratio("600519") → 返回结果 → 生成最终回答
```

### 工作原理

```
用户问题 → [LLM + tools] → 返回 tool_calls["get_pe_ratio", {"symbol": "600519"}]
         → [ToolExecutor] → 执行 get_pe_ratio，返回 {"pe": 28.5}
         → [tool result 回传 LLM] → "贵州茅台当前市盈率 28.5，低于行业平均 35.2"
```

- **自动选择**: LLM 判断需要哪个能力，自动填充参数
- **多轮调用**: 支持 LLM 在单次对话中调用多个工具
- **并行执行**: 同一轮多个 tool_calls 并行执行（ThreadPoolExecutor）
- **打分集成**: 每个工具调用独立评分，汇总到全链路打分卡

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key | 否（无 Key 自动降级本地模式） |

## 许可证

MIT License
