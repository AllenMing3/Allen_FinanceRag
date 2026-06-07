# Financial RAG 使用手册

> 面向使用者的操作指南。架构细节请看 `README.md`。

---

## 目录

1. [环境准备](#1-环境准备)
2. [快速上手](#2-快速上手)
3. [Pipeline — 一句话出分析报告](#3-pipeline--一句话出分析报告)
4. [交互查询 — 连续提问](#4-交互查询--连续提问)
5. [新闻抓取 — 实时拉财经新闻](#5-新闻抓取--实时拉财经新闻)
6. [ETF K线 — 行情分析](#6-etf-k线--行情分析)
7. [Function Calling — LLM 自动调工具](#7-function-calling--llm-自动调工具)
8. [槽位填充 — 结构化输出](#8-槽位填充--结构化输出)
9. [检索打分 — 诊断检索质量](#9-检索打分--诊断检索质量)
10. [Multi-Agent 分析 — 深度财报分析](#10-multi-agent-分析--深度财报分析)
11. [知识库构建](#11-知识库构建)
12. [常见问题排查](#12-常见问题排查)

---

## 1. 环境准备

```powershell
# 进入项目目录
cd d:\llamaindex

# 激活虚拟环境
.\myenv\Scripts\Activate.ps1

# 首次激活如果报错，先执行这个：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

激活成功后终端前面会有 `(myenv)` 标志。

**验证环境：**

```powershell
python -c "from financial_rag.news_fetcher import HAS_AKSHARE; print('akshare:', HAS_AKSHARE)"
python -c "from financial_rag.llm import get_llm; print('llm ok')"
```

- `akshare: True` = 数据获取 OK
- `llm ok` = LLM 调用 OK

**API Key 配置：**

编辑项目根目录的 `.env` 文件：
```
DASHSCOPE_API_KEY=sk-你的密钥
```

没有 Key 也能跑，会降级到纯本地模式（BM25 关键词检索 + Jaccard 相似度）。

---

## 2. 快速上手

```powershell
# 演示模式 — 跑一遍内置示例，看看系统能干什么
python -m financial_rag.main demo
```

输出会展示：
- 配置信息（LLM/Embedding/Rerank 模型）
- 5 个 Agent 的注册和流水线
- 全链路检索测试（BM25 → Vector → RRF → Rerank）
- 每个环节的打分卡

---

## 3. Pipeline — 一句话出分析报告

**最推荐的用法**，一条命令跑完：数据获取 → RAG检索 → Agent分析 → 格式化输出 → 质量打分。

```powershell
# 基本用法
python -m financial_rag.main pipeline "茅台2024年利润增长情况"

# 用新闻简报模板 + 详细日志
python -m financial_rag.main pipeline "新能源板块最近有什么利好" -t news -v

# 用深度分析模板 + 输出到文件
python -m financial_rag.main pipeline "降准对银行股的影响" -t deep -o ./output

# 指定模板 + 控制获取数量
python -m financial_rag.main pipeline "人工智能ETF走势" -t fin --max-fetch 20 --max-retrieve 10
```

**4 种输出模板：**

| 模板 | 参数 | 适合场景 |
|------|------|---------|
| 快答 | `-t quick` | 快速回答简单问题（默认） |
| 财报 | `-t fin` | 财报数据分析 |
| 新闻 | `-t news` | 新闻摘要和热点 |
| 深度 | `-t deep` | 多维度深度分析 |

**输出说明：**
- 终端会打印每个阶段的耗时（获取/索引/加工/输出/总计）
- 如果加了 `-v`，会显示每个 Agent 的详细执行日志
- 用 `-o ./output` 可以把报告保存为 Markdown 文件

---

## 4. 交互查询 — 连续提问

```powershell
python -m financial_rag.main query -i
```

进入交互模式后：

```
输入问题: 茅台2024年营收多少
→ 系统返回分析结果

输入问题: 净利润呢
→ 继续基于同一知识库回答

输入: fin
→ 切换到财报模板

输入: news
→ 切换到新闻模板

输入: score
→ 查看上次查询的全链路打分卡

输入: q
→ 退出
```

**单次查询（不进交互模式）：**

```powershell
python -m financial_rag.main query -q "茅台毛利率多少"
```

---

## 5. 新闻抓取 — 实时拉财经新闻

从东方财富等数据源拉取实时新闻，保存为格式化的 Markdown 文档。

```powershell
# 搜索主题新闻
python -m financial_rag.main news "AI人工智能"

# 搜索 + LLM 生成摘要（推荐）
python -m financial_rag.main news "今天最大的AI新闻" -s

# 指定输出目录和文件名
python -m financial_rag.main news "降准" -o ./output -n 降准新闻.md
```

**输出：**
- 在 `./output/` 目录下生成 Markdown 文件
- 包含新闻标题、来源、时间、正文
- 加 `-s` 会在文末附上 AI 生成的摘要

**数据来源：** akshare 的 `stock_info_global_em` 接口（东方财富）。

---

## 6. ETF K线 — 行情分析

拉取 ETF 历史 K 线数据，生成技术分析报告。

```powershell
# 搜索 ETF + 拉近30天K线
python -m financial_rag.main kline "人工智能ETF"

# 指定回溯天数
python -m financial_rag.main kline "新能源" --days 60

# 直接指定 ETF 代码
python -m financial_rag.main kline "沪深300" --code 510300

# K线 + LLM 技术分析（推荐）
python -m financial_rag.main kline "人工智能ETF" --days 30 -s
```

**输出包含：**
- ETF 代码和名称
- 最新价、涨跌幅
- 区间统计：最高/最低/收盘/MA5/MA10
- 加 `-s` 会附上 AI 技术分析（趋势/支撑位/压力位）

---

## 7. Function Calling — LLM 自动调工具

LLM 根据问题自动选择并调用注册的工具函数。

```powershell
# 列出所有已注册的工具
python -m financial_rag.main toolcall -l

# 单次调用
python -m financial_rag.main toolcall "茅台营收增长多少"

# 详细日志
python -m financial_rag.main toolcall "茅台营收增长多少" -v

# 多轮调用（LLM 可以连续调多个工具）
python -m financial_rag.main toolcall "茅台和五粮液利润对比" --multi-turn -v

# 强制必须调用工具
python -m financial_rag.main toolcall "计算茅台毛利率" --tool-choice required
```

**工具策略：**

| 策略 | 参数 | 说明 |
|------|------|------|
| auto | `--tool-choice auto` | LLM 自己决定要不要调工具（默认） |
| required | `--tool-choice required` | 强制必须调工具 |
| none | `--tool-choice none` | 禁止调工具，纯 LLM 回答 |

---

## 8. 槽位填充 — 结构化输出

用预定义模板约束 LLM 输出格式，每个"槽位"独立填充，降低幻觉风险。

```powershell
# 对比测试：自由生成 vs 槽位填充
python -m financial_rag.main slot "茅台2024年利润" -t financial_report

# 只看槽位填充（跳过对照组）
python -m financial_rag.main slot "茅台营收" -t quick_qa --no-freeform

# 用深度分析模板
python -m financial_rag.main slot "降准对银行股影响" -t deep_analysis
```

**可用模板：**

| 模板名 | 用途 | 槽位数 |
|--------|------|--------|
| `quick_qa` | 快速问答 | 少 |
| `financial_report` | 财报分析 | 多（营收/净利/毛利率/EPS/ROE...） |
| `news_brief` | 新闻简报 | 中 |
| `deep_analysis` | 深度分析 | 多 |

**输出会显示：**
- 总耗时 + 每个槽位的 TTFT（首 Token 延迟）
- 槽位填充率（填了几个/总共几个）
- 并行增益（并行填充比串行快多少）
- 与自由生成的对比（耗时/Token 数）

---

## 9. 检索打分 — 诊断检索质量

不调 LLM，纯测检索链路，快速定位哪个环节有问题。

```powershell
# 基本打分
python -m financial_rag.main score "茅台营收增长" -k 5

# 导出打分 JSON
python -m financial_rag.main score "汇率走势" --json scores.json

# 纯本地模式（不依赖 API）
python -m financial_rag.main score "测试查询" --local
```

**打分卡会显示每个环节的评分：**

| 环节 | 含义 | 分低怎么办 |
|------|------|-----------|
| Jieba 分词 | 中文分词质量 | 检查文本编码 |
| BM25 检索 | 关键词匹配 | 加同义词到文档 |
| 向量检索 | 语义匹配 | 检查 Embedding 模型 |
| RRF 融合 | 两路融合效果 | 调 `bm25_weight` / `vector_weight` |
| Rerank | 精排 | 检查 gte-rerank 是否可用 |

评分等级：A (>=0.9) / B (>=0.75) / C (>=0.6) / D (>=0.4) / F (<0.4)

---

## 10. Multi-Agent 分析 — 深度财报分析

把一份财报文件交给 5 个 Agent 依次处理。

```powershell
# 串行分析（默认）
python -m financial_rag.main analyze ./data/financial/maotai_2024.pdf

# 并行执行（更快，但 Agent 之间不传递上下文）
python -m financial_rag.main analyze ./report.pdf --parallel

# 指定输出路径
python -m financial_rag.main analyze ./report.pdf --output ./output/分析报告.md
```

**5 个 Agent 依次执行：**

```
IngestionAgent   读取文件 → 清洗文本 → 提取元数据（公司/日期/类型）
ExtractionAgent  抽取财务指标（营收/净利/EPS/ROE...）+ 实体
AnalysisAgent    5 维度分析（盈利/成长/健康/效率/估值）
ForecastAgent    3 种情景预测（乐观/基准/悲观）
ReportAgent      生成报告（摘要/详细/PPT大纲）
```

---

## 11. 知识库构建

把本地文件加载为 RAG 知识库。

```powershell
python -m financial_rag.main build --dir ./data/financial
```

支持的文件格式：JSONL、TXT、PDF。

构建完后会自动跑两条测试查询，展示检索效果和打分。

---

## 12. 常见问题排查

### "DASHSCOPE_API_KEY 未设置"

```powershell
# 检查 .env 文件是否存在
cat .env

# 如果没有，创建它
Copy-Item .env.example .env
notepad .env
# 填入 DASHSCOPE_API_KEY=sk-xxx
```

### "Rerank 403 Access Denied"

`gte-rerank` 模型需要在阿里云百炼控制台手动开通：
1. 打开 https://bailian.console.aliyun.com/
2. 找到「模型广场」→ 搜索 `gte-rerank`
3. 点击开通

不开通也不影响使用，系统会自动降级到 RRF 融合排序。

### "akshare 返回 0 条新闻"

- 该股票当天确实没有新闻 → 换个热门股票试试
- 网络问题 → akshare 走东方财富接口，检查网络
- 需要代理 → 设置 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量

### "中文乱码"

- **用 PowerShell 而不是 CMD**（CMD 默认 GBK 编码）
- 如果必须用 CMD：`set PYTHONUTF8=1` 然后再跑命令

### "检索结果不准"

```powershell
# 先跑打分看哪个环节差
python -m financial_rag.main score "你的查询" -k 5

# 如果 BM25 分低 → 文档里缺少相关关键词
# 如果向量检索分低 → 知识库内容和查询不匹配
# 如果 RRF 分低 → 两个检索器方向不一致
```

### "LLM 回答有幻觉"

1. 降低温度：改 `config.py` 里 `temperature` 为 `0.0`
2. 提高忠实度阈值：改 `min_faithfulness` 为 `0.8`
3. 用槽位填充代替自由生成（`slot` 命令比 `query` 更可控）

### "想看详细执行日志"

大部分命令加 `-v` 就能看到详细日志：
```powershell
python -m financial_rag.main pipeline "查询" -v
python -m financial_rag.main toolcall "查询" -v
```

或者改 `config.py` 里的 `config.coordinator.verbose = True`。

---

## 命令速查表

| 命令 | 用途 | 最常用写法 |
|------|------|-----------|
| `web` | 启动 Web UI | `python -m financial_rag.main web` |
| `demo` | 跑内置演示 | `python -m financial_rag.main demo` |
| `pipeline` | 端到端分析 | `python -m financial_rag.main pipeline "查询" -t quick -v` |
| `query` | 交互/单次查询 | `python -m financial_rag.main query -i` |
| `news` | 拉取新闻 | `python -m financial_rag.main news "主题" -s` |
| `kline` | ETF K线分析 | `python -m financial_rag.main kline "ETF" --days 30 -s` |
| `toolcall` | Function Calling | `python -m financial_rag.main toolcall "查询" -v` |
| `slot` | 槽位填充测试 | `python -m financial_rag.main slot "查询" -t financial_report` |
| `score` | 检索质量打分 | `python -m financial_rag.main score "查询" -k 5` |
| `analyze` | Multi-Agent 分析 | `python -m financial_rag.main analyze 文件.pdf` |
| `build` | 构建知识库 | `python -m financial_rag.main build --dir ./data/financial` |
