# Financial RAG 使用手册

> 架构细节请看 `README.md`。

---

## 1. 环境准备

```cmd
cd d:\llamaindex
myenv\Scripts\activate.bat
```

编辑 `.env`：
- `DASHSCOPE_API_KEY=sk-xxx` — LLM/Embedding/Rerank（必须）
- `TUSHARE_TOKEN=xxx` — K线数据（可选）
- `MOCK_MODE=true` — 开启 Mock 模式（可选，详见第 3 节）

---

## 2. 快速开始

```cmd
python -m financial_rag.main web
```

打开 `http://127.0.0.1:8000`，按顺序操作：

| 步骤 | 页面 | 操作 | 说明 |
|------|------|------|------|
| 1 | 数据源 | 搜索新闻（如"AI人工智能"） | 收集元数据，辅助后续文件解析。新闻**不进知识库** |
| 2 | 数据源 | 分析并导入文件 | Agent 链分析文件 → 抽取指标/实体 → 存入知识库 |
| 3 | 构建知识库 | 构建索引 | BM25 + 向量双通道索引 |
| 4 | RAG 查询 | 提问 | 检索知识库 + 匹配新闻 + LLM 回答（带引用和 RRF 分数明细） |
| 5 | 工具 | K线分析 | 输入"茅台"或"600519"，生成技术分析报告 |

**数据来源：**
- 文件放 `./data/financial` 目录
- 新闻来自国内直连 API（同花顺/新浪财经/东方财富）
- K线来自 Tushare Pro（需 120+ 积分）

---

## 3. Mock 模式

无需 Tushare Token，K线和新闻使用内置模拟数据，LLM 仍走真实 API。

`.env` 中加 `MOCK_MODE=true`，或命令行启动：

```cmd
set MOCK_MODE=true && python -m financial_rag.main web
```

| 数据源 | Mock 效果 |
|--------|----------|
| 新闻 | 25 条内置 AI 行业新闻 |
| K线 | 8 只股票 + 7 只 ETF 的模拟行情（几何布朗运动） |
| LLM/Embedding | **真实 DashScope API**（需 Key） |

---

## 4. 测试

```cmd
:: 全量（103 tests，无需 API Key）
python -m pytest tests/ -v

:: 只看 Agent 链
python -m pytest tests/test_agents.py -v

:: 只看 Mock 数据
python -m pytest tests/test_mock_data.py -v
```

测试只 mock 数据源，LLM/Embedding/Rerank 保持真实。Agent 抽取工具走 regex fallback，无需 API Key。

---

## 5. 命令速查表

| 命令 | 用途 | 示例 |
|------|------|------|
| `web` | Web UI | `python -m financial_rag.main web` |
| `pipeline` | 端到端分析 | `python -m financial_rag.main pipeline "商汤营收" -v` |
| `news` | 拉新闻 | `python -m financial_rag.main news "AI" -s` |
| `kline` | K线分析 | `python -m financial_rag.main kline "商汤" --days 30` |
| `toolcall` | Function Calling | `python -m financial_rag.main toolcall "商汤营收" -v` |
| `slot` | 槽位填充 | `python -m financial_rag.main slot "商汤营收" -t fin` |
| `score` | 检索打分 | `python -m financial_rag.main score "商汤算力" -k 5` |
| `build` | 构建知识库 | `python -m financial_rag.main build --dir ./data/financial` |
| `query` | 交互查询 | `python -m financial_rag.main query -i` |

Pipeline 模板：`-t quick`（默认）/ `-t fin`（财报）/ `-t news`（新闻）/ `-t deep`（深度）

---

## 6. 常见问题

**Rerank 403** — `gte-rerank` 需在[阿里云百炼控制台](https://bailian.console.aliyun.com/)手动开通，不开通自动降级为 RRF 融合。

**新闻获取不到** — 关键词太偏或 API 频率限制，换热门主题重试。

**中文乱码** — CMD 下执行 `set PYTHONUTF8=1`。

**详细日志** — 大部分命令加 `-v`。
