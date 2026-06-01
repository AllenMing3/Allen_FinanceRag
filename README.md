# 日志智能分析系统 (Log Intelligence)

基于 LlamaIndex 的日志自动解析和知识库问答系统。支持多种日志格式解析、向量索引构建、智能问答和故障排查。

## 功能特性

- **多格式日志解析**: 支持 Nginx、Syslog、JSON、通用日志格式
- **智能知识库**: 基于 LlamaIndex 的向量索引和检索
- **RAG 问答**: 结合日志内容和知识库进行智能问答
- **故障排查**: 自动分析日志错误并提供解决方案
- **交互式界面**: 支持命令行交互和批量处理

## 项目结构

```
llamaindex/
├── config.py              # 配置文件
├── log_parser.py          # 日志解析模块
├── knowledge_base.py      # 知识库构建模块
├── rag_engine.py          # RAG 查询引擎
├── main.py                # 主程序入口
├── example_usage.py       # 使用示例
├── requirements.txt       # 依赖列表
├── .env.example          # 环境变量示例
├── data/                 # 日志文件目录
├── knowledge_base/       # 知识库文档目录
├── storage/              # 向量存储目录
└── output/               # 输出目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填写你的 OpenAI API Key:

```bash
cp .env.example .env
# 编辑 .env 文件
OPENAI_API_KEY=your_api_key_here
```

### 3. 快速演示

```bash
python main.py demo
```

### 4. 处理日志文件

```bash
# 解析日志并构建知识库
python main.py logs ./data/app.log --build-kb

# 指定日志格式
python main.py logs ./data/access.log --format nginx --build-kb
```

### 5. 构建知识库

```bash
# 从文档目录构建
python main.py build --docs-dir ./knowledge_base
```

### 6. 交互式查询

```bash
# 启动交互式问答
python main.py query -i

# 单次查询
python main.py query -q "日志中有哪些错误？"
```

### 7. 日志分析

```bash
# 生成分析报告并进入交互模式
python main.py analyze -r -i
```

## 使用示例

### 基础使用

```python
from log_parser import parse_log_file
from knowledge_base import LogKnowledgeBase
from rag_engine import RAGEngine

# 1. 解析日志
entries = parse_log_file("./data/app.log")

# 2. 构建知识库
log_kb = LogKnowledgeBase()
log_kb.add_logs(entries).build_index().save_index()

# 3. 智能问答
engine = RAGEngine(log_kb)
result = engine.query("为什么会出现数据库连接失败？")
print(result.answer)
```

### 高级分析

```python
from rag_engine import LogAnalyzer

# 创建分析器
analyzer = LogAnalyzer(log_kb)

# 获取概览
overview = analyzer.get_overview()

# 生成完整报告
report = analyzer.generate_report()

# 查找相似错误
similar = analyzer.find_similar_errors("connection timeout")
```

## 支持的日志格式

| 格式 | 说明 | 自动检测 |
|------|------|----------|
| Nginx | Nginx access log | ✅ |
| Syslog | 系统日志 | ✅ |
| JSON | JSON 格式日志 | ✅ |
| Generic | 通用格式（兜底） | ✅ |

## API 参考

### LogParser 模块

- `parse_log_file(file_path, log_format=None)` - 解析日志文件
- `LogParserFactory.get_parser(file_path)` - 自动检测并获取解析器
- `LogProcessor` - 批量处理日志

### KnowledgeBase 模块

- `KnowledgeBase` - 通用知识库
- `LogKnowledgeBase` - 专用日志知识库
- `add_documents_from_directory(dir_path)` - 从目录加载文档
- `add_logs(log_entries)` - 添加日志条目
- `build_index()` - 构建向量索引
- `save_index()` / `load_index()` - 保存/加载索引

### RAGEngine 模块

- `query(question)` - 执行查询
- `chat(message)` - 聊天模式
- `analyze_logs()` - 自动分析日志
- `troubleshoot_error(error_message)` - 故障排查
- `summarize_logs()` - 日志总结

## 配置说明

编辑 `config.py` 或设置环境变量:

```python
# OpenAI 配置
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://api.openai.com/v1

# 模型配置
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

# RAG 配置
similarity_top_k=5
response_mode=compact
```

## 自定义扩展

### 添加自定义日志解析器

```python
from log_parser import LogParser, LogEntry, LogParserFactory
import re

class MyParser(LogParser):
    def parse(self, line: str):
        # 实现解析逻辑
        pass
    
    def can_parse(self, sample_lines):
        # 检测是否支持
        pass

# 注册到工厂
LogParserFactory.PARSERS.append(MyParser())
```

## 命令行帮助

```bash
# 查看所有命令
python main.py --help

# 查看具体命令帮助
python main.py logs --help
python main.py query --help
python main.py analyze --help
```

## 注意事项

1. **API Key**: 确保设置了有效的 OpenAI API Key
2. **日志编码**: 默认使用 UTF-8 编码读取日志文件
3. **大文件处理**: 大日志文件会自动分块处理
4. **向量存储**: 索引默认保存在 `./storage/vector_store` 目录

## 许可证

MIT License
