"""
使用示例 - 展示如何使用日志解析和知识库问答系统
"""
import os
from log_parser import LogParserFactory, LogProcessor, parse_log_file
from knowledge_base import KnowledgeBase, LogKnowledgeBase, create_knowledge_base_from_logs
from rag_engine import RAGEngine, LogAnalyzer


def example_1_basic_log_parsing():
    """示例1: 基础日志解析"""
    print("=" * 60)
    print("示例1: 基础日志解析")
    print("=" * 60)
    
    # 假设有一个 nginx 日志文件
    log_file = "./data/nginx_access.log"
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        print("跳过此示例\n")
        return
    
    # 方法1: 使用便捷函数
    entries = parse_log_file(log_file)
    print(f"解析到 {len(entries)} 条日志")
    
    # 方法2: 使用处理器（更灵活）
    parser = LogParserFactory.get_parser(log_file)
    processor = LogProcessor(parser)
    entries = processor.process_file(log_file)
    
    # 获取统计信息
    stats = processor.get_statistics()
    print(f"日志统计: {stats}")
    
    # 筛选错误日志
    errors = processor.get_error_entries()
    print(f"错误日志数量: {len(errors)}")
    
    print()


def example_2_build_knowledge_base():
    """示例2: 构建知识库"""
    print("=" * 60)
    print("示例2: 构建知识库")
    print("=" * 60)
    
    # 创建知识库
    kb = KnowledgeBase()
    
    # 从目录加载文档
    kb.add_documents_from_directory("./knowledge_base")
    
    # 或者直接添加文本
    documents = [
        "系统错误代码 500 表示服务器内部错误",
        "系统错误代码 404 表示资源未找到",
        "数据库连接超时通常由网络问题或数据库负载过高引起",
    ]
    kb.add_text_documents(documents, metadata_list=[
        {"category": "error_code"},
        {"category": "error_code"},
        {"category": "troubleshooting"},
    ])
    
    # 构建索引
    kb.build_index()
    
    # 保存索引
    kb.save_index()
    
    print("知识库构建完成\n")


def example_3_log_rag():
    """示例3: 基于日志的 RAG 问答"""
    print("=" * 60)
    print("示例3: 基于日志的 RAG 问答")
    print("=" * 60)
    
    # 假设已有解析好的日志条目
    log_file = "./data/app.log"
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        print("跳过此示例\n")
        return
    
    # 解析日志
    entries = parse_log_file(log_file)
    
    # 创建日志知识库
    log_kb = LogKnowledgeBase()
    log_kb.add_logs(entries).build_index()
    
    # 创建 RAG 引擎
    engine = RAGEngine(log_kb)
    
    # 执行查询
    questions = [
        "日志中有哪些错误？",
        "为什么会出现数据库连接失败？",
        "系统性能如何？"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = engine.query(question)
        print(f"A: {result.answer}")
        print(f"置信度: {result.confidence:.2f}")
    
    print()


def example_4_log_analysis():
    """示例4: 高级日志分析"""
    print("=" * 60)
    print("示例4: 高级日志分析")
    print("=" * 60)
    
    log_file = "./data/app.log"
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        print("跳过此示例\n")
        return
    
    # 解析日志
    entries = parse_log_file(log_file)
    
    # 创建日志知识库和分析器
    log_kb = LogKnowledgeBase()
    log_kb.add_logs(entries).build_index()
    analyzer = LogAnalyzer(log_kb)
    
    # 获取概览
    overview = analyzer.get_overview()
    print("日志概览:")
    print(overview['error_summary'])
    
    # 查找相似错误
    similar = analyzer.find_similar_errors("connection timeout", top_k=3)
    print(f"\n找到 {len(similar)} 个相似错误")
    
    # 生成完整报告
    report = analyzer.generate_report()
    print("\n分析报告已生成")
    
    print()


def example_5_chat_mode():
    """示例5: 聊天模式"""
    print("=" * 60)
    print("示例5: 聊天模式")
    print("=" * 60)
    
    # 加载已有知识库
    kb = KnowledgeBase()
    if not kb.load_index():
        print("知识库不存在，请先构建知识库")
        return
    
    engine = RAGEngine(kb)
    
    # 模拟对话
    conversation = [
        "系统有哪些错误？",
        "这些错误怎么解决？",
        "还有其他需要注意的问题吗？"
    ]
    
    print("模拟对话:")
    for message in conversation:
        print(f"\n用户: {message}")
        result = engine.chat(message)
        print(f"助手: {result.answer}")
    
    print()


def example_6_custom_parser():
    """示例6: 自定义日志解析器"""
    print("=" * 60)
    print("示例6: 自定义日志解析器")
    print("=" * 60)
    
    from log_parser import LogParser, LogEntry
    import re
    from datetime import datetime
    
    class MyCustomParser(LogParser):
        """自定义解析器示例"""
        
        PATTERN = re.compile(
            r'\[(?P<time>[^\]]+)\]\s+'
            r'(?P<level>\w+)\s+-\s+'
            r'(?P<message>.*)'
        )
        
        def parse(self, line: str):
            match = self.PATTERN.match(line)
            if match:
                data = match.groupdict()
                return LogEntry(
                    timestamp=datetime.now(),  # 简化处理
                    level=data['level'],
                    source="custom_app",
                    message=data['message'],
                    raw_line=line
                )
            return None
        
        def can_parse(self, sample_lines):
            return any(self.parse(line) for line in sample_lines)
    
    # 使用自定义解析器
    custom_parser = MyCustomParser()
    processor = LogProcessor(custom_parser)
    
    # 处理自定义格式日志
    # entries = processor.process_file("custom.log")
    
    print("自定义解析器已创建")
    print("可以通过 LogParserFactory.PARSERS.append(MyCustomParser()) 注册到工厂\n")


def create_sample_files():
    """创建示例文件"""
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./knowledge_base", exist_ok=True)
    
    # 创建示例日志文件
    sample_log = """2024-01-15 10:23:45 INFO server Starting application server on port 8080
2024-01-15 10:23:46 INFO db Connected to database at localhost:5432
2024-01-15 10:24:12 WARN auth Failed login attempt for user 'admin' from 192.168.1.100
2024-01-15 10:24:15 ERROR payment Payment processing failed: Connection timeout
2024-01-15 10:24:16 ERROR payment Retry payment processing failed
2024-01-15 10:25:30 INFO server Health check passed
2024-01-15 10:26:45 WARN memory High memory usage detected: 85%
2024-01-15 10:27:00 ERROR db Database connection lost, attempting reconnection
2024-01-15 10:27:05 INFO db Database connection restored
2024-01-15 10:28:00 ERROR payment Payment gateway returned 503 error
2024-01-15 10:28:30 WARN cache Cache miss rate is high: 45%
2024-01-15 10:30:00 INFO server Request processed successfully
"""
    
    with open("./data/app.log", "w", encoding="utf-8") as f:
        f.write(sample_log)
    
    # 创建知识库文档
    kb_doc = """# 系统故障排查手册

## 数据库连接问题

### 症状
- 数据库连接超时
- 连接丢失
- 无法建立新连接

### 可能原因
1. 网络问题
2. 数据库服务器负载过高
3. 连接池配置不当
4. 防火墙设置

### 解决方案
1. 检查网络连接
2. 监控数据库性能指标
3. 调整连接池大小
4. 检查防火墙规则

## 支付系统故障

### 症状
- 支付处理失败
- 网关返回 503 错误
- 连接超时

### 可能原因
1. 支付网关服务不可用
2. 网络连接问题
3. 请求参数错误

### 解决方案
1. 检查支付网关状态页面
2. 查看网关 API 文档
3. 实现重试机制
4. 添加熔断降级

## 内存使用过高

### 症状
- 内存使用率超过 80%
- 系统响应变慢
- OOM 错误

### 解决方案
1. 优化内存分配
2. 增加物理内存
3. 检查内存泄漏
4. 调整垃圾回收策略
"""
    
    with open("./knowledge_base/troubleshooting.md", "w", encoding="utf-8") as f:
        f.write(kb_doc)
    
    print("示例文件已创建: ./data/app.log, ./knowledge_base/troubleshooting.md\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("日志解析和知识库问答系统 - 使用示例")
    print("=" * 60 + "\n")
    
    # 创建示例文件
    create_sample_files()
    
    # 运行示例
    # example_1_basic_log_parsing()
    example_2_build_knowledge_base()
    example_3_log_rag()
    example_4_log_analysis()
    # example_5_chat_mode()
    example_6_custom_parser()
    
    print("\n" + "=" * 60)
    print("所有示例执行完成!")
    print("=" * 60)
