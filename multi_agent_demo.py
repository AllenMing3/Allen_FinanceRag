"""
Multi-Agent 系统演示

展示如何使用 Multi-Agent 架构进行日志分析
"""
import os
from agents import (
    LogParserAgent,
    LogAnalystAgent,
    SolutionAgent,
    ReportAgent,
    AgentOrchestrator,
    ExecutionMode
)


def demo_basic_workflow():
    """演示基础工作流"""
    print("=" * 70)
    print("演示1: 基础 Multi-Agent 工作流")
    print("=" * 70)
    
    # 创建协调器
    orchestrator = AgentOrchestrator()
    
    # 注册所有 Agent
    orchestrator.register_agents(
        LogParserAgent(),
        LogAnalystAgent(),
        SolutionAgent(),
        ReportAgent()
    )
    
    # 创建示例日志
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
2024-01-15 10:32:15 ERROR payment Payment processing failed: Connection timeout
2024-01-15 10:35:00 ERROR db Query timeout after 30 seconds
2024-01-15 10:40:00 ERROR payment Payment gateway returned 503 error
"""
    
    # 保存示例日志
    os.makedirs("./data", exist_ok=True)
    log_file = "./data/sample_multi_agent.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(sample_log)
    
    print(f"\n创建示例日志文件: {log_file}")
    
    # 执行分析
    print("\n开始执行 Multi-Agent 分析流程...\n")
    result = orchestrator.execute(log_file)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("执行结果")
    print("=" * 70)
    
    print(f"\n✓ 执行成功: {result.success}")
    print(f"✓ 总耗时: {result.execution_time:.2f} 秒")
    print(f"✓ Agent 执行数量: {len(result.agent_results)}")
    
    print("\n各 Agent 执行结果:")
    for agent_result in result.agent_results:
        status = "✓" if agent_result.success else "✗"
        print(f"  {status} {agent_result.agent_name}: {agent_result.message}")
    
    # 显示最终报告
    if result.final_output:
        print("\n" + "=" * 70)
        print("最终报告（前1000字符）")
        print("=" * 70)
        print(result.final_output[:1000])
        print("...")
    
    return result


def demo_conditional_execution():
    """演示条件执行"""
    print("\n" + "=" * 70)
    print("演示2: 条件执行模式")
    print("=" * 70)
    
    orchestrator = AgentOrchestrator()
    orchestrator.config.execution_mode = ExecutionMode.CONDITIONAL
    
    orchestrator.register_agents(
        LogParserAgent(),
        LogAnalystAgent(),
    )
    
    # 只有解析成功后才执行分析
    print("\n条件执行: 只有 LogParserAgent 成功后才执行 LogAnalystAgent")
    
    # 测试正常日志
    normal_log = "2024-01-15 10:00:00 INFO server Server started successfully"
    result = orchestrator.execute(normal_log)
    
    print(f"\n正常日志执行结果: {result.success}")
    print(f"执行了 {len(result.agent_results)} 个 Agent")


def demo_tool_usage():
    """演示工具使用"""
    print("\n" + "=" * 70)
    print("演示3: Agent 工具使用")
    print("=" * 70)
    
    from agents.tools import (
        generate_statistics,
        extract_error_patterns,
        analyze_error_frequency
    )
    from log_parser import parse_log_file
    
    # 解析日志
    log_file = "./data/sample_multi_agent.log"
    if os.path.exists(log_file):
        entries = parse_log_file(log_file)
        
        print(f"\n解析到 {len(entries)} 条日志")
        
        # 使用工具
        print("\n1. 生成统计信息:")
        stats = generate_statistics(entries)
        print(f"   总数: {stats['total_entries']}")
        print(f"   错误率: {stats['error_rate']}%")
        print(f"   按级别: {stats['by_level']}")
        
        print("\n2. 提取错误模式:")
        patterns = extract_error_patterns(entries)
        for i, pattern in enumerate(patterns[:3], 1):
            print(f"   模式 {i}: 出现 {pattern['count']} 次")
            print(f"   示例: {pattern['sample_messages'][0][:80]}...")
        
        print("\n3. 错误频率分析:")
        freq = analyze_error_frequency(entries, time_window_minutes=30)
        print(f"   总错误数: {freq['error_count']}")
        print(f"   平均频率: {freq['average_frequency']} 次/窗口")


def demo_custom_agent():
    """演示自定义 Agent"""
    print("\n" + "=" * 70)
    print("演示4: 自定义 Agent")
    print("=" * 70)
    
    from agents.base_agent import BaseAgent, AgentContext, AgentResponse
    
    class AlertAgent(BaseAgent):
        """告警 Agent - 检查是否需要发送告警"""
        
        def __init__(self):
            super().__init__(
                name="AlertAgent",
                description="检查是否需要发送告警通知"
            )
        
        def process(self, context: AgentContext) -> AgentResponse:
            # 检查分析结果
            analysis = context.analysis_results
            
            alerts = []
            
            # 检查错误数量
            if analysis and "analyses" in analysis:
                for a in analysis["analyses"]:
                    if a.get("type") == "error_analysis":
                        error_count = a.get("error_count", 0)
                        if error_count > 5:
                            alerts.append(f"错误数量过多: {error_count}")
            
            # 检查解决方案
            solutions = context.metadata.get('solutions', [])
            high_priority = [s for s in solutions if s.get('priority') == 'high']
            if len(high_priority) > 2:
                alerts.append(f"有 {len(high_priority)} 个高优先级问题需要处理")
            
            should_alert = len(alerts) > 0
            
            return AgentResponse(
                success=True,
                data={"should_alert": should_alert, "alerts": alerts},
                message=f"{'需要' if should_alert else '不需要'}发送告警",
                context_updates={"alerts": alerts}
            )
        
        def can_handle(self, context: AgentContext) -> bool:
            return context.analysis_results is not None
    
    # 使用自定义 Agent
    orchestrator = AgentOrchestrator()
    orchestrator.register_agents(
        LogParserAgent(),
        LogAnalystAgent(),
        SolutionAgent(),
        AlertAgent(),  # 自定义 Agent
        ReportAgent()
    )
    
    print("\n注册自定义 AlertAgent，用于检查是否需要发送告警")
    print("执行流程: Parser → Analyst → Solution → Alert → Report")
    
    log_file = "./data/sample_multi_agent.log"
    if os.path.exists(log_file):
        result = orchestrator.execute(log_file)
        
        # 查看告警结果
        alert_result = result.get_agent_result("AlertAgent")
        if alert_result:
            print(f"\n告警检查结果:")
            print(f"  需要告警: {alert_result.data.get('should_alert')}")
            for alert in alert_result.data.get('alerts', []):
                print(f"  - {alert}")


def demo_save_report():
    """演示保存报告"""
    print("\n" + "=" * 70)
    print("演示5: 保存分析报告")
    print("=" * 70)
    
    from agents.orchestrator import quick_analyze_with_agents
    
    log_file = "./data/sample_multi_agent.log"
    if os.path.exists(log_file):
        print(f"\n分析日志: {log_file}")
        result = quick_analyze_with_agents(log_file, verbose=False)
        
        if result.success:
            # 获取 ReportAgent 并保存报告
            report_agent_result = result.get_agent_result("ReportAgent")
            if report_agent_result:
                from agents.report_agent import ReportAgent
                
                # 创建 ReportAgent 实例来保存
                report_agent = ReportAgent()
                
                # 恢复上下文
                from agents.base_agent import AgentContext
                context = AgentContext()
                context.analysis_results = result.agent_results[1].data if len(result.agent_results) > 1 else {}
                context.metadata['solutions'] = result.agent_results[2].data.get('solutions', []) if len(result.agent_results) > 2 else []
                
                report_agent.process(context)
                
                # 保存不同格式
                os.makedirs("./output", exist_ok=True)
                
                md_path = "./output/report.md"
                report_agent.save_report(md_path, "markdown")
                print(f"✓ Markdown 报告已保存: {md_path}")
                
                json_path = "./output/report.json"
                report_agent.save_report(json_path, "json")
                print(f"✓ JSON 报告已保存: {json_path}")
                
                html_path = "./output/report.html"
                report_agent.save_report(html_path, "html")
                print(f"✓ HTML 报告已保存: {html_path}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("LlamaIndex Multi-Agent 日志分析系统 - 演示")
    print("=" * 70)
    
    # 运行所有演示
    demo_basic_workflow()
    demo_conditional_execution()
    demo_tool_usage()
    demo_custom_agent()
    demo_save_report()
    
    print("\n" + "=" * 70)
    print("所有演示执行完成!")
    print("=" * 70)
    print("\n你可以:")
    print("1. 查看生成的报告文件: ./output/")
    print("2. 修改示例日志内容，重新运行演示")
    print("3. 使用自己的日志文件: python multi_agent_demo.py")
    print("4. 查看详细文档: README_MULTI_AGENT.md")


if __name__ == "__main__":
    main()
