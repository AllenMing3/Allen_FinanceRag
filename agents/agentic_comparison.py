"""
传统 Agent vs Agentic RAG Agent 对比演示

目的：展示 Agentic RAG 的核心改进
"""
from typing import List, Dict, Any


# ==================== 传统 Agent 流程 ====================

def traditional_agent_flow(user_query: str, knowledge_base_results: List[Dict]) -> str:
    """
    传统 Agent 的处理流程

    特点：
    1. 一次检索就完事
    2. 没有反思机制
    3. 直接基于检索结果回答
    4. 不知道答案质量如何
    """
    print("\n" + "=" * 50)
    print("传统 Agent 流程")
    print("=" * 50)

    # 步骤1：用户提问
    print(f"\n[1] 用户提问: {user_query}")

    # 步骤2：一次检索
    print(f"\n[2] 执行检索...")
    results = knowledge_base_results  # 假设这是检索结果
    print(f"    检索到 {len(results)} 条结果")

    # 步骤3：直接回答（没有反思）
    print(f"\n[3] 生成回答（无反思）")
    answer = f"根据检索结果：{results[0]['text'][:100]}..."
    print(f"    {answer}")

    # 步骤4：返回答案（不知道质量如何）
    print(f"\n[4] 返回答案")
    print(f"    问题：无法评估回答质量")

    return answer


# ==================== Agentic RAG Agent 流程 ====================

def agentic_rag_flow(user_query: str, knowledge_base_results: List[Dict]) -> str:
    """
    Agentic RAG Agent 的处理流程

    特点：
    1. 多轮检索，根据需要检索
    2. 有反思机制
    3. 评估答案质量
    4. 支持查询优化
    """
    print("\n" + "=" * 50)
    print("Agentic RAG Agent 流程")
    print("=" * 50)

    # ReAct 循环
    retrieval_count = 0
    all_results = []
    confidence = 0.0
    step = 0

    while retrieval_count < 3 and confidence < 0.7 and step < 5:
        step += 1
        print(f"\n--- ReAct 步骤 {step} ---")

        # THINK: 思考当前状态
        if retrieval_count == 0:
            thought = "首先检索错误相关信息"
            query = "错误异常信息"
        elif retrieval_count == 1:
            thought = "检索到一些结果，现在检索解决方案"
            query = "错误解决方法"
        else:
            thought = "已有足够信息，评估是否需要更多检索"
            query = None

        print(f"[Think] {thought}")

        if query:
            # RETRIEVE: 检索
            print(f"[Retrieve] 查询: {query}")
            # 模拟检索
            step_results = knowledge_base_results[retrieval_count:retrieval_count+2]
            all_results.extend(step_results)
            retrieval_count += 1
            print(f"    检索到 {len(step_results)} 条新结果")

            # OBSERVE: 观察结果
            print(f"[Observe] 累计 {len(all_results)} 条结果")

        # JUDGE: 判断是否继续
        if retrieval_count == 2:
            confidence = 0.75
            print(f"[Judge] 置信度: {confidence:.2f} >= 0.7，停止")
            break
        else:
            print(f"[Judge] 置信度不足，继续检索")

    # 最终综合
    print(f"\n[Synthesize] 综合 {len(all_results)} 条结果生成答案")

    # 质量评估
    print(f"\n[评估] 置信度: {confidence:.2f}")
    if confidence >= 0.7:
        print("    [OK] 回答质量良好")
    else:
        print("    [WARN] 建议人工复核")

    return "综合分析结果..."


# ==================== 对比演示 ====================

def demo_comparison():
    """演示对比"""
    print("\n" + "=" * 70)
    print("传统 Agent vs Agentic RAG Agent 对比")
    print("=" * 70)

    # 模拟检索结果
    mock_results = [
        {"text": "错误1: 数据库连接超时", "score": 0.9},
        {"text": "错误2: 内存溢出", "score": 0.8},
        {"text": "解决方法: 增加连接池", "score": 0.7},
        {"text": "类似案例: 用户A的问题", "score": 0.6},
    ]

    user_query = "系统报错：连接超时和内存溢出"

    # 传统 Agent
    traditional_agent_flow(user_query, mock_results)

    # Agentic RAG Agent
    agentic_rag_flow(user_query, mock_results)

    # 对比总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)

    comparison = """
    | 维度         | 传统 Agent         | Agentic RAG Agent      |
    |-------------|-------------------|------------------------|
    | 检索次数     | 1次               | 多次（根据需要）        |
    | 查询优化     | 无                | 支持（多角度查询）       |
    | 反思机制     | 无                | 有（Judge 阶段）        |
    | 质量评估     | 无                | 有（置信度评估）         |
    | 答案生成     | 直接生成           | 综合多次检索结果         |
    | 适用场景     | 简单问答           | 复杂问题分析             |

    Agentic RAG 的核心价值：
    1. 不是"一次检索就完事"，而是根据需要多次检索
    2. 反思机制让 Agent 知道"自己不知道什么"
    3. 查询优化让检索更精准
    4. 置信度评估让回答更有保障
    """
    print(comparison)


if __name__ == "__main__":
    demo_comparison()
