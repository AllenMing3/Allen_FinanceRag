"""
Agent Tool-Delegation 规则守卫
检查 agents/ 下的 *_agent.py 是否违规直接调用 LLM / 外部 API / Guard 等业务逻辑。

用法：python scripts/check_agent_delegation.py
退出码：0 = 通过，1 = 有违规
"""
import sys
import pathlib

FORBIDDEN_PATTERNS = [
    "llm.chat",
    "llm.chat_with_tools",
    "HallucinationGuard(",
    "PipelineScoreCard(",
    "tushare_client",
    "dashscope",
]

AGENT_DIR = pathlib.Path("financial_rag/agents")


def main() -> int:
    agent_files = list(AGENT_DIR.glob("*_agent.py"))
    if not agent_files:
        print("agent 目录不存在或无 *_agent.py 文件，跳过检查")
        return 0

    violations = []
    for f in agent_files:
        src = f.read_text(encoding="utf-8")
        # 跳过注释行
        lines = [
            line for line in src.splitlines()
            if not line.strip().startswith("#")
        ]
        active_src = "\n".join(lines)
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in active_src:
                violations.append(f"{f.name}: found '{pattern}'")

    if violations:
        print("Agent tool-delegation rule VIOLATED:")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"Agent delegation guard: {len(agent_files)} agent(s) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
