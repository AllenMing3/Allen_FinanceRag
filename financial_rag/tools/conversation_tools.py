"""
追问问答工具 — 开卷追问（第一期质量闭环）

新闻解读/话题调研后的追问不再是"闭卷硬答":
- 用追问本身做 KB 检索，供给新证据（复用主分析 _search_kb，诊断恒返回）
- 案件档案(原始新闻 + 存档 KB) + 新检索 → ContextBuilder 统一组装
- LLM 与 Guard 同源（source_items 同一份），回答强制 [N] 引用并过 L1-L4
- 块编号 1-based，与 Guard L3 引用契约 (1 <= idx <= n_sources) 对齐

业务逻辑在工具；ConversationManager 只负责档案存储与历史管理。
"""
import json
import logging
import re
import time
from typing import Any, Dict, List

from financial_rag.tools.core import FunctionDef

logger = logging.getLogger(__name__)

# 依赖注入引用（cm 在 app_state 中晚于 registry 工厂创建，需二次全量注入）
_conv_refs: Dict[str, Any] = {"llm": None, "retriever": None, "kb_built": False, "cm": None}


def inject_conversation_deps(llm=None, retriever=None, kb_built: bool = False,
                             conversation_manager=None):
    """注入 LLM / Retriever / ConversationManager 供追问工具使用"""
    _conv_refs["llm"] = llm
    _conv_refs["retriever"] = retriever
    _conv_refs["kb_built"] = kb_built
    _conv_refs["cm"] = conversation_manager


# ===================== 证据组装（纯函数，可单测） =====================


def assemble_followup_context(session_context: Dict[str, Any],
                              new_results: List[Dict[str, Any]]):
    """案件档案 + 新检索 → ContextBuilder 统一组装。

    - 档案项: news_text (score=1.0 排首) + 建会话时的 kb_sources
    - 按 text 内容去重（新检索可能命中档案里已有的同一篇）
    - 块编号 0-based → 1-based，对齐 L3 引用契约；source_items 顺序不变
    """
    from financial_rag.context_builder import ContextBuilder

    items: List[Dict[str, Any]] = []
    seen = set()

    def _add(text: str, source: str, score: float):
        text = (text or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        items.append({"text": text, "source": source, "score": score})

    _add(session_context.get("news_text", ""), "原始新闻", 1.0)
    for s in session_context.get("kb_sources", []) or []:
        _add(s.get("text", ""), s.get("source", "知识库"),
             float(s.get("score", 0.5) or 0.5))
    for r in new_results or []:
        _add(r.get("text", ""), r.get("source", "知识库"),
             float(r.get("score", 0) or 0))

    assembled = ContextBuilder.for_path("quick_qa").build(items)
    # ContextBuilder 块标号 0-based；Guard L3 引用契约 1-based → 行首标号 +1
    assembled.grounding_text = re.sub(
        r"(?m)^\[(\d+)\]",
        lambda m: f"[{int(m.group(1)) + 1}]",
        assembled.grounding_text,
    )
    return assembled


def check_followup_answer(answer: str, assembled) -> Dict[str, Any]:
    """对追问回答跑 L1-L4 Guard（与 LLM 证据同源），返回序列化报告。

    统一走 ReliabilityService：空回答/空证据 → 显式 skipped（未校验）。
    """
    from financial_rag.guard.service import ReliabilityService
    from financial_rag.guard.serializer import serialize_guard_result

    guard_result = ReliabilityService().check(
        answer, assembled.source_items, mode="rag")
    return serialize_guard_result(guard_result)


# ===================== 追问主流程 =====================


def answer_followup(session_id: str, message: str) -> Dict[str, Any]:
    """对已有新闻解读/话题调研会话进行开卷追问: KB 检索 + 档案组装 + LLM 生成 + 防幻觉校验。

    Args:
        session_id: 会话 ID
        message: 用户追问内容
    """
    from financial_rag.observability.tracing import observe_span

    cm = _conv_refs.get("cm")
    if cm is None:
        return {"error": "ConversationManager 未初始化"}
    session = cm.get_session(session_id)
    if session is None:
        return {"error": f"会话不存在: {session_id}"}
    if not (message or "").strip():
        return {"error": "追问内容为空"}

    with observe_span("conversation.followup",
                      attributes={"session_id": session_id}):
        return _followup_impl(cm, session, session_id, message)


def _followup_impl(cm, session, session_id: str, message: str) -> Dict[str, Any]:
    from financial_rag.llm.caller import LLMCaller
    from financial_rag.prompts import FOLLOWUP_ANSWER_SYSTEM
    from financial_rag.services.analysis import _search_kb

    t0 = time.time()

    # 1. 用户消息先落盘
    cm.add_message(session_id, "user", message)

    # 2. KB 检索（追问本身供新证据；失败/为空不阻断，仅用档案证据）
    new_results, kb_search_info = _search_kb(
        _conv_refs.get("retriever"), message, _conv_refs.get("kb_built", False))
    logger.info(f"[followup] 检索: {kb_search_info.get('total_results', 0)} 召回, "
                f"{kb_search_info.get('above_threshold', 0)} 过阈值")

    llm = _conv_refs.get("llm")
    if llm is None:
        return {"error": "追问需要 LLM（未配置 DASHSCOPE_API_KEY）",
                "kb_search_info": kb_search_info}

    # 3. 证据组装（档案 + 新检索，LLM/Guard 同源）
    assembled = assemble_followup_context(session.context, new_results)
    logger.info(f"[followup] 组装: {assembled.trace.included_items} 项, "
                f"保真率 {assembled.trace.fidelity:.0%}")

    # 4. 生成（结构化结论 + 证据 + 压缩历史）
    structured = session.context.get("structured", {})
    system = FOLLOWUP_ANSWER_SYSTEM.format(
        assessment=session.context.get("assessment", "") or "未知",
        confidence=session.context.get("confidence", "") or "未知",
        structured=json.dumps(structured, ensure_ascii=False) if structured else "无",
        grounding=assembled.grounding_text or "(无可用证据)",
    )
    messages = cm._build_messages(session, message, llm=llm)
    try:
        resp = LLMCaller(llm).call(messages=messages, system=system,
                                   max_tokens=1024, temperature=0.3)
        answer = resp.content
    except Exception as e:
        logger.error(f"[followup] LLM 失败: {e}")
        return {"error": f"分析过程中出现错误: {e}",
                "kb_search_info": kb_search_info}

    # 5. L1-L4 校验（与证据同源）
    hallucination = check_followup_answer(answer, assembled)
    elapsed_ms = round((time.time() - t0) * 1000)
    logger.info(f"[followup] 完成: {elapsed_ms}ms, "
                f"guard risk={hallucination.get('risk')}")

    # 6. 助手回答落盘（带诊断摘要 + 完整证据，供 eval/backflow 回流取样）
    cm.add_message(session_id, "assistant", answer, metadata={
        "elapsed_ms": elapsed_ms,
        "guard_risk": "skipped" if hallucination.get("skipped") else hallucination.get("risk", ""),
        "guard_score": hallucination.get("overall_score", 0),
        "source_count": len(assembled.source_items),
        "sources": assembled.source_items,
    })

    return {
        "answer": answer,
        "elapsed_ms": elapsed_ms,
        "sources": assembled.source_items,
        "hallucination": hallucination,
        "kb_search_info": kb_search_info,
        "context_trace": assembled.trace.to_dict(),
    }


# ===================== 工具定义 =====================

CONVERSATION_TOOLS = [
    FunctionDef(
        name="answer_followup",
        description="对已有新闻解读/话题调研会话进行开卷追问: 检索知识库补充新证据，"
                    "结合案件档案生成带 [N] 引用的回答，并经 L1-L4 防幻觉校验。",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "会话 ID"},
                "message": {"type": "string", "description": "用户追问内容"},
            },
            "required": ["session_id", "message"],
        },
        callback=answer_followup,
        category="analysis",
        timeout_sec=60.0,
        tags=["追问", "对话", "问答"],
    ),
]