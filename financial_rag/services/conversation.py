"""
ConversationManager — 多轮对话会话管理

为新闻解读和话题调研提供 follow-up 追问能力:
- Session CRUD (创建/读取/列出/删除)
- 消息历史管理
- Follow-up LLM 调用 (注入原始分析上下文 + 历史消息)
- JSON 文件持久化

用法:
    from financial_rag.services.conversation import ConversationManager
    cm = ConversationManager()
    sid = cm.create_session("news", "茅台新闻分析", initial_analysis, context)
    answer = cm.followup(sid, "这个公司最近有什么风险？", llm)
"""
import json
import os
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# 持久化目录
_CONV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         "data", "knowledge_base", "conversations")


# ===================== 数据模型 =====================


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ConversationSession:
    id: str
    title: str
    type: str  # "news" | "topic"
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationSession":
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            type=d.get("type", "news"),
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            context=d.get("context", {}),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ===================== ConversationManager =====================


class ConversationManager:
    """
    会话管理器 — 管理多个对话 session 的生命周期

    - 内存 + 磁盘双写
    - follow-up 调用: 注入原始新闻 + 分析结果 + 最近消息 → LLM
    """

    def __init__(self, conv_dir: str = _CONV_DIR):
        self._conv_dir = conv_dir
        self._sessions: Dict[str, ConversationSession] = {}
        self._load_all()

    # ---- CRUD ----

    def create_session(
        self,
        session_type: str,
        title: str,
        initial_analysis: str,
        context: Dict[str, Any],
    ) -> str:
        """创建新会话，initial_analysis 作为首条 assistant 消息"""
        session_id = uuid.uuid4().hex[:12]
        session = ConversationSession(
            id=session_id,
            title=title[:50],
            type=session_type,
            context=context,
        )
        if initial_analysis:
            session.messages.append(Message(
                role="assistant",
                content=initial_analysis,
                metadata={"type": "initial_analysis"},
            ))
        self._sessions[session_id] = session
        self._save_session(session)
        logger.info(f"[Conversation] 创建会话: {session_id} ({session_type}: {title})")
        return session_id

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict]:
        """列出所有会话 (按 updated_at 倒序)"""
        result = []
        for s in sorted(self._sessions.values(), key=lambda x: x.updated_at, reverse=True):
            result.append({
                "id": s.id,
                "title": s.title,
                "type": s.type,
                "message_count": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            })
        return result

    def delete_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        path = os.path.join(self._conv_dir, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
        logger.info(f"[Conversation] 删除会话: {session_id}")
        return True

    def add_message(self, session_id: str, role: str, content: str,
                    metadata: Optional[Dict] = None) -> Optional[Message]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        msg = Message(role=role, content=content, metadata=metadata or {})
        session.messages.append(msg)
        session.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_session(session)
        return msg

    # ---- Follow-up LLM ----

    def followup(self, session_id: str, user_message: str, llm) -> Dict[str, Any]:
        """
        追问: 注入会话上下文 + 消息历史 → LLM 生成回答

        Args:
            session_id: 会话 ID
            user_message: 用户追问内容
            llm: DashScopeLLM 实例

        Returns:
            {"answer": str, "elapsed_ms": float}
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"answer": "会话不存在", "error": True}

        # 保存用户消息
        self.add_message(session_id, "user", user_message)

        # 构建 LLM 上下文
        import time
        from financial_rag.llm.caller import LLMCaller

        t0 = time.time()

        system_prompt = self._build_system_prompt(session)
        messages = self._build_messages(session, user_message)

        caller = LLMCaller(llm)
        try:
            resp = caller.call(
                messages=messages,
                system=system_prompt,
                max_tokens=1024,
                temperature=0.3,
            )
            answer = resp.content
        except Exception as e:
            logger.error(f"[Conversation] followup LLM 失败: {e}")
            answer = f"抱歉，分析过程中出现错误: {e}"

        elapsed_ms = (time.time() - t0) * 1000

        # 保存助手回答
        self.add_message(session_id, "assistant", answer,
                         metadata={"elapsed_ms": round(elapsed_ms)})

        return {"answer": answer, "elapsed_ms": round(elapsed_ms)}

    def _build_system_prompt(self, session: ConversationSession) -> str:
        """构建 follow-up 的 system prompt (XML 结构)"""
        ctx = session.context

        # 原始上下文
        context_parts = []
        news_text = ctx.get("news_text", "")
        if news_text:
            context_parts.append(f"<original_news>\n{news_text[:3000]}\n</original_news>")

        structured = ctx.get("structured", {})
        if structured:
            context_parts.append(
                f"<analysis_result>\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n</analysis_result>"
            )

        metrics = ctx.get("metrics", {})
        if metrics:
            context_parts.append(
                f"<extracted_metrics>\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n</extracted_metrics>"
            )

        entities = ctx.get("entities", {})
        if entities:
            context_parts.append(
                f"<extracted_entities>\n{json.dumps(entities, ensure_ascii=False, indent=2)}\n</extracted_entities>"
            )

        kb_sources = ctx.get("kb_sources", [])
        if kb_sources:
            kb_text = "\n".join(
                f"  - [{s.get('source', '?')}] {s.get('text', '')[:200]}"
                for s in kb_sources[:5]
            )
            context_parts.append(f"<kb_context>\n{kb_text}\n</kb_context>")

        context_block = "\n\n".join(context_parts) if context_parts else "(无原始上下文)"

        return f"""你是一位专业的 AI/科技/金融行业分析师。你之前已经对用户提供的新闻或话题进行了深度分析。
现在用户想基于之前的分析结果，追问更多细节或相关问题。

{context_block}

<rules>
1. 基于上面提供的原始新闻和分析结果回答追问，不要编造数据或数字
2. 如果追问的内容在已有信息中找不到依据，明确说明"根据已有信息无法判断"
3. 回答要有具体数据和因果推理，避免"可能""或许"等模糊表述
4. 如果用户问的是相关公司或行业，结合 extracted_entities 和 KB 背景回答
5. 保持专业客观，不做投资建议
</rules>

<output_format>
- 用 Markdown 格式回答
- 关键数据加粗
- 重要结论用 > 引用块
- 如涉及多维度，用列表或表格组织
</output_format>"""

    def _build_messages(self, session: ConversationSession, new_question: str) -> List[Dict]:
        """构建 LLM messages 列表 (最近 6 条历史 + 新问题)"""
        history = session.messages[-6:]  # 最近 3 轮对话

        messages = []
        for msg in history:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

        # 新追问 (add_message 已经存了，但 LLM 需要在这里收到)
        # 注意: 最后一条 history 可能就是刚存入的 user message，避免重复
        if not messages or messages[-1].get("content") != new_question:
            messages.append({"role": "user", "content": new_question})

        return messages

    # ---- 持久化 ----

    def _save_session(self, session: ConversationSession):
        os.makedirs(self._conv_dir, exist_ok=True)
        path = os.path.join(self._conv_dir, f"{session.id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Conversation] 保存会话失败: {session.id} — {e}")

    def _load_all(self):
        if not os.path.isdir(self._conv_dir):
            return
        for filename in os.listdir(self._conv_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self._conv_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = ConversationSession.from_dict(data)
                self._sessions[session.id] = session
            except Exception as e:
                logger.warning(f"[Conversation] 加载会话失败: {filename} — {e}")
        if self._sessions:
            logger.info(f"[Conversation] 加载了 {len(self._sessions)} 个会话")
