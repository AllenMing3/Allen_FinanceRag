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
    compressed_summary: str = ""  # 压缩后的历史摘要
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
            "compressed_summary": self.compressed_summary,
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
            compressed_summary=d.get("compressed_summary", ""),
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

        system_prompt = self._build_system_prompt(session, user_message)
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

    def _build_system_prompt(self, session: ConversationSession, question: str = "") -> str:
        """构建 follow-up 的 system prompt (XML 结构)，带上下文预算裁剪"""
        ctx = self._prune_context(session.context, question)

        context_parts = []
        news_text = ctx.get("news_text", "")
        if news_text:
            context_parts.append(f"<original_news>\n{news_text}\n</original_news>")

        structured = ctx.get("structured", "")
        if structured:
            context_parts.append(f"<analysis_result>\n{structured}\n</analysis_result>")

        metrics = ctx.get("metrics", "")
        if metrics:
            context_parts.append(f"<extracted_metrics>\n{metrics}\n</extracted_metrics>")

        entities = ctx.get("entities", "")
        if entities:
            context_parts.append(f"<extracted_entities>\n{entities}\n</extracted_entities>")

        kb_text = ctx.get("kb_sources", "")
        if kb_text:
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
        """构建 LLM messages 列表 — 压缩摘要 + 最近 N 轮 + 新问题"""
        # 触发压缩: 消息超过阈值时，压缩旧消息为摘要
        self._maybe_compress(session)

        messages = []

        # 注入压缩摘要 (如果有)
        if session.compressed_summary:
            messages.append({
                "role": "system",
                "content": f"[历史对话摘要]\n{session.compressed_summary}"
            })

        # 最近 4 轮对话 (8 条消息) 保留原文
        recent = self._get_recent_messages(session, max_turns=4)
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 新追问 — 避免重复 (add_message 已存入)
        if not messages or messages[-1].get("content") != new_question:
            messages.append({"role": "user", "content": new_question})

        return messages

    # ---- 历史压缩 ----

    _COMPRESS_THRESHOLD = 10   # 消息数 >= 此值时触发压缩
    _KEEP_RECENT_TURNS = 4     # 保留最近 N 轮 (2N 条消息)

    def _maybe_compress(self, session: ConversationSession):
        """当消息超过阈值时，压缩旧消息为摘要"""
        total = len(session.messages)
        if total < self._COMPRESS_THRESHOLD:
            return
        if session.compressed_summary and total <= self._COMPRESS_THRESHOLD + 2:
            # 已经压缩过且新增不多，跳过
            return

        keep_count = self._KEEP_RECENT_TURNS * 2  # 保留的消息条数
        to_compress = session.messages[:-keep_count] if total > keep_count else []
        if not to_compress:
            return

        # 提取待压缩的 Q/A 对
        pairs = self._extract_qa_pairs(to_compress)
        if not pairs:
            return

        # 如果已有旧摘要，合并进去
        existing = session.compressed_summary
        new_summary = self._summarize_pairs(pairs, existing)
        if new_summary:
            session.compressed_summary = new_summary
            self._save_session(session)
            logger.info(
                f"[Conversation] 压缩历史: {len(pairs)} 轮 → "
                f"{len(new_summary)} 字摘要 (会话 {session.id})"
            )

    def _extract_qa_pairs(self, messages: List[Message]) -> List[Dict]:
        """从消息列表中提取 Q/A 对"""
        pairs = []
        i = 0
        while i < len(messages):
            if messages[i].role == "user":
                q = messages[i].content
                a = ""
                if i + 1 < len(messages) and messages[i + 1].role == "assistant":
                    a = messages[i + 1].content
                    i += 1
                if q.strip():
                    pairs.append({"q": q[:200], "a": a[:500]})
            i += 1
        return pairs

    def _summarize_pairs(self, pairs: List[Dict], existing_summary: str = "") -> str:
        """将 Q/A 对压缩为摘要文本 (LLM 优先，规则回退)"""
        # 尝试 LLM 压缩
        try:
            from financial_rag.llm.caller import LLMCaller
            from financial_rag.llm import get_llm
            llm = get_llm()
            if llm:
                caller = LLMCaller(llm)
                pair_text = "\n".join(
                    f"Q: {p['q']}\nA: {p['a'][:300]}" for p in pairs
                )
                if existing_summary:
                    pair_text = f"[已有摘要]\n{existing_summary}\n\n[新增轮次]\n{pair_text}"

                resp = caller.call(
                    messages=[{"role": "user", "content": pair_text}],
                    system=(
                        "将以下对话历史压缩为一段简洁的摘要（200字以内）。"
                        "保留关键结论、数据和用户关注点，去掉寒暄和重复内容。"
                        "直接输出摘要文本，不要加标题。"
                    ),
                    max_tokens=300,
                    temperature=0.1,
                )
                return resp.content.strip()
        except Exception as e:
            logger.debug(f"[Conversation] LLM 压缩失败，使用规则回退: {e}")

        # 规则回退: 提取式摘要
        return self._extractive_summary(pairs, existing_summary)

    def _extractive_summary(self, pairs: List[Dict], existing: str = "") -> str:
        """提取式摘要 — 取每轮 Q + A 首句"""
        parts = []
        if existing:
            parts.append(existing)
        for p in pairs:
            q_short = p["q"][:80]
            # 取 A 的第一行非空内容
            a_lines = [l.strip() for l in p["a"].split("\n") if l.strip()]
            a_short = a_lines[0][:150] if a_lines else "(无回答)"
            parts.append(f"- 问: {q_short} → {a_short}")
        summary = "\n".join(parts)
        # 硬截断防止无限膨胀
        return summary[:2000] if len(summary) > 2000 else summary

    def _get_recent_messages(self, session: ConversationSession,
                             max_turns: int = 4) -> List[Dict]:
        """获取最近 N 轮消息 (2N 条)"""
        count = max_turns * 2
        recent = session.messages[-count:]
        return [
            {"role": m.role, "content": m.content}
            for m in recent
            if m.role in ("user", "assistant")
        ]

    # ---- 上下文裁剪 ----

    def _prune_context(self, ctx: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        """按预算裁剪 system prompt 中的上下文数据"""
        pruned = {}

        # 各段预算 (字符数)
        news_budget = 1500
        struct_budget = 1000
        metrics_budget = 500
        entities_budget = 500
        kb_budget = 500

        news_text = ctx.get("news_text", "")
        if news_text:
            pruned["news_text"] = news_text[:news_budget]

        structured = ctx.get("structured", {})
        if structured:
            text = json.dumps(structured, ensure_ascii=False, indent=2)
            pruned["structured"] = text[:struct_budget]

        metrics = ctx.get("metrics", {})
        if metrics:
            text = json.dumps(metrics, ensure_ascii=False, indent=2)
            pruned["metrics"] = text[:metrics_budget]

        entities = ctx.get("entities", {})
        if entities:
            text = json.dumps(entities, ensure_ascii=False, indent=2)
            pruned["entities"] = text[:entities_budget]

        kb_sources = ctx.get("kb_sources", [])
        if kb_sources:
            lines = []
            total = 0
            for s in kb_sources[:5]:
                line = f"  - [{s.get('source', '?')}] {s.get('text', '')[:200]}"
                if total + len(line) > kb_budget:
                    break
                lines.append(line)
                total += len(line)
            pruned["kb_sources"] = "\n".join(lines)

        return pruned

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
            logger.debug(f"[Conversation] 加载了 {len(self._sessions)} 个会话")
