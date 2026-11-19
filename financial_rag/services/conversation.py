"""
ConversationManager — 多轮对话会话管理（案件档案存储层）

为新闻解读和话题调研提供 follow-up 追问的档案与历史管理:
- Session CRUD (创建/读取/列出/删除)
- 消息历史管理 + 历史压缩
- JSON 文件持久化（state 层：可删、不入库）

追问的业务逻辑（KB 检索 + 证据组装 + LLM + Guard）在
tools/conversation_tools.py 的 answer_followup 工具中，本模块不直接调 LLM。

用法:
    from financial_rag.services.conversation import ConversationManager
    cm = ConversationManager()
    sid = cm.create_session("news", "茅台新闻分析", initial_analysis, context)
    msgs = cm._build_messages(cm.get_session(sid), "追问内容", llm=llm)
"""
import json
import os
import threading
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


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
    compressed_upto: int = 0      # 水位线：前 N 条消息已压缩进摘要（档案不删，只缩 LLM 视野）
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
            "compressed_upto": self.compressed_upto,
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
            compressed_upto=d.get("compressed_upto", 0),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ===================== ConversationManager =====================


class ConversationManager:
    """
    会话管理器 — 管理多个对话 session 的生命周期

    - 内存 + 磁盘双写（默认落 config.state_dir/conversations，跟随 KB 分层）
    - 历史消息 + 压缩摘要，供 answer_followup 工具组装 LLM 上下文
    - 档案全量保留，LLM 视野分层压缩：水位线之前的旧轮次进摘要，
      最近 N 轮保留原文；磁盘/UI 永远持有完整历史
    """

    def __init__(self, conv_dir: Optional[str] = None):
        if conv_dir is None:
            # 惰性读 config：会话属运行状态层（state），与 KB 分层重构对齐
            from financial_rag.config import config
            conv_dir = os.path.join(config.state_dir, "conversations")
        self._conv_dir = conv_dir
        self._sessions: Dict[str, ConversationSession] = {}
        # 会话级锁：并发追问下消息追加/整文件覆写串行化（RLock 允许嵌套保存）
        self._locks: Dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._load_all()

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            if session_id not in self._locks:
                self._locks[session_id] = threading.RLock()
            return self._locks[session_id]

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
        with self._locks_guard:
            self._locks.pop(session_id, None)
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
        with self._lock_for(session_id):
            session.messages.append(msg)
            session.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_session(session)
        return msg

    # ---- 历史消息组装（供 answer_followup 工具调用） ----

    def _build_messages(self, session: ConversationSession, new_question: str,
                        llm=None) -> List[Dict]:
        """构建 LLM messages 列表 — 压缩摘要 + 最近 N 轮 + 新问题

        Args:
            llm: 可选 LLM 实例，用于历史压缩；None 时压缩走提取式回退
        """
        # 触发压缩: 消息超过阈值时，压缩旧消息为摘要
        self._maybe_compress(session, llm=llm)

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

    # ---- 历史压缩（水位线增量：档案不删，只缩 LLM 视野） ----

    _COMPRESS_THRESHOLD = 10   # 消息数 >= 此值时触发压缩
    _KEEP_RECENT_TURNS = 4     # 保留最近 N 轮 (2N 条消息)
    _SUMMARY_MAX_CHARS = 2000  # 摘要硬截断：LLM 与提取式回退同口径

    def _maybe_compress(self, session: ConversationSession, llm=None):
        """水位线增量压缩：只总结 compressed_upto 之后的新增旧轮次。

        旧实现每次把已压缩的旧消息再喂一遍 → 摘要反复嵌套失真；
        水位线推进后同一批消息绝不重复总结，无新增时天然幂等。
        """
        if len(session.messages) < self._COMPRESS_THRESHOLD:
            return
        keep_count = self._KEEP_RECENT_TURNS * 2  # 保留的消息条数
        with self._lock_for(session.id):
            total = len(session.messages)
            start = session.compressed_upto
            end = total - keep_count
            if end <= start:
                return  # 无新增可压缩内容
            to_compress = session.messages[start:end]

            # 提取待压缩的 Q/A 对
            pairs = self._extract_qa_pairs(to_compress)
            if not pairs:
                # 无 Q/A（如仅 initial_analysis）：推进水位线避免卡死，
                # 其内容已由 system prompt 结论 + grounding 覆盖
                session.compressed_upto = end
                self._save_session(session)
                return

            # 旧摘要作为 [已有摘要] 与新增轮次合并总结
            new_summary = self._summarize_pairs(
                pairs, session.compressed_summary, llm=llm)
            if new_summary:
                session.compressed_summary = new_summary
                session.compressed_upto = end
                self._save_session(session)
                logger.info(
                    f"[Conversation] 压缩历史: {len(pairs)} 轮 → "
                    f"{len(new_summary)} 字摘要, 水位线 {end} (会话 {session.id})"
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

    def _summarize_pairs(self, pairs: List[Dict], existing_summary: str = "",
                         llm=None) -> str:
        """将 Q/A 对压缩为摘要文本 (注入 llm 优先，无 llm 规则回退)"""
        # 尝试 LLM 压缩（仅使用注入实例，不另起全局 get_llm，避免 mock/测试行为分裂）
        if llm is not None:
            try:
                from financial_rag.llm.caller import LLMCaller
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
                return resp.content.strip()[:self._SUMMARY_MAX_CHARS]
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
        # 硬截断防止无限膨胀（与 LLM 路径同口径）
        return summary[:self._SUMMARY_MAX_CHARS]

    def _get_recent_messages(self, session: ConversationSession,
                             max_turns: int = 4) -> List[Dict]:
        """获取最近 N 轮消息 (2N 条)；initial_analysis 出窗（system prompt
        已含结构化结论 + grounding 原始新闻，避免同一信息三重冗余）。"""
        count = max_turns * 2
        recent = session.messages[-count:]
        return [
            {"role": m.role, "content": m.content}
            for m in recent
            if m.role in ("user", "assistant")
            and m.metadata.get("type") != "initial_analysis"
        ]

    # ---- 压缩可观测 ----

    def history_info(self, session: ConversationSession) -> Dict[str, Any]:
        """历史压缩诊断快照 — 供工具层透出，水位/摘要规模当场可查。"""
        return {
            "total_messages": len(session.messages),
            "compressed_upto": session.compressed_upto,
            "summary_chars": len(session.compressed_summary),
            "recent_window": self._KEEP_RECENT_TURNS * 2,
        }

    # ---- 持久化 ----

    def _save_session(self, session: ConversationSession):
        os.makedirs(self._conv_dir, exist_ok=True)
        path = os.path.join(self._conv_dir, f"{session.id}.json")
        try:
            with self._lock_for(session.id):
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
