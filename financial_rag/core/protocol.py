"""
统一 Agent 间交互协议 — AgentMessage + MessageBus + MessageAdapter

设计目标:
1. 显式的消息协议，Agent 之间通过标准消息通信，而不是隐式 dict
2. 上游 Agent 写了什么字段，通过消息 payload 显式声明
3. 通过 parent_msg_ids 形成 DAG，出错时可追溯完整数据链路
4. 通过 ttl 和 msg_type 区分强依赖（必须等）与弱依赖（可选）
5. 通过 MessageAdapter 桥接现有 AgentResult.context_updates，保持向后兼容
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict
import uuid
import time


# ===================== 1. AgentMessage =====================

@dataclass
class AgentMessage:
    """
    Agent 间通信的标准消息体。

    Attributes:
        msg_id: UUID 唯一标识
        sender: 发送 Agent 名称
        receiver: 接收方名称 ("all" 表示广播给所有 Agent)
        msg_type: 消息类型:
            - "data": 携带业务数据
            - "request": 向其他 Agent 请求数据
            - "error": 错误信息
            - "ack": 确认收到
            - "done": 标记处理完成
        payload: 实际数据载荷
        parent_msg_ids: 上游消息 ID 列表，形成 DAG 链路
        timestamp: 消息创建时间戳 (time.time())
        ttl: 生存时间（秒），0 表示不过期，>0 表示过期秒数
    """
    msg_id: str
    sender: str
    receiver: str
    msg_type: str
    payload: Dict[str, Any]
    parent_msg_ids: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    ttl: int = 0

    @staticmethod
    def create(
        sender: str,
        receiver: str,
        msg_type: str,
        payload: Dict[str, Any],
        parent_msg_ids: Optional[List[str]] = None,
        ttl: int = 0,
    ) -> "AgentMessage":
        """
        工厂方法：创建一个带有自动生成 msg_id 和时间戳的 AgentMessage。

        Args:
            sender: 发送方 Agent 名称
            receiver: 接收方 ("all" 表示广播)
            msg_type: 消息类型 ("data" | "request" | "error" | "ack" | "done")
            payload: 数据载荷
            parent_msg_ids: 上游消息 ID 列表
            ttl: 生存时间（秒）

        Returns:
            新的 AgentMessage 实例
        """
        return AgentMessage(
            msg_id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            parent_msg_ids=parent_msg_ids or [],
            timestamp=time.time(),
            ttl=ttl,
        )

    def is_expired(self) -> bool:
        """检查消息是否已过期（基于 ttl）。ttl=0 永不过期。"""
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """将消息序列化为字典，便于 JSON 序列化和日志记录。"""
        return {
            "msg_id": self.msg_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "msg_type": self.msg_type,
            "payload": self.payload,
            "parent_msg_ids": self.parent_msg_ids,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }

    def to_log(self) -> str:
        """
        生成人类可读的单行日志。

        Returns:
            格式化的日志字符串，包含发送方、接收方、消息类型和 payload 键名。
        """
        payload_keys = list(self.payload.keys())
        parents = self.parent_msg_ids[:3]  # 最多展示 3 个上游
        parent_str = " → ".join(parents) if parents else "(root)"
        return (
            f"[{self.msg_type.upper():5s}] {self.sender} → {self.receiver} "
            f"| msg={self.msg_id[:8]}... | parent={parent_str} "
            f"| keys={payload_keys}"
        )

    def __repr__(self) -> str:
        return self.to_log()


# ===================== 2. MessageBus =====================

class MessageBus:
    """
    消息总线：在 AgentOrchestrator 级别管理消息流。

    职责:
    - 接收并存储所有 AgentMessage
    - 支持按接收方、消息类型过滤消费
    - 支持基于 parent_msg_ids 追溯完整数据链路
    - 支持快照导出（调试/审计）

    Usage::

        bus = MessageBus()
        msg = AgentMessage.create("IngestionAgent", "all", "data", {"parsed_data": ...})
        bus.publish(msg)
        msgs = bus.consume("ExtractionAgent", msg_type="data")
        chain = bus.get_chain(msg.msg_id)
    """

    def __init__(self):
        """初始化空的消息总线。"""
        self._messages: Dict[str, AgentMessage] = {}
        self._inbox: Dict[str, List[str]] = defaultdict(list)  # receiver -> [msg_id, ...]
        self._outbox: Dict[str, List[str]] = defaultdict(list)  # sender -> [msg_id, ...]

    def publish(self, msg: AgentMessage) -> str:
        """
        发布一条消息到总线。

        Args:
            msg: 要发布的 AgentMessage 实例

        Returns:
            消息的 msg_id
        """
        self._messages[msg.msg_id] = msg
        self._outbox[msg.sender].append(msg.msg_id)

        if msg.receiver == "all":
            # 广播：所有已知的 receiver 都会收到
            for receiver in self._inbox:
                self._inbox[receiver].append(msg.msg_id)
            # 也记录到特殊的 "all" 槽位，方便后续 consume("all")
            self._inbox["all"].append(msg.msg_id)
        else:
            self._inbox[msg.receiver].append(msg.msg_id)

        return msg.msg_id

    def consume(
        self,
        receiver: str,
        msg_type: Optional[str] = None,
        remove: bool = False,
    ) -> List[AgentMessage]:
        """
        消费指定接收方的消息。

        Args:
            receiver: 接收方 Agent 名称（或 "all" 获取广播消息）
            msg_type: 可选，按消息类型过滤（"data" | "request" | "error" | "ack" | "done"）
            remove: 是否在消费后移除消息（默认 False，消息可被多次消费）

        Returns:
            匹配的消息列表（按时间戳排序），自动过滤已过期消息
        """
        msg_ids = self._inbox.get(receiver, [])
        result = []

        for mid in msg_ids:
            msg = self._messages.get(mid)
            if msg is None:
                continue
            if msg.is_expired():
                continue
            if msg_type and msg.msg_type != msg_type:
                continue
            result.append(msg)

        if remove:
            removed_ids = {m.msg_id for m in result}
            self._inbox[receiver] = [mid for mid in msg_ids if mid not in removed_ids]

        # 按时间戳排序
        result.sort(key=lambda m: m.timestamp)
        return result

    def get_chain(self, msg_id: str) -> List[AgentMessage]:
        """
        追溯从根节点到指定消息的完整上游链路（BFS 向上遍历 parent_msg_ids）。

        Args:
            msg_id: 目标消息 ID

        Returns:
            从根到目标消息的有序列表（最早在前）
        """
        if msg_id not in self._messages:
            return []

        visited = set()
        chain = []

        def _dfs(mid: str):
            """后序遍历收集上游，保证根在前。"""
            if mid in visited or mid not in self._messages:
                return
            visited.add(mid)
            msg = self._messages[mid]
            for parent_id in msg.parent_msg_ids:
                _dfs(parent_id)
            chain.append(msg)

        _dfs(msg_id)
        return chain

    def get_downstream(self, msg_id: str) -> List[AgentMessage]:
        """
        查找指定消息的所有下游消息（哪些消息引用了它作为 parent）。

        Args:
            msg_id: 上游消息 ID

        Returns:
            下游消息列表
        """
        downstream = []
        for msg in self._messages.values():
            if msg_id in msg.parent_msg_ids:
                downstream.append(msg)
        downstream.sort(key=lambda m: m.timestamp)
        return downstream

    def snapshot(self) -> Dict[str, Any]:
        """
        获取当前总线状态的快照，用于调试/审计。

        Returns:
            包含消息数量、inbox/outbox 统计等信息的字典。
        """
        return {
            "total_messages": len(self._messages),
            "inbox_stats": {k: len(v) for k, v in self._inbox.items()},
            "outbox_stats": {k: len(v) for k, v in self._outbox.items()},
            "msg_types": self._count_by_type(),
        }

    def _count_by_type(self) -> Dict[str, int]:
        """统计各消息类型的数量。"""
        counts: Dict[str, int] = defaultdict(int)
        for msg in self._messages.values():
            counts[msg.msg_type] += 1
        return dict(counts)

    def clear(self):
        """清空总线中的所有消息和路由表。"""
        self._messages.clear()
        self._inbox.clear()
        self._outbox.clear()

    def __len__(self) -> int:
        return len(self._messages)


# ===================== 3. MessageAdapter =====================

class MessageAdapter:
    """
    适配器：桥接现有 AgentResult.context_updates 与 AgentMessage 协议。

    提供两个方向的转换:
    - from_agent_result: 将 AgentResult.context_updates 拆分为 AgentMessage 列表
    - to_context_updates: 将 AgentMessage 列表合并回 context_updates dict

    这样可以在不修改现有 Agent 代码的前提下，接入 MessageBus 协议。
    """

    @staticmethod
    def from_agent_result(
        result: "AgentResult",
        parent_msg_ids: Optional[List[str]] = None,
    ) -> List[AgentMessage]:
        """
        将 AgentResult.context_updates 转换为 AgentMessage 列表。

        每个 context_update 的 key-value 对生成一条 "data" 类型的消息，
        接收方为 "all"（广播），下游 Agent 可以按需消费。

        Args:
            result: 现有的 AgentResult 对象
            parent_msg_ids: 上游消息 ID 列表（可选）

        Returns:
            AgentMessage 列表，每个 context_update 对应一条消息
        """
        messages = []
        for key, value in result.context_updates.items():
            msg = AgentMessage.create(
                sender=result.agent_name,
                receiver="all",
                msg_type="data",
                payload={"key": key, "value": value},
                parent_msg_ids=parent_msg_ids or [],
            )
            messages.append(msg)

        # 额外发送一条 done 消息标记此 Agent 完成
        done_msg = AgentMessage.create(
            sender=result.agent_name,
            receiver="all",
            msg_type="done",
            payload={
                "success": result.success,
                "message": result.message,
                "execution_time": result.execution_time,
            },
            parent_msg_ids=[m.msg_id for m in messages] if messages else (parent_msg_ids or []),
        )
        messages.append(done_msg)
        return messages

    @staticmethod
    def to_context_updates(
        messages: List[AgentMessage],
        context: "AgentContext",
    ) -> Dict[str, Any]:
        """
        将 AgentMessage 列表合并为 context_updates 字典。

        只提取 msg_type="data" 的消息，将其 payload 中的 key-value 对
        应用到 context 的对应字段上。

        Args:
            messages: 要合并的消息列表
            context: 当前的 AgentContext（用于校验字段是否存在）

        Returns:
            Dict[str, Any] 可以传入 AgentResult.context_updates
        """
        updates: Dict[str, Any] = {}
        for msg in messages:
            if msg.msg_type != "data":
                continue
            key = msg.payload.get("key")
            value = msg.payload.get("value")
            if key is not None:
                if hasattr(context, key):
                    updates[key] = value
                else:
                    # 不存在的属性放入 metadata
                    if "metadata" not in updates:
                        updates["metadata"] = {}
                    updates["metadata"][key] = value
        return updates
