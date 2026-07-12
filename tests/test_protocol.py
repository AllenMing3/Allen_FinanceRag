"""
Test core/protocol.py — AgentMessage + MessageBus + MessageAdapter

Pure data structure tests, no mocking needed.
"""
import time
import pytest

from financial_rag.core.protocol import AgentMessage, MessageBus, MessageAdapter
from financial_rag.core.base import AgentResult, AgentContext


# ===================== AgentMessage =====================


class TestAgentMessage:

    def test_create_auto_id_and_timestamp(self):
        msg = AgentMessage.create("A", "B", "data", {"key": "val"})
        assert len(msg.msg_id) == 36  # UUID format
        assert msg.timestamp > 0
        assert msg.sender == "A"
        assert msg.receiver == "B"
        assert msg.msg_type == "data"
        assert msg.payload == {"key": "val"}
        assert msg.parent_msg_ids == []
        assert msg.ttl == 0

    def test_create_with_parent_ids(self):
        msg = AgentMessage.create("A", "B", "data", {}, parent_msg_ids=["p1", "p2"])
        assert msg.parent_msg_ids == ["p1", "p2"]

    def test_is_expired_ttl_zero_never(self):
        msg = AgentMessage.create("A", "B", "data", {}, ttl=0)
        assert msg.is_expired() is False

    def test_is_expired_ttl_future(self):
        msg = AgentMessage.create("A", "B", "data", {}, ttl=3600)
        assert msg.is_expired() is False

    def test_is_expired_ttl_past(self):
        msg = AgentMessage(
            msg_id="x", sender="A", receiver="B", msg_type="data",
            payload={}, timestamp=time.time() - 100, ttl=10,
        )
        assert msg.is_expired() is True

    def test_to_dict(self):
        msg = AgentMessage.create("A", "B", "data", {"k": 1})
        d = msg.to_dict()
        assert d["sender"] == "A"
        assert d["receiver"] == "B"
        assert d["msg_type"] == "data"
        assert d["payload"] == {"k": 1}
        assert "msg_id" in d
        assert "timestamp" in d

    def test_to_log_format(self):
        msg = AgentMessage.create("Ingestion", "Analysis", "data", {"text": "hello"})
        log = msg.to_log()
        assert "DATA" in log
        assert "Ingestion" in log
        assert "Analysis" in log
        assert "(root)" in log  # no parents

    def test_to_log_with_parents(self):
        msg = AgentMessage.create("A", "B", "done", {}, parent_msg_ids=["p1", "p2", "p3", "p4"])
        log = msg.to_log()
        assert "p1" in log
        assert "p3" in log
        assert "p4" not in log  # max 3 parents shown

    def test_repr_equals_to_log(self):
        msg = AgentMessage.create("A", "B", "data", {})
        assert repr(msg) == msg.to_log()


# ===================== MessageBus =====================


class TestMessageBus:

    def test_publish_and_len(self):
        bus = MessageBus()
        msg = AgentMessage.create("A", "B", "data", {"x": 1})
        bus.publish(msg)
        assert len(bus) == 1

    def test_consume_by_receiver(self):
        bus = MessageBus()
        msg = AgentMessage.create("A", "B", "data", {"x": 1})
        bus.publish(msg)
        msgs = bus.consume("B")
        assert len(msgs) == 1
        assert msgs[0].payload == {"x": 1}

    def test_consume_empty_receiver(self):
        bus = MessageBus()
        msgs = bus.consume("nobody")
        assert msgs == []

    def test_consume_filter_by_type(self):
        bus = MessageBus()
        bus.publish(AgentMessage.create("A", "B", "data", {"k": 1}))
        bus.publish(AgentMessage.create("A", "B", "done", {"ok": True}))
        data_msgs = bus.consume("B", msg_type="data")
        assert len(data_msgs) == 1
        assert data_msgs[0].msg_type == "data"

    def test_consume_with_remove(self):
        bus = MessageBus()
        bus.publish(AgentMessage.create("A", "B", "data", {}))
        msgs = bus.consume("B", remove=True)
        assert len(msgs) == 1
        assert bus.consume("B") == []

    def test_consume_filters_expired(self):
        bus = MessageBus()
        expired = AgentMessage(
            msg_id="exp", sender="A", receiver="B", msg_type="data",
            payload={}, timestamp=time.time() - 100, ttl=10,
        )
        fresh = AgentMessage.create("A", "B", "data", {"fresh": True}, ttl=3600)
        bus.publish(expired)
        bus.publish(fresh)
        msgs = bus.consume("B")
        assert len(msgs) == 1
        assert msgs[0].payload == {"fresh": True}

    def test_broadcast_to_all(self):
        bus = MessageBus()
        # Pre-populate inbox for C
        bus.publish(AgentMessage.create("X", "C", "data", {}))
        # Now broadcast
        bus.publish(AgentMessage.create("A", "all", "data", {"broadcast": True}))
        all_msgs = bus.consume("all")
        assert len(all_msgs) == 1
        assert all_msgs[0].payload == {"broadcast": True}
        # C should also receive the broadcast
        c_msgs = bus.consume("C")
        assert len(c_msgs) == 2  # 1 direct + 1 broadcast

    def test_get_chain_upstream_tracing(self):
        bus = MessageBus()
        root = AgentMessage.create("A", "B", "data", {"stage": 1})
        bus.publish(root)
        child = AgentMessage.create("B", "C", "data", {"stage": 2}, parent_msg_ids=[root.msg_id])
        bus.publish(child)
        grandchild = AgentMessage.create("C", "D", "done", {"stage": 3}, parent_msg_ids=[child.msg_id])
        bus.publish(grandchild)

        chain = bus.get_chain(grandchild.msg_id)
        assert len(chain) == 3
        stages = [m.payload["stage"] for m in chain]
        assert stages == [1, 2, 3]

    def test_get_chain_unknown_msg(self):
        bus = MessageBus()
        assert bus.get_chain("nonexistent") == []

    def test_get_downstream(self):
        bus = MessageBus()
        root = AgentMessage.create("A", "all", "data", {"root": True})
        bus.publish(root)
        child1 = AgentMessage.create("B", "C", "data", {}, parent_msg_ids=[root.msg_id])
        child2 = AgentMessage.create("B", "D", "data", {}, parent_msg_ids=[root.msg_id])
        bus.publish(child1)
        bus.publish(child2)

        downstream = bus.get_downstream(root.msg_id)
        assert len(downstream) == 2

    def test_snapshot(self):
        bus = MessageBus()
        bus.publish(AgentMessage.create("A", "B", "data", {}))
        bus.publish(AgentMessage.create("A", "B", "done", {}))
        snap = bus.snapshot()
        assert snap["total_messages"] == 2
        assert snap["msg_types"]["data"] == 1
        assert snap["msg_types"]["done"] == 1
        assert "B" in snap["inbox_stats"]
        assert "A" in snap["outbox_stats"]

    def test_clear(self):
        bus = MessageBus()
        bus.publish(AgentMessage.create("A", "B", "data", {}))
        bus.clear()
        assert len(bus) == 0
        assert bus.consume("B") == []


# ===================== MessageAdapter =====================


class TestMessageAdapter:

    def test_from_agent_result(self):
        ar = AgentResult(
            success=True,
            agent_name="IngestionAgent",
            message="OK",
            execution_time=1.5,
            context_updates={"metadata": {"ts_code": "600519.SH"}, "final_answer": "done"},
        )
        msgs = MessageAdapter.from_agent_result(ar)
        # 2 data messages (metadata + final_answer) + 1 done message
        assert len(msgs) == 3
        data_msgs = [m for m in msgs if m.msg_type == "data"]
        done_msgs = [m for m in msgs if m.msg_type == "done"]
        assert len(data_msgs) == 2
        assert len(done_msgs) == 1
        # All from IngestionAgent
        assert all(m.sender == "IngestionAgent" for m in msgs)
        # All broadcast
        assert all(m.receiver == "all" for m in msgs)
        # Done msg has data msgs as parents
        done = done_msgs[0]
        assert len(done.parent_msg_ids) == 2

    def test_from_agent_result_empty_updates(self):
        ar = AgentResult(success=True, agent_name="A", context_updates={})
        msgs = MessageAdapter.from_agent_result(ar)
        # Only 1 done message
        assert len(msgs) == 1
        assert msgs[0].msg_type == "done"

    def test_from_agent_result_with_parent_ids(self):
        ar = AgentResult(success=True, agent_name="A", context_updates={"x": 1})
        msgs = MessageAdapter.from_agent_result(ar, parent_msg_ids=["p1"])
        data_msg = [m for m in msgs if m.msg_type == "data"][0]
        assert data_msg.parent_msg_ids == ["p1"]

    def test_to_context_updates_known_keys(self):
        ctx = AgentContext(raw_input="test", metadata={"old": 1})
        msgs = [
            AgentMessage.create("A", "all", "data", {"key": "metadata", "value": {"new": 2}}),
            AgentMessage.create("A", "all", "data", {"key": "final_answer", "value": "hello"}),
        ]
        updates = MessageAdapter.to_context_updates(msgs, ctx)
        # metadata is a known AgentContext attribute → direct assignment
        assert updates["metadata"] == {"new": 2}
        # final_answer is also known
        assert updates["final_answer"] == "hello"

    def test_to_context_updates_unknown_keys(self):
        ctx = AgentContext(raw_input="test")
        msgs = [
            AgentMessage.create("A", "all", "data", {"key": "custom_field", "value": 42}),
        ]
        updates = MessageAdapter.to_context_updates(msgs, ctx)
        # Unknown key goes to metadata dict inside updates
        assert updates["metadata"]["custom_field"] == 42

    def test_to_context_updates_skips_non_data(self):
        ctx = AgentContext(raw_input="test")
        msgs = [
            AgentMessage.create("A", "all", "done", {"success": True}),
            AgentMessage.create("A", "all", "data", {"key": "final_answer", "value": "ok"}),
        ]
        updates = MessageAdapter.to_context_updates(msgs, ctx)
        assert updates.get("final_answer") == "ok"
        # done message should not appear in updates
        assert "done" not in updates
