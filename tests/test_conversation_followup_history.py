"""
会话管理与上下文压缩重构测试 — 档案全量保留，LLM 视野分层压缩。

设计契约：
  - 磁盘 JSON 与前端 UI 永远持有完整消息历史；压缩只决定"喂给 LLM 看什么"
  - 水位线增量压缩：compressed_upto 之前的消息已进摘要，绝不重复总结
  - initial_analysis 出 LLM 窗（system prompt 已含结论 + grounding），但档案保真
  - 摘要封顶 2000 字：LLM 路径与提取式回退同一口径
  - 会话级锁：并发 add_message 零丢失
  - 错误路径落盘：assistant error 消息入档，档案与 UI 不分叉

对应生产代码：
  - financial_rag/services/conversation.py（水位线压缩 + 会话锁 + history_info）
  - financial_rag/tools/conversation_tools.py（错误落盘 + history_info 透出）

离线约束：tmp_path 隔离会话目录；StubLLM 记录调用（经 LLMCaller 进程级缓存，
各用例内容唯一防串扰）；不碰真实 LLM/网络。

按项目四条测试标准组织：① 证明效果 ② 可观测 ③ 一致性 ④ 回归锁定
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from financial_rag.services.conversation import ConversationManager
from financial_rag.tools.conversation_tools import (
    answer_followup,
    inject_conversation_deps,
)


# ===================== StubLLM（记录压缩调用） =====================

class CompressStubLLM:
    """记录每次 chat 的 messages/system，返回 canned 摘要。"""

    def __init__(self, summary="压缩摘要"):
        self.summary = summary
        self.calls = []

    def chat(self, messages, system=None, **kwargs):
        self.calls.append({"messages": messages, "system": system})
        return SimpleNamespace(
            content=self.summary, usage={"total_tokens": len(self.summary)})


@pytest.fixture
def cm(tmp_path):
    return ConversationManager(conv_dir=str(tmp_path / "conversations"))


def _mk_session(cm, tag, initial="初步分析：利好。"):
    """每个用例唯一 tag → 压缩 prompt 唯一 → LLMCaller 缓存键隔离。"""
    return cm.create_session(
        session_type="news", title=f"会话{tag}",
        initial_analysis=initial, context={"news_text": f"新闻{tag}"},
    )


def _add_rounds(cm, sid, tag, start, n):
    for i in range(start, start + n):
        cm.add_message(sid, "user", f"问题{tag}-{i}：第{i}轮追问")
        cm.add_message(sid, "assistant", f"回答{tag}-{i}：第{i}轮结论")


# ======================================================================
# ④ 回归锁定：水位线增量压缩 — 同一批旧消息绝不重复总结
# ======================================================================

class TestWatermarkCompression:
    pytestmark = pytest.mark.integration

    @pytest.mark.critical
    @pytest.mark.regression
    def test_second_compress_excludes_already_summarized(self, cm):
        """BUG: 旧实现 messages 从不删除，每次压缩把已总结的旧轮次再喂一遍。
        水位线修复后：第二次压缩 prompt 只含新增轮次 + 已有摘要。"""
        sid = _mk_session(cm, "WM")
        stub = CompressStubLLM(summary="摘要V1")
        _add_rounds(cm, sid, "WM", 0, 6)          # 1 + 12 = 13 条 → 触发压缩
        s = cm.get_session(sid)
        cm._build_messages(s, "新问题WM-1", llm=stub)
        assert len(stub.calls) == 1
        assert s.compressed_upto == 5             # 13 - 8(最近4轮)
        first_prompt = str(stub.calls[0]["messages"])
        assert "问题WM-0" in first_prompt and "问题WM-1" in first_prompt

        _add_rounds(cm, sid, "WM", 6, 4)          # 21 条 → 第二次压缩
        stub.summary = "摘要V2"
        cm._build_messages(s, "新问题WM-2", llm=stub)
        assert len(stub.calls) == 2
        assert s.compressed_upto == 13            # 单调推进
        second_prompt = str(stub.calls[1]["messages"])
        # 核心断言：首轮 Q/A 不再出现（旧实现此处必红）
        assert "问题WM-0" not in second_prompt
        assert "问题WM-1" not in second_prompt
        assert "问题WM-2" in second_prompt        # 只总结新增轮次
        assert "[已有摘要]" in second_prompt       # 旧摘要合并而非重算
        assert "摘要V1" in second_prompt

    @pytest.mark.critical
    def test_watermark_no_overlap_with_recent_window(self, cm):
        """压缩区 [0, upto) 与最近窗口 [total-8, total) 不重叠、无缝衔接。"""
        sid = _mk_session(cm, "OV")
        _add_rounds(cm, sid, "OV", 0, 8)          # 17 条
        s = cm.get_session(sid)
        cm._build_messages(s, "新问题OV", llm=None)
        total = len(s.messages)
        assert s.compressed_upto == total - 8
        # 摘要覆盖的最后一轮与窗口首轮相邻不重复
        assert s.compressed_upto > 0

    def test_idempotent_when_nothing_new_to_compress(self, cm):
        """水位线到位后重复调用 _build_messages 不再触发压缩调用。"""
        sid = _mk_session(cm, "ID")
        stub = CompressStubLLM()
        _add_rounds(cm, sid, "ID", 0, 6)
        s = cm.get_session(sid)
        cm._build_messages(s, "新问题ID-1", llm=stub)
        n_calls = len(stub.calls)
        cm._build_messages(s, "新问题ID-2", llm=stub)
        assert len(stub.calls) == n_calls, "无新增可压缩内容时不应再调 LLM"


# ======================================================================
# ①③ initial_analysis 出 LLM 窗，但档案保真（UI 走 to_dict 全量）
# ======================================================================

class TestInitialAnalysisOutOfWindow:
    pytestmark = pytest.mark.integration

    @pytest.mark.critical
    def test_initial_analysis_excluded_from_llm_messages(self, cm):
        marker = "INITIAL全文标记XZ"
        sid = _mk_session(cm, "IA", initial=f"{marker}：这是一段很长的初步分析。")
        _add_rounds(cm, sid, "IA", 0, 2)
        s = cm.get_session(sid)
        msgs = cm._build_messages(s, "新问题IA", llm=None)
        assert all(marker not in m["content"] for m in msgs), (
            "initial_analysis 已由 system prompt 结论+grounding 覆盖，不应再占窗口"
        )
        assert msgs[-1] == {"role": "user", "content": "新问题IA"}

    def test_archive_keeps_initial_analysis(self, cm):
        marker = "INITIAL档案标记QW"
        sid = _mk_session(cm, "IA2", initial=f"{marker}：初步分析全文。")
        s = cm.get_session(sid)
        d = s.to_dict()
        assert d["messages"][0]["content"].startswith(marker)
        assert d["messages"][0]["metadata"]["type"] == "initial_analysis"


# ======================================================================
# ③ 一致性：摘要封顶 LLM 路径与提取式回退同口径（2000 字）
# ======================================================================

class TestSummaryCapConsistency:
    pytestmark = pytest.mark.integration

    def test_llm_summary_also_capped(self, cm):
        sid = _mk_session(cm, "CAP")
        stub = CompressStubLLM(summary="摘" * 3000)   # LLM 返回超长摘要
        _add_rounds(cm, sid, "CAP", 0, 6)
        s = cm.get_session(sid)
        cm._build_messages(s, "新问题CAP", llm=stub)
        assert len(stub.calls) == 1, "应走 LLM 压缩路径"
        assert len(s.compressed_summary) <= 2000, "LLM 路径也必须过硬截断"


# ======================================================================
# ④ 回归锁定：旧会话 JSON 无 compressed_upto 字段 → 兼容加载
# ======================================================================

class TestLegacySessionCompat:
    pytestmark = pytest.mark.integration

    def test_load_legacy_json_without_watermark(self, tmp_path):
        conv_dir = tmp_path / "conversations"
        conv_dir.mkdir()
        legacy = {
            "id": "legacy000001", "title": "旧会话", "type": "news",
            "messages": [
                {"role": "assistant", "content": "旧分析", "timestamp": "",
                 "metadata": {"type": "initial_analysis"}},
            ],
            "context": {}, "compressed_summary": "",
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
        }
        (conv_dir / "legacy000001.json").write_text(
            json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        cm = ConversationManager(conv_dir=str(conv_dir))
        s = cm.get_session("legacy000001")
        assert s is not None
        assert s.compressed_upto == 0
        # 旧会话继续追问 → 压缩机制正常工作
        _add_rounds(cm, "legacy000001", "LG", 0, 6)
        cm._build_messages(s, "新问题LG", llm=None)
        assert s.compressed_summary
        assert s.compressed_upto > 0


# ======================================================================
# ① 会话级锁：并发 add_message 零丢失 + 磁盘完整
# ======================================================================

class TestConcurrentAddMessage:
    pytestmark = pytest.mark.integration

    @pytest.mark.critical
    def test_concurrent_appends_no_loss(self, cm, tmp_path):
        sid = _mk_session(cm, "CC")
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(
                lambda i: cm.add_message(sid, "user", f"并发消息CC-{i}"),
                range(20)))
        s = cm.get_session(sid)
        assert len(s.messages) == 21               # initial + 20，零丢失
        contents = {m.content for m in s.messages}
        assert all(f"并发消息CC-{i}" in contents for i in range(20))
        # 磁盘最终态完整可加载（整文件覆盖写不丢消息）
        cm2 = ConversationManager(conv_dir=str(tmp_path / "conversations"))
        s2 = cm2.get_session(sid)
        assert len(s2.messages) == 21


# ======================================================================
# ①③ 错误路径落盘：档案与 UI 一致，不留悬空 user 问句
# ======================================================================

class TestErrorPersistence:
    pytestmark = pytest.mark.integration

    def test_no_llm_error_persisted_as_assistant(self, cm):
        sid = _mk_session(cm, "ERR")
        inject_conversation_deps(llm=None, retriever=None, kb_built=False,
                                 conversation_manager=cm)
        result = answer_followup(sid, "追问ERR：为什么？")
        assert "error" in result
        s = cm.get_session(sid)
        assert s.messages[-2].role == "user"
        assert s.messages[-1].role == "assistant"
        assert s.messages[-1].metadata.get("error") is True
        assert result["error"] in s.messages[-1].content


# ======================================================================
# ② 可观测：history_info 诊断字段与实际状态一致
# ======================================================================

class TestHistoryInfoObservable:
    pytestmark = pytest.mark.integration

    def test_history_info_matches_state(self, cm):
        sid = _mk_session(cm, "HI")
        s = cm.get_session(sid)
        info = cm.history_info(s)
        assert info == {
            "total_messages": 1, "compressed_upto": 0,
            "summary_chars": 0, "recent_window": 8,
        }
        _add_rounds(cm, sid, "HI", 0, 6)
        cm._build_messages(s, "新问题HI", llm=None)
        info = cm.history_info(s)
        assert info["total_messages"] == len(s.messages)
        assert info["compressed_upto"] == s.compressed_upto
        assert info["summary_chars"] == len(s.compressed_summary)
        assert info["summary_chars"] > 0