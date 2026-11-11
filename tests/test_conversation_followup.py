"""
追问质量闭环测试 — 第一期开卷追问（新闻解读/话题调研后的持续对话）。

链路位置：新闻解读 → 建会话(案件档案) → 追问 → KB检索 → ContextBuilder组装 → LLM → Guard
对应生产代码：
  - financial_rag/services/conversation.py（档案存储 + 历史压缩，LLM 逻辑已迁出）
  - financial_rag/tools/conversation_tools.py（answer_followup 工具 + 证据组装）
  - financial_rag/prompts.py（FOLLOWUP_ANSWER_SYSTEM）

按项目四条测试标准组织：① 证明效果 ② 可观测 ③ 一致性 ④ 回归锁定

核心契约：
  - 持久化路径跟随 KB 分层重构（state/conversations），旧路径即红
  - 证据组装：档案(新闻 score=1.0 排首) + 新检索，去重，ContextBuilder 统一算法
  - 编号对齐：grounding 块标号 1-based，与 L3 引用契约(1 <= idx <= n_sources)一致
  - LLM 与 Guard 同源：source_items 同一份喂两者
  - 诊断恒返回：kb_search_info / context_trace 即使检索为空也在

离线约束：conftest 已删 DASHSCOPE_API_KEY + 禁 Phoenix；检索用
HybridRetriever(embedder=None, reranker=None) 纯本地 BM25+Jaccard；
不 mock LLM —— 无 llm 走错误路径，Guard 用 canned answer（同 step7 模式）。
"""
import os

import pytest

from financial_rag.config import config
from financial_rag.services.conversation import ConversationManager
from financial_rag.tools.conversation_tools import (
    assemble_followup_context,
    check_followup_answer,
    answer_followup,
    inject_conversation_deps,
)
from financial_rag.retrievers.hybrid_engine import HybridRetriever
from tests.conftest import (
    SAMPLE_AI_FINANCIAL_REPORT,
    SAMPLE_AI_NEWS,
    SAMPLE_AI_FUNDING,
)


# ===================== 共享样本 =====================

def _dossier_context():
    """案件档案上下文：新闻原文 + 建会话时的 KB 来源（与 analysis_router 存入结构同构）。"""
    return {
        "news_text": SAMPLE_AI_NEWS,
        "structured": {"verdict": "利好", "key_signals": ["B200 性能提升 4 倍"]},
        "assessment": "利好",
        "confidence": "高",
        "kb_sources": [
            {"text": SAMPLE_AI_FINANCIAL_REPORT, "source": "kb_report", "score": 0.8},
        ],
    }


def _new_results():
    """追问触发的新 KB 检索结果。"""
    return [{"text": SAMPLE_AI_FUNDING, "source": "kb_funding", "score": 0.9}]


# 完全有据回答：事实全部来自档案+新检索，[N] 1-based + 4 个 Markdown 段落
GROUNDED_FOLLOWUP = """# 摘要
英伟达发布Blackwell B200 GPU，单卡AI训练性能较H100提升4倍[1]。

# 要点
- 智谱AI完成B+轮融资，融资金额超50亿元，投后估值超200亿元[2]。
- 商汤科技2024年营业收入50.3亿元，同比增长36.4%[3]。

# 分析
微软Azure已部署10万张B200用于训练GPT-5，新一代模型推理成本将显著降低[1]。

# 风险提示
全球AI算力投资预计2025年超3000亿美元，行业竞争加剧可能带来不确定性[1]。
"""

FAB_SENTENCE = "腾讯于2024年第四季度全资收购了月之暗面公司，交易金额尚未披露。"
FABRICATED_FOLLOWUP = GROUNDED_FOLLOWUP.replace("# 分析\n", f"# 分析\n{FAB_SENTENCE}\n")


@pytest.fixture
def cm(tmp_path):
    """隔离会话管理器 — 不碰生产 state 目录。"""
    return ConversationManager(conv_dir=str(tmp_path / "conversations"))


@pytest.fixture
def session_id(cm):
    return cm.create_session(
        session_type="news", title="英伟达B200发布",
        initial_analysis="初步分析：利好。", context=_dossier_context(),
    )


# ======================================================================
# ④ 回归锁定：持久化路径必须跟随 KB 分层（d1c038e 重构遗留 bug）
# ======================================================================

class TestPersistencePathRegression:
    """BUG: _CONV_DIR 硬编码旧路径 data/knowledge_base/conversations，
    分层重构后会话实际在 state/conversations → 历史会话全部不可见。"""

    @pytest.mark.regression
    def test_default_conv_dir_follows_state_layer(self):
        cm = ConversationManager()
        assert cm._conv_dir == os.path.join(config.state_dir, "conversations"), (
            f"默认会话目录应跟随 config.state_dir，实际: {cm._conv_dir}"
        )


# ======================================================================
# ① 证明效果：档案 CRUD + 持久化往返
# ======================================================================

class TestSessionCRUD:

    def test_create_stores_initial_analysis_as_assistant(self, cm, session_id):
        s = cm.get_session(session_id)
        assert s is not None
        assert s.type == "news"
        assert len(s.messages) == 1
        assert s.messages[0].role == "assistant"
        assert s.messages[0].metadata.get("type") == "initial_analysis"
        assert s.context["news_text"] == SAMPLE_AI_NEWS

    def test_persistence_roundtrip(self, tmp_path, session_id, cm):
        cm.add_message(session_id, "user", "为什么是利好？")
        # 模拟重启：新实例从磁盘加载
        cm2 = ConversationManager(conv_dir=str(tmp_path / "conversations"))
        s = cm2.get_session(session_id)
        assert s is not None
        assert len(s.messages) == 2
        assert s.messages[1].content == "为什么是利好？"
        assert s.context["assessment"] == "利好"

    def test_delete_removes_disk_file(self, tmp_path, cm, session_id):
        path = tmp_path / "conversations" / f"{session_id}.json"
        assert path.exists()
        assert cm.delete_session(session_id) is True
        assert not path.exists()
        assert cm.get_session(session_id) is None

    def test_list_sessions_sorted_by_updated(self, cm):
        s1 = cm.create_session("news", "会话A", "", {})
        s2 = cm.create_session("topic", "会话B", "", {})
        cm.add_message(s1, "user", "追问")  # s1 变为最新
        listed = cm.list_sessions()
        assert listed[0]["id"] == s1
        assert listed[0]["message_count"] == 1


# ======================================================================
# ① 证明效果：历史压缩（无 llm 时提取式回退，离线确定性）
# ======================================================================

class TestHistoryCompression:

    def test_compress_triggers_and_keeps_recent(self, cm, session_id):
        # 灌入 12 条消息（超过阈值 10）
        for i in range(6):
            cm.add_message(session_id, "user", f"问题{i}：第{i}轮追问内容")
            cm.add_message(session_id, "assistant", f"回答{i}：第{i}轮分析结论")
        s = cm.get_session(session_id)
        msgs = cm._build_messages(s, "新问题", llm=None)
        # 压缩发生：摘要非空（提取式回退，无需 LLM）
        assert s.compressed_summary, "超过阈值应触发压缩（无 llm 走提取式回退）"
        # 最近 4 轮(8条) + 新问题，摘要以 system 注入
        assert len(msgs) <= 10
        assert msgs[-1] == {"role": "user", "content": "新问题"}
        assert msgs[0]["role"] == "system"
        assert "历史对话摘要" in msgs[0]["content"]

    def test_extractive_summary_bounded(self, cm, session_id):
        for i in range(20):
            cm.add_message(session_id, "user", f"问题{i}" + "长" * 100)
            cm.add_message(session_id, "assistant", f"回答{i}" + "析" * 300)
        s = cm.get_session(session_id)
        cm._build_messages(s, "新问题", llm=None)
        assert len(s.compressed_summary) <= 2000, "摘要必须有硬截断防膨胀"


# ======================================================================
# ①②③ 证据组装：档案+新检索、去重、可观测、1-based 编号对齐
# ======================================================================

class TestEvidenceAssembly:

    def test_dossier_and_new_results_both_included(self):
        assembled = assemble_followup_context(_dossier_context(), _new_results())
        # 档案新闻排首（score=1.0），新检索与档案 KB 均纳入
        assert "[1]" in assembled.grounding_text
        assert "Blackwell" in assembled.grounding_text
        assert "智谱AI" in assembled.grounding_text      # 新检索
        assert "商汤科技" in assembled.grounding_text    # 档案 KB
        assert len(assembled.source_items) == 3

    def test_dedup_by_text(self):
        # 新检索命中档案里已有的同一篇 → 不重复注入
        dup = [{"text": SAMPLE_AI_FINANCIAL_REPORT, "source": "kb_report", "score": 0.95}]
        assembled = assemble_followup_context(_dossier_context(), dup)
        assert len(assembled.source_items) == 2
        assert assembled.grounding_text.count("商汤科技集团股份有限公司") == 1

    def test_numbering_is_one_based_aligned_with_sources(self):
        """L3 契约 1 <= idx <= n_sources：块标号必须 1-based 且与 source_items 对齐。"""
        assembled = assemble_followup_context(_dossier_context(), _new_results())
        assert "[0]" not in assembled.grounding_text
        n = len(assembled.source_items)
        assert f"[{n}]" in assembled.grounding_text
        # 首块必须是档案新闻（score=1.0 排首）
        first_block = assembled.grounding_text.split("\n\n")[0]
        assert first_block.startswith("[1]")
        assert "原始新闻" in first_block

    def test_trace_observable(self):
        assembled = assemble_followup_context(_dossier_context(), _new_results())
        trace = assembled.trace
        assert trace.fidelity > 0
        assert trace.included_items == 3
        d = trace.to_dict()
        assert "fidelity" in d and "items" in d

    def test_empty_new_results_still_assembles_dossier(self):
        assembled = assemble_followup_context(_dossier_context(), [])
        assert "Blackwell" in assembled.grounding_text
        assert len(assembled.source_items) == 2


# ======================================================================
# ③ 一致性：LLM 与 Guard 同源（source_items 直接喂 Guard）
# ======================================================================

class TestGuardSameSource:

    def test_grounded_answer_passes(self):
        assembled = assemble_followup_context(_dossier_context(), _new_results())
        report = check_followup_answer(GROUNDED_FOLLOWUP, assembled)
        assert report["passed"] is True
        assert report["risk"] == "low"
        assert report["layers"]["L3_citation_integrity"]["invalid"] == 0

    def test_fabricated_sentence_caught(self):
        assembled = assemble_followup_context(_dossier_context(), _new_results())
        report = check_followup_answer(FABRICATED_FOLLOWUP, assembled)
        unanchored = report["layers"]["L1_source_grounding"]["unanchored"]
        assert any("月之暗面" in u for u in unanchored), (
            f"编造句应被 L1 抓住，实际 unanchored={unanchored}"
        )


# ======================================================================
# ① 工具行为：错误路径 + 诊断恒返回 + 用户消息落盘
# ======================================================================

class TestFollowupTool:

    def test_missing_session_returns_error(self, cm):
        inject_conversation_deps(llm=None, conversation_manager=cm)
        result = answer_followup("nonexistent", "追问")
        assert "error" in result

    def test_no_llm_returns_error_with_diagnostics(self, cm, session_id):
        inject_conversation_deps(llm=None, retriever=None, kb_built=False,
                                 conversation_manager=cm)
        result = answer_followup(session_id, "为什么是利好？")
        assert "error" in result
        # 诊断恒返回：即使 KB 未构建
        assert "kb_search_info" in result
        assert result["kb_search_info"]["kb_built"] is False
        # 用户消息先落盘
        s = cm.get_session(session_id)
        assert s.messages[-1].role == "user"
        assert s.messages[-1].content == "为什么是利好？"

    def test_offline_retrieval_wiring(self, cm, session_id):
        """纯本地 BM25+Jaccard 检索接线：追问能召回 KB 文档（诊断可观测）。"""
        r = HybridRetriever(embedder=None, reranker=None)
        r.index([
            {"text": SAMPLE_AI_NEWS,
             "meta": {"doc_id": "d_news", "source": "test", "doc_type": "news"}},
            {"text": SAMPLE_AI_FUNDING,
             "meta": {"doc_id": "d_funding", "source": "test", "doc_type": "news"}},
        ])
        inject_conversation_deps(llm=None, retriever=r, kb_built=True,
                                 conversation_manager=cm)
        result = answer_followup(session_id, "英伟达 Blackwell GPU 性能提升")
        info = result["kb_search_info"]
        assert info["total_results"] > 0, f"离线检索应召回文档，实际: {info}"

    def test_tool_registered_in_registry(self, registry):
        assert registry.can_handle("answer_followup")


# ======================================================================
# ③ 一致性：domain prompt 归属 prompts.py（Prompt 与工具分层规范）
# ======================================================================

class TestPromptLayering:

    def test_followup_prompt_lives_in_prompts_module(self):
        from financial_rag import prompts
        assert hasattr(prompts, "FOLLOWUP_ANSWER_SYSTEM")
        tpl = prompts.FOLLOWUP_ANSWER_SYSTEM
        # 强制引用 + 无据拒答 + 占位符
        assert "[N]" in tpl
        assert "{grounding}" in tpl
        assert "{structured}" in tpl