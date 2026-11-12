"""
P0-3 bad-case 回流通道单测 — eval/backflow.py。

会话档案（state/conversations/*.json）→ golden_candidate.jsonl 待标注候选：
  - initial_analysis + 追问 Q/A 对提取（带证据、guard 分数、origin 定位）
  - 证据忠实：sources 必须取自落盘 metadata / context，不得凭空构造
  - origin 去重可重复跑；guard 分升序（最差优先待标注）
全部离线确定，tmp_path 隔离，不碰真实会话目录。
"""
import json

import pytest

from financial_rag.eval.backflow import (
    export_candidates,
    session_to_candidates,
)


def _mk_session(sid="s1", followups=True):
    """构造落盘形状的会话 dict（ConversationSession.to_dict 同构）。"""
    messages = [
        {"role": "assistant", "content": "初始解读：营收增长是利好。",
         "timestamp": "2026-01-01 10:00:00",
         "metadata": {"type": "initial_analysis"}},
    ]
    if followups:
        messages += [
            {"role": "user", "content": "为什么是利好？",
             "timestamp": "2026-01-01 10:01:00", "metadata": {}},
            {"role": "assistant", "content": "因为营收超预期[1]。",
             "timestamp": "2026-01-01 10:01:05",
             "metadata": {"guard_risk": "high", "guard_score": 0.3,
                          "source_count": 1,
                          "sources": [{"text": "营收同比增长36.4%。"}]}},
            {"role": "user", "content": "研发投入多少？",
             "timestamp": "2026-01-01 10:02:00", "metadata": {}},
            {"role": "assistant", "content": "研发投入45亿元[1]。",
             "timestamp": "2026-01-01 10:02:05",
             "metadata": {"guard_risk": "low", "guard_score": 0.9,
                          "source_count": 1,
                          "sources": [{"text": "研发投入达45亿元。"}]}},
        ]
    return {
        "id": sid, "title": "商汤年报解读", "type": "news",
        "messages": messages,
        "context": {"news_text": "商汤科技发布2024年年度报告。",
                    "kb_sources": [{"text": "存档KB证据。"}]},
        "created_at": "2026-01-01 10:00:00",
        "updated_at": "2026-01-01 10:02:05",
    }


def _write_session(conv_dir, session):
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / f"{session['id']}.json").write_text(
        json.dumps(session, ensure_ascii=False), encoding="utf-8")


@pytest.mark.unit
class TestSessionToCandidates:
    def test_initial_analysis_extracted(self):
        cands = session_to_candidates(_mk_session(followups=False))
        assert len(cands) == 1
        c = cands[0]
        assert c["kind"] == "initial_analysis"
        assert c["answer"] == "初始解读：营收增长是利好。"
        assert c["origin"] == {"session_id": "s1", "message_index": 0}
        # initial 未记录 guard 分 → None（排序置后），不得伪造 0 分污染最差优先
        assert c["guard_score"] is None

    def test_followup_qa_pairs_with_guard(self):
        cands = session_to_candidates(_mk_session())
        fu = [c for c in cands if c["kind"] == "followup"]
        assert len(fu) == 2
        assert fu[0]["query"] == "为什么是利好？"
        assert fu[0]["answer"] == "因为营收超预期[1]。"
        assert fu[0]["guard_score"] == 0.3 and fu[0]["guard_risk"] == "high"
        assert fu[0]["origin"] == {"session_id": "s1", "message_index": 2}

    def test_sources_faithful_to_archive(self):
        """证据忠实：追问 sources 取自落盘 metadata；initial 取自 context 档案。"""
        cands = {c["kind"]: c for c in session_to_candidates(_mk_session())
                 if c["origin"]["message_index"] in (0, 2)}
        fu = next(c for c in session_to_candidates(_mk_session())
                  if c["kind"] == "followup")
        assert fu["sources"] == [{"text": "营收同比增长36.4%。"}]
        init = cands["initial_analysis"]
        texts = [s["text"] for s in init["sources"]]
        assert "商汤科技发布2024年年度报告。" in texts
        assert "存档KB证据。" in texts

    def test_skipped_guard_carried_not_fabricated(self):
        """skipped 追问（guard_risk='skipped'）原样透出，不当成 0 分坏 case。"""
        s = _mk_session()
        s["messages"][2]["metadata"].update({"guard_risk": "skipped",
                                             "guard_score": 0.0})
        fu = [c for c in session_to_candidates(s) if c["kind"] == "followup"][0]
        assert fu["guard_risk"] == "skipped"


@pytest.mark.unit
class TestExportCandidates:
    def test_worst_first_sorting(self, tmp_path):
        conv_dir = tmp_path / "conversations"
        _write_session(conv_dir, _mk_session())
        out = tmp_path / "candidates.jsonl"
        n = export_candidates(conv_dir, out)
        assert n == 3  # 1 initial + 2 追问
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        scored = [r["guard_score"] for r in rows if r["guard_score"] is not None]
        assert scored == sorted(scored), "必须按 guard 分升序（最差优先待标注）"
        assert rows[-1]["guard_score"] is None, "未记录分的 initial 置后"

    def test_origin_dedup_rerunnable(self, tmp_path):
        conv_dir = tmp_path / "conversations"
        _write_session(conv_dir, _mk_session())
        out = tmp_path / "candidates.jsonl"
        assert export_candidates(conv_dir, out) == 3
        assert export_candidates(conv_dir, out) == 0, "重复跑不得重复追加"
        rows = out.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 3

    def test_new_followup_appended_on_rerun(self, tmp_path):
        """会话新增追问后再跑：只追加新 origin，旧候选不动。"""
        conv_dir = tmp_path / "conversations"
        s = _mk_session()
        _write_session(conv_dir, s)
        out = tmp_path / "candidates.jsonl"
        export_candidates(conv_dir, out)
        s["messages"] += [
            {"role": "user", "content": "新追问", "metadata": {}},
            {"role": "assistant", "content": "新回答",
             "metadata": {"guard_risk": "medium", "guard_score": 0.5,
                          "sources": [{"text": "新证据。"}]}},
        ]
        _write_session(conv_dir, s)
        assert export_candidates(conv_dir, out) == 1
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 4

    def test_empty_or_missing_dir(self, tmp_path):
        out = tmp_path / "candidates.jsonl"
        assert export_candidates(tmp_path / "not_exist", out) == 0
        empty = tmp_path / "empty"
        empty.mkdir()
        assert export_candidates(empty, out) == 0
        assert not out.exists(), "无候选不得产生空文件"

    def test_limit_caps_worst_first(self, tmp_path):
        conv_dir = tmp_path / "conversations"
        _write_session(conv_dir, _mk_session())
        out = tmp_path / "candidates.jsonl"
        assert export_candidates(conv_dir, out, limit=1) == 1
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["guard_score"] == 0.3, "limit 必须截最差的，不是随机截"