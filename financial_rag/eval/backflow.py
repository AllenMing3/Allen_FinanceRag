"""
P0-3 bad-case 回流通道 — 会话档案 → golden_candidate.jsonl 待标注候选。

线上坏 case 无法回流成测试样本是评估集腐化的结构性缺口：golden 永远是
"你以为的失败模式"，不是"实际发生的失败模式"。本模块把会话档案里的
initial_analysis + 追问 Q/A 对（带证据、guard 分数、origin 定位）导出为
待标注候选，人工精编 expected 后并入 golden。

设计原则：
- 纯离线确定：只读 state/conversations/*.json 落盘档案，不碰 LLM/检索器。
- 证据忠实：追问 sources 取自落盘 metadata（conversation_tools 已补存），
  initial 取自 context 档案（news_text + kb_sources），不得凭空构造。
- origin 去重可重复跑：(session_id, message_index) 唯一键，已导出不重复追加。
- 最差优先待标注：按 guard 分升序；skipped（未校验）与未记录分（initial）
  置后 —— 未校验 ≠ 0 分坏 case（P1-2 显式 skipped 契约的延续）。
- 候选不是 golden：expected 需人工标注，本模块只产候选不自动入 golden。

局限：kb-query 通路不落盘会话档案，本次不覆盖。

CLI: python -m financial_rag.eval.backflow [--conv-dir DIR] [--limit N] [--out PATH]
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = _EVAL_DIR / "golden_candidate.jsonl"


def _default_conv_dir() -> Path:
    """惰性读 config：会话属运行状态层（与 ConversationManager 同源）。"""
    from financial_rag.config import config
    return Path(config.state_dir) / "conversations"


# ===================== 会话 → 候选（纯函数） =====================

def _initial_sources(context: Dict) -> List[Dict]:
    """initial_analysis 的证据取自会话档案：原始新闻 + 存档 KB 来源。"""
    sources: List[Dict] = []
    news_text = (context.get("news_text") or "").strip()
    if news_text:
        sources.append({"text": news_text})
    for s in context.get("kb_sources") or []:
        if isinstance(s, dict) and (s.get("text") or "").strip():
            sources.append({"text": s["text"]})
        elif isinstance(s, str) and s.strip():
            sources.append({"text": s})
    return sources


def session_to_candidates(session: Dict) -> List[Dict]:
    """单个落盘会话 → 待标注候选列表（initial_analysis + 追问 Q/A 对）。

    候选形状: {origin, kind, session_title, query, answer, sources,
               guard_score, guard_risk}
    - origin = {session_id, message_index}：去重键 + 回档案定位
    - initial 未记录 guard 分 → guard_score=None（不伪造 0 分污染最差优先）
    - 追问 guard_score/guard_risk/sources 原样取自落盘 metadata
    """
    sid = session.get("id", "")
    title = session.get("title", "")
    context = session.get("context") or {}
    msgs = session.get("messages") or []
    init_sources = _initial_sources(context)

    candidates: List[Dict] = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        answer = (m.get("content") or "").strip()
        if not answer:
            continue
        meta = m.get("metadata") or {}
        if meta.get("type") == "initial_analysis":
            candidates.append({
                "origin": {"session_id": sid, "message_index": i},
                "kind": "initial_analysis",
                "session_title": title,
                "query": title,
                "answer": answer,
                "sources": init_sources,
                "guard_score": None,
                "guard_risk": "",
            })
        else:
            # 追问：向前找最近的 user 消息作为 query
            query = ""
            for j in range(i - 1, -1, -1):
                if msgs[j].get("role") == "user":
                    query = msgs[j].get("content", "")
                    break
            candidates.append({
                "origin": {"session_id": sid, "message_index": i},
                "kind": "followup",
                "session_title": title,
                "query": query,
                "answer": answer,
                "sources": meta.get("sources") or [],
                "guard_score": meta.get("guard_score"),
                "guard_risk": meta.get("guard_risk", ""),
            })
    return candidates


def _worst_first_key(c: Dict) -> Tuple[int, float]:
    """排序键：有分且非 skipped 的按分升序（最差优先）；skipped 次之；无分置后。"""
    score = c.get("guard_score")
    if score is None:
        return (2, 0.0)
    if c.get("guard_risk") == "skipped":
        return (1, 0.0)
    return (0, float(score))


# ===================== 导出（去重落盘） =====================

def _load_seen_origins(out_path: Path) -> set:
    """读已导出候选的 origin 键集合；损坏行跳过不阻断。"""
    seen = set()
    if not out_path.exists():
        return seen
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        o = row.get("origin") or {}
        seen.add((o.get("session_id"), o.get("message_index")))
    return seen


def export_candidates(conv_dir, out_path=DEFAULT_OUT,
                      limit: Optional[int] = None) -> int:
    """扫描会话目录 → 新候选按最差优先追加落盘，返回本次新增条数。

    origin 去重保证可重复跑；limit 截最差的 N 条（不是随机截）。
    无新候选不产生/不改动输出文件。
    """
    conv_dir = Path(conv_dir)
    out_path = Path(out_path)
    if not conv_dir.is_dir():
        return 0

    seen = _load_seen_origins(out_path)
    fresh: List[Dict] = []
    for p in sorted(conv_dir.glob("*.json")):
        try:
            session = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning(f"[backflow] 损坏会话档案跳过: {p.name}")
            continue
        for c in session_to_candidates(session):
            key = (c["origin"]["session_id"], c["origin"]["message_index"])
            if key in seen:
                continue
            seen.add(key)
            fresh.append(c)

    if not fresh:
        return 0
    fresh.sort(key=_worst_first_key)
    if limit is not None:
        fresh = fresh[:limit]
    with open(out_path, "a", encoding="utf-8") as f:
        for c in fresh:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(fresh)


# ===================== CLI =====================

def main(argv=None) -> int:
    import argparse
    import sys
    # Windows GBK 控制台 print 中文会 UnicodeEncodeError → 强制 std 流 utf-8
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        prog="python -m financial_rag.eval.backflow",
        description="bad-case 回流：会话档案 → golden_candidate.jsonl 待标注候选（最差优先）")
    ap.add_argument("--conv-dir", default=None,
                    help="会话目录（默认 config.state_dir/conversations）")
    ap.add_argument("--out", default=None,
                    help=f"候选输出路径（默认 {DEFAULT_OUT}）")
    ap.add_argument("--limit", type=int, default=None,
                    help="本次最多导出 N 条新候选（截最差的）")
    args = ap.parse_args(argv)

    conv_dir = Path(args.conv_dir) if args.conv_dir else _default_conv_dir()
    out_path = Path(args.out) if args.out else DEFAULT_OUT
    n = export_candidates(conv_dir, out_path, limit=args.limit)
    if n == 0:
        print(f"无新候选（{conv_dir} 已全部导出或目录为空）")
    else:
        print(f"已导出 {n} 条新候选 → {out_path}（guard 分升序，最差优先待标注）")
        print("下一步：人工精编 expected 后并入 eval/golden/，候选本身不是 golden。")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())