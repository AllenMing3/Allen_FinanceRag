"""
L2 输出层离线评估运行器 — 对称 runner.py（L1 检索层）。

加载 golden output_news_brief.jsonl → 逐 case 跑 output_metrics.evaluate_case
→ 汇总 macro（全样本）/ per_tier（good/bad）/ per_tag → 区分度 gap → baseline 对比 + CLI。

设计原则（对称 L1，但更简）：
- 纯离线确定、免 API、不跑检索器/LLM —— 评估对象是预制 golden 输出样本，
  同输入必同输出，CLI 与 pytest 两种入口结果一致。
- 回归锁三维度：① 各质量指标 macro 回退超阈值；② separation_gap（正例均分−负例均分）
  回退超阈值（区分度是输出评估器的命脉，掉下去说明评估器失去分辨力）；
  ③ critical_cases 逐 case 下界（P0-2：单 case 0.9→0.3 会被 macro 均值冲淡，
  手工 pin 的关键正例必须单独锁，掉破下界或 case 消失都报警）。
- baseline 只存 macro/per_tier/per_tag/gap/counts + critical_cases（不含 per_case 明细）；
  critical_cases 为手工 curated 段，--update-baseline 重生成时保留不覆盖。

CLI: python -m financial_rag.eval.output_runner
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from financial_rag.eval.output_metrics import evaluate_case

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_GOLDEN_OUTPUT = _GOLDEN_DIR / "output_news_brief.jsonl"

# L2 质量指标（全离线确定）——output_metrics.evaluate_case 的数值返回键
METRIC_NAMES = [
    "numerical_fidelity", "source_grounding",
    "field_completeness", "output_quality",
]
# 回归锁核心指标：L2 指标少而关键，全部纳入
GUARDED_METRICS = list(METRIC_NAMES)
DEFAULT_TOL = 0.02
BASELINE_PATH = Path(__file__).resolve().parent / "output_baseline.json"


# ===================== golden 数据加载 =====================

def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cases() -> List[Dict]:
    """加载输出测评 golden（正例 + 三类负例，多模板：news_brief/quick_qa/deep_analysis）。"""
    return _load_jsonl(_GOLDEN_OUTPUT)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# ===================== 报告结构 =====================

@dataclass
class OutputReport:
    case_count: int
    good_count: int
    bad_count: int
    macro: Dict[str, float]                  # {metric: macro-avg over all cases}
    per_tier: Dict[str, Dict[str, float]]    # {"good"/"bad": {metric: macro}}
    per_tag: Dict[str, Dict[str, float]]     # {tag: {metric: macro over subset}}
    separation_gap: float                    # 正例 output_quality 均分 − 负例均分
    per_case: Dict[str, Dict] = field(default_factory=dict)  # {case_id: evaluate_case 结果}

    def to_dict(self, with_per_case: bool = False) -> Dict:
        d = {
            "case_count": self.case_count,
            "good_count": self.good_count,
            "bad_count": self.bad_count,
            "macro": self.macro,
            "per_tier": self.per_tier,
            "per_tag": self.per_tag,
            "separation_gap": self.separation_gap,
        }
        if with_per_case:
            d["per_case"] = self.per_case
        return d


# ===================== 评估主流程 =====================

def evaluate() -> OutputReport:
    """加载 golden → 逐 case 跑 evaluate_case → macro/per_tier/per_tag 汇总 + 区分度 gap。"""
    cases = load_cases()
    per_case: Dict[str, Dict] = {
        c["case_id"]: evaluate_case(
            c["brief_output"], c["source_docs"],
            template=c.get("template", "news_brief"))
        for c in cases
    }

    macro = {m: _mean(per_case[cid][m] for cid in per_case) for m in METRIC_NAMES}

    good_ids = [c["case_id"] for c in cases if c["expected"]["quality_tier"] == "good"]
    bad_ids = [c["case_id"] for c in cases if c["expected"]["quality_tier"] == "bad"]
    per_tier = {
        "good": {m: _mean(per_case[i][m] for i in good_ids) for m in METRIC_NAMES},
        "bad": {m: _mean(per_case[i][m] for i in bad_ids) for m in METRIC_NAMES},
    }

    all_tags = sorted({t for c in cases for t in c["expected"]["tags"]})
    per_tag = {
        tag: {
            m: _mean(per_case[c["case_id"]][m] for c in cases if tag in c["expected"]["tags"])
            for m in METRIC_NAMES
        }
        for tag in all_tags
    }

    gap = per_tier["good"]["output_quality"] - per_tier["bad"]["output_quality"]

    return OutputReport(
        case_count=len(cases), good_count=len(good_ids), bad_count=len(bad_ids),
        macro=macro, per_tier=per_tier, per_tag=per_tag,
        separation_gap=gap, per_case=per_case,
    )


def format_report(report: OutputReport) -> str:
    """概览 + macro 表 + per_tier 表 + 三类缺陷子集，供 CLI/调试打印。"""
    lines = [
        f"L2 输出评估  |  {report.case_count} 条样本 "
        f"(good={report.good_count} / bad={report.bad_count})  "
        f"|  区分度 gap={report.separation_gap:.4f}",
        "",
        f"{'scope':<10}" + "".join(f"{m:>20}" for m in METRIC_NAMES),
        "-" * (10 + 20 * len(METRIC_NAMES)),
        f"{'macro':<10}" + "".join(f"{report.macro[m]:>20.4f}" for m in METRIC_NAMES),
        f"{'good':<10}" + "".join(f"{report.per_tier['good'][m]:>20.4f}" for m in METRIC_NAMES),
        f"{'bad':<10}" + "".join(f"{report.per_tier['bad'][m]:>20.4f}" for m in METRIC_NAMES),
        "",
        "缺陷子集 output_quality（应显著低于正例）:",
    ]
    for tag in ("fabricated_number", "missing_field", "ungrounded_claim"):
        pt = report.per_tag.get(tag)
        if pt:
            lines.append(f"  {tag:<20} output_quality={pt['output_quality']:.4f}  "
                         f"numerical_fidelity={pt['numerical_fidelity']:.4f}  "
                         f"field_completeness={pt['field_completeness']:.4f}")
    return "\n".join(lines)


# ===================== 基线对比 =====================

def save_baseline(report: OutputReport, path: Path = BASELINE_PATH) -> None:
    """写入 output_baseline.json（macro/per_tier/per_tag/gap/counts；不含 per_case）。

    critical_cases 是手工 curated 的关键用例下界段，重生成时保留旧值不覆盖
    （首次生成无旧段则不写，需手工 pin）。
    """
    payload = report.to_dict(with_per_case=False)
    old = load_baseline(path)
    if old and old.get("critical_cases"):
        payload["critical_cases"] = old["critical_cases"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_baseline(path: Path = BASELINE_PATH) -> Optional[Dict]:
    """读取 output_baseline.json；不存在返回 None。"""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Comparison:
    deltas: Dict[str, float]      # {metric: current_macro - baseline_macro}
    gap_delta: float              # separation_gap: current - baseline
    regressions: List[Dict]       # 核心指标 / 区分度回退超阈值明细
    tol: float

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(report: OutputReport, baseline: Dict, tol: float = DEFAULT_TOL) -> Comparison:
    """对比当前 report 与 baseline：macro 回退 + 区分度 gap 回退 + 关键用例逐 case 下界。"""
    deltas: Dict[str, float] = {}
    regressions: List[Dict] = []
    base_macro = baseline.get("macro", {})
    for m in METRIC_NAMES:
        cur = report.macro[m]
        base = base_macro.get(m, cur)   # baseline 缺该指标 -> delta=0，不误报
        delta = cur - base
        deltas[m] = delta
        if m in GUARDED_METRICS and delta < -tol:
            regressions.append({
                "scope": "macro", "metric": m,
                "baseline": round(base, 4), "current": round(cur, 4),
                "delta": round(delta, 4),
            })

    base_gap = baseline.get("separation_gap", report.separation_gap)
    gap_delta = report.separation_gap - base_gap
    if gap_delta < -tol:
        regressions.append({
            "scope": "separation", "metric": "gap",
            "baseline": round(base_gap, 4), "current": round(report.separation_gap, 4),
            "delta": round(gap_delta, 4),
        })

    # P0-2 关键用例锁：pin 下界是 curated 硬约束，不吃 tol（掉破即报警）；
    # case 消失同样报警 —— 删样本绕过回归锁是静默正确性漏洞。
    for cid, floor in baseline.get("critical_cases", {}).items():
        if cid not in report.per_case:
            regressions.append({
                "scope": "case", "metric": cid,
                "baseline": floor, "current": None, "delta": None,
            })
            continue
        cur = report.per_case[cid]["output_quality"]
        if cur < floor:
            regressions.append({
                "scope": "case", "metric": cid,
                "baseline": floor, "current": round(cur, 4),
                "delta": round(cur - floor, 4),
            })

    return Comparison(deltas=deltas, gap_delta=gap_delta,
                      regressions=regressions, tol=tol)


def format_comparison(cmp: Comparison) -> str:
    """打印 macro delta 表（带符号）+ gap delta + 回归结论。"""
    lines = [
        "", f"对比 baseline (tol={cmp.tol}，数值为 current - baseline)：",
        f"{'scope':<10}" + "".join(f"{m:>20}" for m in METRIC_NAMES),
        "-" * (10 + 20 * len(METRIC_NAMES)),
        f"{'macro':<10}" + "".join(f"{cmp.deltas[m]:>+20.4f}" for m in METRIC_NAMES),
        "", f"区分度 gap delta: {cmp.gap_delta:+.4f}",
    ]
    if cmp.regressions:
        lines.append("回归告警（核心指标 / 区分度 / 关键用例回退超阈值）：")
        for r in cmp.regressions:
            if r["current"] is None:
                lines.append(f"  [{r['scope']}] {r['metric']}: "
                             f"关键用例消失（下界 {r['baseline']}）")
            else:
                lines.append(f"  [{r['scope']}] {r['metric']}: "
                             f"{r['baseline']} -> {r['current']} (delta {r['delta']})")
    else:
        lines.append("无回归：质量指标 macro、区分度 gap 与关键用例下界均未回退。")
    return "\n".join(lines)


# ===================== CLI =====================

def main(argv=None) -> int:
    import argparse
    import logging
    logging.getLogger("financial_rag").setLevel(logging.WARNING)
    logging.getLogger("jieba").setLevel(logging.ERROR)

    ap = argparse.ArgumentParser(
        prog="python -m financial_rag.eval.output_runner",
        description="L2 输出层离线评估（多模板质量指标 + 区分度 + 关键用例锁 + 基线对比）")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="核心指标 / 区分度回退容忍阈值（默认 0.02；关键用例下界不吃 tol）")
    ap.add_argument("--update-baseline", action="store_true",
                    help="用本次结果覆盖生成 output_baseline.json（保留手工 critical_cases）")
    args = ap.parse_args(argv)

    report = evaluate()
    print(format_report(report))

    if args.update_baseline:
        save_baseline(report)
        print(f"\nbaseline 已写入 {BASELINE_PATH.name}")
        return 0

    baseline = load_baseline()
    if baseline is None:
        print("\n（尚无 output_baseline.json，用 --update-baseline 生成首个基线）")
        return 0

    cmp = compare(report, baseline, tol=args.tol)
    print(format_comparison(cmp))
    return 0 if cmp.ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
