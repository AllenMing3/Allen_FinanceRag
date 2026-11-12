"""
L3 LLM-as-judge 在线评估（辅助观测）— RAGAS faithfulness（claim 分解 + NLI 逐条判定）。

替代旧 guard L5/L6 手写 prompt 审计层（已淘汰删除）：深度语义评判前移到
离线 eval，运行时路径只保留 L1-L4 确定性规则层（零 LLM 成本）。

定位（辅，非主门禁）：
- 质量主门禁 = L1/L2 离线尺子（output_runner / runner）+ step7 回归锁；
  本模块是补充的 LLM-as-judge 观测通道，把 RAGAS 架子搭好、落盘得分供人工查看。
- 在线 judge 对中文改写声明判定有噪声（实测 good−defect mean-gap 仅 0.04-0.06），
  低得分/弱区分度在当前阶段可接受；judge 精度改善为后续项。
- 因此 calibrate 的区分度信号仅报告、不作为硬门禁：跑通产出得分即 exit 0。

评估对象与 output_runner（L2 离线规则指标）同一份 golden：
    eval/golden/output_news_brief.jsonl（9 正例 + 13 负例，多源多模板，负例三类缺陷标签）
每 case 按模板提取声明型字段（与 L2 尺子 _TEMPLATE_SPECS 同源）：
NEWS_BRIEF 取 title/市场/展望，quick_qa 取回答/数据支撑，deep_analysis 取全部段落。

区分度信号（对称 output_runner 的 separation_gap 思路，报告用、不阻断）：
1. good tier faithfulness 均值 − 缺陷子集均值（gap）
2. 缺陷检出率 detection_rate：每条 faithfulness 缺陷是否被拆出无据声明(score<1.0)
3. fabricated_number 子集均值是否低于 good tier（与 L2 numerical_fidelity 交叉印证）

设计原则：
- 使用 ragas 0.4.x 现代 v2 接口（metrics.collections.Faithfulness + llm_factory
  instructor 结构化输出）；旧 ragas.metrics faithfulness 与 evaluate() 均已废弃。
- 在线评估有波动，结果写 ragas_results.json，**不进** output_baseline.json，
  不破坏 pytest 离线确定性；CI 不跑本模块（需 API key + 重依赖）。
- 禁止静默跳过：缺 DASHSCOPE_API_KEY / 未装 ragas / 得分 NaN 等**结构性故障**
  明确报错退出（exit 2），不得静默返回成功；但区分度得分低不算故障（exit 0）。
- 依赖隔离：ragas 依赖链见 requirements-eval.txt，不进主 requirements.txt。

CLI: python -m financial_rag.eval.ragas_judge [--judge-model M] [--gap-threshold X]
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from financial_rag.eval.output_metrics import _get_template_spec

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_OUTPUT = _EVAL_DIR / "golden" / "output_news_brief.jsonl"
RESULTS_PATH = _EVAL_DIR / "ragas_results.json"

# 区分度校准默认阈值：good−bad faithfulness 均值差下限。
# 在线 judge 有波动，首跑实测后可用 --update-expected 按实测 gap 重校。
DEFAULT_GAP_THRESHOLD = 0.15

# faithfulness 类缺陷标签（参与校准）；missing_field 与忠实度正交，另行报告
FAITHFULNESS_DEFECT_TAGS = ("fabricated_number", "ungrounded_claim")
ORTHOGONAL_DEFECT_TAGS = ("missing_field",)

# DashScope OpenAI 兼容端点（复用 experiments/ragas_eval.py 已验证的适配）
_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 默认 judge 模型（可用 RAGAS_JUDGE_MODEL 环变量 / --judge-model 覆盖）。
# 实测：qwen-plus 能产出有效结构化判定（对中文改写声明有噪声）；
# qwen-max 对 RAGAS instructor schema 返回残缺 JSON，校验失败，不适作 judge。
_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "qwen-plus")


# ===================== golden → RAGAS 样本映射（纯函数，离线可测） =====================

# 声明型字段提取：复用 L2 尺子的模板字段规格（output_metrics._TEMPLATE_SPECS），
# 保证 judge 与规则层评的是同一份声明内容。分类元数据字段（日期/影响级别/
# 主体列表/行业标签/来源标注/可信度自评）不是可验证声明，喂 faithfulness judge
# 会被当作无据声明扣分 —— 实测 good 档 auto_04 三条真实声明全部有据却只得 0.6，
# gap 被压到贴阈值边缘。故按模板只提取声明型字段。
_CLAIM_META_FIELDS = {
    "news_brief": {"event_date", "impact_level", "key_players", "sector_impact"},
    "quick_qa": {"source_note", "confidence_note"},
    "deep_analysis": set(),
}
# 占位符（未有效填充）不作为声明
_BRIEF_PLACEHOLDER_RE = re.compile(r"^（.*暂缺）$|^无$")


def extract_brief_claims(brief_output: str, template: str = "news_brief") -> str:
    """按模板提取声明型字段，丢弃分类元数据（与 L2 尺子字段规格同源）。

    faithfulness = supported_claims / total_claims；若把元数据碎片也当声明，
    单个编造数字会被大量样板行稀释（缺陷档得分虚高），同时 good 档因元数据
    无法溯源被误扣。只喂真实声明可显著拉开区分度。

    多模板（P0-1）：news_brief 取 title/市场/展望；quick_qa 取回答/数据支撑；
    deep_analysis 取全部 6 段落。旧实现只适配 NEWS_BRIEF，quick_qa/deep 会被
    标题正则截成空标题（'# 回答'/'深度分析报告'）喂 judge —— 已修复。

    Raises:
        ValueError: 提取不到任何声明字段 —— brief 格式异常，显式报错而非喂空文本。
    """
    text = brief_output or ""
    spec = _get_template_spec(template)
    meta = _CLAIM_META_FIELDS.get(template, set())
    claims: List[str] = []
    for key, pat in spec["fields"]:
        if key in meta:
            continue
        m = pat.search(text)
        if not m:
            continue
        v = m.group(1).strip()
        if v and not _BRIEF_PLACEHOLDER_RE.match(v):
            claims.append(v)
    if not claims:
        raise ValueError(
            f"无法从 brief_output 提取声明型字段(template={template})，"
            f"格式可能不匹配模板: {text[:60]!r}"
        )
    return "\n".join(claims)


def load_cases(path: Path = _GOLDEN_OUTPUT) -> List[Dict]:
    """加载 NEWS_BRIEF 输出测评 golden（与 output_runner 同一份）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_ragas_samples(cases: List[Dict]) -> List[Dict]:
    """golden case → RAGAS faithfulness 所需样本形状。

    faithfulness 只消费 answer + contexts；question 一并携带供报告展示。
    contexts = source_docs 的 text 列表（与 Guard l1/l2 的 sources 同源）。

    brief_output 喂 judge 前用 extract_brief_claims 按模板只提取声明型字段，
    丢弃分类元数据 —— 元数据碎片会被 judge 当作无据声明扣分，实测把 good 档
    压到 0.6、gap 贴阈值边缘。

    Raises:
        ValueError: case 缺 answer 或 contexts 为空 —— faithfulness 无法评判，
            显式报错而非喂给 RAGAS 产生难以归因的 NaN。
    """
    samples = []
    for c in cases:
        contexts = [d["text"] for d in c.get("source_docs", []) if d.get("text")]
        answer = extract_brief_claims(
            c.get("brief_output") or "",
            template=c.get("template", "news_brief")).strip()
        question = (c.get("query") or "").strip()
        if not question:
            raise ValueError(f"case {c.get('case_id', '?')} 的 query 为空，"
                             f"Faithfulness.ascore 需要非空 user_input")
        if not answer:
            raise ValueError(f"case {c.get('case_id', '?')} 的 brief_output 为空，无法评判")
        if not contexts:
            raise ValueError(f"case {c.get('case_id', '?')} 的 source_docs 无有效 text，"
                             f"faithfulness 需要非空 contexts")
        samples.append({
            "case_id": c["case_id"],
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "quality_tier": c["expected"]["quality_tier"],
            "tags": c["expected"].get("tags", []),
        })
    return samples


# ===================== Judge LLM 配置 =====================

def get_judge_llm(model: str = None):
    """配置 RAGAS v2 judge LLM（DashScope OpenAI 兼容接口 + instructor 结构化输出）。

    缺 API key 时显式抛错（禁止静默跳过契约）。model 为 None 时用默认 _JUDGE_MODEL。
    """
    # CLI 直跑时环境里通常没有 key —— 与生产 config 同源，从 .env 自动加载
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY 环境变量 — RAGAS judge 无法执行。"
            "请在 .env 或环境中配置后重试（本模块不允许静默跳过）。"
        )
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(api_key=api_key, base_url=_DASHSCOPE_BASE)
    return llm_factory(model or _JUDGE_MODEL, client=client, temperature=0.0, max_tokens=2048)


def _build_cn_faithfulness_prompts():
    """中文化 RAGAS Faithfulness 两步 prompt（声明分解 + NLI），修复 good 档假阴性。

    B 诊断（claim 级 dump）坐实：RAGAS 默认英文 prompt 使 qwen-plus 把中文声明翻译成
    英文再判定，翻译引入语义偏移（省略主语错绑相邻主体、"计划/布局"强化为"已部署"）
    并过度拆分并列成分 → auto_04=0.5 / api_05=0.667 假阴性。中文化后声明保持中文、
    不过度拆分、主语补全更准（实测 api_05→1.0）。NLI 阶段本身判得准，失效只在分解。

    懒加载 import ragas 保持"未装也不报错"契约；属 eval judge 内部实现，不进 prompts.py
    （业务生成 prompt 不应引入 ragas 依赖）。
    """
    from ragas.metrics.collections.faithfulness.util import (
        NLIStatementInput,
        NLIStatementOutput,
        NLIStatementPrompt,
        StatementFaithfulnessAnswer,
        StatementGeneratorInput,
        StatementGeneratorOutput,
        StatementGeneratorPrompt,
    )

    class StatementGeneratorPromptCN(StatementGeneratorPrompt):
        instruction = (
            "给定一个问题和一个回答，把回答拆分成若干条可独立理解、可验证的原子声明。规则："
            "(1) 保持与回答相同的语言（中文），严禁翻译成英文；"
            "(2) 若某句省略主语，补全时必须用回答/问题中真实指代的主体，严禁错绑到相邻句的其它主体；"
            "(3) 严禁改变原意强度，如'计划/拟/布局'不得强化为'已完成/已部署/已实现'；"
            "(4) 并列成分（如'东南亚和中东'）保持一条，不拆成多条改变原意的声明；"
            "(5) 不使用代词。"
        )
        examples = [
            (
                StatementGeneratorInput(
                    question="商汤科技2024年年报披露了哪些业务进展？",
                    answer="商汤绝影与超30家车企合作，覆盖L2+到L4级别方案。公司计划布局东南亚和中东市场。",
                ),
                StatementGeneratorOutput(statements=[
                    "商汤绝影自动驾驶平台已与超过30家车企合作。",
                    "商汤绝影自动驾驶方案覆盖L2+到L4级别。",
                    "商汤科技计划布局东南亚和中东市场。",
                ]),
            ),
        ]

    class NLIStatementPromptCN(NLIStatementPrompt):
        instruction = (
            "你的任务是依据给定上下文判断一系列声明的忠实度。对每条声明：能由上下文直接推出则 "
            "verdict=1，不能直接推出则 verdict=0。保持中文语义判断，注意：'计划/布局'不等于"
            "'已完成/已部署'；上下文未提及的信息一律 verdict=0；主体张冠李戴也判 verdict=0。"
        )
        examples = [
            (
                NLIStatementInput(
                    context="商汤科技发布2024年年报，全年营业收入50.3亿元，同比增长36.4%。"
                            "公司计划进一步扩大算力基础设施投资，并在东南亚和中东市场布局海外业务。",
                    statements=[
                        "商汤科技2024年营业收入50.3亿元。",
                        "商汤科技已在东南亚市场部署海外业务。",
                        "商汤科技2024年净利润为10亿元。",
                    ],
                ),
                NLIStatementOutput(statements=[
                    StatementFaithfulnessAnswer(
                        statement="商汤科技2024年营业收入50.3亿元。",
                        reason="上下文明确写明'全年营业收入50.3亿元'，可直接推出。", verdict=1),
                    StatementFaithfulnessAnswer(
                        statement="商汤科技已在东南亚市场部署海外业务。",
                        reason="上下文说的是'计划…布局海外业务'，属计划而非已部署，不能推出'已部署'。", verdict=0),
                    StatementFaithfulnessAnswer(
                        statement="商汤科技2024年净利润为10亿元。",
                        reason="上下文未提及净利润具体数字，无法推出。", verdict=0),
                ]),
            ),
        ]

    return StatementGeneratorPromptCN(), NLIStatementPromptCN()


# ===================== 评估主流程 =====================

def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def build_metric_inputs(samples: List[Dict]) -> List[Dict]:
    """RAGAS 样本 → Faithfulness.ascore 的 kwargs 形状（v2 接口约定键名）。"""
    return [
        {
            "user_input": s["question"],
            "response": s["answer"],
            "retrieved_contexts": s["contexts"],
        }
        for s in samples
    ]


def run_faithfulness(samples: List[Dict], judge_model: str = None) -> List[float]:
    """对样本列表跑 RAGAS v2 Faithfulness（claim 分解 + NLI），返回同序逐 case 得分。

    并发批量评判（abatch_score = asyncio.gather），11 条 golden 样本墙钟耗时
    约为单条的 1-2 倍而非 11 倍。

    Raises:
        RuntimeError: 存在 NaN 得分（judge 无法从回答拆出任何原子声明，
            通常是 LLM 输出解析失败）—— 显式报错而非静默填 0 污染校准。
    """
    import asyncio

    from ragas.metrics.collections import Faithfulness

    model = judge_model or _JUDGE_MODEL
    metric = Faithfulness(llm=get_judge_llm(model))
    # B 诊断坐实：默认英文 prompt 使 judge 把中文声明翻译扭曲（主语错绑/过度拆分），
    # 注入中文化两步 prompt 修复（详见 _build_cn_faithfulness_prompts）
    metric.statement_generator_prompt, metric.nli_statement_prompt = (
        _build_cn_faithfulness_prompts())
    inputs = build_metric_inputs(samples)

    logger.info(f"开始 RAGAS faithfulness 评估: {len(samples)} 条样本, judge={model}")
    t0 = time.time()
    results = asyncio.run(metric.abatch_score(inputs))
    logger.info(f"RAGAS 评估完成, 耗时 {time.time() - t0:.1f}s")

    if len(results) != len(samples):
        raise RuntimeError(f"RAGAS 结果数 {len(results)} != 样本数 {len(samples)}，无法对齐")
    scores = []
    nan_cases = []
    for s, r in zip(samples, results):
        v = r.value
        if v is None or v != v:  # None / NaN（NaN != 自身）
            nan_cases.append(s["case_id"])
            continue
        scores.append(float(v))
    if nan_cases:
        raise RuntimeError(
            f"RAGAS 对 {len(nan_cases)} 条样本返回 NaN 得分: {nan_cases} —— "
            f"通常是 judge LLM 输出解析失败或回答拆不出声明，"
            f"请检查 API 配额/模型可用性后重跑"
        )
    return scores


def calibrate(samples: List[Dict], scores: List[float],
              gap_threshold: float = DEFAULT_GAP_THRESHOLD) -> Dict:
    """缺陷检出率 + 方向性校准（断言设计依据见模块 docstring）。

    返回 {good_mean, defect_mean, bad_mean, missfield_mean, gap, detection_rate,
          gap_ok, detection_ok, fabnum_ok, per_case, violations}
    """
    per_case = [
        {"case_id": s["case_id"], "tier": s["quality_tier"],
         "tags": s["tags"], "faithfulness": round(sc, 4)}
        for s, sc in zip(samples, scores)
    ]

    def _subset(pred):
        return [sc for s, sc in zip(samples, scores) if pred(s)]

    good = _subset(lambda s: s["quality_tier"] == "good")
    bad = _subset(lambda s: s["quality_tier"] == "bad")
    defect_cases = [(s, sc) for s, sc in zip(samples, scores)
                    if any(t in s["tags"] for t in FAITHFULNESS_DEFECT_TAGS)]
    defect = [sc for _, sc in defect_cases]
    fabnum = _subset(lambda s: "fabricated_number" in s["tags"])
    missfield = _subset(lambda s: any(t in s["tags"] for t in ORTHOGONAL_DEFECT_TAGS))

    good_mean, bad_mean = _mean(good), _mean(bad)
    defect_mean = _mean(defect)
    gap = good_mean - defect_mean

    # 检出率：每条 faithfulness 缺陷都必须被拆出至少一条无据声明（score < 1.0）
    missed = [s["case_id"] for s, sc in defect_cases if sc >= 1.0]
    detection_rate = 1.0 - len(missed) / len(defect_cases) if defect_cases else 0.0

    violations = []
    if not good or not defect:
        violations.append(
            f"golden 分层缺失: good={len(good)} 条 / faithfulness缺陷={len(defect)} 条，无法校准"
        )
    if missed:
        violations.append(
            f"缺陷未检出 (score=1.0): {missed} —— judge 未拆出任何无据声明"
        )
    if gap < gap_threshold:
        violations.append(
            f"方向性不足: good−defect gap={gap:.4f} < 阈值 {gap_threshold}"
        )
    if fabnum and _mean(fabnum) >= good_mean:
        violations.append(
            f"编造数字未抓住: fabricated_number 均值 {_mean(fabnum):.4f} "
            f">= good 均值 {good_mean:.4f}"
        )

    return {
        "good_mean": round(good_mean, 4),
        "defect_mean": round(defect_mean, 4),
        "bad_mean": round(bad_mean, 4),
        "missfield_mean": round(_mean(missfield), 4),
        "gap": round(gap, 4),
        "gap_threshold": gap_threshold,
        "detection_rate": round(detection_rate, 4),
        "gap_ok": gap >= gap_threshold,
        "detection_ok": not missed,
        "fabnum_ok": (not fabnum) or _mean(fabnum) < good_mean,
        "per_case": per_case,
        "violations": violations,
    }


def format_calibration(calib: Dict) -> str:
    """打印校准报告，供 CLI 输出。"""
    lines = [
        "",
        "RAGAS faithfulness 校准（缺陷检出率 + 方向性）：",
        f"  good 均值        : {calib['good_mean']:.4f}",
        f"  缺陷子集均值    : {calib['defect_mean']:.4f}  (fabnum+ungrounded)",
        f"  missfield 均值   : {calib['missfield_mean']:.4f}  (正交缺陷，不参与校准)",
        f"  缺陷检出率      : {calib['detection_rate']:.0%}"
        f"  {'PASS' if calib['detection_ok'] else 'FAIL'}",
        f"  gap (good−defect): {calib['gap']:+.4f}  (阈值 {calib['gap_threshold']})"
        f"  {'PASS' if calib['gap_ok'] else 'FAIL'}",
        "",
        "逐 case 得分：",
    ]
    for pc in calib["per_case"]:
        lines.append(f"  [{pc['tier']:<4}] {pc['case_id']:<32} faithfulness={pc['faithfulness']:.4f}")
    if calib["violations"]:
        lines.append("")
        lines.append("校准告警：")
        for v in calib["violations"]:
            lines.append(f"  - {v}")
    else:
        lines.append("")
        lines.append("校准通过：judge 区分度与编造数字捕获均达标。")
    return "\n".join(lines)


def save_results(calib: Dict, judge_model: str = None, path: Path = None) -> None:
    """落盘 ragas_results.json（带时间戳；在线评估有波动，不参与离线基线）。

    path 延迟绑定模块全局 RESULTS_PATH（而非默认参数），便于测试重定向落盘路径。
    """
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": judge_model or _JUDGE_MODEL,
        "metric": "faithfulness",
        "role": "auxiliary",  # 辅：区分度信号仅观测，主门禁在 L1/L2 尺子 + step7
        **{k: v for k, v in calib.items()},
    }
    with open(path or RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_expected(path: Path = RESULTS_PATH) -> Optional[Dict]:
    """读取上次校准结果；不存在返回 None。"""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ===================== CLI =====================

def main(argv=None) -> int:
    import argparse
    # Windows GBK 控制台 print 中文 / U+2212(−) 会 UnicodeEncodeError → 强制 std 流 utf-8
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    logging.getLogger("ragas").setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(
        prog="python -m financial_rag.eval.ragas_judge",
        description="L3 LLM-as-judge 在线评估（辅助观测：RAGAS faithfulness + 区分度信号）")
    ap.add_argument("--judge-model", default=None,
                    help=f"judge 模型（默认 {_JUDGE_MODEL}；qwen-max 对 instructor schema 不兼容）")
    ap.add_argument("--gap-threshold", type=float, default=None,
                    help=f"good−defect 区分度参考线（仅报告用，不阻断；默认 {DEFAULT_GAP_THRESHOLD}）")
    args = ap.parse_args(argv)

    try:
        cases = load_cases()
        samples = build_ragas_samples(cases)
        scores = run_faithfulness(samples, judge_model=args.judge_model)
    except ImportError as e:
        print(f"\nRAGAS 依赖未安装: {e}\n"
              f"请运行: pip install -r requirements-eval.txt", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as e:
        # 结构性故障（缺 key / NaN / golden 格式错）—— 禁止静默跳过，硬报错
        print(f"\n{e}", file=sys.stderr)
        return 2

    calib = calibrate(samples, scores, gap_threshold=args.gap_threshold or DEFAULT_GAP_THRESHOLD)
    save_results(calib, judge_model=args.judge_model)

    print(format_calibration(calib))
    # 辅定位：跑通产出得分即成功；区分度/低分仅观测，不作硬门禁（主门禁在 L1/L2 + step7）
    if calib["violations"]:
        print("\n[辅助观测] 上述区分度信号未达参考线 —— 在当前阶段可接受（judge 精度改善为后续项）。")
        print("质量主门禁仍以 L1/L2 离线尺子（output_runner / runner）+ step7 回归锁为准。")
    print(f"\n结果已写入 {RESULTS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
