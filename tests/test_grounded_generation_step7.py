"""
Test Step 7 生成有据化 — 防幻觉护栏质量验证（RAG 流水线第七步）。

链路位置：过滤/门控 → 上下文组装(已测: test_context_builder.py) → LLM 生成 → **有据化校验(Guard)**
对应生产代码：
  - financial_rag/guard/rule_layers.py（L1-L4 确定性规则层）
  - financial_rag/guard/reflector.py（HallucinationGuard 编排：加权/风险/LLM层触发）
  - financial_rag/guard/serializer.py（3 个调用点统一输出格式）

按项目四条测试标准组织：① 证明效果 ② 可观测 ③ 一致性 ④ 回归锁定

核心发现（探针 experiments/probe_step7_guard.py + probe_step7_fix_dir.py 实然证据）：
  - ✅ BUG(已修复): L1 来源锚定阈值 0.15 过宽 —— 编造句"腾讯收购月之暗面"仅凭
    领域共享词(公司/收购/交易/披露/2024)overlap 0.182 即锚定成功，整篇 0.964/risk=low
    放行。分离度实测：编造句 overlap ≤0.25 < 合法改写句 ≥0.50，阈值提到 0.30 干净分离。
  - ✅ BUG(已修复): extract_numbers 把引用标记 [N] 当数字 —— L3 要求的规范引用
    反而拖累 L2（满分回答被扣到 0.85，unmatched=['1','1','3']）。现已先剥离 [N]。
  - ⚠️ 已知局限: L2 阈值 0.5 —— 半数数字编造仍 passed（特征化锁定，未改行为）
  - ✅ D/E/F/G case（无效引用/无结构/analysis模式/空sources）行为符合契约

离线约束：不注入 LLM（llm=None），L5/L6 只测触发/跳过逻辑与显式标记；
检索无关，全部用 conftest 样本文本构造 sources。
"""
import pytest

from financial_rag.guard.reflector import HallucinationGuard
from financial_rag.guard.rule_layers import (
    GROUNDING_THRESHOLD,
    extract_numbers,
    l1_source_grounding,
    l2_numerical_fidelity,
    l3_citation_integrity,
    l4_structure_compliance,
)
from financial_rag.guard.serializer import serialize_guard_result
from tests.conftest import (
    SAMPLE_AI_FINANCIAL_REPORT,
    SAMPLE_AI_NEWS,
    SAMPLE_AI_FUNDING,
)


# ===================== 共享样本 =====================

def _sources():
    """3 篇真实金融文本作为 grounding 来源（与生产调用点同构：{"text": ...}）。"""
    return [
        {"text": SAMPLE_AI_FINANCIAL_REPORT},
        {"text": SAMPLE_AI_NEWS},
        {"text": SAMPLE_AI_FUNDING},
    ]


# 完全有据回答：事实全部来自 sources，含 [N] 引用 + 4 个 Markdown 段落
GROUNDED_ANSWER = """# 摘要
商汤科技2024年实现营业收入50.3亿元，同比增长36.4%，生成式AI业务收入占比达60%[1]。

# 要点
- 全年净亏损42.1亿元，同比收窄28.3%，研发投入18.7亿元[1]。
- 英伟达发布Blackwell B200 GPU，单卡AI训练性能较H100提升4倍[2]。

# 分析
智谱AI完成B+轮融资，融资金额超50亿元，投后估值超200亿元，GLM系列模型累计服务企业客户超3000家[3]。

# 风险提示
全球AI算力投资预计2025年超3000亿美元，行业竞争加剧可能带来不确定性[2]。
"""

# 编造事实句：主体/事件均不在任何 source 中，仅含领域共享词
FAB_SENTENCE = "腾讯于2024年第四季度全资收购了月之暗面公司，交易金额尚未披露。"

FABRICATED_ANSWER = GROUNDED_ANSWER.replace("# 分析\n", f"# 分析\n{FAB_SENTENCE}\n")

# 篡改数字：50.3亿→9000亿、36.4%→80%（其余保持有据）
TAMPERED_ANSWER = GROUNDED_ANSWER.replace("50.3亿元", "9000亿元").replace("36.4%", "80%")


# ======================================================================
#  ④ 回归锁定：探针抓出的 2 个 live bug（修复后转常绿，退回旧实现即红）
# ======================================================================

class TestGroundingThresholdRegression:
    """BUG: L1 阈值 0.15 过宽，编造句凭领域共享词锚定成功（探针实测 0.964/low 放行）。"""

    @pytest.mark.regression
    def test_fabricated_sentence_not_anchored(self):
        """编造句必须进 unanchored —— 旧阈值 0.15 下 overlap 0.182 会锚定成功即红。"""
        r = l1_source_grounding(FABRICATED_ANSWER, _sources())
        assert any("月之暗面" in u for u in r["unanchored"]), (
            f"编造句应被 L1 抓住，实际 unanchored={r['unanchored']}"
        )

    @pytest.mark.regression
    def test_fabricated_answer_not_low_risk(self):
        """含编造句的回答不得再以 risk=low 放行（探针实然: overall 0.964 / low）。"""
        result = HallucinationGuard(llm=None).check(FABRICATED_ANSWER, _sources())
        assert result["risk"] != "low" or result["checks"]["L1_source_grounding"]["unanchored"], (
            "编造内容 risk=low 且零 unanchored —— 护栏核心保证失效"
        )

    @pytest.mark.regression
    def test_legit_paraphrase_still_anchored(self):
        """阈值收紧不得误杀合法改写句（防矫枉过正：overlap 实测 ≥0.50）。"""
        legit = (
            "# 摘要\n商汤科技去年营收达到50.3亿元，比前年增长了36.4%[1]。\n"
            "# 要点\n英伟达推出的B200显卡训练性能是H100的四倍[2]。\n"
            "# 分析\n智谱AI这轮融资拿到了超过50亿元，估值突破200亿元[3]。\n"
            "# 风险提示\n微软的云平台上已经部署了十万张B200用于训练GPT-5，资本开支存在不确定性[2]。\n"
        )
        r = l1_source_grounding(legit, _sources())
        assert r["unanchored"] == [], f"合法改写句被误杀: {r['unanchored']}"
        assert r["passed"]

    def test_threshold_value_locked(self):
        """阈值钉在分离间隙 (0.25, 0.50) 内 —— 改回 0.15 或超过 0.50 都即红。"""
        assert 0.25 < GROUNDING_THRESHOLD <= 0.35


class TestCitationNumberRegression:
    """BUG: extract_numbers 把 [N] 引用标记当数字，规范引用反而拖累 L2。"""

    @pytest.mark.regression
    def test_citation_markers_not_extracted_as_numbers(self):
        nums = extract_numbers("营收50.3亿元，同比增长36.4%[1]。估值超200亿元[3]。")
        assert "1" not in nums and "3" not in nums, f"引用标记被当数字: {nums}"
        assert "50.3亿" in nums and "36.4%" in nums and "200亿" in nums

    @pytest.mark.regression
    def test_grounded_answer_l2_perfect(self):
        """满分有据回答的 L2 必须 1.0 —— 旧行为被 [1][3] 污染扣到 0.85 即红。"""
        r = l2_numerical_fidelity(GROUNDED_ANSWER, _sources())
        assert r["score"] == 1.0, f"unmatched={r['unmatched']}"
        assert r["unmatched"] == []

    def test_source_side_still_extracts(self):
        """剥离 [N] 不影响 source 侧正常数字提取（如 '占60%'、'超3000家'）。"""
        nums = extract_numbers(SAMPLE_AI_FINANCIAL_REPORT)
        assert "50.3亿" in nums and "36.4%" in nums


# ======================================================================
#  ① 证明效果：护栏对 4 类幻觉/缺陷的判别力（端到端 check）
# ======================================================================

class TestGuardDiscrimination:
    """核心保证：有据回答放行，各类缺陷被抓。"""

    def test_grounded_answer_passes_low_risk(self):
        result = HallucinationGuard(llm=None).check(GROUNDED_ANSWER, _sources())
        assert result["passed"] is True
        assert result["risk"] == "low"
        assert result["overall_score"] >= 0.9

    def test_tampered_numbers_caught_by_l2(self):
        """篡改数字必须出现在 L2 unmatched，score 显著低于满分回答。"""
        r = l2_numerical_fidelity(TAMPERED_ANSWER, _sources())
        assert "9000亿" in r["unmatched"] and "80%" in r["unmatched"]
        clean = l2_numerical_fidelity(GROUNDED_ANSWER, _sources())
        assert r["score"] < clean["score"]

    def test_invalid_citations_fail_l3(self):
        bad = GROUNDED_ANSWER.replace("[1]", "[7]").replace("[3]", "[99]")
        r = l3_citation_integrity(bad, _sources())
        assert r["passed"] is False
        assert r["invalid"] == 3 and r["valid"] == 2

    def test_missing_structure_fails_l4(self):
        flat = "商汤科技2024年实现营业收入50.3亿元，同比增长36.4%。全年净亏损42.1亿元。"
        r = l4_structure_compliance(flat)
        assert r["passed"] is False
        assert len(r["missing_sections"]) == 4

    def test_empty_sources_no_crash_and_fail(self):
        """边界：sources 为空 → 不崩、判 fail/high（探针 case G 契约）。"""
        result = HallucinationGuard(llm=None).check(
            "智谱AI完成融资超50亿元，投后估值超200亿元[1]。", [])
        assert result["passed"] is False
        assert result["risk"] == "high"
        assert result["overall_score"] == 0.0


# ======================================================================
#  ② 可观测：跳过显式打标 + 报告透出 + 序列化白名单
# ======================================================================

class TestObservability:
    """禁止静默：L5/L6 跳过必须带 reason；报告/序列化必须透出诊断。"""

    def test_llm_layers_skipped_with_explicit_reason(self):
        result = HallucinationGuard(llm=None).check(GROUNDED_ANSWER, _sources())
        for layer in ("L5_llm_critique", "L6_llm_assist"):
            c = result["checks"][layer]
            assert c["skipped"] is True
            assert c["skip_reason"], f"{layer} 跳过但无 reason —— 违反禁止静默契约"
        assert result["llm_layers_active"] is False

    def test_report_exposes_warnings_and_skip_reasons(self):
        result = HallucinationGuard(llm=None).check(TAMPERED_ANSWER, _sources())
        report = result["report"]
        assert "质量评分" in report
        assert "数值无来源" in report, "篡改数字必须出现在报告警告区"
        assert "未执行" in report, "L5/L6 跳过必须在报告中显式可见"

    def test_serializer_whitelist_and_no_raw_leak(self):
        result = HallucinationGuard(llm=None).check(GROUNDED_ANSWER, _sources())
        api = serialize_guard_result(result)
        assert set(api) == {"overall_score", "risk", "passed", "layers"}
        assert set(api["layers"]) == {
            "L1_source_grounding", "L2_numerical_fidelity",
            "L3_citation_integrity", "L4_structure_compliance",
            "L5_llm_critique", "L6_llm_assist",
        }
        # 跳过层的诊断字段透出
        l5 = api["layers"]["L5_llm_critique"]
        assert l5["skipped"] is True and l5["skip_reason"]
        # 不泄漏 raw 字段
        for layer in api["layers"].values():
            assert "raw" not in layer


# ======================================================================
#  ③ 一致性：同输入同输出 / 跨层同源不变量 / 双模式契约
# ======================================================================

class TestConsistency:
    def test_deterministic_same_input_same_output(self):
        g = HallucinationGuard(llm=None)
        r1 = g.check(GROUNDED_ANSWER, _sources())
        r2 = g.check(GROUNDED_ANSWER, _sources())
        assert r1["overall_score"] == r2["overall_score"]
        assert r1["risk"] == r2["risk"]

    def test_passed_threshold_invariant(self):
        """overall.passed 与顶层 passed 必须同源（同一 threshold 判定）。"""
        g = HallucinationGuard(llm=None)
        for ans in (GROUNDED_ANSWER, TAMPERED_ANSWER, FABRICATED_ANSWER):
            r = g.check(ans, _sources())
            assert r["passed"] == (r["overall_score"] >= g.threshold)

    def test_analysis_mode_lenient_contract(self):
        """analysis 模式：无 [N] 但有文字引用 → L3=0.8；【】段落 → L4 高分。"""
        text = (
            "【关键信号】英伟达Blackwell B200训练性能较H100提升4倍，推理效率提升30倍。"
            "【影响分析】微软Azure部署10万张B200，谷歌TPU v6同步对标，算力竞争白热化。"
            "【风险提示】全球AI算力投资2025年预计超3000亿美元，资本开支存在不确定性。"
            "【后续关注】持续跟踪OpenAI新一代模型的推理成本变化与量产进度。"
        )
        l3 = l3_citation_integrity(text, _sources(), mode="analysis")
        assert l3["score"] == 0.8 and l3["passed"] is True
        l4 = l4_structure_compliance(text, mode="analysis")
        assert l4["score"] == 1.0 and len(l4["found_sections"]) == 4

    def test_rag_vs_analysis_mode_same_answer_different_leniency(self):
        """同一【】段落文本：analysis 模式 L4 得分必须高于 rag 模式（宽严契约）。"""
        text = "【关键信号】训练性能提升4倍。【风险提示】资本开支存在不确定性。"
        assert (l4_structure_compliance(text, mode="analysis")["score"]
                > l4_structure_compliance(text, mode="rag")["score"])


# ======================================================================
#  已知局限特征化锁定（不改行为，防悄悄"修复"引起语义漂移）
# ======================================================================

class TestKnownLimitations:
    """⚠️ 特征化：这些是当前设计的已知宽松点，测试锁定现状而非认可其正确。"""

    def test_l2_half_fabricated_numbers_still_passes(self):
        """已知局限: L2 阈值 0.5 —— 半数数字编造仍 passed（探针 P3 实证）。"""
        half = "商汤科技2024年实现营业收入9000亿元，同比增长36.4%，净亏损42.1亿元，研发投入99999亿元。"
        r = l2_numerical_fidelity(half, _sources())
        assert r["verified"] == 3 and r["total"] == 5
        assert r["passed"] is True, "若此处变红说明 L2 阈值被调整 —— 需同步更新本特征化用例"
        assert "9000亿" in r["unmatched"]

    def test_l1_short_sentences_skipped_as_anchored(self):
        """已知局限: L1 对 <2 token 的短句直接计为锚定（宽松跳过）。"""
        r = l1_source_grounding("是的。好的。", _sources())
        assert r["score"] == 1.0 and r["total"] == 0

    def test_l1_semantic_reversal_not_caught(self):
        """已知局限（实测锁定）: 词级极性反转抓不住。

        "下降15%" vs source "增长15%"：整句仅一词之差，其余 token 高度重叠，
        overlap 远超 0.30 阈值 → 仍判 score=1.0/passed。提阈只能抓整句编造（低重叠），
        抓不住据实句中的语义反转——需词级极性/方向校验（见 ARCHITECTURE.md §8 P1-6）。
        若此处变红说明 L1 已能识别极性反转 → 需同步更新本特征化用例与 backlog。
        """
        src = [{"text": "商汤科技2024年营业收入同比增长15%，业务持续扩张。"}]
        reversed_claim = "商汤科技2024年营业收入同比下降15%，业务持续扩张。"
        r = l1_source_grounding(reversed_claim, src)
        assert r["passed"] is True and r["unanchored"] == [], (
            "L1 竟然抓住了语义反转？需核实并更新本特征化用例与 ARCHITECTURE P1-6"
        )
