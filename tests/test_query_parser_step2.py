"""
Test Step 2 query 解析 — 原始 query → 结构化 QueryResult（RAG 流水线第二步）。

沿 RAG 每一步建探针的第二个永久资产。方法：先从契约推"应然"，再用一次性探针
观察"实然"，吻合的钉成断言，抓到的 bug 用 xfail(strict) 回归锁。按四条标准组织：
  ① 证明效果  ② 可观测  ③ 一致性  ④ 回归锁定

断言只锚定契约级不变量（正则/权重/注入词典），不做易碎的快照断言。
对应生产代码：financial_rag/retrievers/query_parser.py
"""
from datetime import datetime

import pytest

from financial_rag.retrievers.query_parser import QueryParser
from financial_rag.retrievers.dictionaries import CONCEPT_MAP

# 注入受控 stock_map → 确定性验证"词典命中"路径，不依赖真实 STOCK_MAP 内容
_CTRL_MAP = {"茅台": ("600519.SH", "贵州茅台")}


@pytest.fixture
def parser():
    return QueryParser()


@pytest.fixture
def ctrl_parser():
    return QueryParser(stock_map=_CTRL_MAP)


# ===========================================================================
# ① 证明效果：实体/日期抽取符合契约
# ===========================================================================
@pytest.mark.unit
class TestParseEffect:
    def test_dict_stock_match(self, ctrl_parser):
        # 词典命中 → stock_code/name/keyword 三件套齐活
        r = ctrl_parser.parse("茅台最近走势")
        assert r.stock_code == "600519.SH"
        assert r.stock_name == "贵州茅台"
        assert r.stock_keyword == "茅台"

    @pytest.mark.parametrize("query,expected", [
        ("600519 的走势", "600519.SH"),   # 6 开头 → 上交所
        ("000001 走势", "000001.SZ"),     # 0 开头 → 深交所
        ("300750 表现", "300750.SZ"),     # 3 开头 → 创业板
    ])
    def test_six_digit_code_suffix_rule(self, parser, query, expected):
        assert parser.parse(query).stock_code == expected

    @pytest.mark.parametrize("query", [
        "2024年6月1日发生了什么",
        "2024-06-01 公告",
        "2024/06/01",
        "20240601",
    ])
    def test_absolute_date_normalized(self, parser, query):
        # 多种写法统一归一到 ISO 日期
        assert parser.parse(query).date == "2024-06-01"

    def test_relative_date_to_range(self, parser):
        r = parser.parse("最近一周走势")
        assert r.date == ""                      # 相对日期不填绝对 date
        assert r.date_range is not None
        assert r.date_range["lte"] == datetime.now().strftime("%Y-%m-%d")
        assert r.date_range["gte"] < r.date_range["lte"]


# ===========================================================================
# ② 可观测：过滤条件/扩展查询等派生字段可验证
# ===========================================================================
@pytest.mark.unit
class TestParseObservability:
    def test_filters_reflect_absolute_date(self, parser):
        assert parser.parse("2024年6月1日").get_filters() == {"publish_date": "2024-06-01"}

    def test_filters_reflect_date_range(self, parser):
        f = parser.parse("最近一周").get_filters()
        assert "publish_date" in f and isinstance(f["publish_date"], dict)

    def test_no_filter_when_no_date(self, parser):
        # stock_code 不进过滤（文档侧无此字段）→ 无日期时过滤为空
        assert parser.parse("600519 的走势").get_filters() == {}

    def test_expanded_query_prefixed_by_raw(self, ctrl_parser):
        # expanded_query 始终以原始 query 打头（供 embedding 用，不丢原意）
        r = ctrl_parser.parse("茅台走势")
        assert r.expanded_query.startswith(r.raw_query)


# ===========================================================================
# ③ 一致性：保真、确定性、权重契约
# ===========================================================================
@pytest.mark.unit
class TestParseConsistency:
    @pytest.mark.parametrize("query", ["600519 的走势", "商汤科技营收", "你好啊", ""])
    def test_raw_query_preserved(self, parser, query):
        assert parser.parse(query).raw_query == query

    def test_deterministic(self, parser):
        a = parser.parse("600519 的走势")
        b = parser.parse("600519 的走势")
        assert a.keywords == b.keywords
        assert a.stock_code == b.stock_code

    def test_stock_keyword_highest_weight_and_repeats(self, ctrl_parser):
        # 股票词权重 3.0（最高），get_weighted_terms 按 int(weight) 重复 → 出现 3 次
        r = ctrl_parser.parse("茅台走势")
        assert dict(r.keywords).get("茅台") == 3.0
        assert r.get_weighted_terms().count("茅台") == 3

    def test_empty_query_no_crash(self, parser):
        r = parser.parse("")
        assert r.keywords == []
        assert r.query_type == "other"


# ===========================================================================
# ⑤ 核心词保真（P0）：部分词典命中时，未登录核心词不得被静默丢弃
#
# Step 2 最重要的能力：解析必须无损保留用户的核心意图词/实体词，
# 转成下游 BM25/vector 能用的信号。解析阶段丢词 = 下游再强也召不回。
#
# 暴露的生产缺陷：_extract_keywords 的 fallback 分词是 if not unique_keywords
# 的全有或全无分支。只要命中一个词典词，未登录的核心词（融资/昇腾/投资/财报）
# 就永远不进分词 → 静默丢弃 → 这是 Step 3 "大模型融资" MISS、"昇腾"问题的上游根因。
# ===========================================================================
@pytest.mark.regression
class TestCoreWordFidelity:
    @pytest.mark.parametrize("query,core_word", [
        ("大模型融资", "融资"),      # 大模型命中词典，意图词"融资"不得丢
        ("昇腾芯片进展", "昇腾"),      # 芯片命中词典，关键实体"昇腾"不得丢
        ("AI算力投资", "投资"),        # 算力/AI 命中，意图词"投资"不得丢
        ("宁德时代财报", "财报"),      # 股票命中，意图词"财报"不得丢
    ])
    def test_unlogged_core_word_not_dropped(self, parser, query, core_word):
        """部分词典命中时，未登录的核心词仍须进入 keywords（喂给 BM25）。

        修复前：fallback 分词仅在零命中时触发 → 核心词被静默丢弃 → 本用例红。
        修复后：对残余文本（去掉已覆盖词典词）分词补充 → 核心词保留。
        """
        r = parser.parse(query)
        terms = {t for t, _ in r.keywords}
        assert core_word in terms, (
            f"query={query!r} 的核心词 {core_word!r} 被静默丢弃，"
            f"实际 keywords={r.keywords}"
        )
        # 核心词也必须进入喂给 BM25 的 weighted terms
        assert core_word in r.get_weighted_terms(), (
            f"{core_word!r} 未进入 get_weighted_terms()，BM25 拿不到该信号"
        )

    def test_dictionary_word_still_highest_weight(self, parser):
        """修复不得本末倒置：词典命中词权重仍高于残余补充词(1.0)。"""
        r = parser.parse("大模型融资")
        kw = dict(r.keywords)
        # "大模型"是 INDUSTRY_TERMS(1.5)，应 >= 残余补充的"融资"(1.0)
        assert kw.get("大模型", 0) >= kw.get("融资", 0), (
            f"词典词权重应不低于残余词: {kw}"
        )

    def test_residual_supplement_no_stopword_noise(self, parser):
        """残余补充不得引入停用词或滑窗跨词垃圾。

        注："走势"/"怎么样"是 ACTION_TERMS 合法动作词（现有设计就抽，权1.0），
        不属于噪声。本用例只验证纯停用词（的/最近）和滑窗跨词垃圾（技的/的最）不泄漏。
        """
        r = parser.parse("商汤科技的最近走势怎么样")
        terms = {t for t, _ in r.keywords}
        # 纯停用词不得出现
        for sw in ["的", "最近"]:
            assert sw not in terms, f"停用词 {sw!r} 不应进入 keywords: {terms}"
        # 滑窗跨词垃圾不得出现（残余文本未去停用词时会产生"技的""的最"）
        for garbage in ["技的", "的最", "的最近"]:
            assert garbage not in terms, f"滑窗跨词垃圾 {garbage!r} 不应进入 keywords: {terms}"


# ===========================================================================
# ⑥ 概念扩展意图感知门控（Step4 根因#2 修复）
#
# _expand_query 的概念关联扩展(CONCEPT_MAP, weight=0.6)是泛化关联：主语"大模型"
# 会注入 GPT/LLM/算力/训练/推理/微调/Token。当 query 带明确判别意图(未登录核心词，
# 如"大模型融资"的"融资")时，这些泛化词淹没意图 → 下游 vector 通道被稀释 →
# fusion recall 坍塌(两单通道满分、融合掉档)。修复：仅在"纯概念探索"时扩展概念。
# 同义词扩展(精确等价)不稀释意图，始终不受门控影响。
# ===========================================================================
@pytest.mark.unit
class TestConceptExpansionGating:
    def test_concept_suppressed_when_specific_intent(self, parser):
        # "大模型融资"：融资 是未登录核心意图词 -> 抑制"大模型"的概念扩展
        r = parser.parse("大模型融资")
        leaked = set(CONCEPT_MAP["大模型"]) & set(r.expanded_terms)
        assert not leaked, f"有判别意图'融资'时概念词仍被注入，会稀释意图: {leaked}"
        assert "融资" in {t for t, _ in r.keywords}, "意图词'融资'必须保留"

    def test_concept_kept_for_pure_exploration(self, parser):
        # 纯概念探索"大模型"(无残余意图词) -> 概念扩展照常补召回，不得因修复而一刀切关掉
        r = parser.parse("大模型")
        assert set(CONCEPT_MAP["大模型"]) & set(r.expanded_terms), (
            f"纯概念探索应保留概念扩展，实测 expanded_terms={r.expanded_terms}")

    def test_synonym_unaffected_by_intent_gating(self, parser):
        # 同义词(华为->Huawei)精确等价、不稀释意图；即便 query 带残余意图词触发门控，
        # 同义词扩展仍照常，不被概念门控误伤。
        r = parser.parse("华为昇腾国产替代")
        assert "Huawei" in set(r.expanded_terms), (
            f"同义词扩展不应被意图门控抑制，实测 {r.expanded_terms}")


# ===========================================================================
# ④ 回归锁定：探针抓出并已修复的 bug，钉死防回退
# ===========================================================================
@pytest.mark.regression
class TestFixedBugLocks:
    def test_eight_digit_date_not_parsed_as_stock(self, parser):
        """回归锁：8 位纯日期只应解析为日期，不得污染 stock_code。

        曾违反：_extract_stock 的 6 位数字正则抓取 YYYYMMDD 前 6 位 → '20240601' 误为 '202406'。
        已修：正则加 (?<!\\d)..(?!\\d) 边界，要求 6 位为独立数字串。
        退回旧实现→本用例即红。
        """
        r = parser.parse("20240601 的公告")
        assert r.date == "2024-06-01"          # 日期本身解析正确
        assert r.stock_code == "", f"8 位日期不应产生股票代码，实测 {r.stock_code!r}"
