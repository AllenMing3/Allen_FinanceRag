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
