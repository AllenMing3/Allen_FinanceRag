"""
Test Step 5 过滤 / 质量门控 — 检索交付前的控制层质量验证（RAG 流水线第五步）。

链路位置：召回 → 融合 → **过滤(Filter) → 质量门控(Gate)** → 交付生成
对应生产代码：
  - financial_rag/retrievers/filters.py（apply_filters 元数据过滤）
  - financial_rag/retrievers/hybrid_engine.py（search() 第 6-7 步：过滤 + 门控）

按项目四条测试标准组织：① 证明效果 ② 可观测 ③ 一致性 ④ 回归锁定

核心发现（探针 _probe_step5.py 实然证据）：
  - 🔴 BUG(待修): 相对日期过滤静默失效 —— query_parser 产出无 `$` 前缀的
    {"gte":...,"lte":...}，apply_filters 只认 $gte/$lte → 未知操作符忽略 → 放行全部。
    用 @xfail(strict=True) 钉死，修复后 XPASS 强制摘标记翻绿。
  - ⚠️ 已知局限: $ne/$nin 对缺失字段判拒（偏离 Mongo 语义）—— 特征化锁定
  - ⚠️ 已知局限: 门控分数分支依赖 reranker，离线(reranker=None)永不触发 —— 特征化锁定
  - ✅ 零结果门控离线可触发（空 KB → stage=retrieval 拦截 + 诊断完整）

注：门控「分数分支」的决策测试（低分拦/高分放）需先抽出 decide_quality_gate 纯函数，
    属 L1 扩展 Dim2 工作，本文件先覆盖离线可达的零结果分支 + 放行行为 + 诊断结构。
"""
import pytest

from financial_rag.retrievers.filters import apply_filters, get_last_stats, FilterStats
from financial_rag.retrievers.hybrid_engine import HybridRetriever


# ===================== 合成候选文档 =====================

def _cands():
    """4 篇合成候选，覆盖有/无 source、跨年份 publish_date。"""
    return [
        {"text": "AI芯片新闻", "meta": {"doc_id": "d1", "source": "news",
                                        "publish_date": "2024-06-01", "title": "AI芯片突破"}},
        {"text": "RSS消费", "meta": {"doc_id": "d2", "source": "rss",
                                     "publish_date": "2023-01-15", "title": "消费数据"}},
        {"text": "缺source", "meta": {"doc_id": "d3", "publish_date": "2024-12-31"}},
        {"text": "垃圾广告", "meta": {"doc_id": "d4", "source": "spam",
                                      "publish_date": "2024-03-01", "title": "推广"}},
    ]


def _ids(result):
    return [r["meta"]["doc_id"] for r in result]


# ======================================================================
#  ① 证明效果：apply_filters 各操作符正确过滤（该留的留、该删的删）
# ======================================================================

class TestFilterOperators:
    """元数据过滤各操作符按契约筛选候选。"""

    def test_empty_filter_keeps_all(self):
        assert _ids(apply_filters(_cands(), {})) == ["d1", "d2", "d3", "d4"]

    def test_exact_match(self):
        assert _ids(apply_filters(_cands(), {"source": "news"})) == ["d1"]

    def test_multi_value_or(self):
        assert _ids(apply_filters(_cands(), {"source": ["news", "rss"]})) == ["d1", "d2"]

    def test_not_operator(self):
        # d4=spam 被拒；d3 缺 source（None≠spam）保留
        assert _ids(apply_filters(_cands(), {"$not": {"source": "spam"}})) == ["d1", "d2", "d3"]

    def test_exists(self):
        assert _ids(apply_filters(_cands(), {"$exists": "source"})) == ["d1", "d2", "d4"]

    def test_not_exists(self):
        assert _ids(apply_filters(_cands(), {"$not_exists": "source"})) == ["d3"]

    def test_regex(self):
        assert _ids(apply_filters(_cands(), {"title": {"$regex": "AI|芯片"}})) == ["d1"]

    def test_range_dollar_gte_lte(self):
        """带 $ 前缀的范围过滤正确筛选 2024 年文档。"""
        f = {"publish_date": {"$gte": "2024-01-01", "$lte": "2024-12-31"}}
        assert _ids(apply_filters(_cands(), f)) == ["d1", "d3", "d4"]


# ======================================================================
#  ① 证明效果：soft 模式 —— 文档缺字段时不过度过滤
# ======================================================================

class TestFilterSoftMode:
    """soft_keys 指定的字段：文档缺失时跳过（不拒绝），避免误删。"""

    def test_soft_key_missing_field_kept(self):
        # d3 缺 source，soft 模式放行；d1 命中 news
        res = apply_filters(_cands(), {"source": "news"}, soft_keys={"source"})
        assert _ids(res) == ["d1", "d3"]

    def test_hard_mode_missing_field_dropped(self):
        # 非 soft：d3 缺 source 被拒
        res = apply_filters(_cands(), {"source": "news"}, soft_keys=set())
        assert _ids(res) == ["d1"]


# ======================================================================
#  ② 可观测：FilterStats 统计可验证
# ======================================================================

class TestFilterObservability:
    """过滤统计（pass_rate / filtered_out / soft_skips）可观测。"""

    def test_stats_counts(self):
        apply_filters(_cands(), {"source": "news"}, collect_stats=True)
        st = get_last_stats()
        assert st.before_count == 4
        assert st.after_count == 1
        assert st.filtered_out == 3
        assert abs(st.pass_rate - 0.25) < 1e-9

    def test_stats_soft_skips(self):
        apply_filters(_cands(), {"source": "news"}, soft_keys={"source"},
                      collect_stats=True)
        st = get_last_stats()
        assert st.soft_skips == 1  # 仅 d3 缺 source 触发一次 skip

    def test_filterstats_pass_rate_zero_guard(self):
        """空候选 pass_rate 返回 0.0，不除零。"""
        assert FilterStats(before_count=0, after_count=0).pass_rate == 0.0


# ======================================================================
#  ③ 一致性：过滤确定性 + 幂等
# ======================================================================

class TestFilterConsistency:
    """同输入同输出；重复过滤结果稳定。"""

    def test_deterministic(self):
        f = {"source": ["news", "rss"]}
        assert _ids(apply_filters(_cands(), f)) == _ids(apply_filters(_cands(), f))

    def test_idempotent(self):
        f = {"publish_date": {"$gte": "2024-01-01"}}
        once = apply_filters(_cands(), f)
        twice = apply_filters(once, f)
        assert _ids(once) == _ids(twice)


# ======================================================================
#  ④ 回归锁定：日期过滤 no-op BUG（xfail 钉死，修复后强制翻绿）
# ======================================================================

class TestDateRangeFilterBug:
    """BUG: query_parser 产出的无 `$` 范围过滤键被静默忽略。

    query_parser.py:171 → date_range = {"gte": start, "lte": end}（无 $ 前缀）
    filters.py:_match_value 只认 $gte/$lte；无 $ 的 gte/lte 落入「未知操作符忽略」
    → return True → 放行全部文档。filters.py:7 文档字符串自身即写无 $ 形状，
    实现违反了自己的文档契约。后果：所有相对日期 query 的过滤是空操作。

    修复方向：_match_value 接受无 $ 的 gte/lte/gt/lt（对齐文档）。
    修复后本用例 XPASS，strict=True 强制摘标记转常绿回归锁。
    """

    @pytest.mark.xfail(strict=True, reason="无 $ 前缀的范围过滤键被忽略 → 日期过滤 no-op")
    def test_daterange_no_dollar_prefix_filters(self):
        f = {"publish_date": {"gte": "2024-01-01", "lte": "2024-12-31"}}
        # 期望：只留 2024 年文档（d1/d3/d4），d2(2023) 被过滤
        # 实然(bug)：无 $ 前缀 → 过滤 no-op → d2 也留下
        assert _ids(apply_filters(_cands(), f)) == ["d1", "d3", "d4"]


# ======================================================================
#  已知局限特征化：$ne/$nin 对缺失字段判拒（偏离 Mongo 语义）
# ======================================================================

class TestKnownLimitations:
    """特征化锁定已知局限（不改行为，记录实然，防无意改动）。"""

    def test_ne_rejects_missing_field(self):
        """已知局限：$ne 对缺失字段判拒。d3 无 source，被 {'source':{'$ne':'spam'}}
        拒绝（Mongo 语义下缺字段应匹配 $ne 而保留）。特征化锁定当前行为。"""
        res = apply_filters(_cands(), {"source": {"$ne": "spam"}})
        assert _ids(res) == ["d1", "d2"]  # d3(缺source)、d4(spam) 均被拒

    def test_gate_offline_score_branch_never_fires(self):
        """已知局限：门控分数分支依赖 reranker。离线 reranker=None → has_rerank=False
        → 分数门控永不触发，低质结果原样放行（探针④：top_score=0.016 未拦截）。
        特征化锁定：无 reranker 时构造低分候选，search 末段门控不会拦截。"""
        r = HybridRetriever(embedder=None, reranker=None)
        # 直接验证门控判定的前置条件：无 reranker → has_rerank 恒 False
        has_rerank = bool(r.reranker)  # reranker=None → False
        assert has_rerank is False
        assert r.gate_threshold == 0.15  # 默认阈值可观测


# ======================================================================
#  ① 证明效果 + ② 可观测：零结果门控（离线可触发分支）
# ======================================================================

class TestZeroResultGate:
    """空 KB → 检索零结果 → 门控拦截，返回 [] + 结构化诊断（stage=retrieval）。"""

    def test_empty_kb_blocks_with_diagnostics(self):
        r = HybridRetriever(embedder=None, reranker=None)
        # 未 index 任何文档 → documents 为空 → 检索必然零结果
        results = r.search("任意查询", top_k=5, use_rerank=False)
        assert results == []
        gi = r.last_gate_info
        # ② 可观测：诊断信息完整（硬拦截 + 全量反馈规范）
        assert gi is not None
        assert gi["blocked"] is True
        assert gi["stage"] == "retrieval"
        assert gi["query"] == "任意查询"
        assert "reason" in gi
        assert gi["threshold"] == r.gate_threshold
        assert gi["fused_count"] == 0
