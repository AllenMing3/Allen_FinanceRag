r"""
Step 4 融合层测试资产 —— RRF 多通道融合的健全性与增值保证。

════════════════════════════════════════════════════════════════════
【这个阶段的目的是什么】
════════════════════════════════════════════════════════════════════
Step 4 的核心保证：**融合取长补短，而非取短拖后腿**。
RRF 把 bm25(关键词) + vector(语义) 两条通道合成一个排序，价值在于：
  - 健壮性兜底：某条通道失效时，另一条把相关文档捞回来；
  - 增量增值：在单通道弱处（如长文多 chunk）超越任一单通道。

但 L1 eval 量化出当前 fusion 是"负增值"——全面劣于 bm25 单通道。
改制前的分（起点，与 baseline.json 逐位一致）：
    channel   recall    mrr      ndcg     map
    bm25      0.9683   1.0000   0.9742   0.9405   <- 最强单通道
    vector    0.9683   0.9167   0.9150   0.8929
    fusion    0.9365   0.8968   0.8978   0.8706   <- 比两个输入都差
    fusion 子集 ndcg: unlogged_term 0.4946 / bug_scene 0.6837 / seed 0.7668
本阶段目标：把"负增值"扭为"至少不拖后腿 + 保住兜底增值"，并建 fusion 专属回归锁
（补足 test_eval_retrieval.py 未覆盖的"fusion 与单通道相对关系"不变量）。
改制后的分（根因修复：#1 bm25 token 对齐 + #2 概念扩展意图感知抑制 + #3 权重翻正）：
    fusion    0.9841   1.0000   0.9940   0.9643   <- 反超 bm25 单通道，取长补短
    fusion 子集 ndcg: unlogged_term 0.9957 / bug_scene 0.9901 / seed 1.0000 / long_doc 0.9860(保住)

════════════════════════════════════════════════════════════════════
【原则底线是什么】（必须遵守，违反即红）
════════════════════════════════════════════════════════════════════
P1 反坍塌底线：fusion 核心指标不得低于"两通道较差者" min(bm25,vector) - tol。
   融合两个排序器，结果不该比最差的输入还差——这是融合的根本健全性。
P2 增量价值底线：bm25 单通道弱处（long_doc 多 chunk / financial_report），
   fusion 必须 >= bm25，证明 vector 通道不是摆设、融合确有兜底增值。
P3 词表对齐底线：喂 bm25 的 query token 必须与建索引的分词器(jieba)同源，
   不得用 QueryParser 的双字滑窗 bigram + 同义/概念扩展词（根因#1）。
P4 权重不倒挂底线：强通道(bm25)的 RRF 权重不得低于弱通道(vector)（根因#3）。
P5 确定性底线：同 query 多次运行，fusion 结果一致（离线、无随机、无网络）。
P6 回归锁必须真锁：每个修复锁定用例须经 revert-diff 自检——回退修复 -> 该用例
   指标应下降；否则是空转锁（见易错点 E3）。

════════════════════════════════════════════════════════════════════
【常见易错点矩阵】（扫到了吗？-> 对应用例）
════════════════════════════════════════════════════════════════════
| 易错点 | 症状 | 本文件如何覆盖 |
|--------|------|----------------|
| E1 "只调权重"陷阱 | 权重换来换去 ndcg 卡 0.89-0.90 不动 | A 组锁 token 污染的真实效果；C1 只锁权重不倒挂，不假装权重是主因（Step3 教训：无可观测效果的改动） |
| E2 recall/排序混淆 | 修 bm25 token 只涨 ndcg/mrr，recall 纹丝不动 | A1 锁 macro recall、A2 锁子集 recall，与 ndcg 分开断言（recall 根因在 vector #2，排序根因在 bm25 #1） |
| E3 回归锁空转 | 子集全绿但锁不住它声称保护的修复 | P6 原则：建锁后 monkeypatch 回退修复做 revert-diff，只有 delta 真变的 query 才纳入子集 |
| E4 golden 偏袒 bm25 | 拿"fusion>bm25"当靶子 -> 永远红 | 底线用 min(bm25,vector)（P1），不用 bm25 上限；fusion 天然追不平 bm25 单通道 |
| E5 jaccard/embedding 模式混淆 | eval 是 jaccard，生产是 embedding，expanded_query 利害相反 | 全部用例显式跑本地 jaccard 模式(build_local_retriever: embedder=None)，不据此断言生产 embedding 行为 |
| E6 RRF 依赖 rank 字段 | 通道结果缺 rank -> rrf 退化 r.get("rank",1) 全 1 | D3 锁各通道结果带连续 rank；D2 锁 RRF 共识提升性质 |

优先级映射（核心能力 -> 用例）：
  P0 反坍塌(核心保证)   -> A1 A2 A3   [已修复翻绿，回归锁]
  P1 增量价值(防修复弄丢) -> B1 B2      [常绿锁，修复全程保住]
  P1 装配不变量(根因防线) -> C1 D3      [C1 已修复翻绿 / D3 常绿]
  P2 融合机制健全        -> D2         [常绿单元锁]
"""
import pytest

from financial_rag.eval.runner import (
    DEFAULT_TOP_K, DEFAULT_TOL, GUARDED_METRICS,
    build_local_retriever, evaluate, load_golden_docs, load_qrels,
)
from financial_rag.retrievers.fusion import rrf_fusion


# ===================== fixtures =====================

@pytest.fixture(scope="module")
def report():
    """模块内共享一次 evaluate()（~0.7s），避免每个用例重复装配检索器。"""
    return evaluate(top_k=DEFAULT_TOP_K)


@pytest.fixture(scope="module")
def indexed_retriever(tmp_path_factory):
    """本地 jaccard 模式装配 + 索引 golden（E5：显式离线、无 embedding）。"""
    workdir = tmp_path_factory.mktemp("step4_fusion")
    r = build_local_retriever(str(workdir))
    r.index(load_golden_docs(), precompute_embeddings=False, use_chunker=True)
    return r


def _floor(report, tag, metric):
    """P1 反坍塌底线：两通道较差者 min(bm25,vector) - tol。"""
    bm25 = report.channels["bm25"].per_tag[tag][metric]
    vec = report.channels["vector"].per_tag[tag][metric]
    return min(bm25, vec) - DEFAULT_TOL


# ===================== A. P0 反坍塌底线（核心保证）=====================

@pytest.mark.eval
class TestFusionNoCollapse:
    """P1：fusion 不得低于其任一输入通道。曾全面违反（根因#1 bm25 token 污染 +
    #2 vector 概念扩展稀释），已在源头修复 -> 现为回归锁，退回修复即翻红。"""

    def test_fusion_macro_not_below_worst_channel(self, report):
        for m in GUARDED_METRICS:
            worst = min(report.macro("bm25", m), report.macro("vector", m))
            fus = report.macro("fusion", m)
            assert fus >= worst - DEFAULT_TOL, (
                f"fusion macro {m}={fus:.4f} < min(bm25,vector) {worst:.4f} "
                f"- tol {DEFAULT_TOL}：融合比最差输入通道还差")

    def test_fusion_unlogged_term_not_collapse(self, report):
        fus = report.channels["fusion"].per_tag["unlogged_term"]
        for m in GUARDED_METRICS:
            floor = _floor(report, "unlogged_term", m)
            assert fus[m] >= floor, (
                f"unlogged_term fusion.{m}={fus[m]:.4f} < min(bm25,vector)-tol "
                f"{floor:.4f}：未登录词场景融合坍塌（概念扩展淹没判别词）")

    def test_fusion_bug_scene_not_collapse(self, report):
        fus = report.channels["fusion"].per_tag["bug_scene"]
        for m in GUARDED_METRICS:
            floor = _floor(report, "bug_scene", m)
            assert fus[m] >= floor, (
                f"bug_scene fusion.{m}={fus[m]:.4f} < min(bm25,vector)-tol "
                f"{floor:.4f}：bug 场景融合坍塌")


# ===================== B. P1 增量价值底线（防修复弄丢）=====================

@pytest.mark.eval
class TestFusionValueAdd:
    """P2：bm25 单通道弱处，fusion 必须反超——锁住融合的兜底增值，
    防止修复 bm25 污染时把 vector 通道的贡献一起弄丢。现状已绿 -> 常绿锁。"""

    def test_fusion_beats_bm25_on_long_doc(self, report):
        # long_doc: 长文多 chunk，bm25 单通道 ndcg 弱(0.862)，fusion 应反超(0.986)
        bm25 = report.channels["bm25"].per_tag["long_doc"]["ndcg"]
        fus = report.channels["fusion"].per_tag["long_doc"]["ndcg"]
        assert fus >= bm25, (
            f"long_doc fusion ndcg={fus:.4f} < bm25 {bm25:.4f}："
            "融合丢失了长文多 chunk 场景的兜底增值")

    def test_fusion_beats_bm25_on_financial_report(self, report):
        # financial_report: bm25 单通道 ndcg 0.904，fusion 应反超(0.975)
        bm25 = report.channels["bm25"].per_tag["financial_report"]["ndcg"]
        fus = report.channels["fusion"].per_tag["financial_report"]["ndcg"]
        assert fus >= bm25, (
            f"financial_report fusion ndcg={fus:.4f} < bm25 {bm25:.4f}："
            "融合丢失了财报场景的兜底增值")


# ===================== C. P1 装配不变量（根因防线）=====================

@pytest.mark.eval
class TestFusionAssembly:
    """根因防线：权重不倒挂(P4) + 各通道结果带 rank(E6/RRF 依赖)。"""

    def test_rrf_weights_reflect_channel_strength(self, report, indexed_retriever):
        # 前提：eval 证明 bm25 是更强通道
        assert report.macro("bm25", "ndcg") >= report.macro("vector", "ndcg"), (
            "本用例前提失效：bm25 不再是更强通道，需重新评估权重方向")
        # P4：强通道权重不得低于弱通道
        assert indexed_retriever.bm25_weight >= indexed_retriever.vector_weight, (
            f"RRF 权重倒挂：bm25_weight={indexed_retriever.bm25_weight} < "
            f"vector_weight={indexed_retriever.vector_weight}，强通道被压制")

    def test_channel_results_carry_contiguous_rank(self, indexed_retriever):
        # E6：RRF 用 r.get("rank",1)，通道结果必须带连续 rank，否则融合退化
        query = load_qrels()[0]["query"]
        bm25_items = indexed_retriever._bm25.search(
            indexed_retriever.documents, query, DEFAULT_TOP_K)
        assert [it["rank"] for it in bm25_items] == list(range(1, len(bm25_items) + 1)), (
            "bm25 通道结果 rank 不连续，RRF 会退化")
        vec_items = indexed_retriever._vector.search_jaccard(
            indexed_retriever.documents, query, DEFAULT_TOP_K,
            tokenize_fn=indexed_retriever._bm25.tokenize)
        assert [it["rank"] for it in vec_items] == list(range(1, len(vec_items) + 1)), (
            "vector 通道结果 rank 不连续，RRF 会退化")


# ===================== D. P2 融合机制健全性（纯单元）=====================

@pytest.mark.unit
class TestRrfMechanism:
    """RRF 融合算法本身的性质锁（不依赖 eval/检索器，纯 fusion.py 单元）。"""

    def test_rrf_consensus_outranks_single_channel_hit(self):
        # 双通道都命中的文档(consensus)应排在只命中单通道的文档之前——RRF 核心性质
        ch1 = [{"text": "A", "rank": 2, "score": 1.0},
               {"text": "B", "rank": 1, "score": 2.0}]
        ch2 = [{"text": "A", "rank": 1, "score": 1.0}]
        fused, stats = rrf_fusion(
            [(ch1, 0.5, "bm25"), (ch2, 0.5, "vector")], top_k=5, rrf_k=60)
        texts = [f["text"] for f in fused]
        assert texts.index("A") < texts.index("B"), (
            f"consensus 文档 A 未排在单通道文档 B 之前：{texts}")
        assert stats.consensus_count == 1, "consensus_count 统计错误"

    def test_rrf_deterministic_same_input(self):
        # P5：同输入两次融合，排序逐位一致（无随机）
        ch1 = [{"text": "X", "rank": 1, "score": 1.0},
               {"text": "Y", "rank": 2, "score": 0.5}]
        ch2 = [{"text": "Y", "rank": 1, "score": 1.0},
               {"text": "Z", "rank": 2, "score": 0.5}]
        lists = [(ch1, 0.5, "bm25"), (ch2, 0.5, "vector")]
        r1, _ = rrf_fusion(lists, top_k=5, rrf_k=60)
        r2, _ = rrf_fusion(lists, top_k=5, rrf_k=60)
        assert [f["text"] for f in r1] == [f["text"] for f in r2], "RRF 非确定"
