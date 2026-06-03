"""
全链路打分系统 — 每个管道阶段独立评分，精确诊断薄弱环节

设计目标:
- 每一级管道都有独立分数 (0~1.0)，能明确看到是哪个环节不好
- 每级分数附带诊断信息 (diagnosis)，解释分数为什么低
- 分数链可追溯，从 metadata解析 → jieba分词 → BM25 → Vector → RRF → Rerank → LLM → 防幻觉

使用方式:
    scorecard = PipelineScoreCard()
    # 逐阶段记录
    scorecard.record_metadata(stage="metadata_parse", score=0.85, ...)
    scorecard.record_retrieval("bm25", score=0.72, ...)
    ...
    # 最终输出
    print(scorecard.summary())
    print(scorecard.table())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import time


# ===================== 评分等级 =====================

class ScoreGrade(Enum):
    EXCELLENT = "A"    # >= 0.90
    GOOD      = "B"    # >= 0.75
    FAIR      = "C"    # >= 0.60
    POOR      = "D"    # >= 0.40
    FAIL      = "F"    # <  0.40

    @classmethod
    def from_score(cls, score: float) -> "ScoreGrade":
        if score >= 0.90: return cls.EXCELLENT
        if score >= 0.75: return cls.GOOD
        if score >= 0.60: return cls.FAIR
        if score >= 0.40: return cls.POOR
        return cls.FAIL

    @property
    def color(self) -> str:
        return {ScoreGrade.EXCELLENT: "green", ScoreGrade.GOOD: "blue",
                ScoreGrade.FAIR: "yellow", ScoreGrade.POOR: "orange",
                ScoreGrade.FAIL: "red"}[self]

    @property
    def cn(self) -> str:
        return {ScoreGrade.EXCELLENT: "优秀", ScoreGrade.GOOD: "良好",
                ScoreGrade.FAIR: "一般", ScoreGrade.POOR: "较差",
                ScoreGrade.FAIL: "不合格"}[self]


# ===================== 单阶段评分记录 =====================

@dataclass
class StageScore:
    """单个管道阶段的评分快照"""
    stage: str                         # 阶段名，如 "metadata_parse" / "bm25_retrieval"
    display_name: str                  # 展示名，如 "元数据解析"
    score: float                       # 0.0 ~ 1.0
    grade: ScoreGrade = ScoreGrade.FAIR
    elapsed_ms: float = 0.0
    # 子指标详情
    details: Dict[str, Any] = field(default_factory=dict)
    # 诊断：分数低的原因
    diagnosis: str = ""
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.grade = ScoreGrade.from_score(self.score)


# ===================== 大类阶段组 =====================

class StageGroup(Enum):
    """阶段大类"""
    INGESTION   = ("ingestion",   "数据摄取 & 元数据解析")
    PREPROCESS  = ("preprocess",  "文本预处理 & 分词")
    RETRIEVAL   = ("retrieval",   "混合检索 (BM25+Vector+RRF+Rerank)")
    GENERATION  = ("generation",  "LLM 生成 & 防幻觉校验")
    COORDINATE  = ("coordinate",  "Multi-Agent 协调调度")


# ===================== 打分卡 =====================

@dataclass
class PipelineScoreCard:
    """
    全链路打分卡 — 记录每个阶段的评分

    阶段链路:
    ┌─────────── INGESTION ────────────┐
    │ metadata_parse    元数据解析      │
    │ text_clean        文本清洗        │
    │ chunk_quality     分块质量        │
    ├─────────── PREPROCESS ───────────┤
    │ tokenization      jieba 分词     │
    │ keyword_extract   关键词抽取      │
    │ query_rewrite     查询改写        │
    ├─────────── RETRIEVAL ────────────┤
    │ bm25_retrieval    BM25 检索      │
    │ vector_retrieval  向量检索        │
    │ rrf_fusion        RRF 融合       │
    │ rerank            重排序          │
    ├─────────── GENERATION ───────────┤
    │ llm_generate      LLM 生成       │
    │ l1_source         L1 来源验证     │
    │ l2_consistency    L2 一致性       │
    │ l3_fact           L3 事实核查     │
    │ l4_completeness   L4 完整性       │
    │ l5_citation       L5 引用准确     │
    │ l6_overall        L6 综合         │
    └──────────────────────────────────┘
    """

    query: str = ""
    stages: List[StageScore] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # 小类加权 → 大类汇总
    group_weights: Dict[StageGroup, float] = field(default_factory=lambda: {
        StageGroup.INGESTION:   0.15,
        StageGroup.PREPROCESS:  0.20,
        StageGroup.RETRIEVAL:   0.35,
        StageGroup.GENERATION:  0.30,
    })

    # ===================== 记录方法 =====================

    def record(self, stage: str, display_name: str, score: float,
               elapsed_ms: float = 0.0, details: Dict = None,
               diagnosis: str = "", warnings: List[str] = None,
               suggestions: List[str] = None):
        """通用记录"""
        s = StageScore(
            stage=stage, display_name=display_name,
            score=max(0.0, min(1.0, score)),
            elapsed_ms=elapsed_ms,
            details=details or {},
            diagnosis=diagnosis,
            warnings=warnings or [],
            suggestions=suggestions or [],
        )
        self.stages.append(s)
        return s

    # --- 摄取阶段 ---
    def record_metadata(self, score: float, fields_found: int, fields_expected: int,
                        elapsed_ms: float = 0.0, extra: Dict = None):
        """记录元数据解析评分"""
        ratio = fields_found / max(fields_expected, 1)
        diag = ""
        warns = []
        sug = []
        if ratio < 0.8:
            diag = f"仅解析出 {fields_found}/{fields_expected} 个字段"
            sug.append("检查文档格式是否规范，或增加正则/NER 解析规则")
        if ratio < 0.5:
            warns.append(f"元数据覆盖率过低 ({ratio:.0%})")
        return self.record(
            "metadata_parse", "元数据解析", score,
            elapsed_ms=elapsed_ms,
            details={"fields_found": fields_found, "fields_expected": fields_expected,
                     "coverage": round(ratio, 3), **(extra or {})},
            diagnosis=diag, warnings=warns, suggestions=sug,
        )

    def record_clean(self, score: float, original_len: int, cleaned_len: int,
                     elapsed_ms: float = 0.0):
        """记录文本清洗评分"""
        ratio = cleaned_len / max(original_len, 1)
        diag = ""
        warns = []
        if ratio < 0.3:
            warns.append(f"清洗后文本仅剩 {ratio:.0%}，可能过度清洗")
            diag = f"清洗率 {ratio:.0%}，大量文本被丢弃"
        return self.record(
            "text_clean", "文本清洗", score,
            elapsed_ms=elapsed_ms,
            details={"original_len": original_len, "cleaned_len": cleaned_len,
                     "retention": round(ratio, 3)},
            diagnosis=diag, warnings=warns,
        )

    def record_chunk(self, score: float, chunk_count: int, avg_chunk_len: float,
                     elapsed_ms: float = 0.0):
        """记录分块质量"""
        warns = []
        diag = ""
        if chunk_count == 0:
            warns.append("分块数为 0，无可用文本")
            diag = "未生成任何文本块"
        elif avg_chunk_len < 50:
            warns.append(f"平均块长度仅 {avg_chunk_len:.0f} 字符，信息量不足")
            diag = "分块过小，信息密度低"
        return self.record(
            "chunk_quality", "分块质量", score,
            elapsed_ms=elapsed_ms,
            details={"chunk_count": chunk_count, "avg_chunk_len": round(avg_chunk_len, 1)},
            diagnosis=diag, warnings=warns,
        )

    # --- 预处理阶段 ---
    def record_tokenization(self, score: float, token_count: int, unique_tokens: int,
                            avg_token_len: float, elapsed_ms: float = 0.0):
        """记录 jieba 分词评分"""
        diag = ""
        warns = []
        sug = []
        if token_count == 0:
            diag = "分词结果为空，可能是文本编码或语言不匹配"
            warns.append("分词失败：无有效 token")
            sug.append("检查输入文本编码是否为 UTF-8，是否包含有效中文/英文")
        elif unique_tokens / max(token_count, 1) < 0.1:
            diag = "分词去重率过低，可能存在大量重复词"
            warns.append(f"唯一词占比仅 {unique_tokens/max(token_count,1):.0%}")
        if avg_token_len < 1.5:
            diag = "平均词长过短，可能是过度切分"
            sug.append("检查 jieba 词典是否包含领域专有词汇")
        return self.record(
            "tokenization", "Jieba 分词", score,
            elapsed_ms=elapsed_ms,
            details={"token_count": token_count, "unique_tokens": unique_tokens,
                     "uniqueness": round(unique_tokens / max(token_count, 1), 3),
                     "avg_token_len": round(avg_token_len, 1)},
            diagnosis=diag, warnings=warns, suggestions=sug,
        )

    def record_keyword_extract(self, score: float, keyword_count: int,
                               elapsed_ms: float = 0.0):
        """记录关键词抽取"""
        diag = ""
        warns = []
        if keyword_count == 0:
            diag = "未抽取出关键词，检索可能无方向"
            warns.append("关键词抽取为空")
        elif keyword_count < 3:
            warns.append(f"关键词仅 {keyword_count} 个，检索覆盖可能不足")
        return self.record(
            "keyword_extract", "关键词抽取", score,
            elapsed_ms=elapsed_ms,
            details={"keyword_count": keyword_count},
            diagnosis=diag, warnings=warnings,
        )

    def record_query_rewrite(self, score: float, original_query: str,
                             rewritten_queries: List[str], elapsed_ms: float = 0.0):
        """记录查询改写"""
        diag = ""
        warns = []
        if not rewritten_queries:
            diag = "查询改写未生成新查询"
            warns.append("查询改写失败")
        return self.record(
            "query_rewrite", "查询改写", score,
            elapsed_ms=elapsed_ms,
            details={"original": original_query[:80], "rewritten_count": len(rewritten_queries)},
            diagnosis=diag, warnings=warns,
        )

    # --- 检索阶段 ---
    def record_retrieval(self, stage: str, display_name: str, score: float,
                         result_count: int, top_score: float, avg_score: float,
                         elapsed_ms: float = 0.0, extra: Dict = None):
        """通用检索阶段记录"""
        diag = ""
        warns = []
        sug = []
        if result_count == 0:
            diag = f"{display_name} 未返回任何结果"
            warns.append(f"{display_name} 检索为空")
            sug.append("检查索引是否构建完成，或查询词是否在知识库范围内")
        elif top_score < 0.3:
            diag = f"{display_name} 最高分仅 {top_score:.2f}，与查询相关性弱"
            warns.append(f"最高相关度 {top_score:.2f}，检索质量低")
            sug.append("尝试调整 BM25 参数 (k1, b) 或扩充知识库")
        return self.record(
            stage, display_name, score,
            elapsed_ms=elapsed_ms,
            details={"result_count": result_count, "top_score": round(top_score, 4),
                     "avg_score": round(avg_score, 4), **(extra or {})},
            diagnosis=diag, warnings=warns, suggestions=sug,
        )

    def record_bm25(self, result_count: int, top_score: float, avg_score: float,
                    query_terms: int, matched_terms: int,
                    elapsed_ms: float = 0.0):
        """记录 BM25 检索"""
        # BM25 评分 = 匹配率 × 结果质量
        match_rate = matched_terms / max(query_terms, 1)
        quality = min(1.0, top_score * 2) if top_score > 0 else 0.0
        score = 0.4 * match_rate + 0.6 * quality
        extra = {"query_terms": query_terms, "matched_terms": matched_terms,
                 "match_rate": round(match_rate, 3)}
        return self.record_retrieval(
            "bm25_retrieval", "BM25 关键词检索", score,
            result_count=result_count, top_score=top_score, avg_score=avg_score,
            elapsed_ms=elapsed_ms, extra=extra,
        )

    def record_vector(self, result_count: int, top_similarity: float,
                      avg_similarity: float, embedding_dim: int = 0,
                      elapsed_ms: float = 0.0):
        """记录向量检索"""
        quality = min(1.0, top_similarity * 1.5) if top_similarity > 0 else 0.0
        coverage = min(1.0, result_count / 5)
        score = 0.3 * coverage + 0.7 * quality
        return self.record_retrieval(
            "vector_retrieval", "向量语义检索", score,
            result_count=result_count, top_score=top_similarity,
            avg_score=avg_similarity, elapsed_ms=elapsed_ms,
            extra={"embedding_dim": embedding_dim},
        )

    def record_rrf(self, fused_count: int, bm25_count: int, vector_count: int,
                   consensus_count: int, elapsed_ms: float = 0.0):
        """记录 RRF 融合"""
        # 共识度 = 两个检索器共同返回的文档数 / 融合总数
        consensus_rate = consensus_count / max(fused_count, 1)
        redundancy = 1.0 - abs(bm25_count - vector_count) / max(bm25_count + vector_count, 1)
        score = 0.5 * consensus_rate + 0.3 * redundancy + 0.2 * min(1.0, fused_count / 5)
        diag = ""
        warns = []
        if consensus_rate < 0.3:
            diag = f"BM25 和 Vector 共识度仅 {consensus_rate:.0%}，两个检索器方向不一致"
            warns.append(f"检索器共识度低 ({consensus_rate:.0%})")
        return self.record(
            "rrf_fusion", "RRF 融合排序", score,
            elapsed_ms=elapsed_ms,
            details={"bm25_count": bm25_count, "vector_count": vector_count,
                     "fused_count": fused_count, "consensus_count": consensus_count,
                     "consensus_rate": round(consensus_rate, 3)},
            diagnosis=diag, warnings=warns,
        )

    def record_rerank(self, result_count: int, top_rerank_score: float,
                      avg_rerank_score: float, high_count: int,
                      elapsed_ms: float = 0.0):
        """记录 Rerank 重排序"""
        high_ratio = high_count / max(result_count, 1)
        quality = min(1.0, top_rerank_score * 1.2) if top_rerank_score > 0 else 0.0
        score = 0.5 * quality + 0.5 * high_ratio
        diag = ""
        warns = []
        if high_ratio < 0.3:
            diag = f"高相关文档占比仅 {high_ratio:.0%}，Rerank 后仍无优质候选"
            warns.append(f"Rerank 高相关率低 ({high_ratio:.0%})")
            suggests = ["知识库内容可能与查询不匹配，建议扩充相关文档"]
            return self.record(
                "rerank", "Rerank 精排", score,
                elapsed_ms=elapsed_ms,
                details={"result_count": result_count, "top_score": round(top_rerank_score, 4),
                         "avg_score": round(avg_rerank_score, 4),
                         "high_count": high_count, "high_ratio": round(high_ratio, 3)},
                diagnosis=diag, warnings=warns, suggestions=suggests,
            )
        return self.record(
            "rerank", "Rerank 精排", score,
            elapsed_ms=elapsed_ms,
            details={"result_count": result_count, "top_score": round(top_rerank_score, 4),
                     "avg_score": round(avg_rerank_score, 4),
                     "high_count": high_count, "high_ratio": round(high_ratio, 3)},
            diagnosis=diag, warnings=warns,
        )

    # --- 生成阶段 ---
    def record_llm(self, score: float, token_count: int, model: str = "",
                   elapsed_ms: float = 0.0):
        """记录 LLM 生成"""
        diag = ""
        warns = []
        if token_count == 0:
            diag = "LLM 未生成任何内容"
            warns.append("LLM 生成为空")
        elif token_count < 50:
            warns.append(f"LLM 输出仅 {token_count} tokens，回答可能不完整")
        return self.record(
            "llm_generate", "LLM 生成", score,
            elapsed_ms=elapsed_ms,
            details={"token_count": token_count, "model": model},
            diagnosis=diag, warnings=warns,
        )

    def record_hallucination(self, overall_score: float, layer_scores: Dict[str, float],
                             risk: str = "", elapsed_ms: float = 0.0):
        """记录防幻觉综合打分"""
        diag = ""
        warns = []
        sug = []
        if overall_score < 0.6:
            diag = f"防幻觉综合评分 {overall_score:.2f}，存在明显幻觉风险"
            sug.append("增加检索召回数量 top_k，或降低 LLM 温度")
        for layer, s in layer_scores.items():
            if s < 0.5:
                warns.append(f"{layer} 评分过低 ({s:.2f})")
        return self.record(
            "hallucination_check", "六层防幻觉校验", overall_score,
            elapsed_ms=elapsed_ms,
            details={"layer_scores": layer_scores, "risk": risk},
            diagnosis=diag, warnings=warns, suggestions=sug,
        )

    def record_hallucination_layers(self, layer_scores: Dict[str, float],
                                    elapsed_ms: float = 0.0):
        """逐层记录防幻觉分数（可选细粒度）"""
        layer_names = {
            "L1": "L1 来源验证", "L2": "L2 一致性检查",
            "L3": "L3 事实核查", "L4": "L4 完整性检查",
            "L5": "L5 引用准确性", "L6": "L6 综合评分",
        }
        for key, display in layer_names.items():
            if key in layer_scores:
                self.record(
                    f"hal_{key.lower()}", display, layer_scores[key],
                    elapsed_ms=elapsed_ms,
                    diagnosis="" if layer_scores[key] >= 0.5 else f"{display}未通过",
                )

    # ===================== 聚合计算 =====================

    def _group_stages(self) -> Dict[StageGroup, List[StageScore]]:
        """将 stage 映射到 StageGroup"""
        mapping = {
            "metadata_parse": StageGroup.INGESTION,
            "text_clean": StageGroup.INGESTION,
            "chunk_quality": StageGroup.INGESTION,
            "tokenization": StageGroup.PREPROCESS,
            "keyword_extract": StageGroup.PREPROCESS,
            "query_rewrite": StageGroup.PREPROCESS,
            "bm25_retrieval": StageGroup.RETRIEVAL,
            "vector_retrieval": StageGroup.RETRIEVAL,
            "rrf_fusion": StageGroup.RETRIEVAL,
            "rerank": StageGroup.RETRIEVAL,
            "llm_generate": StageGroup.GENERATION,
            "hallucination_check": StageGroup.GENERATION,
            "slot_filling_summary": StageGroup.GENERATION,
            "tool_session_summary": StageGroup.GENERATION,
        }
        groups: Dict[StageGroup, List[StageScore]] = {g: [] for g in StageGroup}
        for s in self.stages:
            g = mapping.get(s.stage)
            if g is None:
                if s.stage.startswith("slot_"):
                    g = StageGroup.GENERATION
                elif s.stage.startswith("tool_"):
                    g = StageGroup.GENERATION
            if g:
                groups[g].append(s)
        return groups

    def overall_score(self) -> float:
        """加权总分"""
        groups = self._group_stages()
        total = 0.0
        weight_sum = 0.0
        for g, stages in groups.items():
            if not stages:
                continue
            avg = sum(s.score for s in stages) / len(stages)
            w = self.group_weights.get(g, 0.1)
            total += avg * w
            weight_sum += w
        return total / max(weight_sum, 1) if weight_sum > 0 else 0.0

    def group_scores(self) -> Dict[str, Dict]:
        """每个大类摘要"""
        groups = self._group_stages()
        result = {}
        for g, stages in groups.items():
            if not stages:
                result[g.value[0]] = {"score": None, "grade": "-", "stages": 0}
                continue
            avg = sum(s.score for s in stages) / len(stages)
            result[g.value[0]] = {
                "score": round(avg, 3),
                "grade": ScoreGrade.from_score(avg).value,
                "name": g.value[1],
                "stages": len(stages),
                "worst_stage": min(stages, key=lambda s: s.score).display_name
                if any(s.score < 0.6 for s in stages) else None,
            }
        self._overall = self.overall_score()
        self._overall_grade = ScoreGrade.from_score(self._overall)
        return result

    def get_by_name(self, name: str) -> Optional[StageScore]:
        """按 stage 名查找"""
        for s in self.stages:
            if s.stage == name:
                return s
        return None

    # ===================== 输出 =====================

    def summary(self) -> str:
        """文本摘要"""
        overall = self.overall_score()
        grade = ScoreGrade.from_score(overall)
        lines = [
            "=" * 72,
            f"  Pipeline 全链路打分卡 — 综合: {overall:.2f}  ({grade.cn}, {grade.value})",
            "=" * 72,
        ]
        if self.query:
            lines.append(f"  查询: {self.query[:60]}")

        groups = self._group_stages()
        for g in [StageGroup.INGESTION, StageGroup.PREPROCESS,
                   StageGroup.RETRIEVAL, StageGroup.GENERATION]:
            stages = groups.get(g, [])
            if not stages:
                continue
            avg = sum(s.score for s in stages) / len(stages)
            g_grade = ScoreGrade.from_score(avg)
            lines.append(f"\n── {g.value[1]} [{g_grade.value}] 均分 {avg:.2f} ──")
            for s in stages:
                marker = "[FAIL]" if s.score < 0.4 else ("[WARN]" if s.score < 0.6 else ("[ OK ]" if s.score < 0.75 else "[GOOD]"))
                lines.append(f"  {marker} {s.display_name:<14s} {s.score:.2f} "
                             f"({s.elapsed_ms:6.0f}ms)")
                if s.diagnosis:
                    lines.append(f"     [!] {s.diagnosis}")
                for w in s.warnings[:1]:
                    lines.append(f"     [W] {w}")
                for sug in s.suggestions[:1]:
                    lines.append(f"     [>] {sug}")

        lines.append(f"\n{'='*72}")
        lines.append(f"  综合评分: {overall:.2f} / 1.00  ({grade.cn})")
        lines.append(f"  共 {len(self.stages)} 个评分点, 耗时 {self.total_elapsed():.0f}ms")
        # 找最薄弱的三个环节
        worst = sorted(self.stages, key=lambda s: s.score)[:3]
        if worst and worst[0].score < 0.7:
            lines.append(f"  [!] 最需改进: {', '.join(s.display_name for s in worst)}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def table(self) -> str:
        """表格形式"""
        rows = [
            f"{'阶段':<18s} {'分数':>6s} {'等级':>4s} {'耗时(ms)':>8s} {'诊断'}",
            "-" * 72,
        ]
        for s in self.stages:
            rows.append(
                f"{s.display_name:<18s} {s.score:>5.2f}  {s.grade.value:>3s}  "
                f"{s.elapsed_ms:>7.0f}  {s.diagnosis[:30] if s.diagnosis else '—'}"
            )
        overall = self.overall_score()
        rows.append("-" * 72)
        rows.append(f"{'【综合】':<18s} {overall:>5.2f}  {ScoreGrade.from_score(overall).value:>3s}")
        return "\n".join(rows)

    def to_dict(self) -> Dict:
        """导出为字典"""
        groups = self._group_stages()
        group_summaries = {}
        for g, stages in groups.items():
            if stages:
                group_summaries[g.value[0]] = {
                    "name": g.value[1],
                    "avg_score": round(sum(s.score for s in stages) / len(stages), 3),
                    "stages": [s.stage for s in stages],
                }
        return {
            "query": self.query,
            "overall_score": round(self.overall_score(), 3),
            "overall_grade": ScoreGrade.from_score(self.overall_score()).value,
            "stage_count": len(self.stages),
            "total_elapsed_ms": self.total_elapsed(),
            "groups": group_summaries,
            "stages": [
                {
                    "stage": s.stage, "display": s.display_name,
                    "score": round(s.score, 3), "grade": s.grade.value,
                    "elapsed_ms": round(s.elapsed_ms, 1),
                    "diagnosis": s.diagnosis, "warnings": s.warnings,
                    "suggestions": s.suggestions, "details": s.details,
                }
                for s in self.stages
            ],
        }

    def total_elapsed(self) -> float:
        return sum(s.elapsed_ms for s in self.stages)

    def worst_stages(self, n: int = 3) -> List[StageScore]:
        return sorted(self.stages, key=lambda s: s.score)[:n]


    def print_summary(self, title: str = None):
        """打印评分卡详细信息（便捷方法）"""
        if not self.stages:
            print("\n[无评分数据]")
            return
        print()
        if title:
            print(title)
        print(self.summary())


# 便捷工厂
def create_scorecard(query: str = "") -> PipelineScoreCard:
    return PipelineScoreCard(query=query)
