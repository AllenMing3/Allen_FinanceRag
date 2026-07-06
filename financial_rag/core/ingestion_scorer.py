"""
索引管道评分卡 — IngestionScoreCard

评估文档从原始文本到可检索索引的全链路质量。
4 个阶段独立评分，精确定位瓶颈。

阶段:
1. 预处理质量 (preprocessing): 源文档内容完整性
2. 切片质量 (chunking): chunk 大小分布、边界质量
3. 分词质量 (tokenization): 词汇丰富度、金融术语覆盖、垃圾 token 率
4. 索引健康度 (index_health): 词汇多样性、引擎覆盖、空文档率

用法:
    card = IngestionScoreCard()
    card.record_preprocessing(docs)
    card.record_chunking(original_count, chunked_docs)
    card.record_tokenization(corpus_tokens)
    card.record_index(doc_count, total_vocab, bm25_built, chroma_built)
    card.compute()
    card.log_summary()
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from financial_rag.core.scorer import ScoreGrade

logger = logging.getLogger(__name__)

_CHINESE = re.compile(r'[\u4e00-\u9fff]')
_SENTENCE_END = set('。！？；.!?\n')


# ===================== 辅助函数 =====================


def _chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CHINESE.findall(text)) / len(text)


def _bar(score: float, width: int = 12) -> str:
    filled = int(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ===================== 单阶段评分 =====================


@dataclass
class IngestionStageScore:
    """单个索引阶段的评分"""
    name: str
    display: str
    score: float
    grade: ScoreGrade
    metrics: Dict[str, Any]
    diagnosis: str = ""
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# ===================== 评分卡 =====================


class IngestionScoreCard:
    """
    索引管道评分卡 — 评估 预处理→切片→分词→索引 四阶段质量

    每个阶段独立评分 (0~1.0)，附诊断信息和建议。
    综合得分 = 加权平均 (预处理 20% + 切片 25% + 分词 30% + 索引 25%)
    """

    _WEIGHTS = (0.20, 0.25, 0.30, 0.25)

    def __init__(self):
        self.stages: List[IngestionStageScore] = []
        self._overall: float = 0.0
        self._overall_grade: ScoreGrade = ScoreGrade.FAIR

    # ===================== 阶段评分方法 =====================

    def record_preprocessing(self, docs: List[Dict]):
        """
        评估源文档内容质量

        指标: 中文占比、段落密度、平均长度、内容丰富度
        """
        if not docs:
            self.stages.append(IngestionStageScore(
                "preprocessing", "预处理质量", 0.0, ScoreGrade.FAIL,
                {"doc_count": 0}, diagnosis="无文档",
                warnings=["无文档可供索引"],
            ))
            return

        texts = [d.get("text", "") for d in docs]
        lengths = [len(t) for t in texts]
        avg_len = sum(lengths) / len(lengths)

        # 中文占比
        ratios = [_chinese_ratio(t) for t in texts]
        avg_chinese = sum(ratios) / len(ratios)

        # 段落密度 (每文档有效段落数，>20字算有效)
        para_counts = [
            len([p for p in t.split("\n\n") if len(p.strip()) > 20])
            for t in texts
        ]
        avg_paras = sum(para_counts) / len(para_counts)

        # 空文档占比
        empty_count = sum(1 for t in texts if len(t.strip()) < 10)
        empty_ratio = empty_count / len(texts)

        # 评分
        # content: 中文占比 (权重 0.5) + 段落密度 (权重 0.5)
        content = min(1.0, (avg_chinese / 0.3) * 0.5 + min(avg_paras / 5, 1.0) * 0.5)
        # size: 平均长度在 100-5000 之间为理想
        if avg_len < 50:
            size_score = avg_len / 50
        elif avg_len <= 5000:
            size_score = 1.0
        else:
            size_score = max(0.5, 1.0 - (avg_len - 5000) / 20000)
        # completeness: 非空文档占比
        completeness = 1.0 - empty_ratio

        score = content * 0.4 + size_score * 0.3 + completeness * 0.3

        warnings = []
        suggestions = []
        if avg_chinese < 0.1:
            warnings.append(f"中文占比低 ({avg_chinese:.0%})")
            suggestions.append("检查文档编码或内容是否匹配中文")
        if empty_ratio > 0.1:
            warnings.append(f"{empty_count}/{len(docs)} 文档内容过短 (<10字)")
        if avg_paras < 1:
            suggestions.append("文档段落过少，可能结构不够丰富")

        diagnosis = ""
        if score < 0.6:
            diagnosis = f"内容质量不足: 中文占比 {avg_chinese:.0%}, 平均段落 {avg_paras:.1f}"

        self.stages.append(IngestionStageScore(
            "preprocessing", "预处理质量", score, ScoreGrade.from_score(score),
            {
                "doc_count": len(docs),
                "avg_length": round(avg_len),
                "chinese_ratio": round(avg_chinese, 3),
                "avg_paragraphs": round(avg_paras, 1),
                "empty_docs": empty_count,
            },
            diagnosis=diagnosis, warnings=warnings, suggestions=suggestions,
        ))

    def record_chunking(self, original_count: int, chunked_docs: List[Dict],
                        chunk_size: int = 500):
        """
        评估切片质量

        指标: 大小均匀度、边界质量 (句子/段落边界)、小 chunk 占比
        """
        if not chunked_docs:
            self.stages.append(IngestionStageScore(
                "chunking", "切片质量", 0.0, ScoreGrade.FAIL,
                {"original_count": original_count, "chunk_count": 0},
                diagnosis="切片后无输出", warnings=["切片后无可用 chunk"],
            ))
            return

        sizes = [len(d.get("text", "")) for d in chunked_docs]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)

        # 大小均匀度: CV = std/mean, 越低越均匀
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std = variance ** 0.5
        cv = std / max(avg_size, 1)
        uniformity = max(0.0, 1.0 - cv * 0.8)

        # 边界质量: chunk 末尾是否为句子结束符
        good_endings = sum(
            1 for d in chunked_docs
            if d.get("text", "") and d["text"][-1] in _SENTENCE_END
        )
        boundary = good_endings / len(chunked_docs)

        # 过短 chunk 占比 (<50字)
        small_count = sum(1 for s in sizes if s < 50)
        small_ratio = small_count / len(sizes)
        small_score = max(0.0, 1.0 - small_ratio * 3)

        # 超长 chunk 占比
        oversized = sum(1 for s in sizes if s > chunk_size * 1.5)
        oversized_ratio = oversized / len(sizes)

        score = uniformity * 0.35 + boundary * 0.35 + small_score * 0.30

        warnings = []
        suggestions = []
        if small_ratio > 0.2:
            warnings.append(f"{small_ratio:.0%} chunks 过短 (<50字)")
            suggestions.append("考虑增大 min_chunk_size 或减小 chunk_size")
        if oversized_ratio > 0.1:
            warnings.append(f"{oversized_ratio:.0%} chunks 超过 {chunk_size * 1.5} 字符")
        if boundary < 0.5:
            suggestions.append("切片边界质量低，硬切分占比高")

        diagnosis = ""
        if score < 0.6:
            diagnosis = f"切片不佳: 边界完整率 {boundary:.0%}, 小chunk占 {small_ratio:.0%}"

        self.stages.append(IngestionStageScore(
            "chunking", "切片质量", score, ScoreGrade.from_score(score),
            {
                "original_count": original_count,
                "chunk_count": len(chunked_docs),
                "avg_size": round(avg_size),
                "min_size": min_size,
                "max_size": max_size,
                "std_dev": round(std),
                "uniformity": round(uniformity, 3),
                "boundary_quality": round(boundary, 3),
                "small_chunk_ratio": round(small_ratio, 3),
            },
            diagnosis=diagnosis, warnings=warnings, suggestions=suggestions,
        ))

    def record_tokenization(self, corpus_tokens: List[List[str]]):
        """
        评估分词质量

        指标: 词汇丰富度 (TTR)、金融术语覆盖率、垃圾 token 占比
        """
        if not corpus_tokens:
            self.stages.append(IngestionStageScore(
                "tokenization", "分词质量", 0.0, ScoreGrade.FAIL,
                {"doc_count": 0}, diagnosis="无 token 数据",
            ))
            return

        # 延迟加载金融术语
        try:
            from financial_rag.retrievers.dictionaries import FINANCIAL_TERMS, INDUSTRY_TERMS
            domain_terms = FINANCIAL_TERMS | INDUSTRY_TERMS
        except Exception:
            domain_terms = set()

        all_tokens = []
        for tokens in corpus_tokens:
            all_tokens.extend(tokens)

        total = len(all_tokens)
        if total == 0:
            self.stages.append(IngestionStageScore(
                "tokenization", "分词质量", 0.0, ScoreGrade.FAIL,
                {"total_tokens": 0}, diagnosis="所有文档分词结果为空",
                warnings=["分词器可能未正确初始化"],
            ))
            return

        unique = set(all_tokens)
        ttr = len(unique) / total

        # 金融术语覆盖: 含至少 1 个金融术语的文档占比
        if domain_terms:
            docs_with_terms = sum(
                1 for tokens in corpus_tokens
                if any(t in domain_terms for t in tokens)
            )
            financial_coverage = docs_with_terms / len(corpus_tokens)
        else:
            financial_coverage = 0.5  # 无词典时取中性值

        # 垃圾 token: 单字、纯数字、常见停用词
        _stop = {'的', '了', '是', '在', '和', '有', '我', '他', '她', '它',
                 '这', '那', '不', '也', '都', '就', '要', '会', '能', '对', '把', '被'}
        junk = sum(1 for t in all_tokens
                   if len(t) <= 1 or t.isdigit() or t in _stop)
        junk_ratio = junk / total

        # 平均 token 长度
        avg_token_len = sum(len(t) for t in all_tokens) / total

        # 评分
        richness = min(1.0, ttr / 0.3)
        coverage_score = financial_coverage
        junk_score = max(0.0, 1.0 - junk_ratio * 2)
        # token 长度甜区: 2-6 字符
        if avg_token_len < 1.5:
            len_score = avg_token_len / 1.5
        elif avg_token_len <= 6:
            len_score = 1.0
        else:
            len_score = max(0.5, 1.0 - (avg_token_len - 6) / 10)

        score = richness * 0.3 + coverage_score * 0.25 + junk_score * 0.25 + len_score * 0.20

        warnings = []
        suggestions = []
        if junk_ratio > 0.3:
            warnings.append(f"垃圾 token 占比 {junk_ratio:.0%}")
            suggestions.append("加强分词后过滤或使用更严格长度阈值")
        if financial_coverage < 0.3 and domain_terms:
            warnings.append(f"仅 {financial_coverage:.0%} 文档含金融术语")
            suggestions.append("文档可能非金融领域，或金融词典需扩展")
        if ttr < 0.1:
            suggestions.append("词汇丰富度极低，可能存在大量重复内容")

        diagnosis = ""
        if score < 0.6:
            diagnosis = (
                f"分词不足: TTR={ttr:.3f}, 金融覆盖={financial_coverage:.0%}, "
                f"垃圾率={junk_ratio:.0%}"
            )

        self.stages.append(IngestionStageScore(
            "tokenization", "分词质量", score, ScoreGrade.from_score(score),
            {
                "total_tokens": total,
                "unique_tokens": len(unique),
                "ttr": round(ttr, 4),
                "financial_coverage": round(financial_coverage, 3),
                "junk_ratio": round(junk_ratio, 3),
                "avg_token_length": round(avg_token_len, 2),
                "doc_count": len(corpus_tokens),
            },
            diagnosis=diagnosis, warnings=warnings, suggestions=suggestions,
        ))

    def record_index(self, doc_count: int, total_vocab: int,
                     bm25_built: bool, chroma_built: bool,
                     embedding_dim: int = 0, empty_docs: int = 0):
        """
        评估索引健康度

        指标: 词汇多样性 (vocab/doc)、完整性 (空文档率)、引擎覆盖 (BM25+Chroma)
        """
        # 词汇多样性
        vocab_per_doc = total_vocab / max(doc_count, 1)
        diversity = min(1.0, vocab_per_doc / 100)

        # 完整性
        completeness = (doc_count - empty_docs) / max(doc_count, 1)

        # 引擎覆盖
        engines_up = int(bm25_built) + int(chroma_built)
        engine_score = engines_up / 2

        score = diversity * 0.35 + completeness * 0.35 + engine_score * 0.30

        warnings = []
        suggestions = []
        if empty_docs > doc_count * 0.1 and doc_count > 0:
            warnings.append(f"{empty_docs}/{doc_count} 文档为空")
        if not chroma_built:
            warnings.append("ChromaDB 未构建，仅支持 BM25+Jaccard 检索")
            suggestions.append("配置 DashScope embedding API 以启用语义检索")
        if not bm25_built:
            warnings.append("BM25 未构建")

        diagnosis = ""
        if score < 0.6:
            diagnosis = f"索引不健康: 词汇多样性={diversity:.3f}, 空文档={empty_docs}"

        self.stages.append(IngestionStageScore(
            "index_health", "索引健康度", score, ScoreGrade.from_score(score),
            {
                "doc_count": doc_count,
                "total_vocab": total_vocab,
                "vocab_per_doc": round(vocab_per_doc),
                "bm25_built": bm25_built,
                "chroma_built": chroma_built,
                "embedding_dim": embedding_dim,
                "empty_docs": empty_docs,
            },
            diagnosis=diagnosis, warnings=warnings, suggestions=suggestions,
        ))

    # ===================== 汇总 =====================

    def compute(self) -> float:
        """计算加权综合得分"""
        if not self.stages:
            return 0.0
        total_w = 0.0
        weighted = 0.0
        for i, stage in enumerate(self.stages):
            w = self._WEIGHTS[i] if i < len(self._WEIGHTS) else 0.25
            weighted += stage.score * w
            total_w += w
        self._overall = weighted / total_w if total_w > 0 else 0.0
        self._overall_grade = ScoreGrade.from_score(self._overall)
        return self._overall

    def overall_score(self) -> float:
        return self._overall

    def log_summary(self):
        """输出评分摘要到 logger.info"""
        for line in self._build_lines():
            logger.info(line)

    def summary(self) -> str:
        return "\n".join(self._build_lines())

    def _build_lines(self) -> List[str]:
        lines = [
            "═══ 索引管道评分卡 ═══",
            f"综合: {self._overall:.1%} [{self._overall_grade.value}] "
            f"{self._overall_grade.cn}",
            "─" * 45,
        ]
        for s in self.stages:
            bar = _bar(s.score)
            lines.append(f"  {s.display:8s} {bar} {s.score:.1%} [{s.grade.value}]")
            # 关键指标
            m = s.metrics
            if s.name == "preprocessing":
                lines.append(
                    f"           ↳ {m.get('doc_count', 0)} 文档, "
                    f"均长 {m.get('avg_length', 0)} 字, "
                    f"中文占比 {m.get('chinese_ratio', 0):.0%}"
                )
            elif s.name == "chunking":
                lines.append(
                    f"           ↳ {m.get('original_count', 0)}→"
                    f"{m.get('chunk_count', 0)} chunks, "
                    f"均长 {m.get('avg_size', 0)} 字, "
                    f"边界 {m.get('boundary_quality', 0):.0%}"
                )
            elif s.name == "tokenization":
                lines.append(
                    f"           ↳ {m.get('total_tokens', 0)} tokens, "
                    f"TTR={m.get('ttr', 0):.3f}, "
                    f"金融覆盖 {m.get('financial_coverage', 0):.0%}, "
                    f"垃圾率 {m.get('junk_ratio', 0):.0%}"
                )
            elif s.name == "index_health":
                bm = "✓" if m.get("bm25_built") else "✗"
                ch = "✓" if m.get("chroma_built") else "✗"
                lines.append(
                    f"           ↳ {m.get('doc_count', 0)} 文档, "
                    f"词汇量 {m.get('total_vocab', 0)}, "
                    f"BM25={bm} Chroma={ch}"
                )
            for w in s.warnings:
                lines.append(f"           ⚠ {w}")
            for sg in s.suggestions:
                lines.append(f"           → {sg}")
        lines.append("─" * 45)
        return lines

    def to_dict(self) -> Dict:
        """API 序列化"""
        return {
            "overall_score": round(self._overall, 3),
            "overall_grade": self._overall_grade.value,
            "overall_grade_cn": self._overall_grade.cn,
            "stages": [
                {
                    "name": s.name,
                    "display": s.display,
                    "score": round(s.score, 3),
                    "grade": s.grade.value,
                    "grade_cn": s.grade.cn,
                    "metrics": s.metrics,
                    "diagnosis": s.diagnosis,
                    "warnings": s.warnings,
                    "suggestions": s.suggestions,
                }
                for s in self.stages
            ],
        }
