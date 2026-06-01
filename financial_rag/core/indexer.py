"""
架构二: Indexer — 多文本索引调度流水线

核心设计:
- Clean → Extract → Retrieve → Verify 四阶段流水线
- Hybrid RAG: BM25 + Vector + RRF 融合
- 多角度查询策略: 对同一输入从不同视角多次检索
- 与业务脱钩: 只定义流水线协议，不绑定具体领域
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ===================== 配置 =====================

@dataclass
class PipelineConfig:
    # 执行控制
    max_retries: int = 2
    timeout_per_stage: float = 30.0

    # 检索配置
    hybrid_top_k: int = 10
    rrf_k: int = 60
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # 防幻觉配置
    min_faithfulness: float = 0.7
    min_source_count: int = 2

    # 追踪
    enable_tracing: bool = True


# ===================== 结果类型 =====================

@dataclass
class StageResult:
    stage_name: str
    success: bool
    output: Any
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    status: PipelineStatus
    # 各阶段结果
    clean: Optional[StageResult] = None
    extract: Optional[StageResult] = None
    retrieve: Optional[StageResult] = None
    verify: Optional[StageResult] = None
    # 最终输出
    final_answer: Optional[str] = None
    citations: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    # 元信息
    total_elapsed_ms: float = 0.0
    trace: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


# ===================== 流水线阶段协议 =====================

class PipelineStage:
    """流水线阶段基类 — 子类实现具体处理逻辑"""

    def __init__(self, name: str):
        self.name = name

    def process(self, input_data: Any, context: Dict) -> Dict:
        """子类重写: 返回处理结果 dict"""
        raise NotImplementedError


class CleanerStage(PipelineStage):
    """
    阶段一: 文本清洗
    - 去噪、去重、标准化
    - 保留高信息量文本
    """
    def __init__(self):
        super().__init__("Cleaner")

    def process(self, text: str, context: Dict) -> Dict:
        """子类实现具体清洗逻辑"""
        return {"text": text, "stats": {}}


class ExtractorStage(PipelineStage):
    """
    阶段二: 关键信息抽取
    - 提取实体、数值、时间等结构化数据
    - 意图识别与查询生成
    """
    def __init__(self):
        super().__init__("Extractor")

    def process(self, text: str, context: Dict) -> Dict:
        """子类实现具体抽取逻辑"""
        return {"keywords": [], "entities": {}, "queries": [text[:300]]}


class RetrieverStage(PipelineStage):
    """
    阶段三: 混合检索
    - BM25 关键词 + Vector 语义 + RRF 融合
    - 多角度多轮检索
    """
    def __init__(self):
        super().__init__("Retriever")

    def process(self, queries: List[str], context: Dict) -> Dict:
        """子类实现具体检索逻辑"""
        return {"sources": [], "retrieval_count": 0}


class VerifierStage(PipelineStage):
    """
    阶段四: 引用验证 + 防幻觉校验
    - 逐断言验证是否有来源支撑
    - 六层防幻觉检查
    """
    def __init__(self):
        super().__init__("Verifier")

    def process(self, answer: str, sources: List[Dict], context: Dict) -> Dict:
        """子类实现具体验证逻辑"""
        return {"passed": True, "risk": "low", "confidence": 0.9}


# ===================== 流水线调度器 =====================

class PipelineOrchestrator:
    """
    Indexer — 多文本索引调度引擎

    执行流程:
    1. Cleaner  — 清洗垃圾信息
    2. Extractor — 提取关键信息 + 生成多角度查询
    3. Retriever — Hybrid 检索 + 多轮 Agentic 检索
    4. Verifier  — 引用验证 + 六层防幻觉校验
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stages: Dict[str, PipelineStage] = {
            "clean": CleanerStage(),
            "extract": ExtractorStage(),
            "retrieve": RetrieverStage(),
            "verify": VerifierStage(),
        }

    def set_stage(self, name: str, stage: PipelineStage):
        """替换某个阶段的实现"""
        self.stages[name] = stage
        return self

    def run(self, input_text: str, context: Optional[Dict] = None,
            scorecard=None) -> PipelineResult:
        """执行完整流水线

        Args:
            input_text: 输入文本
            context: 上下文
            scorecard: PipelineScoreCard 实例（可选），传入则自动记录每阶段评分
        """
        context = context or {}
        t0 = time.time()
        trace = []
        result = PipelineResult(status=PipelineStatus.RUNNING)

        try:
            # === Stage 1: Clean ===
            t1 = time.time()
            clean_out = self.stages["clean"].process(input_text, context)
            original_len = len(input_text)
            cleaned_len = len(clean_out.get("text", ""))
            clean_elapsed = (time.time() - t1) * 1000
            result.clean = StageResult(
                stage_name="clean", success=True, output=clean_out,
                elapsed_ms=clean_elapsed,
                metadata={"original_len": original_len, "cleaned_len": cleaned_len}
            )
            trace.append({"stage": 1, "elapsed": result.clean.elapsed_ms})
            if scorecard:
                clean_score = min(1.0, cleaned_len / max(original_len, 1) * 0.8 + 0.2)
                scorecard.record_clean(clean_score, original_len, cleaned_len,
                                       elapsed_ms=clean_elapsed)

            # === Stage 2: Extract ===
            t2 = time.time()
            extract_out = self.stages["extract"].process(clean_out.get("text", input_text), context)
            extract_elapsed = (time.time() - t2) * 1000
            kw_count = len(extract_out.get("keywords", []))
            result.extract = StageResult(
                stage_name="extract", success=True, output=extract_out,
                elapsed_ms=extract_elapsed,
                metadata={"keywords": kw_count}
            )
            trace.append({"stage": 2, "elapsed": result.extract.elapsed_ms})
            if scorecard:
                kw_score = min(1.0, kw_count / 5) if kw_count > 0 else 0.1
                scorecard.record_keyword_extract(kw_score, kw_count,
                                                 elapsed_ms=extract_elapsed)

            # === Stage 3: Retrieve ===
            t3 = time.time()
            retrieve_out = self.stages["retrieve"].process(
                extract_out.get("queries", [input_text]), context
            )
            retrieve_elapsed = (time.time() - t3) * 1000
            sources = retrieve_out.get("sources", [])
            result.retrieve = StageResult(
                stage_name="retrieve", success=bool(sources),
                output=retrieve_out,
                elapsed_ms=retrieve_elapsed,
                metadata={"sources": len(sources)}
            )
            trace.append({"stage": 3, "elapsed": result.retrieve.elapsed_ms})
            if scorecard:
                # 检索评分: 基于来源数量和质量
                src_count = len(sources)
                has_scores = any(s.get("score") is not None for s in sources)
                if has_scores:
                    src_scores = [s.get("score", 0) for s in sources]
                    ret_score = (min(1.0, src_count / 3) * 0.4 +
                                 (sum(src_scores) / max(len(src_scores), 1)) * 0.6)
                else:
                    ret_score = min(1.0, src_count / 3)
                from .scorer import PipelineScoreCard
                # 直接 record 以免覆盖 HybridRetriever 内部已经记录的子阶段评分
                if scorecard.get_by_name("bm25_retrieval") is None:
                    scorecard.record_retrieval(
                        "bm25_retrieval", "BM25 检索", ret_score,
                        result_count=src_count, top_score=src_scores[0] if has_scores and src_scores else 0.0,
                        avg_score=sum(src_scores)/len(src_scores) if has_scores and src_scores else 0.0,
                        elapsed_ms=retrieve_elapsed,
                    )

            # === Stage 4: Verify ===
            t4 = time.time()
            verify_out = self.stages["verify"].process(
                answer=retrieve_out.get("answer", ""),
                sources=sources,
                context=context
            )
            verify_elapsed = (time.time() - t4) * 1000
            result.verify = StageResult(
                stage_name="verify", success=verify_out.get("passed", False),
                output=verify_out,
                elapsed_ms=verify_elapsed,
                metadata={"risk": verify_out.get("risk", "unknown")}
            )
            trace.append({"stage": 4, "elapsed": result.verify.elapsed_ms})

            # === 汇总 ===
            result.final_answer = verify_out.get("verified_answer", retrieve_out.get("answer", ""))
            result.citations = verify_out.get("citations", [])
            result.confidence = verify_out.get("confidence", 0.0)
            result.status = PipelineStatus.COMPLETED

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.total_elapsed_ms = (time.time() - t0) * 1000
        result.trace = trace
        # 附带打分卡
        if scorecard:
            result.scorecard = scorecard
        return result
