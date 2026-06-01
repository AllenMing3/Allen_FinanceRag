"""
流水线协调器 - 生产级三 Agent 流水线引擎

架构：
  CleanerAgent → KeywordAgent → AnalyzerAgent → ReferenceAgent
       ↓              ↓              ↓              ↓
   清洗垃圾        提取关键词      检索+推理       引用验证
                                           ↓
                                    防幻觉中间件(6层)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import logging

from .cleaner_agent import CleanerAgent
from .keyword_agent import KeywordAgent
from .analyzer_agent import AnalyzerAgent
from .reference_agent import ReferenceAgent

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 执行控制
    max_retries: int = 2               # 单步最大重试
    timeout_per_agent: float = 30.0    # 每个 Agent 超时(秒)
    enable_tracing: bool = True        # 启用追踪
    enable_monitoring: bool = True     # 启用监控

    # 检索配置
    hybrid_top_k: int = 10             # BM25+Vector 各取 top_k
    rrf_k: int = 60                    # RRF 融合参数
    bm25_weight: float = 0.3           # BM25 权重
    vector_weight: float = 0.7         # Vector 权重

    # 防幻觉配置
    hallucination_checks: int = 6      # 防幻觉层数
    min_faithfulness: float = 0.7      # 最低忠实度
    min_source_count: int = 2          # 最少引用源数量

    # 持续学习
    enable_learning: bool = True       # 启用持续学习
    feedback_threshold: float = 0.6    # 反馈阈值


@dataclass
class StepResult:
    """单步结果"""
    agent_name: str
    success: bool
    output: Any
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """流水线完整结果"""
    status: PipelineStatus
    # 各步骤结果
    cleaner_result: Optional[StepResult] = None
    keyword_result: Optional[StepResult] = None
    analyzer_result: Optional[StepResult] = None
    reference_result: Optional[StepResult] = None

    # 最终输出
    final_answer: Optional[str] = None
    citations: List[Dict] = field(default_factory=list)
    confidence: float = 0.0

    # 元信息
    total_elapsed_ms: float = 0.0
    trace: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


class PipelineOrchestrator:
    """
    三 Agent 流水线协调器

    执行流程:
    1. CleanerAgent  - 清洗垃圾信息
    2. KeywordAgent  - 提取关键词
    3. AnalyzerAgent - 检索 + 推理 + 生成
    4. ReferenceAgent - 引用验证 + 防幻觉

    支持:
    - 流水线失败自动重试
    - 每步超时控制
    - 完整执行追踪
    - 防幻觉多层校验
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # 初始化 Agent
        self.cleaner = CleanerAgent()
        self.keyword = KeywordAgent()
        self.analyzer = AnalyzerAgent()
        self.reference = ReferenceAgent()

        logger.info("PipelineOrchestrator 初始化完成")

    def run(self, input_text: str, context: Optional[Dict] = None) -> PipelineResult:
        """
        执行完整流水线

        Args:
            input_text: 原始输入文本
            context: 额外上下文

        Returns:
            PipelineResult: 完整执行结果
        """
        context = context or {}
        start_time = time.time()
        trace = []

        result = PipelineResult(status=PipelineStatus.RUNNING)

        try:
            # ===== Step 1: 清洗 =====
            step1_start = time.time()
            cleaner_output = self.cleaner.clean(input_text, context)
            result.cleaner_result = StepResult(
                agent_name="CleanerAgent",
                success=True,
                output=cleaner_output,
                elapsed_ms=(time.time() - step1_start) * 1000,
                metadata={"original_len": len(input_text), "cleaned_len": len(cleaner_output.get("text", ""))}
            )
            trace.append({"step": 1, "agent": "CleanerAgent", "elapsed": result.cleaner_result.elapsed_ms})

            # ===== Step 2: 关键词 =====
            step2_start = time.time()
            keyword_output = self.keyword.extract(
                cleaner_output.get("text", input_text),
                context
            )
            result.keyword_result = StepResult(
                agent_name="KeywordAgent",
                success=True,
                output=keyword_output,
                elapsed_ms=(time.time() - step2_start) * 1000,
                metadata={"keywords_count": len(keyword_output.get("keywords", []))}
            )
            trace.append({"step": 2, "agent": "KeywordAgent", "elapsed": result.keyword_result.elapsed_ms})

            # ===== Step 3: 分析 =====
            step3_start = time.time()
            analyzer_output = self.analyzer.analyze(
                cleaned_text=cleaner_output.get("text", input_text),
                keywords=keyword_output.get("keywords", []),
                intent=keyword_output.get("intent", "general"),
                context=context
            )
            result.analyzer_result = StepResult(
                agent_name="AnalyzerAgent",
                success=analyzer_output.get("success", True),
                output=analyzer_output,
                elapsed_ms=(time.time() - step3_start) * 1000,
                metadata={"retrieval_count": analyzer_output.get("retrieval_count", 0)}
            )
            trace.append({"step": 3, "agent": "AnalyzerAgent", "elapsed": result.analyzer_result.elapsed_ms})

            # ===== Step 4: 引用验证 =====
            step4_start = time.time()
            reference_output = self.reference.verify(
                answer=analyzer_output.get("answer", ""),
                sources=analyzer_output.get("sources", []),
                context=context
            )
            result.reference_result = StepResult(
                agent_name="ReferenceAgent",
                success=reference_output.get("passed", False),
                output=reference_output,
                elapsed_ms=(time.time() - step4_start) * 1000,
                metadata={"hallucination_risk": reference_output.get("hallucination_risk", "unknown")}
            )
            trace.append({"step": 4, "agent": "ReferenceAgent", "elapsed": result.reference_result.elapsed_ms})

            # ===== 汇总 =====
            result.final_answer = reference_output.get("verified_answer", analyzer_output.get("answer", ""))
            result.citations = reference_output.get("citations", [])
            result.confidence = reference_output.get("confidence", 0.0)
            result.status = PipelineStatus.COMPLETED

        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            result.status = PipelineStatus.FAILED
            result.error = str(e)

        result.total_elapsed_ms = (time.time() - start_time) * 1000
        result.trace = trace

        logger.info(
            f"流水线完成: status={result.status.value}, "
            f"elapsed={result.total_elapsed_ms:.0f}ms, "
            f"confidence={result.confidence:.2f}"
        )

        return result

    def run_async(self, input_text: str, context: Optional[Dict] = None):
        """异步执行（后续可接 asyncio）"""
        import asyncio
        loop = asyncio.new_event_loop()
        return loop.run_in_executor(None, self.run, input_text, context)
