"""
FastAPI 入口 - 生产级 API 服务

端点:
  POST /query        - 主查询入口（走完整流水线）
  POST /feedback     - 用户反馈
  GET  /stats        - 系统统计
  GET  /health       - 健康检查
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import logging

from pipeline import PipelineOrchestrator, PipelineConfig
from pipeline.learning import ContinuousLearning

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化
app = FastAPI(
    title="Agentic RAG Pipeline",
    description="三 Agent 流水线 + Reference Agent + 六层防幻觉 + Hybrid RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 核心组件
orchestrator = PipelineOrchestrator(PipelineConfig())
learning = ContinuousLearning()


# ===== Request/Response Models =====

class QueryRequest(BaseModel):
    text: str = Field(..., description="用户输入文本", min_length=1)
    context: Optional[Dict] = Field(default=None, description="额外上下文")


class SourceInfo(BaseModel):
    id: str
    text: str
    score: float
    source: str


class QueryResponse(BaseModel):
    success: bool
    answer: str
    citations: List[SourceInfo] = []
    confidence: float = 0.0
    hallucination_risk: str = "unknown"
    elapsed_ms: float = 0.0
    trace: List[Dict] = []


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    confidence: float = 0.0
    hallucination_risk: str = "unknown"
    tags: List[str] = []
    comment: str = ""


class StatsResponse(BaseModel):
    total_queries: int
    avg_rating: float
    avg_confidence: float
    high_risk_count: int
    improvement_trend: float
    weak_areas: List[str]
    suggestions: List[str]


# ===== Endpoints =====

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    主查询入口 - 走完整三 Agent 流水线
    """
    start = time.time()

    try:
        result = orchestrator.run(request.text, request.context)
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (time.time() - start) * 1000

    return QueryResponse(
        success=result.status.value == "completed",
        answer=result.final_answer or "处理失败",
        citations=[
            SourceInfo(
                id=c.get("id", ""),
                text=c.get("text", "")[:100],
                score=c.get("score", 0),
                source=c.get("source", "unknown"),
            )
            for c in (result.citations or [])
        ],
        confidence=result.confidence,
        hallucination_risk=(
            result.reference_result.output.get("hallucination_risk", "unknown")
            if result.reference_result and result.reference_result.success
            else "unknown"
        ),
        elapsed_ms=elapsed,
        trace=result.trace,
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """用户反馈"""
    learning.record_feedback(
        query=request.query,
        answer=request.answer,
        rating=request.rating,
        confidence=request.confidence,
        hallucination_risk=request.hallucination_risk,
        tags=request.tags,
        user_comment=request.comment,
    )
    return {"status": "ok", "message": "反馈已记录"}


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """系统统计"""
    s = learning.get_stats()
    weak = learning.get_weak_areas()
    suggestions = learning.get_improvement_suggestions()

    return StatsResponse(
        total_queries=s.total_queries,
        avg_rating=round(s.avg_rating, 2),
        avg_confidence=round(s.avg_confidence, 2),
        high_risk_count=s.high_risk_count,
        improvement_trend=round(s.improvement_trend, 2),
        weak_areas=weak,
        suggestions=suggestions,
    )


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
