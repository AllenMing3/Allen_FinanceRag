"""
Pydantic request/response models for the Web API
"""
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    template: str = "quick"
    top_k: int = 5
    max_fetch: int = 10
    max_retrieve: int = 5
    verbose: bool = False


class NewsRequest(BaseModel):
    query: str
    summarize: bool = True
    max_news: int = 30


class KlineRequest(BaseModel):
    query: str
    ts_code: str = ""
    name: str = ""
    days: int = 60
    period: str = "daily"


class SlotRequest(BaseModel):
    query: str
    template: str = "quick_qa"
    top_k: int = 5
    no_freeform: bool = False


class ScoreRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestFilesRequest(BaseModel):
    dir: str = "./data/financial"
    analyze: bool = False
    files: list = []  # Optional: specific filenames to import (empty = all)


class IngestNewsRequest(BaseModel):
    query: str
    max_news: int = 30


class BuildRequest(BaseModel):
    documents: list = []
    skip_test_queries: bool = False  # 跳过 build 后的测试查询（节省 token）


class AnalyzeNewsRequest(BaseModel):
    text: str
    query: str = ""


class AnalyzeTopicRequest(BaseModel):
    topic: str
    max_news: int = 20


class ChatFollowupRequest(BaseModel):
    session_id: str
    message: str


class CreateSessionRequest(BaseModel):
    session_type: str = "news"  # "news" | "topic"
    title: str = ""
    initial_analysis: str = ""
    context: dict = {}


class CleanReportRequest(BaseModel):
    text: str = ""          # 直接传文本（与 doc_index 二选一）
    doc_index: int = -1     # >= 0 时从 KB 取第 N 篇文档


class ChunkDemoRequest(BaseModel):
    text: str = ""          # 直接传文本（与 doc_index 二选一）
    doc_index: int = -1     # >= 0 时从 KB 取第 N 篇文档
    chunk_size: int = 500   # 切片大小
