"""
并发模型回归锁 — 多页签并发下"不冻结全站、结论不串台"的两条底线。

背景：主通路曾是 async def + 同步重活（LLM 数秒）直接跑在事件循环上，
单个分析冻结全站；且 orchestrator/scheduler 是全局单例，实例级可变状态
（pipeline/context、config.verbose）并发下互相覆写。

本资产锁定修复后的契约：
  ① 重端点必须是同步 def（FastAPI 自动扔线程池，事件循环不被阻塞）
  ② app_state 不再持有 orchestrator/scheduler 全局单例
  ③ 分析链与 pipeline 端点每请求新建 orchestrator（请求间天然隔离）

对应生产代码：financial_rag/api/{analysis,query,kb,ingest}_router.py、app_state.py
"""
import inspect
from types import SimpleNamespace

import pytest

from financial_rag.api import analysis_router, query_router, kb_router, ingest_router
from financial_rag.api.app_state import _ensure_init
from financial_rag.api.models import QueryRequest

pytestmark = pytest.mark.unit  # 全部离线：静态契约 + stub 隔离探针，无网络/无落盘


# ── ① 重端点同步 def 契约 ──

# (模块, 端点函数名)：凡含 LLM/索引/文件解析等秒级重活的端点都在此列。
# 若改回 async def 且未把重活包进 to_thread，单请求会冻结全站（含静态资源）。
_HEAVY_ENDPOINTS = [
    (analysis_router, "api_analyze_news"),
    (analysis_router, "api_analyze_topic"),
    (analysis_router, "api_news"),
    (analysis_router, "api_kline"),
    (query_router, "api_pipeline"),
    (query_router, "api_slot"),
    (query_router, "api_score"),
    (kb_router, "api_kb_query"),
    (kb_router, "api_build_kb"),
    (kb_router, "api_pipeline_clean_report"),
    (kb_router, "api_pipeline_chunk_demo"),
    (kb_router, "api_pipeline_dict_stats"),
    (ingest_router, "api_ingest_files"),
    (ingest_router, "api_ingest_upload"),
    (ingest_router, "api_ingest_news"),
]


class TestSyncHeavyEndpoints:
    @pytest.mark.critical
    @pytest.mark.parametrize("module,func_name", _HEAVY_ENDPOINTS,
                             ids=[f"{m.__name__.split('.')[-1]}.{n}" for m, n in _HEAVY_ENDPOINTS])
    def test_heavy_endpoint_is_sync_def(self, module, func_name):
        fn = getattr(module, func_name)
        assert not inspect.iscoroutinefunction(fn), (
            f"{func_name} 改回了 async def：同步重活会阻塞事件循环，"
            f"多页签并发时全站冻结。要么改回 def，要么把重活包进 asyncio.to_thread。"
        )


# ── ② 无共享单例契约 ──


class TestNoSharedSingleton:
    @pytest.mark.critical
    def test_ensure_init_does_not_create_shared_orchestrator(self):
        # orchestrator.pipeline/context 是实例级可变状态，全局单例并发下结论串台
        src = inspect.getsource(_ensure_init)
        assert '_state["orchestrator"]' not in src
        assert '_state["scheduler"]' not in src


# ── ③ 每请求新建 orchestrator（隔离探针） ──


class _StubOrchestrator:
    """只记录身份的最小替身：set_pipeline 空转，execute 返回空链结果。"""

    def set_pipeline(self, agents):
        pass

    def execute(self, raw_input, context=None):
        return {}


class TestPerRequestIsolation:
    @pytest.mark.critical
    def test_analysis_chain_creates_fresh_orchestrator_per_call(self, monkeypatch):
        created = []

        def _spy(retriever=None, llm=None):
            orch = _StubOrchestrator()
            created.append(orch)
            return orch

        # 端点内是调用期局部 import，patch 工厂模块属性即可拦截
        monkeypatch.setattr("financial_rag.core.factory.create_orchestrator", _spy)
        r1 = analysis_router._run_analysis_via_agent_chain("news", "探针A", {})
        r2 = analysis_router._run_analysis_via_agent_chain("news", "探针B", {})
        assert r1 is None and r2 is None  # stub 空链 → 走 fallback 分支，符合预期
        assert len(created) == 2
        assert created[0] is not created[1], "两次请求共享了同一 orchestrator，并发会串台"

    @pytest.mark.critical
    def test_pipeline_endpoint_creates_fresh_scheduler_per_request(self, monkeypatch):
        orchs = []
        sched_orchs = []

        def _orch_spy(retriever=None, llm=None):
            orch = _StubOrchestrator()
            orchs.append(orch)
            return orch

        stub_result = SimpleNamespace(
            query="探针", final_output="ok", scorecard=None, errors=[],
            fetch_elapsed_ms=0.0, index_elapsed_ms=0.0, process_elapsed_ms=0.0,
            output_elapsed_ms=0.0, total_elapsed_ms=0.0,
        )

        def _sched_spy(orchestrator=None, **kwargs):
            sched_orchs.append(orchestrator)
            return SimpleNamespace(run=lambda **kw: stub_result)

        monkeypatch.setattr(query_router, "_ensure_init", lambda: None)
        monkeypatch.setitem(query_router._state, "has_key", True)
        monkeypatch.setattr("financial_rag.core.factory.create_orchestrator", _orch_spy)
        monkeypatch.setattr("financial_rag.core.pipeline.create_pipeline_scheduler", _sched_spy)

        query_router.api_pipeline(QueryRequest(query="并发探针A"))
        query_router.api_pipeline(QueryRequest(query="并发探针B"))

        assert len(orchs) == 2 and orchs[0] is not orchs[1]
        # 每个 scheduler 必须绑定自己那份 orchestrator，而非共享单例
        assert sched_orchs == orchs