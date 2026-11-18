# -*- coding: utf-8 -*-
"""元数据治理测试 — 归一化 + 公司名守卫 + 写入路径接线

钉死三类历史缺陷（2026-11 探针核实）：
1. publish_time/publish_date 双字段失范（91 篇有权威时间戳却只有 13 篇有 publish_date，
   并存的 10 篇全部冲突）→ normalize_doc_meta 归一
2. 公司名正则兜底是垃圾工厂（"人工智能"×14、句子碎片当公司名）→ 词典唯一裁决
3. _save_analysis_to_kb 落库不写 title/doc_type/doc_id/date_added → 出生即齐全
"""
import pytest

from financial_rag.retrievers.metadata import (
    _extract_company,
    extract_metadata,
    normalize_doc_meta,
)

pytestmark = pytest.mark.unit


class TestNormalizeDocMeta:
    """publish_time → publish_date 归一 + doc_type 词表映射"""

    def test_publish_time_to_publish_date(self):
        out = normalize_doc_meta({"publish_time": "2026-06-07 15:02:52"})
        assert out["publish_date"] == "2026-06-07"

    @pytest.mark.critical
    def test_publish_time_overrides_conflicting_date(self):
        # 新闻场景 publish_time 是权威时间戳；正文抠出的年份（如 "2025-12"）必须让位
        out = normalize_doc_meta({
            "publish_time": "2026-06-07 01:15:00",
            "publish_date": "2025-12",
        })
        assert out["publish_date"] == "2026-06-07"

    def test_invalid_publish_time_ignored(self):
        # 非法格式不产生 publish_date，也不破坏已有值
        out = normalize_doc_meta({"publish_time": "昨天", "publish_date": "2024"})
        assert "publish_date" not in out

    def test_no_publish_time_keeps_date(self):
        assert normalize_doc_meta({"publish_date": "2024"}) == {}

    def test_doc_type_legacy_map(self):
        assert normalize_doc_meta({"doc_type": "新闻报道"})["doc_type"] == "news"
        assert normalize_doc_meta({"doc_type": "年报"})["doc_type"] == "financial_report"
        assert normalize_doc_meta({"doc_type": "研究报告"})["doc_type"] == "research"

    def test_doc_type_unknown_preserved(self):
        # 未知非空值原样保留，不破坏现行分类器词表
        assert normalize_doc_meta({"doc_type": "macro_data"}) == {}

    def test_analysis_doc_type_inferred(self):
        out = normalize_doc_meta({"source": "analysis:news:Meta"})
        assert out["doc_type"] == "analysis"
        out2 = normalize_doc_meta({"doc_type": "", "source": "analysis:news:微软"})
        assert out2["doc_type"] == "analysis"

    def test_empty_meta(self):
        assert normalize_doc_meta({}) == {}


class TestCompanyGuard:
    """公司名抽取：词典唯一裁决，正则兜底已删除（垃圾工厂）"""

    @pytest.mark.critical
    def test_generic_term_not_company(self):
        # "人工智能" 曾命中 "智能" 后缀兜底，污染 14 篇文档的 company 字段
        assert _extract_company("本文介绍人工智能的发展") == ""

    @pytest.mark.critical
    def test_fragment_not_company(self):
        # 句子碎片 "周一发送员工内部信息" 曾被当公司名入库
        assert _extract_company("周一发送员工内部信息，讨论AI规划") == ""

    def test_dictionary_hit(self):
        assert _extract_company("商汤科技发布财报") == "商汤科技"

    def test_extract_metadata_respects_existing(self):
        # 已有非空 company 不被覆盖（既有契约）
        out = extract_metadata("科大讯飞发布新品", existing_meta={"company": "商汤科技"})
        assert "company" not in out


class TestWritePathWiring:
    """写入路径出生即归一（mock 数据源，不 mock LLM）"""

    @pytest.mark.critical
    def test_preprocess_normalizes_news_meta(self):
        # 文件与新闻两条入库路径都经 _preprocess_docs（ingest_router L757），
        # 一处接线覆盖两者：publish_time→publish_date + 旧 doc_type 映射
        from financial_rag.api.ingest_router import _preprocess_docs

        text = "商汤科技发布AI大模型，算力与GPU芯片投入持续加码，人工智能业务营收增长。" * 4
        docs = [{
            "text": text,
            "meta": {
                "source": "news:商汤科技发布AI大模型",
                "doc_type": "新闻报道",
                "title": "商汤科技发布AI大模型",
                "publish_time": "2026-06-07 15:02:52",
            },
        }]
        cleaned, stats = _preprocess_docs(docs)
        assert stats["cleaned"] == 1
        meta = cleaned[0]["meta"]
        assert meta["publish_date"] == "2026-06-07"
        assert meta["doc_type"] == "news"

    @pytest.mark.critical
    def test_save_analysis_to_kb_fields(self, monkeypatch):
        # 分析结论落库必须自带完整身份证（此前 title/doc_type/doc_id/date_added 全空）
        from financial_rag.api import kb_router

        saved = {}
        monkeypatch.setattr(kb_router, "_state", {"kb_docs": [], "kb_built": False})
        monkeypatch.setattr(kb_router, "_save_kb", lambda docs: saved.update(docs=docs))
        monkeypatch.setattr(kb_router, "_append_learning_record", lambda **kw: {})
        monkeypatch.setattr(kb_router, "_update_stats", lambda **kw: None)

        kb_router._save_analysis_to_kb(
            topic="Meta", assessment="neutral",
            analysis="Meta 筹资计划强化 AI 基建。", analysis_type="news",
            confidence="high",
        )
        doc = saved["docs"][0]
        meta = doc["meta"]
        assert meta["title"] == "Meta"
        assert meta["doc_type"] == "analysis"
        assert meta["source"] == "analysis:news:Meta"
        assert len(meta["doc_id"]) == 16
        assert len(meta["date_added"]) == 10  # YYYY-MM-DD