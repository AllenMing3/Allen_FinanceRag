"""
EmbeddingCache._hash 撞键缺陷回归锁 — P1-1。

缺陷：_hash 曾取 text[:500] 算 MD5，前 500 字相同的两个 chunk 得到同一 key，
后者静默复用前者向量（金融文档开头高度模板化，高频触发）。
修复：全文 MD5。本文件钉死全文敏感性，全部纯函数、离线确定。
"""
import pytest

from financial_rag.retrievers.embedding_cache import EmbeddingCache


@pytest.mark.unit
class TestHashFullText:
    def test_shared_prefix_different_tail_different_key(self):
        """bug 复现：前 500 字相同、尾部不同的两文本必须得到不同 key。"""
        prefix = "寒武纪科技股份有限公司2024年年度报告。" * 30  # >500 字模板化开头
        assert len(prefix) > 500
        text_a = prefix + "营业收入11.74亿元，同比增长65.56%。"
        text_b = prefix + "营业收入50.30亿元，同比增长36.40%。"
        assert EmbeddingCache._hash(text_a) != EmbeddingCache._hash(text_b)

    def test_identical_text_same_key(self):
        text = "商汤科技2024年营收50.3亿元，同比增长36.4%。"
        assert EmbeddingCache._hash(text) == EmbeddingCache._hash(text)

    def test_short_text_stable(self):
        """短文本（<500 字）hash 稳定且与全文 MD5 一致（旧缓存条目仍有效）。"""
        import hashlib
        text = "日日新大模型API日均调用量突破2000万次。"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert EmbeddingCache._hash(text) == expected

    def test_empty_text(self):
        assert isinstance(EmbeddingCache._hash(""), str)
        assert len(EmbeddingCache._hash("")) == 32

    def test_long_text_tail_sensitive(self):
        """超长文本仅末尾一字之差 → key 必须不同。"""
        base = "算力基础设施投资持续加码。" * 200  # ~2600 字
        assert EmbeddingCache._hash(base + "甲") != EmbeddingCache._hash(base + "乙")