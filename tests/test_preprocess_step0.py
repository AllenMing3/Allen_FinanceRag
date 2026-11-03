r"""
Test Step 0 数据清洗 — 入库前的文本预处理层（RAG 流水线第零步）。

Step 0 位于 Step 1 入库之前：原始文档 → 清洗 → 相关性门控 → 分类 → metadata 抽取。
对应生产代码：
  - financial_rag/retrievers/preprocessor.py（TextPreprocessor / RelevanceGate / DocTypeClassifier）
  - financial_rag/api/ingest_router.py::_preprocess_docs（端到端编排）


============================ 【目标】 ============================
Step 0 是 RAG 质量的第一道关。它的职责是把"原始脏数据"转成"干净、可检索、
不失真、不误杀"的入库文本。清洗层的任何静默失真都会在后续检索/生成里被放大
（脏数据进 → 脏答案出），且极难追溯到源头。核心保证有五条：
  G1 去噪      —— 移除不承载语义的噪声（HTML/URL/email/控制字符/样板行）
  G2 保真      —— 绝不破坏有信息价值的内容（数字/金额/股票代码/术语/语义）
  G3 门控准确  —— 过滤真正无关的，但不误杀有价值的
  G4 可观测    —— 清洗过程产出 CleanStats + warning，出问题能诊断
  G5 确定性    —— 同输入同输出、离线、免 API、可复现


========================= 【必守原则】 =========================
  P1 保真优先于去噪：金融数字/代码/术语是命脉。宁可残留一点噪声，
     也绝不能删掉或改写真实信息。（对应 G2，本文件最高优先级）
  P2 幂等性：clean(clean(x)) == clean(x)。重复清洗不得叠加损伤
     （pipeline 可能对同一文本多次清洗）。
  P3 可观测：每次清洗都能拿到 CleanStats；过度清洗(retention<0.3)/
     清空结果必须触发 warning，不允许"静默破坏"。
  P4 确定性：同输入连续多次结果逐字节一致，离线可复现。
  P5 不误杀优先：门控的假阴性（有价值文档被拒）比假阳性（垃圾漏过）
     更危险——前者是永久召回损失，后者还能靠后续检索排序补救。


======================= 【常见易错点矩阵】 =======================
  易错点                              | 风险           | 用例                                  | 状态
  ------------------------------------|----------------|---------------------------------------|--------
  连续≥5个相同数字被折叠(10万→1000)   | 金额失真       | test_fidelity_large_amount_not_collapsed | ✅已修复
  URL 的 \S+ 吞掉后续中文正文          | 丢正文         | test_url_removal_does_not_eat_following  | ✅已修复
  collapse_whitespace 吃掉中文段落的\n\n| 段落去重失效  | test_chinese_paragraph_dedup             | ✅已修复
  纯英文技术文被 low_chinese_ratio 误杀 | 门控假阴性     | test_gate_false_negative_english_tech    | ⚠已知局限
  单字关键词"电"子串命中"电影"          | 门控假阳性     | test_gate_false_positive_single_char_kw  | ⚠已知局限
  未闭合 HTML 标签(<div 无 >)不被移除   | 残留噪声       | test_html_unclosed_tag_left_as_is        | ✅特征化
  短正文行(前缀≤20字)含样板词被整行误删 | 正文误伤       | test_boilerplate_overmatches_short_line   | ⚠已知局限
  不传 collect_stats 时 get_last_stats 陈旧| 误读诊断    | test_stats_require_collect_flag          | ✅特征化
  NFKC 把全角标点转半角                 | 改变检索文本   | test_nfkc_fullwidth_to_halfwidth         | ✅特征化
  
  ✅已修复 = 曾为 @xfail 缺陷，修复后摘标记转常绿回归锁（strict 保证不回退）。
  ⚠    = 设计权衡/已知局限，用正向断言锁死当前行为，供未来产品决策。
  ✅特征化 = 不是期望行为，但锁定既成事实防无意变更。


按项目四条测试标准组织：①证明效果 ②可观测 ③一致性 ④回归锁定。
"""
import re

import pytest

from financial_rag.retrievers.preprocessor import (
    TextPreprocessor, RelevanceGate, DocTypeClassifier, CleanStats,
)
from financial_rag.api.ingest_router import _preprocess_docs


# ===================== Fixtures =====================

@pytest.fixture
def pp():
    return TextPreprocessor()


@pytest.fixture
def gate():
    return RelevanceGate()


@pytest.fixture
def clf():
    return DocTypeClassifier()


# 真实 AI 新闻（清洗后 >80 字，含领域词，通过门控 + 分类为 news）
REAL_AI_NEWS = (
    "英伟达正式发布Blackwell B200 GPU，单卡AI训练性能较上一代H100提升4倍，"
    "推理效率提升30倍。微软Azure已部署10万张B200用于训练GPT-5，"
    "OpenAI表示新一代模型推理成本将显著降低。"
)

# 真实财报（清洗后 >80 字，含多个财报词，分类为 financial_report）
REAL_FIN_REPORT = (
    "商汤科技2024年年报显示，公司实现营业收入50.3亿元，同比增长36.4%，"
    "净亏损同比收窄28.3%，研发投入占营收比例达37.2%，毛利率保持稳定，"
    "生成式AI业务收入占比首次超过六成成为核心增长引擎。"
)

# 无关生活文（清洗后 >80 字、纯中文、零领域关键词 → 门控以 no_domain_keywords 拒绝）
# 注意避开单字领域词（如 news 的"电"/"讯"），否则会被子串误命中而通过门控
IRRELEVANT_LIFE = (
    "今天天气很好我和妈妈一起去公园散步，吃了好吃的冰淇淋和蛋糕，"
    "还逛了商场买了衣服鞋子包包一大堆东西回来，真是开心的一天，"
    "下次放假还要再来这里玩，顺便叫上朋友一起爬山看风景拍照留念记录生活。"
)


# ============================================================
# A. 保真度（原则 P1：保真优先于去噪）— P0 最高优先级
# ============================================================

def test_fidelity_preserves_financial_numbers(pp):
    """① 证明效果：清洗后数字/百分比/股票代码全部存活，不被误删。"""
    raw = "商汤科技(00020.HK)2024营收50.3亿元，同比增长36.4%！净亏损收窄28.3%。"
    out = pp.process(raw)
    assert "50.3" in out
    assert "36.4%" in out
    assert "00020" in out
    assert "28.3%" in out


def test_fidelity_preserves_chinese_terms(pp):
    """① 证明效果：中文金融/AI 术语清洗后完整保留。"""
    raw = "生成式AI业务收入占比达60%，成为核心增长引擎，大模型推理成本显著下降。"
    out = pp.process(raw)
    for term in ("生成式", "大模型", "推理成本", "增长引擎"):
        assert term in out


def test_repeated_digit_boundary_not_collapsed(pp):
    """④ 回归锁定（保真边界）：≤4 位重复数字保留（(.)\\1{4,} 需 5 个才折叠）。

    与 test_fidelity_large_amount_not_collapsed 形成对照：4 个安全、5 个开始被吃。
    """
    out = pp.process("该股涨幅达5555%，成交额6666亿元，融资10000元。")
    assert "5555%" in out
    assert "6666" in out
    assert "10000元" in out


def test_fidelity_large_amount_not_collapsed(pp):
    """④ 回归锁定（已修复）：大额金额里的连续 0 不被当作"刷屏重复字符"折叠。

    旧缺陷：融资100000元 → 融资1000元（缩水 100 倍）。_REPEATED_CHARS 已排除数字。
    """
    out = pp.process("公司完成融资100000元，年度营收达100000000元。")
    assert "100000元" in out
    assert "100000000" in out


def test_repeated_chars_collapsed_when_excessive(pp):
    """① 证明效果：≥5 个连续重复"字符"（非数字语义）被折叠为 3 个，属正常去噪。"""
    out = pp.process("重要的事情说aaaaaa遍，他哈哈哈哈哈哈")
    assert "aaaaaa" not in out
    assert "哈哈哈" in out and "哈哈哈哈哈哈" not in out


def test_nfkc_fullwidth_to_halfwidth_characterization(pp):
    """✅ 特征化：NFKC 归一化把全角标点转半角（，→, ！→! ？→?）。

    既有行为，会改变进入 BM25/jieba 的文本，钉死防止无意变更。
    """
    out = pp.process("营收增长，利润上升！后市如何？")
    assert "," in out and "，" not in out
    assert "!" in out and "！" not in out
    assert "?" in out and "？" not in out


# ============================================================
# B. 去噪有效性（目标 G1）— P0
# ============================================================

def test_removes_html_tags(pp):
    out = pp.process('<div class="art">英伟达发布B200 GPU</div><p>性能提升4倍</p>')
    assert "<" not in out and ">" not in out
    assert "英伟达发布B200 GPU" in out


def test_removes_urls(pp):
    out = pp.process("详情见 https://example.com/news?a=1 及 www.nvidia.com 官网")
    assert "https://" not in out
    assert "www." not in out
    assert "详情见" in out


def test_url_removal_does_not_eat_following_chinese(pp):
    """④ 回归锁定（已修复）：删 URL 不连带吞掉其后紧跟的中文正文。

    旧缺陷：'详见https://example.com。后续报道' → '详见'（正文全丢）。
    _URL 已收窄为 RFC3986 字符集，遇中文/中文标点自动停止。
    """
    out = pp.process("详见https://example.com。后续报道指出营收增长")
    assert "后续报道" in out
    assert "营收增长" in out


def test_removes_emails(pp):
    out = pp.process("联系 press@nvidia.com 获取资料")
    assert "press@nvidia.com" not in out
    assert "联系" in out


def test_removes_control_and_zero_width_chars(pp):
    raw = "营收\x00增长\x1f50.3亿\u200b\u200c元\ufeff"
    out = pp.process(raw)
    for ch in ("\x00", "\x1f", "\u200b", "\u200c", "\ufeff"):
        assert ch not in out
    assert "50.3" in out


def test_removes_boilerplate_lines(pp):
    """① 证明效果：独立成行的版权/微信推广/免责声明被移除。"""
    raw = (
        "英伟达发布B200 GPU性能提升4倍。\n"
        "扫码关注微信公众号\n"
        "免责声明：本文不构成投资建议\n"
        "版权作者所有 转载必究"
    )
    out = pp.process(raw)
    assert "英伟达发布B200" in out
    assert "扫码关注微信公众号" not in out
    assert "免责声明" not in out
    assert "版权作者所有" not in out


def test_html_unclosed_tag_left_as_is(pp):
    """✅ 特征化：<div 未闭合（无 >）时不被移除——正则 <[^>]+> 要求闭合。

    残留噪声，但未闭合标签在真实数据里罕见，锁定当前行为。
    """
    out = pp.process("正文<div class=x 没有闭合")
    assert "<div class=x" in out


def test_boilerplate_overmatches_short_line(pp):
    """⚠ 已知局限（正文误伤）：样板正则是 ^.{0,20}(样板词).{0,20}$ 的整行匹配。

    当正文前缀 ≤20 字且行内含样板词（如"关注/微信/来源"）时，整行被当样板删除，
    连带吃掉真实正文；前缀 >20 字则整行保留（连样板也不删）。锁定这个边界行为，
    提示金融短标题/短句混排样板词时的误删风险。
    """
    # 短前缀(19字)：整行被删空（正文"营收50.3亿元"也丢了）
    short = "商汤科技营收50.3亿元增长36.4%。扫码关注微信公众号"
    assert pp.process(short) == ""
    # 长前缀(42字)：整行保留，样板词也不删
    long = "商汤科技2024年度财报显示公司实现营业收入50.3亿元同比增长36.4%毛利率稳定。扫码关注微信公众号"
    out = pp.process(long)
    assert "50.3亿元" in out
    assert "扫码关注微信公众号" in out


def test_clean_stats_counters_observable(pp):
    """② 可观测：CleanStats 逐项计数 html/urls/boilerplate 移除量。"""
    raw = (
        '<p>正文内容</p>\n'
        'https://a.com/x\n'
        '扫码关注微信公众号'
    )
    pp.process(raw, collect_stats=True)
    st = pp.get_last_stats()
    assert isinstance(st, CleanStats)
    assert st.html_removed >= 1
    assert st.urls_removed >= 1
    assert st.boilerplate_removed >= 1


# ============================================================
# C. 幂等性 & 确定性（原则 P2、P4）— P0
# ============================================================

def test_idempotent_cleaning(pp):
    """③ 一致性（原则 P2）：clean(clean(x)) == clean(x)，重复清洗不叠加损伤。"""
    raw = "<p>商汤科技营收50.3亿元</p> 详见 https://a.com 增长36.4%。"
    c1 = pp.process(raw)
    c2 = pp.process(c1)
    c3 = pp.process(c2)
    assert c1 == c2 == c3
    assert "50.3亿元" in c1  # 幂等清洗后关键信息仍在


def test_deterministic_output(pp):
    """③ 一致性（原则 P4）：同输入连续多次清洗结果逐字节一致，离线可复现。"""
    raw = '<p>英伟达发布B200</p> https://a.com 扫码关注微信公众号 营收50.3亿元'
    assert pp.process(raw) == pp.process(raw) == pp.process(raw)


# ============================================================
# D. 过度清洗可观测（原则 P3）— P1
# ============================================================

def test_over_cleaned_warning_triggered(pp):
    """② 可观测：retention < 0.3 触发 over_cleaned warning（不允许静默破坏）。"""
    raw = (
        "正文。\n"
        "扫码关注微信公众号获取更多资讯内容\n"
        "免责声明本文不构成任何投资建议据此操作风险自担\n"
        "版权作者所有未经授权不得转载必究"
    )
    pp.process(raw, collect_stats=True)
    st = pp.get_last_stats()
    assert st.retention < 0.3
    assert any("over_cleaned" in w for w in st.warnings)


def test_cleaning_emptied_warning_triggered(pp):
    """② 可观测：纯样板文本被清空 → cleaning_emptied_text warning。"""
    pp.process("扫码关注本公众号", collect_stats=True)
    st = pp.get_last_stats()
    assert st.cleaned_len == 0
    assert st.original_len > 0
    assert any("cleaning_emptied_text" in w for w in st.warnings)


def test_empty_input_returned_as_is(pp):
    """③ 一致性：空/None 输入原样返回，不报错。"""
    assert pp.process("") == ""
    assert pp.process(None) is None


def test_stats_require_collect_flag(pp):
    """✅ 特征化（API 陷阱）：不传 collect_stats=True 时 get_last_stats 不刷新，
    返回上一次的陈旧值。诊断时必须显式传 collect_stats=True。"""
    pp.process("<p>营收50.3亿元增长36.4%</p>", collect_stats=True)
    s1 = pp.get_last_stats()
    assert s1.html_removed >= 1
    pp.process("完全不同的一段文本内容啊啊")  # 不传 collect_stats
    assert pp.get_last_stats() is s1  # 未刷新，仍是上次对象


# ============================================================
# E. 段落去重（原则 P1）— P1
# ============================================================

def test_plain_paragraph_dedup_works(pp):
    """① 证明效果：无中文标点结尾的重复段能去重（\\n\\n 存活时机制有效）。"""
    raw = "para one about AI compute\n\npara one about AI compute\n\npara two about model"
    out = pp.process(raw, collect_stats=True)
    assert pp.get_last_stats().paragraphs_deduped >= 1
    assert out.count("para one about AI compute") == 1
    assert "para two about model" in out


def test_chinese_paragraph_dedup(pp):
    """④ 回归锁定（已修复）：中文标点结尾的重复段能被去重。

    旧缺陷：collapse_whitespace 先吃掉中文标点后的 \\n\\n，dedup 无法分段。
    _MULTI_CJK_SPACE 已收窄为 [ \\t]+，不再吞换行。
    """
    raw = "第一段讲AI算力。\n\n第一段讲AI算力。\n\n第二段讲大模型。"
    out = pp.process(raw, collect_stats=True)
    assert pp.get_last_stats().paragraphs_deduped >= 1
    assert out.count("第一段讲AI算力") == 1


# ============================================================
# F. 门控准确性（原则 P5：不误杀）— P2
# ============================================================

def test_gate_passes_real_ai_news(gate):
    """① 证明效果（真阳）：真实 AI 新闻通过门控。"""
    passed, reason, kw = gate.check(REAL_AI_NEWS)
    assert passed is True
    assert reason == "pass"
    assert kw >= 1


def test_gate_passes_financial_report(gate):
    passed, reason, kw = gate.check(REAL_FIN_REPORT)
    assert passed is True
    assert kw >= 1


def test_gate_rejects_english_garbage(gate):
    """④ 回归锁定（真阴）：纯英文乱码垃圾以 low_chinese_ratio 拒绝。"""
    passed, reason, kw = gate.check(
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor."
    )
    assert passed is False
    assert reason.startswith("low_chinese_ratio")


def test_gate_rejects_irrelevant_chinese(gate):
    """④ 回归锁定（真阴）：无关中文生活文以 no_domain_keywords 拒绝。"""
    passed, reason, kw = gate.check(IRRELEVANT_LIFE)
    assert passed is False
    assert reason.startswith("no_domain_keywords")
    assert kw == 0


def test_gate_rejects_empty(gate):
    passed, reason, kw = gate.check("")
    assert passed is False
    assert reason == "empty_text"


def test_gate_rejects_too_short(gate):
    passed, reason, kw = gate.check("股票涨")
    assert passed is False
    assert reason.startswith("too_short")


def test_gate_false_negative_english_tech(gate):
    """⚠ 已知局限（原则 P5 的反例）：纯英文技术文被 low_chinese_ratio 误杀。

    AI 领域大量有价值内容（论文/技术报告）是英文或中英混合，中文占比阈值会
    造成假阴性召回损失。锁定当前行为，供未来产品决策（是否放宽/加白名单）。
    """
    passed, reason, kw = gate.check(
        "We propose a MoE Transformer architecture with CUDA kernels, "
        "benchmarked on MMLU and HumanEval with 70B parameters."
    )
    assert passed is False
    assert reason.startswith("low_chinese_ratio")


def test_gate_false_positive_single_char_keyword(gate):
    """⚠ 已知局限：门控用子串匹配，单字关键词"电"命中"电影"→无关文误通过。

    锁定当前行为；提示字典里的单字词（电/讯）易造成假阳性。
    """
    passed, reason, kw = gate.check(
        "今天去电影院看了一部非常精彩好看的电影和家人朋友一起度过愉快周末假期"
    )
    assert passed is True  # 因"电"子串命中而误通过
    assert kw >= 1


# ============================================================
# G. 分类准确性（目标 G3）— P2
# ============================================================

def test_classify_news(clf):
    r = clf.classify(REAL_AI_NEWS)
    assert r["doc_type"] == "news"
    assert r["matched_keywords"]


def test_classify_financial_report(clf):
    r = clf.classify(REAL_FIN_REPORT)
    assert r["doc_type"] == "financial_report"
    assert r["confidence"] >= 0.66


def test_classify_query(clf):
    """疑问句式且无领域词竞争 → query（query 只追加 1 个 pattern，需无其它类别命中才胜出）。"""
    r = clf.classify("请问该如何理解这个现象呢")
    assert r["doc_type"] == "query"


def test_classify_other_fallback(clf):
    r = clf.classify("今天天气很好我们去公园散步吃东西")
    assert r["doc_type"] == "other"


def test_classify_empty_returns_other(clf):
    """③ 一致性：空输入返回 other + confidence 0.0。"""
    r = clf.classify("")
    assert r == {"doc_type": "other", "confidence": 0.0, "matched_keywords": []}


# ============================================================
# H. 端到端编排（_preprocess_docs）— E2E
# ============================================================

@pytest.mark.integration
def test_preprocess_docs_stats_accounting():
    """① 证明效果 + ② 可观测：混合输入的 stats 计数正确且守恒。"""
    raw_docs = [
        {"text": REAL_AI_NEWS, "meta": {"title": "英伟达B200", "source": "t"}},  # cleaned
        {"text": REAL_FIN_REPORT, "meta": {"title": "商汤年报", "source": "t"}},  # cleaned
        {"text": "", "meta": {}},                                                # rejected_empty
        {"text": "股票涨了", "meta": {}},                                         # rejected_short(<80)
        {"text": IRRELEVANT_LIFE, "meta": {}},                                   # rejected_irrelevant
    ]
    cleaned, stats = _preprocess_docs(raw_docs)
    assert stats["total"] == 5
    assert stats["cleaned"] == 2
    assert stats["rejected_empty"] == 1
    assert stats["rejected_short"] == 1
    assert stats["rejected_irrelevant"] == 1
    # 守恒：cleaned + 三类 rejected == total
    assert (stats["cleaned"] + stats["rejected_empty"] + stats["rejected_short"]
            + stats["rejected_irrelevant"]) == stats["total"]
    assert len(cleaned) == 2


@pytest.mark.integration
def test_preprocess_docs_enriches_metadata():
    """① 证明效果：通过清洗的 doc 注入 doc_type / text_length / date_added / 关键词数。"""
    cleaned, _ = _preprocess_docs([{"text": REAL_FIN_REPORT, "meta": {"title": "商汤年报"}}])
    assert len(cleaned) == 1
    meta = cleaned[0]["meta"]
    assert meta["doc_type"] == "financial_report"
    assert meta["text_length"] == len(cleaned[0]["text"])
    assert meta["_relevance_keywords"] >= 1
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", meta["date_added"])  # 必填 YYYY-MM-DD


@pytest.mark.integration
def test_preprocess_docs_cleans_text_in_place():
    """① 证明效果：端到端返回的 text 是清洗后的（HTML/URL 已去除，正文保留）。"""
    raw = "<div>" + REAL_AI_NEWS + "</div> 详见 https://example.com/news 报道"
    cleaned, stats = _preprocess_docs([{"text": raw, "meta": {"title": "x"}}])
    assert stats["cleaned"] == 1
    out = cleaned[0]["text"]
    assert "<div>" not in out
    assert "https://" not in out
    assert "英伟达正式发布" in out
