"""
guard/rule_layers.py — 规则层 L1-L4 防幻觉校验

四个确定性校验层，无 LLM 依赖：
- L1 来源锚定: jieba 分词 + token 重叠率
- L2 数值一致: 正则提取数字，交叉比对来源
- L3 引用完整: [N] 标记是否指向有效来源
- L4 结构规范: 输出是否包含预期 Markdown 段落
"""
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ===================== 共享工具 =====================

GROUNDING_THRESHOLD = 0.15  # 来源锚定阈值：句子 token 与某 source 重叠 >= 此值视为"锚定"


_jieba_loaded = False


def _tokenize_zh(text: str) -> List[str]:
    """jieba 分词，无 jieba 时回退到按字分"""
    global _jieba_loaded
    try:
        import jieba
        if not _jieba_loaded:
            # 通过 DictionaryRegistry 注入领域词典（内置 + 外部 JSON）
            try:
                from financial_rag.retrievers.dictionary_registry import get_registry
                get_registry().set_jieba(jieba)
            except Exception:
                pass  # registry 不可用时静默跳过
            _jieba_loaded = True
        return [w for w in jieba.cut(text) if len(w.strip()) > 1]
    except ImportError:
        return [c for c in text if c.strip()]


# 常见停用词（功能词、标点、虚词）
_STOP_WORDS = frozenset(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 "
    "没有 看 好 自己 这 他 她 它 们 那 被 从 以 对 而 与 及 或 但 还 其 之 "
    "为 于 把 向 让 给 用 过 能 可 应 将 所 如 果 因 此 等 且 已 又 再 更 最".split()
)

# 数字+单位 正则（匹配 "50.3亿元"、"20%"、"2024年" 等）
_NUM_UNIT_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*([%％万亿元亿万美元美元美刀港元港币人民币块个台套条家万人人次份月年日季])?',
    re.UNICODE,
)


def extract_numbers(text: str) -> List[str]:
    """提取数字（含单位），返回标准化字符串列表"""
    matches = _NUM_UNIT_RE.findall(text)
    results = []
    for num_str, unit in matches:
        try:
            num = float(num_str)
            normalized = str(int(num)) if num == int(num) else num_str
        except ValueError:
            normalized = num_str
        results.append(f"{normalized}{unit}")
    return results


# ===================== L1: 来源锚定 =====================

def l1_source_grounding(answer: str, sources: List[Dict],
                        threshold: float = GROUNDING_THRESHOLD) -> Dict:
    """每个声明是否能追溯到某篇 source（jieba 分词 + token 重叠率）"""
    if not sources:
        return {"score": 0.0, "passed": False,
                "anchored": 0, "total": 0, "unanchored": []}

    # 预处理: 对每篇 source 做 jieba 分词
    source_tokens_list = []
    for s in sources:
        text = s.get("text", "")
        tokens = set(_tokenize_zh(text)) - _STOP_WORDS
        source_tokens_list.append(tokens)

    # 逐句检查
    sentences = [s.strip() for s in re.split(r'[。！？\n]', answer)
                 if len(s.strip()) > 8]
    if not sentences:
        return {"score": 1.0, "passed": True,
                "anchored": 0, "total": 0, "unanchored": []}

    anchored, unanchored = 0, []
    for sent in sentences:
        sent_tokens = set(_tokenize_zh(sent)) - _STOP_WORDS
        if len(sent_tokens) < 2:
            anchored += 1  # 太短的跳过
            continue

        # 找最佳匹配的 source
        best_overlap = 0.0
        for src_tokens in source_tokens_list:
            if not src_tokens:
                continue
            overlap = len(sent_tokens & src_tokens) / len(sent_tokens)
            best_overlap = max(best_overlap, overlap)

        if best_overlap >= threshold:
            anchored += 1
        else:
            unanchored.append(sent[:80])

    total = len(sentences)
    ratio = anchored / total if total > 0 else 1.0
    return {
        "score": ratio,
        "passed": ratio >= 0.6,
        "anchored": anchored,
        "total": total,
        "unanchored": unanchored,
    }


# ===================== L2: 数值一致性 =====================

def l2_numerical_fidelity(answer: str, sources: List[Dict]) -> Dict:
    """answer 中的数字是否能在 source 中找到"""
    if not sources:
        ans_nums = extract_numbers(answer)
        return {"score": 0.0 if ans_nums else 1.0,
                "passed": not bool(ans_nums),
                "verified": 0, "total": len(ans_nums), "unmatched": ans_nums}

    all_source_text = " ".join(s.get("text", "") for s in sources)
    src_nums = set(extract_numbers(all_source_text))
    ans_nums = extract_numbers(answer)

    if not ans_nums:
        return {"score": 1.0, "passed": True,
                "verified": 0, "total": 0, "unmatched": []}

    verified = sum(1 for n in ans_nums if n in src_nums)
    unmatched = [n for n in ans_nums if n not in src_nums]
    ratio = verified / len(ans_nums)
    return {
        "score": ratio,
        "passed": ratio >= 0.5,
        "verified": verified,
        "total": len(ans_nums),
        "unmatched": unmatched[:5],
    }


# ===================== L3: 引用完整性 =====================

def l3_citation_integrity(answer: str, sources: List[Dict], mode: str = "rag") -> Dict:
    """检查 [N] 引用标记是否存在且对应有效来源

    Args:
        mode: "rag" (RAG 查询，期望 [N] 引用) 或 "analysis" (深度分析，宽松引用)
    """
    citations = re.findall(r'\[(\d+)\]', answer)
    if not citations:
        if mode == "analysis":
            # 深度分析不强制 [N] 引用，检查文字引用或直接给高分
            has_text_ref = any(m in answer for m in
                               ["来源", "引用", "参考", "根据", "数据显示",
                                "据报", "资料", "报告", "新闻", "报道",
                                "分析", "评估", "显示", "表明", "认为"])
            score = 0.8 if has_text_ref else 0.6
            return {"score": score, "passed": True,
                    "citations_found": 0, "valid": 0, "invalid": 0,
                    "has_text_reference": has_text_ref, "mode": mode}

        has_text_ref = any(m in answer for m in
                           ["来源", "引用", "参考", "根据", "数据显示",
                            "据报", "资料", "报告"])
        score = 0.6 if has_text_ref else 0.2
        return {"score": score, "passed": score >= 0.5,
                "citations_found": 0, "valid": 0, "invalid": 0,
                "has_text_reference": has_text_ref}

    n_sources = len(sources)
    valid, invalid = 0, 0
    for c in citations:
        idx = int(c)
        if 1 <= idx <= n_sources:
            valid += 1
        else:
            invalid += 1

    total = valid + invalid
    ratio = valid / total if total > 0 else 0.0
    score = ratio * 0.9 + (0.1 if total >= 2 else 0.0)
    score = min(1.0, score)
    return {
        "score": score,
        "passed": score >= 0.5,
        "citations_found": total,
        "valid": valid,
        "invalid": invalid,
    }


# ===================== L4: 结构规范性 =====================

# RAG 查询期望的结构标记（Markdown 标题）
_EXPECTED_SECTIONS_RAG = [
    (r'(?:^|\n)#+\s*(?:摘要|概述|总结|Summary|Overview)', "摘要/概述"),
    (r'(?:^|\n)#+\s*(?:要点|关键|发现|Key|Finding)', "要点/发现"),
    (r'(?:^|\n)#+\s*(?:分析|Analysis|详情)', "分析/详情"),
    (r'(?:^|\n)#+\s*(?:风险|注意|提示|Risk|Warning|Caution)', "风险/提示"),
]

# 深度分析期望的结构标记（关键词匹配，不要求精确标题格式）
_EXPECTED_SECTIONS_ANALYSIS = [
    (r'(?:关键信号|核心信号|主要信号|重要信号|Key\s*Signal|signal)', "关键信号"),
    (r'(?:影响分析|多维影响|行业影响|市场影响|Impact|冲击|波及|利好|利空|受益|受损)', "影响分析"),
    (r'(?:风险提示|风险因素|不确定因素|注意事项|Risk|Warning|潜在风险|下行风险|风险点)', "风险提示"),
    (r'(?:后续关注|持续关注|投资建议|投资启示|趋势|展望|Watch|建议关注|下一步|值得关注)', "后续关注/趋势"),
]


def l4_structure_compliance(answer: str, mode: str = "rag") -> Dict:
    """输出是否包含预期的结构段落

    Args:
        mode: "rag" (RAG 查询，期望 # Markdown 标题) 或 "analysis" (深度分析，期望 【】括号段落)
    """
    sections = _EXPECTED_SECTIONS_ANALYSIS if mode == "analysis" else _EXPECTED_SECTIONS_RAG
    found = []
    missing = []
    for pattern, name in sections:
        if re.search(pattern, answer, re.IGNORECASE):
            found.append(name)
        else:
            missing.append(name)

    ratio = len(found) / len(sections)
    score = min(1.0, ratio + 0.2) if len(found) >= 2 else ratio
    return {
        "score": score,
        "passed": score >= 0.5,
        "found_sections": found,
        "missing_sections": missing,
        "mode": mode,
    }
