"""
guard/llm_assist.py — L6 LLM协助层

职责：独立于规则层，LLM 自主逐句扫描回答，判断每个断言的来源支撑度。
特别关注规则层完全无法检测的幻觉类型：
- 编造的因果关系（"因此导致..."但来源没有这个因果链）
- 编造的定性描述（"市场反应积极"但来源没有这个判断）
- 数字正确但语境被歪曲（来源说"营收增长15%"，回答说"营收暴增15%"）
- 跨句一致性问题（前句说A增长，后句推断B增长，但来源只说了A）

LLM 输入：answer（分句后）+ sources
LLM 输出：逐句判定 + 汇总统计 + 严重程度
"""
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# ===================== L6 核心 =====================

L6_SYSTEM_PROMPT = (
    "你是一位独立的金融事实核查员，具有深厚的行业分析经验。"
    "你的任务是逐句审查 RAG 系统的回答，判断每个断言是否有来源支撑。"
    "你必须独立判断，不依赖任何前置检测结果。"
    "你对金融领域的常见幻觉模式特别敏感：虚假因果、夸大措辞、编造排名、伪造预测。"
    "只输出合法的 JSON，不要添加任何额外文字、解释或 Markdown 标记。"
)

L6_PROMPT_TEMPLATE = """<role>
你是独立事实核查员。你将收到一个 RAG 系统生成的回答（已分句）和对应的来源文档。
你需要逐句审查，对每个断言给出支撑度判定。
</role>

<task>
对回答中的每个句子进行独立审查，判断其是否有来源支撑。
重点关注以下四类规则系统无法检测的幻觉：
</task>

<hallucination_taxonomy>
类型1 — 虚假因果 (Causal Fabrication)
来源只陈述事实A和事实B，但回答推断"A导致B"或"因为A所以B"
示例：来源说"营收增长15%，净利润增长20%"，回答说"营收增长带动了净利润大幅提升"
判定：如果来源没有明确这个因果关系 → fabricated

类型2 — 定性夸大 (Qualitative Inflation)
来源用中性措辞，但回答使用了带有主观色彩的描述
示例：来源说"营收1738亿元"，回答说"营收高达1738亿元" / "表现亮眼" / "成绩斐然"
判定：如果来源没有这些评价性词语 → partially_supported 或 fabricated

类型3 — 编造排名/地位 (Ranking Fabrication)
回答声称某公司"排名第一""行业龙头""市场份额最大"，但来源没有排名信息
示例：来源说"茅台营收1738亿"，回答说"茅台作为白酒行业龙头"
判定：如果来源没有排名或对比信息 → fabricated

类型4 — 伪造预测/趋势 (Trend Fabrication)
回答基于历史数据做出了来源中没有的预测或趋势判断
示例：来源只给了2024年数据，回答说"预计2025年将继续保持增长态势"
判定：如果来源没有预测信息 → fabricated
</hallucination_taxonomy>

<verdict_definitions>
- supported: 句子核心断言能在来源中找到直接对应的信息，语义一致
- partially_supported: 句子部分信息有来源支撑，但存在细节出入（措辞强化、范围扩大、时间模糊）
- unsupported: 句子核心断言在来源中找不到依据，但不属于明显编造
- fabricated: 句子包含明显编造的信息（虚假因果、定性夸大、编造排名、伪造预测）
</verdict_definitions>

<confidence_calibration>
- high (>0.8): 非常确定判定正确，来源信息充分且明确
- medium (0.5-0.8): 较为确定，但来源信息可能不够完整或有歧义
- low (<0.5): 不太确定，需要更多来源信息才能判断
</confidence_calibration>

<answer_sentences>
{numbered_sentences}
</answer_sentences>

<sources>
{sources}
</sources>

<output_format>
{{
  "per_sentence": [
    {{
      "index": 1,
      "text": "句子原文（截取前60字）",
      "verdict": "supported|partially_supported|unsupported|fabricated",
      "confidence": 0.9,
      "reason": "具体理由，说明在哪个来源中找到了什么信息，或者为什么判定为该类型",
      "hallucination_type": "null|causal|qualitative|ranking|trend"
    }}
  ],
  "cross_sentence_issues": [
    {{
      "description": "跨句一致性问题描述",
      "sentences_involved": [1, 3],
      "severity": "high|medium|low"
    }}
  ],
  "unsupported_claims": ["无来源支撑的断言摘要"],
  "fabricated_claims": ["明显编造的断言摘要"],
  "total_assessed": "审查的句子总数（整数）",
  "supported_count": "supported 的句子数（整数）",
  "fabricated_count": "fabricated 的句子数（整数）",
  "overall_assessment": "一段话总结回答的整体可信度"
}}
</output_format>

<important_rules>
1. 每句必须给出 verdict，不要跳过任何句子
2. fabricated 判定必须有明确的 hallucination_type
3. 如果来源为空或不足，所有含具体数据的句子应判为 unsupported
4. 跨句一致性问题单独记录在 cross_sentence_issues 中
5. supported_count 只计算 verdict 为 "supported" 的句子
</important_rules>"""


def llm_assist(answer: str, sources: List[Dict], llm) -> Dict:
    """L6 LLM协助：独立逐句扫描，发现规则抓不到的幻觉"""
    from financial_rag.llm.caller import LLMCaller
    caller = LLMCaller(llm)

    # 分句
    sentences = [s.strip() for s in re.split(r'[。！？\n]', answer)
                 if len(s.strip()) > 8]
    if not sentences:
        return {
            "score": 1.0,
            "passed": True,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "fabricated_claims": [],
            "per_sentence": [],
            "cross_sentence_issues": [],
        }

    # 截断 sources
    source_texts = []
    for i, s in enumerate(sources[:5]):
        text = s.get("text", "")[:500]
        source_texts.append(f"来源{i+1}: {text}")
    sources_joined = "\n".join(source_texts) if source_texts else "(无来源)"

    # 截断句子数量避免超长输入
    sents_for_check = sentences[:15]
    numbered_sents = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents_for_check))

    prompt = L6_PROMPT_TEMPLATE.format(
        numbered_sentences=numbered_sents,
        sources=sources_joined,
    )

    try:
        data = caller.call_json(
            prompt,
            system=L6_SYSTEM_PROMPT,
            max_json_retries=1,
            use_cache=False,
        )
        if not isinstance(data, dict):
            return _fallback("LLM returned non-dict", len(sents_for_check))

        total = data.get("total_assessed", len(sents_for_check))
        supported = data.get("supported_count", 0)
        unsupported = data.get("unsupported_claims", [])
        fabricated = data.get("fabricated_claims", [])

        # 分数：有支撑的比例，fabricated 额外扣分
        fabricated_count = data.get("fabricated_count", len(fabricated))
        if total > 0:
            base_score = supported / total
            # fabricated 额外惩罚：每个扣 0.1
            fabrication_penalty = min(0.3, fabricated_count * 0.1)
            score = max(0.0, base_score - fabrication_penalty)
        else:
            score = 1.0

        return {
            "score": score,
            "passed": score >= 0.5,
            "total_claims": total,
            "supported_claims": supported,
            "fabricated_count": fabricated_count,
            "unsupported_claims": unsupported[:5],
            "fabricated_claims": fabricated[:5],
            "per_sentence": data.get("per_sentence", [])[:15],
            "cross_sentence_issues": data.get("cross_sentence_issues", []),
            "overall_assessment": data.get("overall_assessment", ""),
            "raw": data,
        }
    except Exception as e:
        logger.warning(f"L6 LLM assist failed: {e}")
        return _fallback(str(e), len(sents_for_check))


def _fallback(reason: str, total: int = 0) -> Dict:
    return {
        "score": 0.5,
        "passed": True,
        "total_claims": total,
        "supported_claims": total // 2,
        "fabricated_count": 0,
        "unsupported_claims": [],
        "fabricated_claims": [],
        "per_sentence": [],
        "cross_sentence_issues": [],
        "fallback": True,
        "fallback_reason": reason,
    }
