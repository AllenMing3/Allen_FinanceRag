"""
guard/llm_critique.py — L5 LLM质疑层

职责：审查 L1-L4 规则层的检测结果，找出规则系统的盲区。
- 假阳性：规则说"已锚定"但语义上不被来源支撑（token 重叠但意思变了）
- 假阴性：规则说"未锚定"但语义上实际被来源支持（换了表述但意思对）
- 语境歪曲：数字正确但描述歪曲了来源原意
- 推断越界：回答加入了来源中没有的因果推断或趋势判断

LLM 输入：answer + sources + L1-L4 检测结果摘要
LLM 输出：结构化 JSON（发现列表 + 严重程度 + 置信度）
"""
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# ===================== L5 核心 =====================

L5_SYSTEM_PROMPT = (
    "你是一位资深的金融事实核查审计员，专门审查 RAG（检索增强生成）系统的输出质量。"
    "你精通中文金融文本的语义分析，能够识别 token 重叠检测无法发现的深层语义问题。"
    "你的工作是找到规则系统的盲区，而非重复规则系统已发现的问题。"
    "只输出合法的 JSON，不要添加任何额外文字、解释或 Markdown 标记。"
)

L5_PROMPT_TEMPLATE = """<role>
你是事实核查审计员，负责审查基于规则的防幻觉检测结果。
规则系统使用 jieba 分词 + token 重叠率来判断回答是否被来源支撑，这种方法有以下已知盲区：
- 只看 token 重叠，不理解语义（"营收下降15%" 和 "营收增长15%" token 高度重叠但意思相反）
- 停用词过滤后丢失关键修饰词（"不""未""非" 被过滤，导致否定句被判为已锚定）
- 无法识别隐含推断（来源说"营收1738亿"，回答说"茅台是行业龙头"，规则无法判断这个推断是否有据）
</role>

<task>
审查以下回答和来源，以及规则层的检测结果，找出三类问题：
1. 假阳性（False Positive）：规则判定"已锚定"，但语义上实际不被支撑
2. 假阴性（False Negative）：规则判定"未锚定"，但语义上实际被支撑
3. 推断越界（Overreach）：回答中包含来源里没有的因果推断、趋势判断或定性评价
</task>

<answer>
{answer}
</answer>

<sources>
{sources}
</sources>

<rule_detection_results>
L1 来源锚定得分: {l1_score}
已锚定 {l1_anchored}/{l1_total} 句
L1 标记为"未锚定"的句子:
{l1_unanchored_list}

L2 数值一致得分: {l2_score}
L2 未匹配的数字: {l2_unmatched}

L3 引用完整得分: {l3_score}
L4 结构规范得分: {l4_score}
</rule_detection_results>

<analysis_dimensions>
请按以下维度逐一分析：

维度A — 假阳性审查
检查规则标记为"已锚定"的句子，找出以下情况：
- 语义反转：回答与来源意思相反（如来源说"下降"，回答说"增长"）
- 程度夸大：回答使用了比来源更强/更弱的措辞（如来源说"小幅增长"，回答说"大幅增长"）
- 主语替换：数字正确但主体不同（如来源说"行业营收"，回答说"公司营收"）
- 时间错位：数字正确但时间段不同（如来源说"2023年"，回答说"2024年"）

维度B — 假阴性审查
检查规则标记为"未锚定"的句子（已在上方列出），判断：
- 是否只是换了表述方式但核心信息一致（如同义词替换、句式变换）
- 是否是合理的概括或归纳（如来源给了多个数字，回答做了汇总）

维度C — 推断越界
检查回答中是否包含：
- 因果关系：如"因此""导致""由于"等，但来源没有这种因果链
- 趋势预测：如"预计将""未来有望"，但来源只给了历史数据
- 定性评价：如"表现优异""令人失望"，但来源只是中性数据
- 行业地位判断：如"龙头""第一"，但来源没有排名信息
</analysis_dimensions>

<severity_definition>
- high: 核心数据或结论被歪曲，可能导致读者做出错误判断
- medium: 措辞有偏差但不影响核心结论
- low: 细微差异，不影响整体可信度
</severity_definition>

<output_format>
{{
  "false_positives": [
    {{"claim": "被误判为已锚定的句子", "source_truth": "来源实际表达的意思", "severity": "high|medium|low", "reason": "具体问题说明"}}
  ],
  "false_negatives": [
    {{"claim": "被误判为未锚定的句子", "matching_source": "对应的来源片段", "reason": "为什么语义上实际被支撑"}}
  ],
  "overreach": [
    {{"claim": "推断越界的句子", "type": "causal|trend|qualitative|ranking", "severity": "high|medium|low", "reason": "为什么这是推断而非事实"}}
  ],
  "critique_summary": "一段话总结规则层的主要盲区和本轮发现的关键问题",
  "rule_blind_spot_count": "发现的规则盲区总数（整数）"
}}
</output_format>"""


def llm_critique(answer: str, sources: List[Dict], rule_results: Dict, llm) -> Dict:
    """L5 LLM质疑：审查规则层结果，找出假阳性/假阴性/推断越界"""
    from financial_rag.llm.caller import LLMCaller
    caller = LLMCaller(llm)

    # 提取规则层关键结果
    l1 = rule_results.get("L1_source_grounding", {})
    l2 = rule_results.get("L2_numerical_fidelity", {})
    l3 = rule_results.get("L3_citation_integrity", {})
    l4 = rule_results.get("L4_structure_compliance", {})
    unanchored = l1.get("unanchored", [])
    l2_unmatched = l2.get("unmatched", [])

    # 截断 sources 避免超长输入
    source_texts = []
    for i, s in enumerate(sources[:5]):
        text = s.get("text", "")[:500]
        source_texts.append(f"来源{i+1}: {text}")
    sources_joined = "\n".join(source_texts) if source_texts else "(无来源)"

    prompt = L5_PROMPT_TEMPLATE.format(
        answer=answer[:2000],
        sources=sources_joined,
        l1_score=f"{l1.get('score', 0):.0%}",
        l1_anchored=l1.get("anchored", 0),
        l1_total=l1.get("total", 0),
        l1_unanchored_list=(
            "\n".join(f"- {s}" for s in unanchored[:5]) if unanchored else "(无)"
        ),
        l2_score=f"{l2.get('score', 0):.0%}",
        l2_unmatched=", ".join(l2_unmatched[:5]) if l2_unmatched else "(无)",
        l3_score=f"{l3.get('score', 0):.0%}",
        l4_score=f"{l4.get('score', 0):.0%}",
    )

    try:
        data = caller.call_json(
            prompt,
            system=L5_SYSTEM_PROMPT,
            max_json_retries=1,
            use_cache=False,
        )
        if not isinstance(data, dict):
            return _fallback("LLM returned non-dict")

        fp = data.get("false_positives", [])
        fn = data.get("false_negatives", [])
        overreach = data.get("overreach", [])

        # 分数计算：加权扣分
        # high = -0.25, medium = -0.15, low = -0.05
        penalty = 0.0
        for item in fp:
            sev = item.get("severity", "medium") if isinstance(item, dict) else "medium"
            penalty += {"high": 0.25, "medium": 0.15, "low": 0.05}.get(sev, 0.15)
        for item in overreach:
            sev = item.get("severity", "medium") if isinstance(item, dict) else "medium"
            penalty += {"high": 0.25, "medium": 0.15, "low": 0.05}.get(sev, 0.15)
        # 假阴性是规则漏判，相对轻微
        penalty += len(fn) * 0.05

        score = max(0.0, 1.0 - penalty)

        # 格式化为简洁字符串列表（兼容 report 输出）
        fp_display = [
            (item.get("claim", str(item))[:80] if isinstance(item, dict) else str(item)[:80])
            for item in fp[:5]
        ]
        fn_display = [
            (item.get("claim", str(item))[:80] if isinstance(item, dict) else str(item)[:80])
            for item in fn[:5]
        ]
        overreach_display = [
            (item.get("claim", str(item))[:80] if isinstance(item, dict) else str(item)[:80])
            for item in overreach[:5]
        ]

        return {
            "score": score,
            "passed": score >= 0.5,
            "false_positives": fp_display,
            "false_negatives": fn_display,
            "overreach": overreach_display,
            "context_distortions": fp_display + overreach_display,  # 向后兼容
            "critique_summary": data.get("critique_summary", ""),
            "rule_blind_spot_count": data.get("rule_blind_spot_count", len(fp) + len(overreach)),
            "raw": data,  # 完整结果供调试
        }
    except Exception as e:
        logger.warning(f"L5 LLM critique failed: {e}")
        return _fallback(str(e))


def _fallback(reason: str) -> Dict:
    return {
        "score": 0.5,
        "passed": True,
        "false_positives": [],
        "false_negatives": [],
        "overreach": [],
        "context_distortions": [],
        "critique_summary": f"LLM质疑层未生效: {reason}",
        "rule_blind_spot_count": 0,
        "fallback": True,
    }
