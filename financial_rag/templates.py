"""
槽位模板定义 — 用槽位填充替代长文自由生成，降低首 token 延迟

设计理念:
- 把长报告拆成多个独立/半独立的短槽位
- 每个槽位 LLM 只需输出 20~80 tokens（vs 原来 300~800 tokens）
- 首 token 延迟从 2~5s 降至 0.3~0.8s
- 独立槽位可并行填充，串行槽位按依赖顺序执行

每个模板:
- slots: 槽位列表
- render: Jinja2 风格模板字符串
- phase: 槽位分组（同组可并行）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ===================== 槽位定义 =====================

@dataclass
class SlotDef:
    """单个槽位的定义"""
    key: str                      # 槽位键，如 "company_name"
    label: str                    # 中文名，如 "公司名称"
    prompt: str                   # 发送给 LLM 的填充指令
    default: str = ""             # LLM 失败时的回退值
    depends_on: List[str] = field(default_factory=list)  # 依赖的其他槽位 key
    max_tokens: int = 80          # 该槽位最大输出 tokens（越小越快）
    required: bool = False        # 是否必须填充


# ===================== 模板定义 =====================

@dataclass
class SlottedTemplate:
    """模板 = 槽位列表 + 渲染模板字符串"""
    name: str                     # 模板名称
    description: str              # 适用场景描述
    slots: List[SlotDef]          # 槽位列表
    phases: List[List[str]]       # 填充阶段（每个 phase 内的槽位可并行）
    render: str                   # 最终渲染模板 {slot_key} 会被替换

    def get_slot(self, key: str) -> Optional[SlotDef]:
        for s in self.slots:
            if s.key == key:
                return s
        return None

    def slot_keys(self) -> List[str]:
        return [s.key for s in self.slots]


# ===================== 预置模板 =====================

# -------- 模板 1: 财报核心摘要 --------

FINANCIAL_REPORT_SLOTS = [
    SlotDef("company_name", "公司名称",
        prompt="只输出公司全称，不要任何解释。例如：贵州茅台酒股份有限公司",
        max_tokens=30, required=True),
    SlotDef("fiscal_period", "报告期间",
        prompt="只输出财报所属期间，格式如 2024年年度/2024Q3。不要任何解释。",
        max_tokens=20, required=True),
    SlotDef("revenue_summary", "营收概况",
        prompt="用1句话概括营收情况（数值+同比增速）。最多50字。",
        max_tokens=60, required=True),
    SlotDef("profit_summary", "利润概况",
        prompt="用1句话概括净利润情况（数值+同比增速）。最多50字。",
        max_tokens=60, required=True),
    SlotDef("key_ratios", "核心指标",
        prompt="列出毛利率、净利率、ROE 三项核心指标及同比变动。用分号分隔。最多60字。",
        max_tokens=80),
    SlotDef("cashflow_brief", "现金流概要",
        prompt="用1句话概括经营活动现金流情况。最多40字。",
        max_tokens=50),
    SlotDef("segment_breakdown", "业务分部",
        prompt="列出主要产品或业务分部的营收占比。用分号分隔。最多60字。",
        max_tokens=80),
    SlotDef("major_changes", "重大变动",
        prompt="指出最值得关注的1-2个同比重大变动。最多60字。",
        max_tokens=80),
    SlotDef("risk_note", "风险提示",
        prompt="基于数据，指出1个最需要注意的风险点。最多40字。",
        max_tokens=60),
]

FINANCIAL_REPORT_PHASES = [
    # Phase 1: 并行 — 所有事实抽取相互独立
    ["company_name", "fiscal_period", "revenue_summary", "profit_summary",
     "key_ratios", "cashflow_brief", "segment_breakdown"],
    # Phase 2: 依赖事实 — 重大变动依赖营收/利润/指标
    ["major_changes"],
    # Phase 3: 依赖分析 — 风险基于所有前序
    ["risk_note"],
]

FINANCIAL_REPORT_RENDER = """## {company_name} 财报核心摘要 ({fiscal_period})

### 营收与利润
- **营收**: {revenue_summary}
- **利润**: {profit_summary}

### 核心指标
{key_ratios}

### 现金流
{cashflow_brief}

### 业务分部
{segment_breakdown}

### 重大变动
{major_changes}

### 风险提示
⚠ {risk_note}
"""

FINANCIAL_REPORT_TEMPLATE = SlottedTemplate(
    name="financial_report",
    description="财报核心摘要 — 快速获取营收/利润/指标/风险等关键信息",
    slots=FINANCIAL_REPORT_SLOTS,
    phases=FINANCIAL_REPORT_PHASES,
    render=FINANCIAL_REPORT_RENDER,
)

# -------- 模板 2: 经济新闻快读 --------

NEWS_BRIEF_SLOTS = [
    SlotDef("event_title", "事件标题",
        prompt="用一句话概括新闻事件，最多25字。",
        max_tokens=35, required=True),
    SlotDef("event_date", "发生时间",
        prompt="输出事件发生日期。格式如 2025-06-02。",
        max_tokens=15),
    SlotDef("key_players", "涉及主体",
        prompt="列出涉及的主体（公司/机构/人物）。用逗号分隔。最多40字。",
        max_tokens=50, required=True),
    SlotDef("impact_level", "影响级别",
        prompt="评估事件对市场的影响：重大/中等/轻微。仅输出一个词。",
        max_tokens=10, required=True),
    SlotDef("sector_impact", "行业影响",
        prompt="受影响的行业板块。用逗号分隔。最多40字。",
        max_tokens=50),
    SlotDef("market_reaction", "市场反应",
        prompt="描述市场即时反应（如涨跌幅、成交量变化）。最多40字。",
        max_tokens=50),
    SlotDef("outlook", "短期展望",
        prompt="1-2句话预测短期影响。最多50字。",
        max_tokens=70),
]

NEWS_BRIEF_PHASES = [
    ["event_title", "event_date", "key_players", "impact_level", "sector_impact"],
    ["market_reaction", "outlook"],
]

NEWS_BRIEF_RENDER = """## 📰 {event_title}
**时间**: {event_date} | **影响**: {impact_level}
**涉及**: {key_players}
**行业**: {sector_impact}
**市场**: {market_reaction}
**展望**: {outlook}
"""

NEWS_BRIEF_TEMPLATE = SlottedTemplate(
    name="news_brief",
    description="经济新闻快读 — 事件核心要素提取",
    slots=NEWS_BRIEF_SLOTS,
    phases=NEWS_BRIEF_PHASES,
    render=NEWS_BRIEF_RENDER,
)

# -------- 模板 3: 快速问答 --------

QUICK_QA_SLOTS = [
    SlotDef("direct_answer", "直接回答",
        prompt="用1-2句话直接回答用户的问题。要求：简洁、准确、只引用可靠信息。最多80字。",
        max_tokens=100, required=True),
    SlotDef("supporting_data", "支撑数据",
        prompt="提供1-2个支撑直接回答的具体数据点。没有则输出'无'。最多40字。",
        max_tokens=60),
    SlotDef("source_note", "数据来源",
        prompt="标注数据来源（如有）。格式：来源：xxx。最多30字。",
        max_tokens=40),
    SlotDef("confidence_note", "可信度",
        prompt="对回答可信度的简短判断：高/中/低，并说明原因（最多15字）。",
        max_tokens=30),
]

QUICK_QA_PHASES = [
    ["direct_answer", "supporting_data", "source_note", "confidence_note"],
]

QUICK_QA_RENDER = """### 回答
{direct_answer}

**数据支撑**: {supporting_data}
**来源**: {source_note}
**可信度**: {confidence_note}
"""

QUICK_QA_TEMPLATE = SlottedTemplate(
    name="quick_qa",
    description="快速问答 — 简洁直接的答案",
    slots=QUICK_QA_SLOTS,
    phases=QUICK_QA_PHASES,
    render=QUICK_QA_RENDER,
)

# -------- 模板 4: 深度分析（多 Agent 汇总用） --------

DEEP_ANALYSIS_SLOTS = [
    SlotDef("executive_summary", "摘要",
        prompt="用2句话总结分析结论，不包含数据细节。最多60字。",
        max_tokens=80, required=True),
    SlotDef("profitability", "盈利能力分析",
        prompt="分析盈利能力：毛利率/净利率/ROE水平及趋势。最多60字。",
        max_tokens=80),
    SlotDef("growth", "成长性分析",
        prompt="分析营收/利润增速趋势及驱动因素。最多60字。",
        max_tokens=80),
    SlotDef("health", "财务健康度",
        prompt="分析资产负债率/现金流/偿债能力。最多60字。",
        max_tokens=80),
    SlotDef("valuation_hint", "估值参考",
        prompt="给出市盈率/市净率等估值参考，判断偏贵/合理/低估。最多40字。",
        max_tokens=60),
    SlotDef("recommendation", "投资建议",
        prompt="基于以上分析给出投资建议：买入/持有/卖出 + 一句话理由。最多40字。",
        max_tokens=60),
]

DEEP_ANALYSIS_PHASES = [
    ["profitability", "growth", "health", "valuation_hint"],
    ["executive_summary"],
    ["recommendation"],
]

DEEP_ANALYSIS_RENDER = """## 深度分析报告

### 摘要
{executive_summary}

### 盈利能力
{profitability}

### 成长性
{growth}

### 财务健康度
{health}

### 估值参考
{valuation_hint}

### 投资建议
{recommendation}
"""

DEEP_ANALYSIS_TEMPLATE = SlottedTemplate(
    name="deep_analysis",
    description="深度分析 — 多维度财报分析",
    slots=DEEP_ANALYSIS_SLOTS,
    phases=DEEP_ANALYSIS_PHASES,
    render=DEEP_ANALYSIS_RENDER,
)


# ===================== 模板注册表 =====================

ALL_TEMPLATES: Dict[str, SlottedTemplate] = {
    "financial_report": FINANCIAL_REPORT_TEMPLATE,
    "news_brief": NEWS_BRIEF_TEMPLATE,
    "quick_qa": QUICK_QA_TEMPLATE,
    "deep_analysis": DEEP_ANALYSIS_TEMPLATE,
}


def get_template(name: str) -> Optional[SlottedTemplate]:
    """获取模板"""
    return ALL_TEMPLATES.get(name)


def list_templates() -> List[Dict]:
    """列出所有模板"""
    return [
        {"name": t.name, "description": t.description,
         "slots": len(t.slots), "phases": len(t.phases)}
        for t in ALL_TEMPLATES.values()
    ]
