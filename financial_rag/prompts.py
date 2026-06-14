"""
AI 板块 RAG 专用 Prompt 模板

所有 prompt 聚焦 AI/科技行业，覆盖：
- 算力 (GPU集群、推理成本、训练规模)
- 模型 (参数量、benchmark、推理延迟)
- 商业 (API调用量、ARR、客户数)
- 融资 (估值、轮次、投资方)
- 技术 (架构、训练数据、上下文窗口)

LLM 调用统一使用 DashScopeLLM (qwen-plus)。
"""

# ===================== 元数据自动提取 =====================

METADATA_EXTRACTION_SYSTEM = """你是一位专业的 AI/科技行业文档元数据提取专家。你的任务是从给定文本中自动识别和提取结构化元数据。

规则：
1. 只提取文本中明确出现的信息，不要推测或编造
2. 日期统一为 YYYY-MM-DD 格式
3. 公司名称使用官方全称（如"商汤科技"而非"商汤"）
4. 无法确定字段留空字符串
5. 严格按照 JSON 格式输出，不要添加任何额外的解释文字"""

METADATA_EXTRACTION_PROMPT = """请从以下 AI/科技行业文本中提取元数据，以 JSON 格式返回：

文本内容：
{text}

需要提取的字段：
- source: 信息来源（如"公司公告"、"科技媒体"、"研究机构"、"交易所公告"等）
- company: 涉及的主要公司名称（全称），若无则留空
- date: 文本中提及的日期或发布日期（YYYY-MM-DD）
- fiscal_period: 财报/报告期间（如"2024年度"、"2024Q1"），非财报则留空
- currency: 货币单位（如"CNY"、"USD"），默认 CNY
- doc_type: 文档类型，从以下选择：
    年报、季报、公告、政策文件、新闻报道、研究报告、
    技术报告、产品发布、融资公告、研究论文、行业分析、其他

仅返回一个合法的 JSON 对象，不要添加任何其他文字。"""


# ===================== 财务/业务指标抽取 =====================

FINANCIAL_METRICS_EXTRACTION_SYSTEM = """你是一位专业的 AI/科技行业分析师，擅长从企业文档中精确抽取结构化业务指标。

核心原则：
1. 只提取文本中明确给出的数值，绝不编造或推测
2. 数字保留原始精度，包括单位（如"亿元"、"万次"、"PFLOPS"）
3. 同时提取数值和对应的增长率（如有）
4. AI 行业指标分四大类，全部关注：
   - 财务指标：营收、净利润、毛利率、研发投入、ARR
   - 算力指标：GPU数量、训练集群规模、推理成本/token、算力利用率
   - 模型指标：参数量、benchmark分数、推理延迟、上下文窗口长度
   - 商业指标：API调用量、客户数、客单价、DAU/MAU
5. 如果某项指标未在文本中出现，不要包含该字段
6. 严格按照 JSON 格式输出"""

FINANCIAL_METRICS_EXTRACTION_PROMPT = """请从以下文本中提取所有可识别的业务指标，以 JSON 格式返回：

文本内容：
{text}

需要关注的指标类别（如文本中存在则提取）：

【财务指标】
- revenue: 营业收入（含同比增长率）
- net_income: 净利润（含同比增长率）
- gross_margin: 毛利率
- rd_expense: 研发费用（含同比增长率）
- arr: 年度经常性收入 (Annual Recurring Revenue)

【算力指标】
- gpu_count: GPU/芯片数量
- training_cluster_size: 训练集群规模（如"万卡"）
- inference_cost_per_token: 推理成本（元/百万token）
- compute_utilization: 算力利用率

【模型指标】
- model_params: 模型参数量（如"70B"、"700亿"）
- context_window: 上下文窗口长度
- inference_latency: 推理延迟
- benchmark_score: benchmark 评测分数

【商业指标】
- api_calls: API 调用量（日均/月均）
- customer_count: 客户数量
- dau: 日活跃用户数
- mau: 月活跃用户数

对于每个指标，格式为：
{{
  "value": 数值(数字或字符串如"70B"),
  "unit": "单位(如亿元、%、万次、token、卡)",
  "yoy_growth": "同比增长率(字符串，如+15.66%或无)",
  "period": "所属期间"
}}

仅返回一个合法的 JSON 对象。"""


# ===================== 实体抽取 =====================

ENTITY_EXTRACTION_SYSTEM = """你是一位 AI/科技行业实体识别专家，擅长从科技新闻和行业报告中抽取关键实体和事件信息。

实体类别：
1. 公司/机构：AI 公司、芯片厂商、云服务商、研究机构、监管机构
2. 人物：创始人、CTO、首席科学家、投资人等
3. AI 产品与模型：大模型名称、AI 产品、开发框架、芯片型号
4. 技术术语：关键技术概念（如 MoE、RLHF、RAG、Agent 等）
5. 事件类型：产品发布、融资、并购、技术突破、政策监管、开源发布等
6. 行业/赛道：涉及的细分领域

规则：
- 只提取文本中明确提及的实体
- 每个实体附带其在文中的简要上下文
- 事件需标注影响范围（正面/负面/中性）
- 严格按照 JSON 格式输出"""

ENTITY_EXTRACTION_PROMPT = """请从以下 AI/科技文本中抽取结构化实体信息，以 JSON 格式返回：

文本内容：
{text}

需要抽取的实体类别：
1. "companies": 涉及的公司/机构列表，每项包含 name、role（角色）、industry（细分行业）
2. "persons": 关键人物列表，每项包含 name、title（职务）
3. "ai_models": AI 模型/产品列表，每项包含 name（模型名）、developer（开发方）、type（类型：大语言模型/多模态/图像生成/语音/...）
4. "tech_terms": 关键技术术语列表（如 MoE、RAG、Agent、RLHF 等）
5. "chips_hardware": 芯片/硬件列表，每项包含 name、vendor（厂商）、type（GPU/TPU/NPU/...）
6. "financial_figures": 关键数字列表，每项包含 label（含义）、value、unit
7. "event": 核心事件，包含 type（事件类型）、description（简要描述）、impact（正面/负面/中性）
8. "industries": 涉及的行业/赛道列表
9. "key_topics": 关键主题词列表（3-5个）

仅返回一个合法的 JSON 对象，不要添加任何其他文字。"""


# ===================== 财经新闻结构化抽取 =====================

FINANCIAL_NEWS_EXTRACTION_SYSTEM = """你是一位 AI/科技行业财经新闻分析专家。你的任务是将科技新闻文本转化为结构化的摘要分析。

分析维度：
1. 事件概述：一句话概括核心事件
2. 关键数据：提取文中所有关键数字和指标
3. 影响分析：对相关公司、行业、技术生态的短期和长期影响
4. 关联方：所有直接和间接涉及的公司/机构
5. 后续关注：需要持续跟踪的关键节点或指标

规则：严格基于文本内容，不做主观推测。以 JSON 格式输出。"""

FINANCIAL_NEWS_EXTRACTION_PROMPT = """请对以下 AI/科技新闻进行结构化分析，以 JSON 格式返回：

新闻文本：
{text}

请从以下维度分析：
1. "headline": 新闻标题（如文本无标题则自行概括）
2. "summary": 一句话事件概述（50字以内）
3. "event_type": 事件类型（产品发布/融资/并购/技术突破/政策监管/开源发布/人事变动/行业动态/其他）
4. "key_data": 关键数据列表 [{label, value, unit}]
5. "impact_analysis": {{
    "short_term": "短期影响描述",
    "long_term": "长期影响描述",
    "affected_sectors": ["受影响的行业/赛道列表"],
    "sentiment": "整体情感倾向(正面/负面/中性)"
  }}
6. "stakeholders": 利益相关方 [{name, role, impact_level(高/中/低)}]
7. "follow_up": ["后续值得关注的事项列表"]

仅返回一个合法的 JSON 对象，不要添加任何其他文字。"""


# ===================== 新闻综合分析 =====================

NEWS_SYNTHESIS_SYSTEM = """你是一位资深 AI/科技行业分析师。你的任务是将多条新闻源综合成一份结构化的分析报告。

分析要求：
1. 从多条新闻中识别关键发现（key findings），每条发现必须标注来源编号 [1][2] 等
2. 分析时间线上的趋势变化：技术演进、市场格局变化、资本动向
3. 评估市场情绪（sentiment）：正面/负面/中性，并说明理由
4. 识别矛盾信息：不同来源之间是否有冲突
5. 列出受影响的行业赛道和公司

输出格式要求：
- 返回严格的 JSON 格式
- 不要添加任何解释性文字，仅返回 JSON
- 每个 finding 必须包含 source_refs 字段（来源编号数组）

JSON 结构：
{
  "title": "报告标题（基于查询主题）",
  "key_findings": [
    {
      "finding": "发现内容",
      "importance": "high/medium/low",
      "source_refs": [1, 2]
    }
  ],
  "trend_analysis": "趋势分析（技术演进、市场反应、资本动向）",
  "sentiment": {
    "overall": "positive/negative/neutral/mixed",
    "reasoning": "理由说明"
  },
  "affected_sectors": ["赛道1", "赛道2"],
  "affected_companies": ["公司1", "公司2"],
  "contradictions": ["矛盾信息1", "矛盾信息2"],
  "summary": "整体总结（200字以内）"
}"""

NEWS_SYNTHESIS_PROMPT = """请基于以下 AI/科技新闻源综合分析，返回结构化 JSON 报告。

查询主题: {query}

=== 新闻源 ===
{sources}

=== 抽取的指标 ===
{metrics}

=== 抽取的实体 ===
{entities}

仅返回一个合法的 JSON 对象。"""


# ===================== Few-Shot 示例 =====================

FEW_SHOT_EXAMPLES = {
    "metrics_extraction": """
示例1 — AI 公司业务指标抽取：

输入文本：
"商汤科技2024年实现营业收入50.3亿元，同比增长36%；生成式AI业务收入占比达60%。
日日新大模型API日均调用量突破2000万次，同比增长400%。企业客户数达5800家。
研发投入42亿元，研发费用率83.5%。训练集群规模达4万卡A100。"

期望输出：
{
  "revenue": {"value": 50.3, "unit": "亿元", "yoy_growth": "+36%", "period": "2024年度"},
  "rd_expense": {"value": 42, "unit": "亿元", "yoy_growth": "无", "period": "2024年度"},
  "api_calls": {"value": 2000, "unit": "万次/日", "yoy_growth": "+400%", "period": "2024年度"},
  "customer_count": {"value": 5800, "unit": "家", "yoy_growth": "无", "period": "2024年度"},
  "gpu_count": {"value": 40000, "unit": "卡", "yoy_growth": "无", "period": "2024年度"},
  "training_cluster_size": {"value": 4, "unit": "万卡A100", "yoy_growth": "无", "period": "2024年度"}
}
""",

    "entity_extraction": """
示例2 — AI 行业实体抽取：

输入文本：
"英伟达发布新一代Blackwell B200 GPU，单卡AI训练性能较H100提升4倍。
微软Azure已部署超过10万张B200用于训练GPT-5。OpenAI CEO Sam Altman表示新架构将显著降低推理成本。
同期，谷歌宣布TPU v6将于Q4量产，直接对标Blackwell。"

期望输出：
{
  "companies": [
    {"name": "英伟达", "role": "芯片发布方", "industry": "AI芯片/半导体"},
    {"name": "微软", "role": "云服务/部署方", "industry": "云计算/AI"},
    {"name": "OpenAI", "role": "模型开发方", "industry": "AI大模型"},
    {"name": "谷歌", "role": "竞品发布方", "industry": "AI芯片/云计算"}
  ],
  "persons": [
    {"name": "Sam Altman", "title": "OpenAI CEO"}
  ],
  "ai_models": [
    {"name": "GPT-5", "developer": "OpenAI", "type": "大语言模型"}
  ],
  "chips_hardware": [
    {"name": "Blackwell B200", "vendor": "英伟达", "type": "GPU"},
    {"name": "H100", "vendor": "英伟达", "type": "GPU"},
    {"name": "TPU v6", "vendor": "谷歌", "type": "TPU"}
  ],
  "tech_terms": ["AI训练", "推理优化"],
  "financial_figures": [
    {"label": "B200训练性能提升", "value": 4, "unit": "倍"},
    {"label": "Azure部署B200数量", "value": 100000, "unit": "张"}
  ],
  "event": {
    "type": "产品发布",
    "description": "英伟达发布Blackwell B200 GPU，微软Azure大规模部署，谷歌TPU v6对标",
    "impact": "正面"
  },
  "industries": ["AI芯片", "云计算", "大模型"],
  "key_topics": ["GPU", "AI训练", "推理成本", "芯片竞争", "Blackwell"]
}
""",

    "metadata_extraction": """
示例3 — 元数据提取：

输入文本：
"智谱AI（Zhipu AI）于2025年3月宣布完成B+轮融资，估值超200亿元人民币。
本轮融资由社保基金领投，北京市政府引导基金跟投。GLM-5系列模型将于Q2发布。"

期望输出：
{
  "source": "公司公告",
  "company": "智谱AI",
  "date": "2025-03-01",
  "fiscal_period": "",
  "currency": "CNY",
  "doc_type": "融资公告"
}
"""
}
