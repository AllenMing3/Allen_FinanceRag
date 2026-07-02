"""
Extraction Tools — 信息抽取能力池

将原本硬编码在 Agent 中的抽取逻辑（正则+LLM）下沉为可注册的工具，
供 IngestionAgent / ExtractionAgent 通过 call_tool() 调用。

设计原则:
    1. LLM-first — 优先用 LLM 结构化输出，鲁棒且易扩展
    2. Regex-fallback — 仅对格式确定性高的字段做轻量正则兜底
    3. Confidence signaling — 每个结果标注置信度 (high/medium/low)

工具列表:
    - extract_financial_metrics: 从财报文本抽取营收/利润/毛利率等指标
    - extract_entities: 从文本抽取公司/人物/事件/金额等实体
    - extract_document_metadata: 从文本抽取元数据 (来源/公司/日期/期间/币种)
    - detect_document_type: 关键词判定文档类型 (年报/季报/公告/...)
    - generate_search_queries: 根据抽取结果生成多角度检索查询
"""
import logging
import re
from typing import Dict, Any, List, Optional

from financial_rag.tools.core import FunctionDef
from financial_rag.llm.caller import (
    LLMCaller,
    parse_json_from_text as _parse_json_from_text,
    parse_json_list_from_text as _parse_json_list_from_text,
)
from financial_rag.prompts import (
    FINANCIAL_METRICS_EXTRACTION_SYSTEM,
    FINANCIAL_METRICS_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_PROMPT,
    METADATA_EXTRACTION_SYSTEM,
    METADATA_EXTRACTION_PROMPT,
    FEW_SHOT_EXAMPLES,
)

logger = logging.getLogger(__name__)


# ===================== LLM 注入 (闭包模式) =====================

_llm_ref = {"llm": None}


def inject_extraction_llm(llm):
    """注入 LLM 实例 — 由 create_financial_registry() 调用"""
    _llm_ref["llm"] = llm


def _get_llm():
    return _llm_ref["llm"]


# ===================== 通用 JSON 解析 =====================
# 已移至 financial_rag.llm.caller，本模块通过 import 引用


# ===================== Tool 1: 财务指标抽取 =====================

def extract_financial_metrics(text: str) -> Dict[str, Any]:
    """从财报文本中抽取结构化财务指标（营收、利润、毛利率等）。

    Args:
        text: 财报/财经文本内容
    """
    if not text or not text.strip():
        return {"_confidence": "none", "_error": "empty input"}

    logger.info(f"[extract_financial_metrics] 入口: text={len(text)}字, llm={'yes' if _get_llm() else 'no'}")

    prompt_text = text[:8000]
    llm = _get_llm()

    # --- 主路径: LLM structured output ---
    if llm:
        try:
            system_prompt = FINANCIAL_METRICS_EXTRACTION_SYSTEM
            few_shot = FEW_SHOT_EXAMPLES.get("metrics_extraction", "")
            if few_shot:
                system_prompt += f"\n\n以下是一些示例供参考：\n{few_shot}"
            few_shot_bad = FEW_SHOT_EXAMPLES.get("metrics_extraction_bad", "")
            if few_shot_bad:
                system_prompt += f"\n\n以下是错误示范，请避免：\n{few_shot_bad}"

            user_prompt = FINANCIAL_METRICS_EXTRACTION_PROMPT.format(text=prompt_text)
            caller = LLMCaller(llm)
            result = caller.call_json(
                user_prompt,
                system=system_prompt,
                max_tokens=1024,
                temperature=0.0,
            )
            if result:
                normalized = _normalize_metric_keys(result)
                normalized["_confidence"] = "high"
                normalized["_source"] = "llm"
                logger.info(f"[extract_financial_metrics] LLM 返回: {len(normalized)} keys")
                return normalized
        except Exception as e:
            logger.warning(f"[extract_financial_metrics] LLM 失败: {e}")

    # --- 兜底: 轻量正则 (仅确定性高的字段) ---
    return _regex_fallback_metrics(text)


# ===================== 指标键名映射表 (模块级，供 _normalize_metric_keys 使用) =====================

_METRIC_KEY_MAP = {
    # === 财务指标 ===
    "revenue": ["revenue", "营业收入", "营收", "营业总收入", "总营收"],
    "net_income": ["net_income", "净利润", "归属净利润", "归母净利润"],
    "gross_margin": ["gross_margin", "毛利率", "综合毛利率"],
    "rd_expense": ["rd_expense", "研发费用", "研发投入", "研发支出"],
    "arr": ["arr", "年度经常性收入", "ARR", "Annual Recurring Revenue"],
    # === 算力指标 ===
    "gpu_count": ["gpu_count", "GPU数量", "GPU数", "芯片数量"],
    "training_cluster_size": ["training_cluster_size", "训练集群规模",
                               "训练集群", "集群规模"],
    "inference_cost_per_token": ["inference_cost_per_token", "推理成本",
                                  "每token成本", "推理单价"],
    "compute_utilization": ["compute_utilization", "算力利用率", "GPU利用率"],
    # === 模型指标 ===
    "model_params": ["model_params", "参数量", "模型参数", "模型规模"],
    "context_window": ["context_window", "上下文窗口", "上下文长度",
                       "context length"],
    "inference_latency": ["inference_latency", "推理延迟", "推理时延",
                          "响应延迟"],
    "benchmark_score": ["benchmark_score", "benchmark分数", "评测分数",
                        "benchmark"],
    # === 商业指标 ===
    "api_calls": ["api_calls", "API调用量", "API调用", "日均调用量",
                  "月均调用量"],
    "customer_count": ["customer_count", "客户数", "客户数量",
                       "企业客户数", "付费客户"],
    "dau": ["dau", "DAU", "日活跃用户", "日活跃用户数", "日活"],
    "mau": ["mau", "MAU", "月活跃用户", "月活跃用户数", "月活"],
}

# 预构建 all-aliases set，O(1) 查找未映射键
_ALL_METRIC_ALIASES = set()
for _aliases in _METRIC_KEY_MAP.values():
    _ALL_METRIC_ALIASES.update(_aliases)


def _normalize_metric_keys(raw_metrics: Dict) -> Dict[str, Any]:
    """将 LLM 返回的中文/混合键名标准化为英文键名 — AI 行业指标体系"""
    normalized = {}
    for eng_key, aliases in _METRIC_KEY_MAP.items():
        for alias in aliases:
            if alias in raw_metrics:
                normalized[eng_key] = raw_metrics[alias]
                break

    # 保留未映射的其他指标 (O(1) lookup via precomputed set)
    for k, v in raw_metrics.items():
        if k not in normalized and k not in _ALL_METRIC_ALIASES:
            normalized[k] = v

    return normalized


def _regex_fallback_metrics(text: str) -> Dict[str, Any]:
    """正则兜底 — 仅提取格式高度确定的通用数值（AI 行业指标格式多样，正则仅作最基础兜底）"""
    metrics = {}

    def _extract_value_unit(match_text: str):
        """从匹配文本中提取数值和单位"""
        num = re.search(r'([\d,.]+)\s*(亿|万|千|百)?\s*(元|美元|港元|次|卡)?', match_text)
        if num:
            val = float(num.group(1).replace(",", ""))
            unit = (num.group(2) or "") + (num.group(3) or "")
            return {"value": val, "unit": unit, "source": "regex", "_confidence": "low"}
        return None

    # 营收
    for pattern in [r'营业(?:总)?收入[约达为]?\s*[\d,.]+\s*亿?元?',
                    r'营收[约达为]?\s*[\d,.]+\s*亿?元?']:
        m = re.search(pattern, text)
        if m:
            parsed = _extract_value_unit(m.group())
            if parsed:
                metrics["revenue"] = parsed
                break

    # 研发费用
    m = re.search(r'研发(?:费用|投入|支出)[约达为]?\s*[\d,.]+\s*亿?元?', text)
    if m:
        parsed = _extract_value_unit(m.group())
        if parsed:
            metrics["rd_expense"] = parsed

    # GPU/集群规模
    m = re.search(r'(?:GPU|芯片|训练集群)[^，。]{0,10}?([\d,.]+)\s*(万)?卡', text)
    if m:
        val = float(m.group(1).replace(",", ""))
        if m.group(2):
            val *= 10000
        metrics["gpu_count"] = {
            "value": val, "unit": "卡",
            "source": "regex", "_confidence": "low",
        }

    # API 调用量
    m = re.search(r'(?:API|接口)[^，。]{0,10}?(?:调用量|日均|月均)[^，。]{0,5}?([\d,.]+)\s*(万|亿)?次', text)
    if m:
        val = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "") + "次"
        metrics["api_calls"] = {
            "value": val, "unit": unit,
            "source": "regex", "_confidence": "low",
        }

    metrics["_confidence"] = "low" if metrics else "none"
    metrics["_source"] = "regex"
    return metrics


# ===================== Tool 2: 实体抽取 =====================

def extract_entities(text: str) -> Dict[str, Any]:
    """从财经文本中抽取公司/人物/事件/金额等实体。

    Args:
        text: 财经文本内容
    """
    if not text or not text.strip():
        return {"_confidence": "none", "_error": "empty input"}

    logger.info(f"[extract_entities] 入口: text={len(text)}字, llm={'yes' if _get_llm() else 'no'}")

    prompt_text = text[:8000]
    llm = _get_llm()

    # --- 主路径: LLM ---
    if llm:
        try:
            system_prompt = ENTITY_EXTRACTION_SYSTEM
            few_shot = FEW_SHOT_EXAMPLES.get("entity_extraction", "")
            if few_shot:
                system_prompt += f"\n\n以下是一些示例供参考：\n{few_shot}"
            few_shot_bad = FEW_SHOT_EXAMPLES.get("entity_extraction_bad", "")
            if few_shot_bad:
                system_prompt += f"\n\n以下是错误示范，请避免：\n{few_shot_bad}"

            user_prompt = ENTITY_EXTRACTION_PROMPT.format(text=prompt_text)
            caller = LLMCaller(llm)
            result = caller.call_json(
                user_prompt,
                system=system_prompt,
                max_tokens=1024,
                temperature=0.0,
            )
            if result:
                result["_confidence"] = "high"
                result["_source"] = "llm"
                logger.info(f"[extract_entities] LLM 返回: {len(result)} keys ({list(result.keys())})")
                return result
        except Exception as e:
            logger.warning(f"[extract_entities] LLM 失败: {e}")

    # --- 兜底: 仅公司名 + 金额 (最可靠的两个 pattern) ---
    logger.info("[extract_entities] 走 regex 兆底")
    return _regex_fallback_entities(text)


def _regex_fallback_entities(text: str) -> Dict[str, Any]:
    """正则兜底实体抽取"""
    entities = {}

    # 公司名
    companies = []
    company_pattern = r'([\u4e00-\u9fa5]{2,6}(?:集团|公司|控股|股份|科技|银行|证券|基金|保险|信托|租赁)(?:有限公司|股份有限公司)?)'
    seen = set()
    for m in re.finditer(company_pattern, text):
        name = m.group(1)
        if name not in seen and len(name) >= 4:
            seen.add(name)
            companies.append({"name": name, "role": "涉及方"})
    if companies:
        entities["companies"] = companies

    # 金额
    figures = []
    amount_pattern = r'([\d,.]+)\s*(亿|万|千|百)?\s*(元|美元|港元|人民币)'
    for m in re.finditer(amount_pattern, text):
        figures.append({
            "label": "金额",
            "value": float(m.group(1).replace(",", "")),
            "unit": (m.group(2) or "") + (m.group(3) or "元"),
        })
    if figures:
        entities["financial_figures"] = figures

    entities["_confidence"] = "low" if (companies or figures) else "none"
    entities["_source"] = "regex"
    return entities


# ===================== Tool 3: 文档元数据抽取 =====================

def extract_document_metadata(text: str) -> Dict[str, str]:
    """从文档文本中提取结构化元数据（来源、公司、日期、期间、币种）。

    Args:
        text: 文档文本内容
    """
    if not text or not text.strip():
        return {"_confidence": "none", "_error": "empty input"}

    prompt_text = text[:8000]
    llm = _get_llm()

    # --- 主路径: LLM ---
    if llm:
        try:
            system_prompt = METADATA_EXTRACTION_SYSTEM
            few_shot = FEW_SHOT_EXAMPLES.get("metadata_extraction", "")
            if few_shot:
                system_prompt += f"\n\n以下是一些示例供参考：\n{few_shot}"
            few_shot_bad = FEW_SHOT_EXAMPLES.get("metadata_extraction_bad", "")
            if few_shot_bad:
                system_prompt += f"\n\n以下是错误示范，请避免：\n{few_shot_bad}"

            user_prompt = METADATA_EXTRACTION_PROMPT.format(text=prompt_text)
            caller = LLMCaller(llm)
            result = caller.call_json(
                user_prompt,
                system=system_prompt,
                max_tokens=512,
                temperature=0.0,
            )
            if result:
                result["_confidence"] = "high"
                result["_source"] = "llm"
                return result
        except Exception as e:
            logger.warning(f"[extract_document_metadata] LLM 失败: {e}")

    # --- 兜底: 正则 (日期 + 期间 + 币种 — 格式确定性高) ---
    return _regex_fallback_metadata(text)


def _regex_fallback_metadata(text: str) -> Dict[str, str]:
    """正则兜底元数据抽取 — 仅提取格式确定性高的字段"""
    metadata = {
        "source": "",
        "company": "",
        "date": "",
        "fiscal_period": "",
        "currency": "CNY",
        "doc_type": "",
    }

    # 日期 (2 patterns)
    date_patterns = [
        (r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日',
         lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
         lambda m: f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
    ]
    for pattern, fmt in date_patterns:
        m = re.search(pattern, text)
        if m:
            try:
                metadata["date"] = fmt(m)
            except (ValueError, IndexError):
                pass
            break

    # 财报期间 (2 patterns)
    period_patterns = [
        (r'(\d{4})\s*年\s*年度报告', lambda g: f"{g[0]}年度"),
        (r'(\d{4})\s*年\s*第?\s*(\d)\s*季度?报告', lambda g: f"{g[0]}Q{g[1]}"),
        (r'(\d{4})\s*年\s*半年报', lambda g: f"{g[0]}H1"),
        (r'(\d{4})\s*年年报', lambda g: f"{g[0]}年度"),
    ]
    for pattern, fmt in period_patterns:
        m = re.search(pattern, text)
        if m:
            metadata["fiscal_period"] = fmt(m.groups())
            break

    # 币种
    if re.search(r'美元|USD|亿美元|百万美元', text):
        metadata["currency"] = "USD"
    elif re.search(r'港元|HKD|亿港元', text):
        metadata["currency"] = "HKD"

    metadata["_confidence"] = "low"
    metadata["_source"] = "regex"
    return metadata


# ===================== Tool 4: 文档类型检测 =====================

# 文档类型关键词集合 (模块级预构建，避免每次调用重建)
_DOC_TECH = frozenset([
    "arXiv", "论文", "技术报告", "benchmark", "评测",
    "消融实验", "ablation", "消融", "数据集", "训练策略",
    "模型架构", "transformer", "attention", "MoE",
])
_DOC_PRODUCT = frozenset([
    "正式发布", "重磅发布", "全新上线", "全新发布",
    "新版本", "V2", "V3", "2.0", "3.0",
    "API开放", "开放调用", "公测", "内测",
])
_DOC_FUNDING = frozenset([
    "融资", "轮", "领投", "跟投", "估值",
    "A轮", "B轮", "C轮", "Pre-IPO", "天使轮",
    "投资方", "投后估值",
])
_DOC_INDUSTRY = frozenset([
    "行业报告", "市场分析", "赛道", "竞争格局",
    "市场规模", "渗透率", "行业趋势", "白皮书",
])
_DOC_FINANCIAL = frozenset([
    "营业收入", "净利润", "每股收益", "毛利率", "ROE",
    "经营活动现金流", "资产负债表", "利润表", "现金流量表",
    "归属于上市公司股东", "基本每股收益",
])
_DOC_POLICY_SRC = frozenset([
    "中国人民银行", "央行", "证监会", "银保监会", "财政部",
    "发改委", "工信部", "科技部", "网信办",
])
_DOC_POLICY_KW = frozenset([
    "下调", "上调", "利率", "监管", "行政处罚",
    "暂行办法", "通知", "决定", "规范",
])
_DOC_RESEARCH = frozenset([
    "评级", "目标价", "买入", "卖出", "增持", "减持",
    "研报", "研究报告", "盈利预测",
])
_DOC_NEWS = frozenset([
    "报道", "记者", "据悉", "消息人士",
    "分析人士", "最新消息", "快讯", "独家",
])

# 合并所有关键词，编译为单个正则做一次全文扫描
_ALL_DOC_KEYWORDS = (
    _DOC_TECH | _DOC_PRODUCT | _DOC_FUNDING | _DOC_INDUSTRY
    | _DOC_FINANCIAL | _DOC_POLICY_SRC | _DOC_POLICY_KW
    | _DOC_RESEARCH | _DOC_NEWS
)
_DOC_TYPE_RE = re.compile("|".join(re.escape(kw) for kw in sorted(_ALL_DOC_KEYWORDS, key=len, reverse=True)))

def detect_document_type(text: str) -> str:
    """基于关键词检测文档类型（覆盖传统财报 + AI 行业文档）。

    Args:
        text: 文档文本内容
    """
    if not text or not text.strip():
        return "其他"
    
    # 用预编译正则做多模式一次扫描，替代 10 次关键词循环
    _hits = _DOC_TYPE_RE.findall(text)
    hit_set = set(_hits)
    
    def _count(kw_set):
        return len(hit_set & kw_set)
    
    # === AI 行业特有文档类型 ===
    if _count(_DOC_TECH) >= 2:
        return "技术报告"
    if _count(_DOC_PRODUCT) >= 2:
        return "产品发布"
    if _count(_DOC_FUNDING) >= 2:
        return "融资公告"
    if _count(_DOC_INDUSTRY) >= 2:
        return "行业分析"
    
    # === 传统财报类型 ===
    if _count(_DOC_FINANCIAL) >= 3:
        return "年报"
    if re.search(r'第?\s*[一二三1-3]\s*季度?报告|Q[1-3]\s*报告', text):
        return "季报"
    if _count(_DOC_POLICY_SRC) >= 1:
        return "政策文件"
    if _count(_DOC_POLICY_KW) >= 2:
        return "政策文件"
    if re.search(r'(公告|通知|声明|决定)\s*(编号|第|如下)', text):
        return "公告"
    if _count(_DOC_RESEARCH) >= 2:
        return "研究报告"
    
    # 新闻特征 (兆底)
    if _count(_DOC_NEWS) >= 1 or len(text) > 100:
        return "新闻报道"

    return "其他"


# ===================== Tool 5: 搜索查询生成 =====================

_QUERY_GEN_SYSTEM = """你是一个专业的 AI/科技行业检索查询生成器。
根据给定的文档内容和已抽取的结构化信息，生成 3-5 个多样化的搜索查询。

规则：
1. 查询应覆盖不同分析角度（如技术能力、商业进展、算力布局、竞争格局、融资估值）
2. 查询应是自然语言，适合用于知识库搜索
3. 只输出 JSON 数组格式，不要添加任何解释"""

_QUERY_GEN_PROMPT = """请根据以下信息生成搜索查询：

文档内容摘要：
{text_summary}

已抽取的财务指标：
{metrics_summary}

已抽取的实体：
{entities_summary}

请输出一个 JSON 数组，包含 3-5 个查询字符串，例如：
["查询1", "查询2", "查询3"]"""


def generate_search_queries(
    text: str,
    metrics: Optional[Dict] = None,
    entities: Optional[Dict] = None,
) -> List[str]:
    """根据文档内容和抽取结果生成多角度检索查询。

    Args:
        text: 文档文本
        metrics: 已抽取的财务指标 (extract_financial_metrics 的输出)
        entities: 已抽取的实体 (extract_entities 的输出)
    """
    metrics = metrics or {}
    entities = entities or {}

    # 纯模板生成 — LLM 调用已移除（产出查询在当前流程中未用于实际检索）
    return _fallback_queries(metrics, entities)


def _fallback_queries(metrics: Dict, entities: Dict) -> List[str]:
    """兜底查询生成 — AI 行业导向"""
    queries = []

    # 基于指标
    for key, label in [("revenue", "营业收入"), ("gpu_count", "GPU算力"),
                       ("api_calls", "API调用量"), ("model_params", "模型参数"),
                       ("rd_expense", "研发投入")]:
        if key in metrics:
            queries.append(f"{label} 分析")
            break

    # 基于公司
    companies = entities.get("companies", [])
    if companies and isinstance(companies[0], dict):
        name = companies[0].get("name", "")
        if name:
            queries.append(f"{name} 业务进展")

    # 保底
    if not queries:
        queries = ["AI业务关键指标分析", "核心技术能力摘要"]

    return queries[:3]


# ===================== FunctionDef 列表 =====================

EXTRACTION_TOOLS: List[FunctionDef] = [
    FunctionDef(
        name="extract_financial_metrics",
        description="从 AI/科技行业文本中抽取结构化业务指标（营收、研发投入、算力、模型参数、API调用量等）。"
                    "输入原始文本，返回包含各指标的 value/unit/yoy_growth 的 JSON 对象。"
                    "当需要从文档中提取具体业务数字时使用。",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "财报或财经文本内容（支持长文本，自动截取前 8000 字符）",
                },
            },
            "required": ["text"],
        },
        callback=extract_financial_metrics,
        category="extraction",
        tags=["财务指标", "抽取", "结构化", "LLM"],
    ),
    FunctionDef(
        name="extract_entities",
        description="从 AI/科技文本中抽取实体信息（公司、人物、AI模型、芯片、技术术语、事件等）。"
                    "返回包含 companies/persons/ai_models/chips_hardware/tech_terms 等字段的 JSON 对象。"
                    "当需要识别文档中涉及的关键参与者和技术要素时使用。",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "财经文本内容",
                },
            },
            "required": ["text"],
        },
        callback=extract_entities,
        category="extraction",
        tags=["实体", "公司", "事件", "抽取"],
    ),
    FunctionDef(
        name="extract_document_metadata",
        description="从文档文本中提取结构化元数据（来源、公司、日期、财报期间、币种、文档类型）。"
                    "返回 JSON 对象，日期统一为 YYYY-MM-DD 格式。"
                    "当需要识别文档的基本属性时使用。",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "文档文本内容",
                },
            },
            "required": ["text"],
        },
        callback=extract_document_metadata,
        category="extraction",
        tags=["元数据", "日期", "公司", "来源"],
    ),
    FunctionDef(
        name="detect_document_type",
        description="基于关键词检测文档类型。返回以下类型之一："
                    "年报、季报、公告、政策文件、新闻报道、研究报告、"
                    "技术报告、产品发布、融资公告、行业分析、其他。"
                    "纯关键词规则，无需 LLM，速度快。",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "文档文本内容",
                },
            },
            "required": ["text"],
        },
        callback=detect_document_type,
        category="extraction",
        tags=["文档类型", "分类", "关键词"],
    ),
    FunctionDef(
        name="generate_search_queries",
        description="根据文档内容和已抽取的结构化信息，生成 3-5 个多样化的搜索查询。"
                    "查询覆盖不同分析角度（增长趋势、盈利能力、行业对比等），"
                    "适合用于知识库检索。",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "文档文本内容",
                },
                "metrics": {
                    "type": "object",
                    "description": "已抽取的财务指标 (extract_financial_metrics 的输出)",
                    "default": {},
                },
                "entities": {
                    "type": "object",
                    "description": "已抽取的实体 (extract_entities 的输出)",
                    "default": {},
                },
            },
            "required": ["text"],
        },
        callback=generate_search_queries,
        category="extraction",
        tags=["查询生成", "搜索", "检索"],
    ),
]
