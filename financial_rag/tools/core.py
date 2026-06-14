"""
能力注册中心 — 集中管理所有 Agent 可通过 Function Calling 调起的能力

设计思路:
    每个能力是一个 FunctionDef（name + description + JSON Schema + callback），
    注册到 FunctionRegistry。
    LLM 根据用户意图通过 function calling 选择能力 → ToolExecutor 执行 → 结果回传。

用法:
    >>> from financial_rag.tools import FunctionRegistry, FunctionDef, ToolExecutor
    >>> registry = FunctionRegistry()
    >>> @registry.register()
    ... def search_data(query: str, top_k: int = 5) -> dict: ...
    >>> executor = ToolExecutor(registry)
    >>> results = executor.run("search_data", {"query": "茅台营收", "top_k": 3})
"""
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ===================== 数据类 =====================

@dataclass
class FunctionDef:
    """单个能力的完整定义 — name + description + JSON Schema + callback"""
    name: str                           # 函数名（LLM 选能力用的 key）
    description: str                    # 功能描述（给 LLM 看）
    parameters: Dict                    # JSON Schema 格式的 parameters
    callback: Callable                  # 实际执行的函数
    category: str = "general"           # 分类: retrieval | analysis | compute | data
    require_context: bool = False       # 是否需要检索上下文
    timeout_sec: float = 30.0           # 执行超时
    tags: List[str] = field(default_factory=list)

    def to_openai_schema(self) -> Dict:
        """转换为 OpenAI/DashScope 兼容的 tool schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallRequest:
    """LLM 返回的工具调用请求"""
    id: str                             # 调用 ID
    name: str                           # 函数名
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_llm(cls, tool_call: Dict) -> "ToolCallRequest":
        func = tool_call.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return cls(
            id=tool_call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        )


@dataclass
class ToolCallResult:
    """工具调用执行结果"""
    call_id: str
    name: str
    success: bool
    result: Any = None
    error: str = ""
    elapsed_ms: float = 0
    token_count: int = 0

    def to_llm_message(self) -> Dict:
        """转换为 tool 角色的消息，回传给 LLM"""
        content = json.dumps(self.result, ensure_ascii=False) if self.success else f"错误: {self.error}"
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": content,
        }


@dataclass
class ToolCallStats:
    """一次 Function Calling 会话的统计"""
    query: str
    calls: List[ToolCallResult] = field(default_factory=list)
    total_elapsed_ms: float = 0
    total_tokens: int = 0
    rounds: int = 0  # 多轮调用轮次

    @property
    def succeeded(self) -> int:
        return sum(1 for c in self.calls if c.success)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.calls if not c.success)

    @property
    def tools_used(self) -> List[str]:
        return list(set(c.name for c in self.calls))


# ===================== 注册中心 =====================

class FunctionRegistry:
    """能力注册中心 — 集中管理所有可调用的能力。

    支持两种注册方式:
    1. 装饰器: @registry.register() 装饰普通函数
    2. 显式注册: registry.add(FunctionDef(...))
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._functions: Dict[str, FunctionDef] = {}

    # ---------- 注册 ----------

    def register(
        self,
        name: str = None,
        description: str = None,
        parameters: Dict = None,
        *,
        category: str = "general",
        require_context: bool = False,
        timeout_sec: float = 30.0,
        tags: List[str] = None,
    ):
        """装饰器 — 自动从函数签名推断 parameters schema"""

        def decorator(fn: Callable):
            func_name = name or fn.__name__
            func_desc = description or (fn.__doc__ or "").strip().split("\n")[0]
            func_params = parameters or _infer_parameters(fn)

            self._functions[func_name] = FunctionDef(
                name=func_name,
                description=func_desc,
                parameters=func_params,
                callback=fn,
                category=category,
                require_context=require_context,
                timeout_sec=timeout_sec,
                tags=tags or [],
            )
            return fn

        return decorator

    def add(self, func_def: FunctionDef):
        """显式注册一个能力"""
        self._functions[func_def.name] = func_def

    def add_batch(self, func_defs: List[FunctionDef]):
        """批量注册"""
        for f in func_defs:
            self._functions[f.name] = f

    # ---------- 查询 ----------

    def get(self, name: str) -> Optional[FunctionDef]:
        return self._functions.get(name)

    def list_by_category(self, category: str = None) -> List[FunctionDef]:
        funcs = list(self._functions.values())
        if category:
            funcs = [f for f in funcs if f.category == category]
        return funcs

    def can_handle(self, name: str) -> bool:
        return name in self._functions

    def to_openai_schemas(self) -> List[Dict]:
        """导出为 DashScope/OpenAI 兼容的 tools 参数"""
        return [f.to_openai_schema() for f in self._functions.values()]

    @property
    def function_names(self) -> List[str]:
        return list(self._functions.keys())

    @property
    def functions(self) -> Dict[str, FunctionDef]:
        return dict(self._functions)

    def __len__(self) -> int:
        return len(self._functions)

    def __repr__(self) -> str:
        cats = {}
        for f in self._functions.values():
            cats.setdefault(f.category, []).append(f.name)
        lines = [f"FunctionRegistry('{self.name}') — {len(self._functions)} 个能力:"]
        for cat, names in sorted(cats.items()):
            lines.append(f"  [{cat}] {', '.join(names)}")
        return "\n".join(lines)


# ===================== 执行器 =====================

class ToolExecutor:
    """能力执行器 — 接收 LLM 的工具调用请求，执行并返回结果"""

    def __init__(self, registry: FunctionRegistry, max_workers: int = 8):
        self.registry = registry
        self.max_workers = max_workers

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行单个工具调用"""
        t0 = time.time()
        func_def = self.registry.get(request.name)

        if not func_def:
            return ToolCallResult(
                call_id=request.id, name=request.name, success=False,
                error=f"未知能力: {request.name}",
                elapsed_ms=(time.time() - t0) * 1000,
            )

        try:
            result = func_def.callback(**request.arguments)
            elapsed = (time.time() - t0) * 1000
            token_count = _estimate_tokens(str(result))
            return ToolCallResult(
                call_id=request.id, name=request.name, success=True,
                result=result, elapsed_ms=elapsed, token_count=token_count,
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.warning(f"工具执行失败 [{request.name}]: {e}")
            return ToolCallResult(
                call_id=request.id, name=request.name, success=False,
                error=str(e), elapsed_ms=elapsed,
            )

    def execute_batch(self, requests: List[ToolCallRequest]) -> List[ToolCallResult]:
        """并行执行多个工具调用（同一 phase 内可并行）"""
        if not requests:
            return []
        if len(requests) == 1:
            return [self.execute(requests[0])]

        results_map: Dict[str, ToolCallResult] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requests))) as pool:
            futures = {pool.submit(self.execute, r): r.id for r in requests}
            for f in as_completed(futures):
                result = f.result()
                results_map[futures[f]] = result
        return [results_map[r.id] for r in requests]

    def execute_and_report(
        self, requests: List[ToolCallRequest], scorecard=None
    ) -> List[ToolCallResult]:
        """执行并自动记录到打分卡"""
        results = self.execute_batch(requests)
        if scorecard:
            for r in results:
                stage_key = f"tool_{r.name}"
                score = 0.9 if r.success else 0.2
                scorecard.record(
                    stage_key,
                    f"[能力] {r.name}",
                    score,
                    elapsed_ms=r.elapsed_ms,
                    details={"success": r.success, "tokens": r.token_count,
                              "error": r.error if not r.success else ""},
                )
        return results


# ===================== 内置金融能力 =====================

# ---- 检索类 ----

def _make_search_tool():
    """创建一个带检索器引用的搜索工具（闭包注入 retriever）"""

    _retriever_ref = {"retriever": None}

    def set_retriever(retriever):
        _retriever_ref["retriever"] = retriever

    def search_financial_data(
        query: str,
        top_k: int = 5,
        source_filter: str = "",
    ) -> Dict:
        """在知识库中搜索金融数据。

        Args:
            query: 搜索查询，用自然语言描述需要的信息
            top_k: 返回结果数量，默认 5
            source_filter: 按来源过滤（如 'maotai_2024'），空表示不过滤
        """
        retriever = _retriever_ref["retriever"]
        if not retriever:
            return {"error": "检索器未初始化", "results": []}

        results = retriever.search(query, top_k=top_k)
        items = []
        for r in results:
            text = r.get("text", "")
            meta = r.get("meta", {})
            if source_filter and meta.get("source", "") != source_filter:
                continue
            items.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "score": round(r.get("score", 0), 4),
            })
        return {
            "query": query,
            "total_found": len(items),
            "results": items,
        }

    return search_financial_data, set_retriever


# ---- 分析类 ----

def calculate_growth_rate(
    current_value: float,
    previous_value: float,
    label: str = "指标",
) -> Dict:
    """计算同比增长率。

    Args:
        current_value: 当期数值
        previous_value: 上期数值
        label: 指标名称标签
    """
    if previous_value == 0:
        return {"label": label, "current": current_value, "previous": previous_value,
                "growth_rate": None, "error": "上期值为0，无法计算增长率"}
    rate = round((current_value - previous_value) / abs(previous_value) * 100, 2)
    direction = "增长" if rate > 0 else "下降" if rate < 0 else "持平"
    return {
        "label": label,
        "current": current_value,
        "previous": previous_value,
        "absolute_change": round(current_value - previous_value, 2),
        "growth_rate": rate,
        "direction": direction,
    }


def calculate_financial_ratio(
    metric_name: str,
    numerator: float,
    denominator: float,
    unit: str = "%",
) -> Dict:
    """计算通用财务比率。

    Args:
        metric_name: 指标名称，如 '毛利率', 'ROE', '资产负债率'
        numerator: 分子（如毛利）
        denominator: 分母（如营收）
        unit: 单位，默认百分号
    """
    if denominator == 0:
        return {"metric": metric_name, "value": None, "error": "分母为零"}
    value = round(numerator / denominator * (100 if unit == "%" else 1), 2)
    return {
        "metric": metric_name,
        "value": f"{value}{unit}",
        "numerator": numerator,
        "denominator": denominator,
    }


def compare_metrics(
    company_a: Dict,
    company_b: Dict,
    metrics: List[str],
) -> Dict:
    """横向对比两家公司的财务指标。

    Args:
        company_a: 公司A的数据，如 {"name": "茅台", "营收": 1738.52, "净利润": 862.28}
        company_b: 公司B的数据，如 {"name": "五粮液", "营收": 832.72, "净利润": 305.64}
        metrics: 要对比的指标列表，如 ["营收", "净利润"]
    """
    comparisons = []
    for m in metrics:
        va = company_a.get(m)
        vb = company_b.get(m)
        if va is not None and vb is not None and vb != 0:
            diff_pct = round((va - vb) / abs(vb) * 100, 2)
            comparisons.append({
                "metric": m,
                "company_a": va,
                "company_b": vb,
                "difference": round(va - vb, 2),
                "diff_percent": diff_pct,
            })
        else:
            comparisons.append({"metric": m, "error": "数据不足"})
    return {
        "company_a": company_a.get("name", "A"),
        "company_b": company_b.get("name", "B"),
        "comparisons": comparisons,
    }


def summarize_financials(
    metrics: Dict[str, float],
    company_name: str = "",
    period: str = "",
) -> str:
    """将多个财务指标汇总为一段自然语言描述。

    Args:
        metrics: 财务指标字典，如 {"营收_亿元": 1738.52, "净利润_亿元": 862.28, "营收增速_%": 15.66}
        company_name: 公司名称
        period: 报告期间
    """
    parts = []
    if company_name:
        parts.append(f"{company_name}")
    if period:
        parts.append(f"({period})")

    detail_parts = []
    for k, v in metrics.items():
        key_display = k.replace("_亿元", "").replace("_%", "")
        if "_亿元" in k:
            detail_parts.append(f"{key_display} {v:.2f} 亿元")
        elif "_%" in k:
            detail_parts.append(f"{key_display} {v:.2f}%")
        else:
            detail_parts.append(f"{key_display} {v}")

    return f"{' '.join(parts)}: {'，'.join(detail_parts)}。"


# ---- 分类映射 ----

CATEGORIES = {
    "retrieval": "检索类 — 从知识库获取数据",
    "analysis": "分析类 — 计算比率、对比、趋势分析",
    "compute": "计算类 — 通用数学/统计计算",
    "data": "数据类 — 实时数据获取（新闻、公告、行情等）",
}


# ===================== 工具函数 =====================

def _infer_parameters(fn: Callable) -> Dict:
    """从函数签名 + docstring 自动推断 parameters JSON Schema"""
    import inspect
    sig = inspect.signature(fn)
    props = {}
    required = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ptype = "string"
        if param.annotation is not inspect.Parameter.empty:
            anno = param.annotation
            if anno is int:
                ptype = "integer"
            elif anno is float:
                ptype = "number"
            elif anno is bool:
                ptype = "boolean"
            elif anno is list or str(anno).startswith("typing.List"):
                ptype = "array"
            elif anno is dict or str(anno).startswith("typing.Dict"):
                ptype = "object"

        prop_def = {"type": ptype}
        if param.default is not inspect.Parameter.empty:
            prop_def["default"] = param.default
        else:
            required.append(pname)
        props[pname] = prop_def

    return {
        "type": "object",
        "properties": props,
        "required": required,
    }


def _estimate_tokens(text: str) -> int:
    """粗略估计文本 token 数（中文约 1 字 1 token，英文约 4 字符 1 token）"""
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = len(text) - cn
    return cn + en // 4


# ===================== 注册中心工厂 =====================

def create_financial_registry(retriever=None, llm=None) -> FunctionRegistry:
    """创建预置金融能力的注册中心

    Args:
        retriever: 可选的 HybridRetriever 实例 (注入搜索工具)
        llm: 可选的 DashScopeLLM 实例 (注入抽取工具)
    """
    registry = FunctionRegistry(name="financial")

    # 注入抽取工具的 LLM
    from financial_rag.tools.extraction_tools import inject_extraction_llm, EXTRACTION_TOOLS
    if llm:
        inject_extraction_llm(llm)

    # 注入检索器
    search_fn, set_retriever = _make_search_tool()
    if retriever:
        set_retriever(retriever)

    # ---- 检索类 ----
    registry.add(FunctionDef(
        name="search_financial_data",
        description="在知识库中搜索金融数据，可用于查询公司财报、经济指标、行业数据等。"
                    "当用户问及具体数字、财务指标、公司数据时优先使用此能力。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言搜索查询"},
                "top_k": {"type": "integer", "description": "返回结果数量，默认 5", "default": 5},
                "source_filter": {"type": "string", "description": "按来源过滤，如 'maotai_2024'，空表示不过滤", "default": ""},
            },
            "required": ["query"],
        },
        callback=search_fn,
        category="retrieval",
        require_context=False,
        tags=["搜索", "检索", "数据"],
    ))

    # ---- 分析类 ----
    registry.add(FunctionDef(
        name="calculate_growth_rate",
        description="计算同比增长率。给定当期和上期数值，返回增长率百分比和变化方向。"
                    "用于计算营收增速、利润增速等指标。",
        parameters={
            "type": "object",
            "properties": {
                "current_value": {"type": "number", "description": "当期数值"},
                "previous_value": {"type": "number", "description": "上期数值"},
                "label": {"type": "string", "description": "指标名称", "default": "指标"},
            },
            "required": ["current_value", "previous_value"],
        },
        callback=calculate_growth_rate,
        category="analysis",
        tags=["计算", "增长率", "同比"],
    ))

    registry.add(FunctionDef(
        name="calculate_financial_ratio",
        description="计算通用财务比率，如毛利率、净利率、ROE、资产负债率等。",
        parameters={
            "type": "object",
            "properties": {
                "metric_name": {"type": "string", "description": "指标名称"},
                "numerator": {"type": "number", "description": "分子数值"},
                "denominator": {"type": "number", "description": "分母数值"},
                "unit": {"type": "string", "description": "单位，默认 %", "default": "%"},
            },
            "required": ["metric_name", "numerator", "denominator"],
        },
        callback=calculate_financial_ratio,
        category="analysis",
        tags=["比率", "财务指标", "计算"],
    ))

    registry.add(FunctionDef(
        name="compare_metrics",
        description="横向对比两家公司的财务指标，如营收、净利润等。",
        parameters={
            "type": "object",
            "properties": {
                "company_a": {"type": "object", "description": "公司A的指标数据"},
                "company_b": {"type": "object", "description": "公司B的指标数据"},
                "metrics": {"type": "array", "items": {"type": "string"},
                            "description": "要对比的指标名称列表"},
            },
            "required": ["company_a", "company_b", "metrics"],
        },
        callback=compare_metrics,
        category="analysis",
        tags=["对比", "横向分析"],
    ))

    registry.add(FunctionDef(
        name="summarize_financials",
        description="将多个财务指标汇总成一段自然语言描述文本。",
        parameters={
            "type": "object",
            "properties": {
                "metrics": {"type": "object", "description": "财务指标字典"},
                "company_name": {"type": "string", "description": "公司名称", "default": ""},
                "period": {"type": "string", "description": "报告期间", "default": ""},
            },
            "required": ["metrics"],
        },
        callback=summarize_financials,
        category="analysis",
        tags=["汇总", "描述", "报告"],
    ))

    # ---- 新闻类 ---- (国内免费 API: 同花顺/新浪/东方财富)
    from financial_rag.rss_fetcher import search_news as _rss_search, fetch_all_news as _rss_all

    def _fetch_stock_news(stock_code: str = "600519", max_news: int = 10) -> Dict:
        """获取个股新闻 (国内免费 API)"""
        result = _rss_search(keyword=stock_code, max_news=max_news)
        return {
            "query": f"个股新闻: {stock_code}",
            "total": result.get("total", 0),
            "items": result.get("items", []),
            "source": "domestic_api",
        }

    def _fetch_financial_news(keyword: str = "", max_news: int = 20) -> Dict:
        """搜索财经新闻 (国内免费 API)"""
        return _rss_search(keyword=keyword, max_news=max_news)

    def _fetch_announcements(stock_code: str = "600519", max_news: int = 20) -> Dict:
        """获取公司公告 (国内免费 API)"""
        result = _rss_search(keyword=stock_code, max_news=max_news)
        return {
            "query": f"公告: {stock_code}",
            "total": result.get("total", 0),
            "items": result.get("items", []),
            "source": "domestic_api",
        }

    registry.add(FunctionDef(
        name="fetch_stock_news",
        description="获取指定股票的近期新闻，涵盖公告、研报、媒体报道等。"
                    "当用户问'XX股票最近有什么新闻'或'XX公司最新动态'时使用。"
                    " [数据源: 同花顺/新浪/东方财富]",
        parameters={
            "type": "object",
            "properties": {
                "stock_code": {"type": "string",
                               "description": "股票代码，如 '600519'(茅台)、'000858'(五粮液)、'300750'(宁德时代)",
                               "default": "600519"},
                "max_news": {"type": "integer", "description": "最大返回条数", "default": 10},
            },
            "required": ["stock_code"],
        },
        callback=_fetch_stock_news,
        category="data",
        tags=["新闻", "个股", "公告"],
    ))

    registry.add(FunctionDef(
        name="fetch_financial_news",
        description="搜索财经新闻或获取最新财经快讯。可按关键词搜索（如'降准''新能源''茅台'），"
                    "空关键词返回最新财经电报。当用户问'最近有什么财经大事'或'搜索XX相关新闻'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string",
                            "description": "搜索关键词，如 '降准'、'茅台'、'新能源'。留空获取最新电报",
                            "default": ""},
                "max_news": {"type": "integer", "description": "最大返回条数", "default": 20},
            },
            "required": [],
        },
        callback=_fetch_financial_news,
        category="data",
        tags=["新闻", "快讯", "搜索", "财经"],
    ))

    registry.add(FunctionDef(
        name="fetch_announcements",
        description="获取上市公司公告（年报、季报、重大事项、分红方案等）。"
                    "当用户问'XX公司发了什么公告'或'XX公司财报'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "stock_code": {"type": "string",
                               "description": "股票代码，如 '600519'(茅台)",
                               "default": "600519"},
                "max_news": {"type": "integer", "description": "最大返回条数", "default": 20},
            },
            "required": ["stock_code"],
        },
        callback=_fetch_announcements,
        category="data",
        tags=["公告", "财报", "年报", "季报"],
    ))

    # ---- 抽取类 (ExtractionAgent / IngestionAgent 调用) ----
    for tool_def in EXTRACTION_TOOLS:
        registry.add(tool_def)

    # ---- 新闻报告类（高级封装，含保存 Markdown）----
    from financial_rag.tools.news_tools import NEWS_REPORT_TOOL
    registry.add(NEWS_REPORT_TOOL)

    # ---- ETF K线报告类（高级封装，含统计分析 + Markdown）----
    from financial_rag.tools.kline_tools import KLINE_REPORT_TOOL
    registry.add(KLINE_REPORT_TOOL)

    return registry


# ===================== 会话管理 =====================

class ToolCallSession:
    """
    Function Calling 会话管理器

    封装完整的 LLM function calling 循环:
    1. 发送用户消息 + tools → LLM 返回 tool_calls
    2. 执行 tool_calls → 结果作为 tool 消息回传
    3. LLM 综合结果生成最终回答
    """

    def __init__(
        self,
        llm,
        registry: FunctionRegistry,
        executor: ToolExecutor = None,
        system_prompt: str = "",
        max_rounds: int = 5,
        verbose: bool = False,
    ):
        self.llm = llm
        self.registry = registry
        self.executor = executor or ToolExecutor(registry)
        self.system_prompt = system_prompt or (
            "你是专业金融分析师。当需要具体数据时，请调用提供的工具函数获取。"
            "如果工具返回了数据，基于数据给出准确的分析结论。"
            "不要捏造任何具体数字，不确定的数据请说明。"
        )
        self.max_rounds = max_rounds
        self.verbose = verbose

    def run(self, query: str, scorecard=None) -> ToolCallStats:
        """
        执行一次完整的 function calling 会话

        Args:
            query: 用户问题
            scorecard: 可选，PipelineScoreCard 用于记录每个工具调用的评分

        Returns:
            ToolCallStats 包含所有工具调用和执行统计
        """
        stats = ToolCallStats(query=query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        tools = self.registry.to_openai_schemas()
        final_answer = ""

        t_session = time.time()

        for round_num in range(self.max_rounds):
            if self.verbose:
                logger.info(f"[Round {round_num + 1}] 发送 {len(tools)} 个 tools")

            # 调用 LLM
            t_call = time.time()
            resp = self._call_llm(messages, tools)
            call_elapsed = (time.time() - t_call) * 1000

            # LLM 没有 tool call → 最终回答
            if not resp.tool_calls:
                final_answer = resp.content
                if self.verbose:
                    logger.info(f"[Round {round_num + 1}] 无 tool call, 最终回答: {resp.content[:100]}...")
                break

            # 执行工具调用
            if self.verbose:
                names = [tc.get("function", {}).get("name", "?") for tc in resp.tool_calls]
                logger.info(f"[Round {round_num + 1}] LLM 选择了: {names}")

            requests = [ToolCallRequest.from_llm(tc) for tc in resp.tool_calls]
            results = self.executor.execute_and_report(requests, scorecard)
            stats.calls.extend(results)
            stats.rounds = round_num + 1

            # 构建下一轮消息
            messages.append({"role": "assistant", "content": resp.content or "",
                             "tool_calls": resp.tool_calls})
            for r in results:
                messages.append(r.to_llm_message())

        stats.total_elapsed_ms = (time.time() - t_session) * 1000
        stats.total_tokens = sum(r.token_count for r in stats.calls)

        if self.verbose:
            logger.info(
                f"会话完成: {stats.rounds} 轮, {stats.succeeded}/{len(stats.calls)} 个工具成功, "
                f"耗时 {stats.total_elapsed_ms:.0f}ms"
            )

        # 记录汇总到打分卡
        if scorecard:
            avg_score = sum(
                (0.9 if c.success else 0.2) for c in stats.calls
            ) / max(len(stats.calls), 1) if stats.calls else 0
            scorecard.record(
                "tool_session_summary",
                "Function Calling 汇总",
                avg_score,
                elapsed_ms=stats.total_elapsed_ms,
                details={
                    "rounds": stats.rounds,
                    "tools_used": stats.tools_used,
                    "succeeded": stats.succeeded,
                    "failed": stats.failed,
                },
            )

        return stats

    def _call_llm(self, messages: List[Dict], tools: List[Dict]):
        """调用 LLM 并解析 tool_calls"""
        try:
            resp = self.llm.chat_with_tools(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            return resp
        except AttributeError:
            # 回退：LLM 不支持 chat_with_tools，用普通 chat + 手动解析
            logger.warning("LLM 不支持 chat_with_tools，回退到普通 chat")
            tool_desc = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools
            )
            fallback_msg = (
                f"你可以调用以下工具:\n{tool_desc}\n\n"
                f"如果需要调用工具，请用 JSON 格式回复: "
                f'{{"tool": "函数名", "args": {{...}}}}\n'
                f"如果直接回答即可，正常回复。"
            )
            messages.append({"role": "system", "content": fallback_msg})
            resp = self.llm.chat(messages=messages)
            # 尝试解析 JSON tool call
            content = resp.content
            # 简单包装：如果检测到 JSON，手动执行
            resp.tool_calls = _try_parse_fallback_tool(content)
            return resp


def _try_parse_fallback_tool(text: str) -> List[Dict]:
    """尝试从文本中解析 JSON tool call（回退模式）"""
    import re
    # 查找 JSON 块: {"tool": "xxx", "args": {...}}
    match = re.search(r'\{[^{}]*"tool"\s*:\s*"(\w+)"[^{}]*"args"\s*:\s*(\{[^}]+\})[^{}]*\}', text)
    if match:
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            return [{
                "id": "fallback_0",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }]
        except json.JSONDecodeError:
            pass
    return []


# ===================== 便捷工厂 =====================

def create_tool_session(
    llm,
    retriever=None,
    registry: FunctionRegistry = None,
    executor: ToolExecutor = None,
    **kwargs,
) -> ToolCallSession:
    """快速创建 ToolCallSession"""
    if registry is None:
        registry = create_financial_registry(retriever=retriever, llm=llm)
    if executor is None:
        executor = ToolExecutor(registry)
    return ToolCallSession(llm=llm, registry=registry, executor=executor, **kwargs)
