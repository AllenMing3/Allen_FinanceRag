"""
AgentRouter — 查询意图分类 + Agent 链路由

根据用户查询自动识别意图，选择最优 Agent 执行链。

意图类型:
- kline: K 线技术分析（走势、MACD、RSI、支撑压力位）
- event_impact: 事件影响分析（利好利空、事件冲击、影响因子）
- report: 财报/深度分析（营收、利润、财报数据）
- news: 新闻综合（行业动态、新闻汇总）
- general: 通用查询（走默认 Pipeline）

用法:
    from financial_rag.core.agent_router import AgentRouter, create_agent_router
    router = create_agent_router()
    decision = router.route("茅台最近走势怎么样")
    # → intent='kline', chain=['KLineAgent', 'ReportAgent']
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ===================== 数据结构 =====================


@dataclass
class QueryIntent:
    """查询意图分类结果"""
    name: str               # 意图名称
    confidence: float       # 置信度 0-1
    matched_keywords: List[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """路由决策"""
    intent: str             # 意图名称
    agent_chain: List[str]  # Agent 执行链
    confidence: float       # 路由置信度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 提取的元数据 (date, stock_code 等)

    def __str__(self) -> str:
        chain_str = " → ".join(self.agent_chain)
        return f"[{self.intent}] ({self.confidence:.0%}) {chain_str}"


# ===================== 意图模式定义 =====================


_INTENT_PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "kline",
        "keywords": [
            "K线", "k线", "走势", "技术分析", "MACD", "RSI", "均线",
            "支撑位", "压力位", "趋势", "布林", "KDJ", "kdj",
            "涨跌", "收盘", "开盘", "成交量", "放量", "缩量",
            "多头", "空头", "震荡", "突破", "跌破",
            "日K", "周K", "月K", "5日线", "10日线", "20日线",
        ],
        "patterns": [
            r"(\d{6})\s*(的)?(走势|行情|K线|技术)",
            r"(茅台|五粮液|宁德时代|比亚迪|招商银行|中国平安|腾讯|阿里巴巴|沪深300|中证500|创业板)\s*(的)?(走势|行情|K线|技术|趋势|涨|跌)",
        ],
        "chain": ["AnalysisAgent", "ScoringAgent"],
    },
    {
        "name": "event_impact",
        "keywords": [
            "利好", "利空", "事件影响", "影响因子", "冲击",
            "发生了什么", "大事", "突发事件", "政策影响",
            "并购", "收购", "重组", "增减持", "回购",
            "暴涨", "暴跌", "涨停", "跌停",
        ],
        "patterns": [
            r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?\s*(发生|出|有什么|新闻|事件|大事)",
            r"(发生|出了|有什么)\s*(了)?(什么)?(大事|事件|新闻|情况)",
            r"(利好|利空)\s*(消息|消息面|影响)",
        ],
        "chain": ["AnalysisAgent", "ScoringAgent"],
    },
    {
        "name": "report",
        "keywords": [
            "财报", "营收", "净利润", "毛利率", "同比", "环比",
            "季度", "年报", "半年报", "利润", "亏损",
            "资产负债表", "现金流", "EPS", "ROE", "市盈率",
        ],
        "patterns": [
            r"(财报|年报|季报|半年报)",
            r"(营收|利润|毛利|净利)\s*(多少|增长|下降|变化)",
        ],
        "chain": ["IngestionAgent", "AnalysisAgent", "ScoringAgent"],
    },
    {
        "name": "news",
        "keywords": [
            "新闻", "最新消息", "行业动态", "要闻", "资讯",
            "报道", "公告", "通知", "政策", "监管",
        ],
        "patterns": [
            r"(最新|今天|本周|本月)\s*(的)?(新闻|消息|动态|资讯|要闻)",
        ],
        "chain": ["IngestionAgent", "AnalysisAgent", "ScoringAgent"],
    },
]

# 通用查询默认链
_DEFAULT_CHAIN = ["IngestionAgent", "AnalysisAgent", "ScoringAgent"]


# ===================== AgentRouter =====================


class AgentRouter:
    """
    查询意图分类 + Agent 链路由

    根据用户查询的关键词和模式匹配，自动识别意图并选择最优 Agent 执行链。

    特性:
    - 多意图优先级匹配（按定义顺序）
    - 支持运行时动态注册新意图
    - 提取查询中的元数据（日期、股票代码）传入路由决策
    - 置信度评分帮助上层做降级决策
    """

    def __init__(self):
        self._intents: List[Dict[str, Any]] = list(_INTENT_PATTERNS)
        self._default_chain: List[str] = list(_DEFAULT_CHAIN)

    # ---- 主入口 ----

    def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """
        分析查询并返回路由决策

        Args:
            query: 用户自然语言查询
            context: 可选的外部上下文 (如 metadata 中已知的 date/stock_code)

        Returns:
            RoutingDecision 包含意图、Agent 链、置信度、提取的元数据
        """
        intent = self._classify(query)
        metadata = self._extract_metadata(query, context)

        # 找到匹配的意图对应的 chain
        chain = self._default_chain
        for intent_def in self._intents:
            if intent_def["name"] == intent.name:
                chain = intent_def["chain"]
                break

        return RoutingDecision(
            intent=intent.name,
            agent_chain=list(chain),
            confidence=intent.confidence,
            metadata=metadata,
        )

    def classify(self, query: str) -> QueryIntent:
        """仅做意图分类（不做完整路由）"""
        return self._classify(query)

    # ---- 运行时配置 ----

    def register_intent(
        self,
        name: str,
        keywords: List[str],
        chain: List[str],
        patterns: Optional[List[str]] = None,
    ):
        """运行时注册新意图"""
        self._intents.append({
            "name": name,
            "keywords": keywords,
            "patterns": patterns or [],
            "chain": chain,
        })
        logger.info(f"[AgentRouter] 注册新意图: {name} → {chain}")

    def set_default_chain(self, chain: List[str]):
        """修改默认链"""
        self._default_chain = list(chain)

    def get_intent_map(self) -> Dict[str, List[str]]:
        """返回意图 → Agent 链的映射表"""
        result = {ip["name"]: list(ip["chain"]) for ip in self._intents}
        result["_default"] = list(self._default_chain)
        return result

    # ---- 内部方法 ----

    def _classify(self, query: str) -> QueryIntent:
        """意图分类 — 按定义顺序优先匹配"""
        best: Optional[QueryIntent] = None

        for intent_def in self._intents:
            intent = self._match_intent(query, intent_def)
            if intent and (best is None or intent.confidence > best.confidence):
                best = intent

        if best:
            return best
        return QueryIntent(name="general", confidence=0.5)

    def _match_intent(self, query: str, intent_def: Dict) -> Optional[QueryIntent]:
        """匹配单个意图"""
        matched_kw = []
        matched_pat = []

        # 关键词匹配
        for kw in intent_def["keywords"]:
            if kw.lower() in query.lower():
                matched_kw.append(kw)

        # 正则匹配
        for pat in intent_def["patterns"]:
            if re.search(pat, query):
                matched_pat.append(pat)

        if not matched_kw and not matched_pat:
            return None

        # 置信度: 关键词权重 0.3/个, 正则权重 0.4/个, 上限 1.0
        raw = len(matched_kw) * 0.3 + len(matched_pat) * 0.4
        confidence = min(1.0, raw)

        return QueryIntent(
            name=intent_def["name"],
            confidence=confidence,
            matched_keywords=matched_kw + matched_pat,
        )

    def _extract_metadata(self, query: str, context: Optional[Dict]) -> Dict:
        """从查询中提取路由相关的元数据"""
        meta = dict(context) if context else {}

        # 日期提取
        if "date" not in meta:
            date = self._extract_date(query)
            if date:
                meta["date"] = date

        # 股票代码提取
        if "stock_code" not in meta and "ts_code" not in meta:
            code, name = self._extract_stock(query)
            if code:
                meta["stock_code"] = code
                meta["ts_code"] = code  # KLineAgent 使用 ts_code
            if name:
                meta["stock_name"] = name

        return meta

    def _extract_date(self, query: str) -> str:
        """提取日期 YYYY-MM-DD"""
        patterns = [
            (r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?', True),
            (r'(\d{4})(\d{2})(\d{2})', False),
        ]
        for pat, _ in patterns:
            m = re.search(pat, query)
            if m:
                y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
                return f"{y}-{mo}-{d}"
        return ""

    def _extract_stock(self, query: str) -> tuple:
        """提取股票代码"""
        try:
            from financial_rag.tools.kline_tools import STOCK_MAP
        except ImportError:
            STOCK_MAP = {}

        for keyword, (ts_code, name) in STOCK_MAP.items():
            if keyword in query:
                return ts_code, name

        m = re.search(r'(\d{6})', query)
        if m:
            code = m.group(1)
            if code.startswith("6"):
                return f"{code}.SH", code
            return f"{code}.SZ", code

        return "", ""


# ===================== 工厂函数 =====================


def create_agent_router() -> AgentRouter:
    """创建默认 AgentRouter 实例"""
    return AgentRouter()
