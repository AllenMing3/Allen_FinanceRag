"""
KLineAgent — K 线技术分析 Agent（按需查询，不进 KB）

功能:
- 从 Tushare 获取股票/ETF K 线数据
- 计算技术指标: MA, MACD, RSI, Bollinger Bands, KDJ
- 使用 LLM 生成自然语言技术分析摘要
- 查询时返回结果，不存入知识库
"""
import re
import logging
import json
from typing import Dict, Any, List, Optional

from financial_rag.config import config
from financial_rag.core.base import BaseAgent, AgentContext, AgentResult
from financial_rag.llm.dashscope_client import get_llm

logger = logging.getLogger(__name__)


# ===================== 股票关键词映射 =====================

STOCK_MAP = {
    "茅台": ("600519.SH", "贵州茅台"),
    "五粮液": ("000858.SZ", "五粮液"),
    "宁德时代": ("300750.SZ", "宁德时代"),
    "比亚迪": ("002594.SZ", "比亚迪"),
    "招商银行": ("600036.SH", "招商银行"),
    "中国平安": ("601318.SH", "中国平安"),
    "腾讯": ("00700.HK", "腾讯控股"),
    "阿里巴巴": ("09988.HK", "阿里巴巴"),
    "沪深300": ("510300.SH", "沪深300ETF"),
    "中证500": ("510500.SH", "中证500ETF"),
    "创业板": ("159915.SZ", "创业板ETF"),
}


# ===================== LLM Prompt =====================

KLINE_ANALYSIS_SYSTEM = """你是一个专业的金融技术分析师。请根据提供的 K 线数据和技术指标，给出专业的技术分析。

分析要点:
1. 趋势判断（多头/空头/震荡）
2. 关键支撑位和压力位
3. 技术指标信号解读
4. 短期操作建议

要求:
- 语言简洁专业，不超过 500 字
- 使用中文
- 给出具体的价位和数值
- 最后给出风险提示"""

KLINE_ANALYSIS_PROMPT = """请对以下 {name} 的 K 线数据进行技术分析：

## 基本信息
- 标的: {name} ({ts_code})
- 数据区间: {date_range}
- 最新收盘价: {latest_close} 元

## 基础统计
- 区间涨跌幅: {change_pct}%
- 区间最高: {period_high}  |  区间最低: {period_low}
- 上涨天数: {up_days}  |  下跌天数: {down_days}
- 5日均线: {ma5}  |  10日均线: {ma10}  |  20日均线: {ma20}
- 平均成交量: {avg_volume}

## 技术指标
### MACD
- DIF: {macd_dif}  |  DEA: {macd_dea}  |  MACD柱: {macd_value}
- 信号: {macd_signal}

### RSI (14日)
- 数值: {rsi_value}
- 信号: {rsi_signal}

### 布林带 (20日)
- 上轨: {boll_upper}  |  中轨: {boll_middle}  |  下轨: {boll_lower}
- 位置: {boll_position}

### KDJ
- K: {kdj_k}  |  D: {kdj_d}  |  J: {kdj_j}
- 信号: {kdj_signal}

请给出你的技术分析。"""


class KLineAgent(BaseAgent):
    """
    K 线技术分析 Agent

    按需查询模式：获取 K 线数据 → 计算指标 → LLM 分析 → 返回结果
    不存入知识库，仅在查询时使用
    """

    def __init__(self):
        super().__init__(
            name="KLineAgent",
            description="K 线技术分析：获取行情数据、计算技术指标、LLM 生成分析报告",
        )
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def process(self, context: AgentContext) -> AgentResult:
        """
        处理 K 线分析请求

        context.raw_input: 用户查询（如 "茅台最近走势"）
        context.metadata 可包含:
            - ts_code: Tushare 股票代码（如 '600519.SH'）
            - name: 股票名称
            - days: 回溯天数（默认 60）
            - period: K 线周期（'daily' 或 'weekly'）
        """
        raw_input = context.raw_input
        ts_code = context.metadata.get("ts_code", "")
        name = context.metadata.get("name", "")
        days = context.metadata.get("days", 60)
        period = context.metadata.get("period", "daily")

        # 如果没有 ts_code，尝试从查询中提取
        if not ts_code:
            ts_code, name = self._extract_stock_code(raw_input)

        if not ts_code:
            return AgentResult(
                success=False,
                message="无法识别股票代码，请提供具体的股票名称或代码",
                context_updates={"final_answer": "无法识别股票代码，请提供具体的股票名称或代码（如：贵州茅台、600519）"},
            )

        # 1. 获取 K 线数据
        from financial_rag.tushare_client import (
            fetch_stock_kline, fetch_etf_kline,
            compute_kline_stats, compute_technical_indicators,
        )

        # 判断是股票还是 ETF
        # ETF 代码规则: 51xxxx.SH (上海), 159xxx.SZ (深圳)
        is_etf = ts_code.startswith("51") or ts_code.startswith("159")
        if is_etf:
            df = fetch_etf_kline(ts_code, days=days)
        else:
            df = fetch_stock_kline(ts_code, days=days, period=period)

        if df.empty:
            return AgentResult(
                success=False,
                message=f"获取 K 线数据失败 ({ts_code})",
                context_updates={"final_answer": f"获取 K 线数据失败 ({ts_code})，请检查股票代码是否正确或 Tushare Token 是否已配置"},
            )

        # 2. 计算统计指标
        stats = compute_kline_stats(df)
        indicators = compute_technical_indicators(df)

        # 3. LLM 分析
        analysis = self._generate_analysis(ts_code, name, df, stats, indicators)

        # 4. 构建结果
        result_data = {
            "ts_code": ts_code,
            "name": name or ts_code,
            "days": days,
            "data_points": len(df),
            "stats": stats,
            "indicators": indicators,
            "analysis": analysis,
        }

        return AgentResult(
            success=True,
            message=f"K 线分析完成: {name or ts_code} ({len(df)} 个交易日)",
            data=result_data,
            context_updates={
                "final_answer": analysis,
                "intermediate_findings": [{
                    "stage": "kline_analysis",
                    "ts_code": ts_code,
                    "data_points": len(df),
                    "stats": stats,
                    "indicators": indicators,
                }],
            }
        )

    def _extract_stock_code(self, query: str) -> tuple:
        """
        从用户查询中提取股票代码

        策略:
        1. 正则匹配 6 位数字代码
        2. 关键词映射常用股票
        3. Tushare search_stock API
        """
        # 1. 正则匹配 6 位代码
        code_match = re.search(r'(\d{6})', query)
        if code_match:
            code = code_match.group(1)
            # 判断交易所
            if code.startswith("6"):
                ts_code = f"{code}.SH"
            else:
                ts_code = f"{code}.SZ"
            return ts_code, code

        # 2. 关键词映射
        for keyword, (ts_code, name) in STOCK_MAP.items():
            if keyword in query:
                return ts_code, name

        # 3. 使用 Tushare 搜索
        try:
            from financial_rag.tushare_client import search_stock
            results = search_stock(query[:6], limit=1)
            if results:
                return results[0]["ts_code"], results[0]["name"]
        except Exception:
            pass

        return "", ""

    def _generate_analysis(
        self,
        ts_code: str,
        name: str,
        df,
        stats: Dict,
        indicators: Dict,
    ) -> str:
        """使用 LLM 生成技术分析摘要"""
        llm = self._get_llm()
        if llm is None:
            return self._fallback_analysis(stats, indicators)

        # 构建 prompt
        macd = indicators.get("macd", {})
        rsi = indicators.get("rsi", {})
        boll = indicators.get("bollinger", {})
        kdj = indicators.get("kdj", {})

        date_range = ""
        if not df.empty and "date" in df.columns:
            date_range = f"{df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}"

        prompt = KLINE_ANALYSIS_PROMPT.format(
            name=name or ts_code,
            ts_code=ts_code,
            date_range=date_range,
            latest_close=stats.get("latest_close", "N/A"),
            change_pct=stats.get("period_change_pct", "N/A"),
            period_high=stats.get("period_high", "N/A"),
            period_low=stats.get("period_low", "N/A"),
            up_days=stats.get("up_days", "N/A"),
            down_days=stats.get("down_days", "N/A"),
            ma5=stats.get("ma5", "N/A"),
            ma10=stats.get("ma10", "N/A"),
            ma20=stats.get("ma20", "N/A"),
            avg_volume=stats.get("avg_volume", "N/A"),
            macd_dif=macd.get("dif", "N/A"),
            macd_dea=macd.get("dea", "N/A"),
            macd_value=macd.get("macd", "N/A"),
            macd_signal=macd.get("signal", "N/A"),
            rsi_value=rsi.get("value", "N/A"),
            rsi_signal=rsi.get("signal", "N/A"),
            boll_upper=boll.get("upper", "N/A"),
            boll_middle=boll.get("middle", "N/A"),
            boll_lower=boll.get("lower", "N/A"),
            boll_position=boll.get("position", "N/A"),
            kdj_k=kdj.get("k", "N/A"),
            kdj_d=kdj.get("d", "N/A"),
            kdj_j=kdj.get("j", "N/A"),
            kdj_signal=kdj.get("signal", "N/A"),
        )

        try:
            response = llm.chat(
                messages=prompt,
                system=KLINE_ANALYSIS_SYSTEM,
                max_tokens=800,
                temperature=0.3,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"[KLineAgent] LLM 分析失败: {e}")
            return self._fallback_analysis(stats, indicators)

    def _fallback_analysis(self, stats: Dict, indicators: Dict) -> str:
        """无 LLM 时的纯数据摘要"""
        lines = [f"## K 线数据摘要", ""]
        lines.append(f"- 最新收盘价: {stats.get('latest_close', 'N/A')}")
        lines.append(f"- 区间涨跌幅: {stats.get('period_change_pct', 'N/A')}%")
        lines.append(f"- 上涨/下跌天数: {stats.get('up_days', 'N/A')} / {stats.get('down_days', 'N/A')}")

        macd = indicators.get("macd", {})
        if macd:
            lines.append(f"- MACD 信号: {macd.get('signal', 'N/A')}")

        rsi = indicators.get("rsi", {})
        if rsi:
            lines.append(f"- RSI(14): {rsi.get('value', 'N/A')} ({rsi.get('signal', 'N/A')})")

        kdj = indicators.get("kdj", {})
        if kdj:
            lines.append(f"- KDJ 信号: {kdj.get('signal', 'N/A')}")

        return "\n".join(lines)
