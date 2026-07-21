"""
Mock 数据生成器 — 为无 API Key / 无网络环境提供逼真的模拟数据

覆盖两个外部数据源:
1. Tushare: K 线、股票搜索、ETF 搜索、财务指标
2. 财经新闻: 新闻搜索、全量新闻（同花顺/新浪财经/东方财富）

所有 mock 函数返回与真实 API 完全相同的数据结构。
"""
import logging
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 固定随机种子，保证同一请求返回一致数据
random.seed(42)


# ============================================================================
#  Tushare Mock
# ============================================================================

# 模拟股票数据库
_MOCK_STOCKS = {
    "600519.SH": {"name": "贵州茅台", "industry": "白酒", "market": "主板", "base_price": 1680.0},
    "000858.SZ": {"name": "五粮液", "industry": "白酒", "market": "主板", "base_price": 145.0},
    "300750.SZ": {"name": "宁德时代", "industry": "电池", "market": "创业板", "base_price": 210.0},
    "002594.SZ": {"name": "比亚迪", "industry": "汽车", "market": "中小板", "base_price": 265.0},
    "600036.SH": {"name": "招商银行", "industry": "银行", "market": "主板", "base_price": 35.0},
    "601318.SH": {"name": "中国平安", "industry": "保险", "market": "主板", "base_price": 45.0},
    "000001.SZ": {"name": "平安银行", "industry": "银行", "market": "主板", "base_price": 11.5},
    "600900.SH": {"name": "长江电力", "industry": "电力", "market": "主板", "base_price": 28.0},
}

_MOCK_ETFS = {
    "510300.SH": {"name": "沪深300ETF", "fund_type": "股票型", "base_price": 3.85},
    "510500.SH": {"name": "中证500ETF", "fund_type": "股票型", "base_price": 5.62},
    "510050.SH": {"name": "上证50ETF", "fund_type": "股票型", "base_price": 2.78},
    "159915.SZ": {"name": "创业板ETF", "fund_type": "股票型", "base_price": 2.15},
    "159995.SZ": {"name": "芯片ETF", "fund_type": "股票型", "base_price": 1.25},
    "512880.SH": {"name": "证券ETF", "fund_type": "股票型", "base_price": 1.05},
    "515790.SH": {"name": "光伏ETF", "fund_type": "股票型", "base_price": 0.85},
}


def _generate_kline(base_price: float, days: int, volatility: float = 0.02) -> pd.DataFrame:
    """生成逼真的 K 线数据（几何布朗运动 + 日内波动）"""
    dates = pd.bdate_range(end=datetime.now(), periods=days, freq="B")

    # 几何布朗运动模拟价格
    np.random.seed(hash(str(base_price) + str(days)) % (2**31))
    returns = np.random.normal(0.0003, volatility, days)  # 微正漂移
    prices = base_price * np.exp(np.cumsum(returns))

    # 日内 OHLC
    opens = prices * (1 + np.random.uniform(-0.005, 0.005, days))
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.normal(0, 0.008, days)))
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.normal(0, 0.008, days)))
    closes = prices

    # 成交量（与价格波动正相关）
    base_vol = base_price * 100000
    vol_noise = np.random.lognormal(0, 0.3, days)
    volumes = (base_vol * vol_noise).astype(int)
    amounts = (volumes * closes).astype(int)

    df = pd.DataFrame({
        "date": dates[:days],
        "open": np.round(opens, 3),
        "high": np.round(highs, 3),
        "low": np.round(lows, 3),
        "close": np.round(closes, 3),
        "volume": volumes,
        "amount": amounts,
    })
    return df


def mock_stock_kline(ts_code: str, days: int = 30, period: str = "daily") -> pd.DataFrame:
    """模拟股票 K 线数据"""
    info = _MOCK_STOCKS.get(ts_code)
    if not info:
        # 未知代码 → 随机生成一个基准价格
        seed = int(hashlib.md5(ts_code.encode()).hexdigest()[:8], 16) % 1000
        base_price = 10 + seed * 0.5
    else:
        base_price = info["base_price"]

    actual_days = days if period == "daily" else days // 5
    vol = 0.02 if period == "daily" else 0.04
    df = _generate_kline(base_price, actual_days, volatility=vol)
    logger.info(f"[Mock] 股票 K 线: {ts_code}, {len(df)} 条 ({period})")
    return df


def mock_etf_kline(ts_code: str, days: int = 30) -> pd.DataFrame:
    """模拟 ETF K 线数据"""
    info = _MOCK_ETFS.get(ts_code)
    if not info:
        seed = int(hashlib.md5(ts_code.encode()).hexdigest()[:8], 16) % 100
        base_price = 0.5 + seed * 0.1
    else:
        base_price = info["base_price"]

    df = _generate_kline(base_price, days, volatility=0.015)
    logger.info(f"[Mock] ETF K 线: {ts_code}, {len(df)} 条")
    return df


def mock_search_stock(keyword: str, limit: int = 10) -> List[Dict]:
    """模拟股票搜索"""
    results = []
    for ts_code, info in _MOCK_STOCKS.items():
        if keyword in info["name"] or keyword in info.get("industry", ""):
            results.append({
                "ts_code": ts_code,
                "symbol": ts_code.split(".")[0],
                "name": info["name"],
                "area": "贵州" if "茅台" in info["name"] else "广东",
                "industry": info["industry"],
                "market": info["market"],
                "list_date": "20010101",
            })
        if len(results) >= limit:
            break

    # 如果没有匹配，返回一个通用结果
    if not results:
        results.append({
            "ts_code": "600000.SH",
            "symbol": "600000",
            "name": f"{keyword}相关",
            "area": "上海",
            "industry": "金融",
            "market": "主板",
            "list_date": "19991110",
        })

    logger.info(f"[Mock] 股票搜索 '{keyword}': {len(results)} 条")
    return results


def mock_search_etf(keyword: str, limit: int = 10) -> List[Dict]:
    """模拟 ETF 搜索"""
    results = []
    for ts_code, info in _MOCK_ETFS.items():
        if keyword in info["name"] or keyword in info.get("fund_type", ""):
            results.append({
                "ts_code": ts_code,
                "name": info["name"],
                "management": "华夏基金" if "510" in ts_code else "易方达",
                "fund_type": info["fund_type"],
                "list_date": "20120504",
            })
        if len(results) >= limit:
            break

    if not results:
        results.append({
            "ts_code": "510300.SH",
            "name": "沪深300ETF",
            "management": "华泰柏瑞",
            "fund_type": "股票型",
            "list_date": "20120504",
        })

    logger.info(f"[Mock] ETF 搜索 '{keyword}': {len(results)} 条")
    return results


def mock_financial_indicators(ts_code: str, periods: int = 4) -> List[Dict]:
    """模拟财务指标数据"""
    info = _MOCK_STOCKS.get(ts_code, {})
    base_price = info.get("base_price", 100.0)

    # 生成最近 4 个季度的报告日期
    now = datetime.now()
    report_dates = []
    for i in range(periods):
        q = (now.month - 1) // 3 - i
        year = now.year + q // 4
        quarter = q % 4
        end_month = (quarter + 1) * 3
        report_dates.append(f"{year}{end_month:02d}30")

    results = []
    for i, period in enumerate(report_dates):
        # 基于股票类型生成合理指标
        eps = round(base_price * random.uniform(0.02, 0.06) * (1 + i * 0.05), 2)
        roe = round(random.uniform(8.0, 35.0), 2)
        gross_margin = round(random.uniform(20.0, 92.0), 2)

        results.append({
            "period": period,
            "eps": eps,
            "dt_eps": round(eps * 0.95, 2),
            "roe": roe,
            "roe_waa": round(roe * 0.9, 2),
            "grossprofit_margin": gross_margin,
            "netprofit_margin": round(gross_margin * 0.6, 2),
            "debt_to_assets": round(random.uniform(20.0, 65.0), 2),
            "current_ratio": round(random.uniform(1.0, 3.5), 2),
            "quick_ratio": round(random.uniform(0.8, 2.5), 2),
            "ocfps": round(eps * random.uniform(0.8, 1.5), 2),
            "bps": round(base_price * random.uniform(0.2, 0.5), 2),
        })

    logger.info(f"[Mock] 财务指标: {ts_code}, {len(results)} 期")
    return results


# ============================================================================
#  RSS News Mock
# ============================================================================

_MOCK_NEWS_POOL = [
    {"title": "商汤科技2024年营收50.3亿元 同比增长36% 生成式AI业务占比超60%", "source": "同花顺", "sentiment": "正面"},
    {"title": "日日新大模型API日均调用量突码2000万次 企业客户数达5800家", "source": "新浪财经", "sentiment": "正面"},
    {"title": "英伟达发布Blackwell B200 GPU 单卡AI训练性能较H100提升4倍", "source": "东方财富", "sentiment": "正面"},
    {"title": "智谱AI完成B+轮融资 估值超200亿元 GLM-5系列模型Q2发布", "source": "同花顺", "sentiment": "正面"},
    {"title": "微软Azure部署10万张B200用于训练GPT-5 推理成本将显著降低", "source": "新浪财经", "sentiment": "正面"},
    {"title": "谷歌宣布TPU v6将于Q4量产 直接对标Blackwell架构", "source": "东方财富", "sentiment": "正面"},
    {"title": "百度文心一言日均调用量突码1亿次 企业客户数达85万", "source": "同花顺", "sentiment": "正面"},
    {"title": "阿里云百炼大模型平台升级 推理成本降至0.5元/百万token", "source": "新浪财经", "sentiment": "正面"},
    {"title": "阿里巴巴2025财年营收9412亿元 云智能AI收入增长三位数", "source": "同花顺", "sentiment": "正面"},
    {"title": "通义千问Qwen2.5发布 MMLU得分91.2位列开源第一", "source": "东方财富", "sentiment": "正面"},
    {"title": "阿里铉铉用户突码7亿 企业客户超2500万家 AI助理覆盖率80%", "source": "新浪财经", "sentiment": "正面"},
    {"title": "阿里巴巴蔡崇信：AI是阿里未来十年核心战略", "source": "同花顺", "sentiment": "正面"},
    {"title": "智谱AI发布AutoGLM智能体平台 支持自主任务规划和工具调用", "source": "东方财富", "sentiment": "正面"},
    {"title": "智谱AI CodeGeeX服务超200万开发者 代码补全准确率92%", "source": "新浪财经", "sentiment": "正面"},
    {"title": "智谱AI GLM-5系列模型将于Q2发布 参数量达万亿级别", "source": "同花顺", "sentiment": "正面"},
    {"title": "国产GPU芯片替代加速 寒武纪思元590芯片量产出货", "source": "东方财富", "sentiment": "正面"},
    {"title": "OpenAI发布GPT-4o 推理速度提升2倍 价格降低50%", "source": "同花顺", "sentiment": "正面"},
    {"title": "科大讯飞星火大模型V4.0发布 数学推理能力超越GPT-4", "source": "新浪财经", "sentiment": "正面"},
    {"title": "半导体设备国产替代加速 北方华创刻蚀机出货量翻倍", "source": "东方财富", "sentiment": "正面"},
    {"title": "AI算力需求强劲拉动 全球GPU市场规模预计突破500亿美元", "source": "同花顺", "sentiment": "正面"},
    {"title": "云从科技发布从容大模型2.0 多模态理解能力达GPT-4V水平", "source": "新浪财经", "sentiment": "正面"},
    {"title": "工信部发布AI产业发展规划 2025年智能算力规模达300EFlops", "source": "东方财富", "sentiment": "正面"},
    {"title": "字节跳动豆包大模型日均调用量超5000万次 内部应用全面接入", "source": "同花顺", "sentiment": "正面"},
    {"title": "Meta发布Llama 3.1 405B参数模型 开源最大规模AI模型", "source": "新浪财经", "sentiment": "正面"},
    {"title": "AI芯片行业景气度持续上升 寒武纪中报营收同比增长43%", "source": "东方财富", "sentiment": "正面"},
    {"title": "大模型推理成本一年内下降90% 行业进入规模化应用拐点", "source": "同花顺", "sentiment": "正面"},
    {"title": "国产AI芯片华为昇腾910C开始量产 性能对标A100", "source": "新浪财经", "sentiment": "正面"},
    {"title": "AI大模型赋能金融科技 智能投顾市场规模快速扩张", "source": "东方财富", "sentiment": "正面"},
    {"title": "Anthropic发布Claude 3.5 Sonnet 代码生成能力超越GPT-4", "source": "同花顺", "sentiment": "正面"},
    {"title": "智能制造产业政策密集出台 工业机器人订单放量增长", "source": "新浪财经", "sentiment": "正面"},
    {"title": "AI Agent技术爆发 LangChain等开源框架下载量突破千万", "source": "东方财富", "sentiment": "正面"},
    {"title": "全球AI算力投资超3000亿美元 中美占比超80%", "source": "同花顺", "sentiment": "正面"},
]

# ============================================================================
#  Long-form AI Sector Articles (realistic 1000-3000 char)
# ============================================================================

MOCK_LONG_ARTICLES = [
    {
        "title": "商汤科技2024年年报深度解读：生成式AI业务爆发，亏损大幅收窄",
        "source": "同花顺",
        "content": """商汤科技集团股份有限公司（0020.HK）于2025年3月28日发布2024年度业绩报告，全年业绩表现超出市场预期。

一、营收与利润
2024年度，商汤科技实现营业收入50.3亿元人民币，同比增长36.4%，连续三年保持30%以上增速。其中，生成式AI业务收入达30.2亿元，占比60%，同比增长150%，成为公司最核心的增长引擎。传统AI解决方案业务收入20.1亿元，同比下降5%，反映出公司战略转型的成效。

全年净亏损42.1亿元，较上年的58.7亿元收窄28.3%。经调整净亏损（剔除股权激励等非现金项目）为28.5亿元，收窄幅度达35%。毛利率提升至44.3%，较上年提升6.2个百分点，主要得益于生成式AI业务的规模化效应。

二、算力基础设施
商汤科技持续加码算力投入。截至2024年末，公司训练集群规模达4万卡A100等效算力，算力利用率从2023年的65%提升至85%。日日新大模型推理成本降至0.5元/百万token，较2023年的1.4元/百万token下降65%。

公司自研的SenseCore AI大装置已完成第三代升级，支持多模态训练和超大规模分布式训练。2024年新增部署国产芯片8000张，国产芯片占比达20%。

三、商业化进展
企业客户数达5800家，同比增长120%。其中，年合同金额超过100万元的大客户数达320家，同比增长85%。三大业务板块表现：
- 智慧商业：营收22.6亿元，占比45%，同比增长25%
- 智慧城市：营收15.1亿元，占比30%，同比增长10%
- 智慧生活：营收12.6亿元，占比25%，同比增长120%

年度经常性收入（ARR）达28亿元，同比增长95%，收入质量显著改善。

四、模型能力
日日新大模型5.5版本在多项权威评测中表现优异：
- MMLU评测得分87.3，超越GPT-4 Turbo
- GSM8K数学推理得分92.1
- HumanEval代码生成得分89.7
- 上下文窗口支持128K tokens，推理延迟优化至200ms以内

公司研发团队约3200人，研发费用18.7亿元，占营收比例37.2%。2024年新增AI相关专利1200件，累计持有专利超8000件。

五、展望
管理层预计2025年营收将突破70亿元，生成式AI业务占比有望提升至70%。公司计划在Q2发布日日新大模型6.0版本，并启动海外市场拓展。""",
    },
    {
        "title": "英伟达Blackwell架构全面解析：AI算力新纪元的开启",
        "source": "东方财富",
        "content": """英伟达（NVIDIA）于2025年GTC大会上正式发布新一代Blackwell架构GPU系列，标志着AI算力进入新纪元。

一、产品矩阵
Blackwell系列包含三款核心产品：
- B200 GPU：面向大规模AI训练，单卡拥有2080亿个晶体管，采用台积电4nm工艺。FP8训练性能达4.5 PFLOPS，较H100提升4倍；FP4推理性能达9 PFLOPS，较H100提升30倍。
- GB200 NVL72：机架级解决方案，集成72颗B200 GPU和36颗Grace CPU，通过NVLink Switch实现全互联，提供720 PFLOPS的FP8算力。
- B100：面向推理优化场景，功耗较B200降低30%，推理性价比提升5倍。

二、关键技术创新
1. 第二代Transformer引擎：支持FP4/FP6精度训练，在不损失模型质量的前提下将训练速度提升2倍。
2. NVLink 5.0：单链路带宽达100GB/s，72卡互联总带宽达130TB/s，较Hopper架构提升12倍。
3. RAS Engine：可靠性增强系统，支持芯片级故障预测和自动恢复，集群可用性从99.9%提升至99.99%。
4. 安全AI：硬件级机密计算，支持TEE（可信执行环境），保护模型权重和用户数据。

三、生态系统
微软Azure已部署10万张B200 GPU用于训练GPT-5，这是目前公开披露的最大规模AI训练集群。Google Cloud、AWS、Oracle Cloud均已签署大额采购订单。

国内方面，由于出口管制，英伟达推出中国特供版B20，算力约为B200的60%。多家中国云厂商已表达采购意向。

四、市场影响
根据第三方机构测算，2025年全球AI GPU市场规模预计达500亿美元，英伟达占据约80%份额。Blackwell架构的推出将进一步巩固其市场主导地位。

AMD、谷歌、英特尔等竞争对手也在加速追赶。谷歌TPU v6预计Q4量产，性能对标Blackwell；AMD MI400系列预计2025年Q3发布。

五、供应链
台积电为Blackwell GPU的独家代工厂，采用4nm制程。CoWoS先进封装产能是主要瓶颈，英伟达已提前锁定2025年70%的CoWoS产能。HBM3E内存由SK海力士和三星供应，单卡搭载192GB HBM3E，带宽达8TB/s。

分析师预计，英伟达2025财年数据中心业务营收将突码1200亿美元，Blackwell系列贡献约40%。""",
    },
    {
        "title": "阿里巴巴集团2025财年深度解析：AI驱动云智能重回增长，电商基本盘稳健",
        "source": "东方财富",
        "content": """阿里巴巴集团控股有限公司（09988.HK / BABA）于2025年5月发布2025财年全年业绩报告，整体表现超出市场预期，AI战略成效显著。

一、整体业绩
2025财年，阿里巴巴实现营收9411.68亿元人民币，同比增长8%。其中，第四季度营收2364.54亿元，同比增长7%。经调整EBITA为1641.02亿元，同比增长12%。净利润1259.76亿元，归属于普通股股东的净利润为1189.45亿元。

集团主席蔡崇信表示，AI是阿里未来十年的核心战略，集团将持续加大在通义千问大模型、阿里云基础设施和AI应用生态方面的投入。

二、云智能集团——AI驱动重回增长
云智能集团全年营收1096.87亿元，同比增长13%，其中AI相关收入连续六个季度保持三位数增长，全年AI收入突破200亿元。

通义千问大模型已迭代至Qwen2.5系列，在多项国际评测中表现优异：
- MMLU评测得分91.2，位列全球开源模型第一梯队
- HumanEval代码生成得分93.5
- GSM8K数学推理得分95.8
- 支持最大128K上下文窗口，推理延迟优化至100ms以内

阿里云百炼平台已服务超50万家企业客户，API日均调用量突码2亿次。推理成本降至0.5元/百万token，较2024年下降75%。ModelScope魔搭社区模型下载量累计突破5亿次，成为中国最大的开源模型社区。

三、淘天集团——电商基本盘稳健
淘天集团全年营收4405.06亿元，同比增长4%。88VIP会员数突破5000万，年度活跃消费者达9.1亿。双11大促GMV再创新高，直播电商贡献占比提升至35%。

客户管理收入（CMR）同比增长9%，货币化率持续改善。跨境业务（速卖通 + Lazada）营收1025.59亿元，同比增长45%，国际数字商业成为新增长引擎。

四、其他业务
蚂蚁集团（支付宝）全年支付GMV超200万亿元，全球活跃用户超13亿。花呗、借呗等消费信贷业务稳健运行。钉钉用户数突破7亿，企业客户数超2500万家，AI助理功能覆盖率达80%。菜鸟全年营收超1000亿元，全球物流网络覆盖200多个国家和地区。

五、AI战略全景
阿里在AI领域的布局覆盖四层：
1. 算力层：自研倚天710芯片 + 含光800推理芯片，训练集群超5万卡
2. 模型层：通义千问Qwen2.5系列（开源+闭源），通义万相（图像生成），通义听悟（语音）
3. 平台层：百炼平台（MaaS模型即服务）、PAI平台（AI开发工具链）
4. 应用层：通义灵码（AI编程）、AI客服、智能搜索，全集团业务已全面接入大模型

达摩院和平头哥在基础研究和芯片设计方面持续投入，为集团AI战略提供底层支撑。

六、展望
管理层预计2026财年云智能营收将突破1300亿元，AI收入占比提升至30%。电商业务将重点发力直播电商和跨境，淘天集团营收有望突破4600亿元。""",
    },
    {
        "title": "智谱AI深度解读：清华系大模型领军者，商业化路径与竞争格局",
        "source": "新浪财经",
        "content": """智谱AI（北京智谱华章科技有限公司）是当前中国AI大模型赛道的头部创业公司，依托清华大学知识图谱与人工智能实验室（KEG）的学术积累，在GLM系列模型和企业级AI应用方面建立了显著优势。

一、公司背景
智谱AI成立于2019年，由清华大学计算机系教授唐杰、副教授刘知远等人联合创立。公司核心技术源自清华KEG实验室超过20年的知识图谱和自然语言处理研究。

核心团队包括：
- 唐杰（首席科学家）：清华大学计算机系教授，AMiner创始人，知识图谱领域国际顶尖学者
- 刘知远（联合创始人）：清华大学计算机系副教授，自然语言处理领域知名学者
- 张鹏（CEO）：前美团技术负责人，负责商业化落地

二、融资与估值
智谱AI累计完成多轮融资：
- 天使轮（2019）：数百万美元，清华系基金
- A轮（2022）：数亿元人民币，启明创投领投
- B轮（2023）：超25亿元人民币，社保基金、北京市政府引导基金参与
- B+轮（2025）：超50亿元人民币，投后估值超200亿元，高瓴创投、红杉中国跟投

智谱AI估值位居中国大模型创业公司前列，与月之暗面（Kimi）、百川智能、MiniMax并称为“AI四小龙”。

三、产品矩阵
1. GLM系列大模型：
   - ChatGLM-6B：开源模型，GitHub星标超2万，成为国内最受欢迎的开源大模型之一
   - GLM-4系列：闭源旗舰模型，参数规模万亿级，性能对标GPT-4
   - GLM-5系列：计划2025年Q2发布，预计参数量达万亿级别，性能对标GPT-4o

2. 智谱清言（ChatGLM App）：
   面向C端的AI助手应用，支持对话、文档分析、代码生成等多模态能力。月活跃用户（MAU）突破500万。

3. CodeGeeX：
   AI编程助手，支持VS Code、JetBrains等主流IDE，代码补全准确率达92%，累计服务超200万开发者。

4. AutoGLM：
   AI Agent平台，支持自主任务规划、工具调用、多步骤执行，面向企业级自动化场景。

5. 企业API服务（bigmodel.cn）：
   提供GLM系列模型的API调用服务，日均调用量超500万次，累计服务企业客户超3000家。

四、技术特色
智谱AI的技术路线与竞品有显著差异：
- 自研GLM架构：不同于GPT的自回归架构，GLM采用自回归填空（Autoregressive Blank Infilling）预训练框架，在多任务场景下表现更均衡
- 知识图谱融合：利用清华KEG的知识图谱积累，在事实性问答和知识推理方面具有独特优势
- 中英双语原生：GLM系列模型原生支持中英文，在中文任务上的表现持续领先

在多项评测中：
- C-Eval中文评测：GLM-4得分89.5，位列前三
- MMLU英文评测：GLM-4得分87.6
- OpenCompass综合评测：GLM-4位列国产模型前二

五、商业化进展
智谱AI的商业化聚焦三大场景：
1. 企业级AI应用：为金融、政务、教育等行业提供大模型私有化部署和API服务，客单价50-200万元/年
2. AI编程工具：CodeGeeX已签约超50家企业客户，包括多家大型金融机构
3. 教育领域：与清华大学等高校合作，提供AI教学平台和科研辅助工具

2024年公司营收约12亿元人民币，同比增长超200%，但仍处于亏损阶段，预计2026年实现盈亏平衡。

六、竞争格局
中国大模型赛道竞争激烈，智谱AI的主要竞争对手包括：
- 月之暗面（Kimi）：C端用户体验领先，MAU超2000万
- 百川智能：企业级市场深耕，已服务超500家企业
- MiniMax：AI社交领域突出，Talkie全球用户超5000万
- DeepSeek：开源策略激进，DeepSeek-V3性能亮眼
- 零一万物：全球化布局，Yi系列模型海外用户增长快

大厂方面，阿里通义千问、百度文心一言、字节豆包、腾讯混元等均在加速追赶，智谱AI面临来自资金和生态的双重竞争压力。

七、展望
管理层表示，2025年将是智谱AI商业化的关键一年。GLM-5系列模型发布后，公司计划启动海外市场拓展，并探索IPO可能性。技术层面，公司将重点投入多模态能力和AI Agent技术，目标是在企业级AI市场建立绝对领先地位。""",
    },
    {
        "title": "2024年中国AI大模型行业融资盘点：总规模超800亿元",
        "source": "新浪财经",
        "content": """2024年是中国AI大模型行业融资的高峰年，全年融资总额超过800亿元人民币，行业估值整体突破3000亿元。

一、头部企业融资
1. 智谱AI：完成B+轮融资，融资金额超50亿元，投后估值超200亿元。投资方包括社保基金、北京市政府引导基金、高瓴创投等。智谱AI旗下GLM系列模型累计服务企业客户超3000家，API日均调用量超500万次。GLM-5系列模型将于2025年Q2发布，参数量达万亿级别。

2. 月之暗面（Moonshot AI）：完成B轮融资，融资金额超10亿美元，估值达30亿美元。Kimi智能助手MAU突破2000万，成为国内最受欢迎的AI助手之一。公司计划2025年推出Kimi 2.0，支持100万字上下文窗口。

3. 百川智能：完成A2轮融资，融资金额超5亿美元。百川大模型在企业级市场占有率持续提升，已服务超500家企业客户。

4. MiniMax：完成B轮融资，融资金额6亿美元。Talkie应用全球用户超5000万，成为AI社交领域的领军企业。

二、融资趋势分析
- 单笔融资金额增大：2024年平均单笔融资超5亿元，较2023年增长300%
- 国资参与度提升：政府引导基金、国有资本参与比例从2023年的15%提升至40%
- 估值分化加剧：头部5家企业估值占行业总量的65%
- 商业化成为估值关键：有实际营收的企业估值溢价达2-3倍

三、商业化进展
2024年中国AI大模型行业整体营收约150亿元，主要收入来源：
- API调用服务：占比45%，日均调用量合计超5亿次
- 私有化部署：占比30%，金融、政务、医疗为主要客户
- SaaS应用：占比25%，AI写作、AI编程、AI客服为主

四、行业挑战
1. 算力瓶颈：高端GPU受限，国产芯片替代仍需2-3年
2. 人才竞争：顶尖AI人才年薪已达200-500万元
3. 数据合规：个人数据保护法对训练数据提出更高要求
4. 盈利难题：行业整体仍处于亏损阶段，预计2026年头部企业实现盈亏平衡

五、2025年展望
预计2025年行业融资规模将达1000亿元，头部企业有望启动IPO。推理成本将继续下降90%，AI Agent技术将迎来爆发。""",
    },
]


def mock_long_article(index: int = 0) -> Dict:
    """获取一篇长文 AI 行业新闻"""
    if index < 0 or index >= len(MOCK_LONG_ARTICLES):
        index = 0
    article = MOCK_LONG_ARTICLES[index]
    return {
        "title": article["title"],
        "content": article["content"],
        "source": article["source"],
        "publish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url": f"https://example.com/long-article/{index+1}",
        "sentiment": "正面",
        "text_length": len(article["content"]),
    }


def mock_search_news(keyword: str, max_news: int = 30) -> Dict:
    """模拟新闻搜索 — 按关键词过滤并返回结构化结果"""
    import time
    t0 = time.time()

    # 从池中选出与关键词相关的新闻（宽松匹配）
    kw_parts = [
        kw.strip()
        for kw in keyword.replace("、", ",").replace("，", ",").split(",")
        if kw.strip()
    ]

    matched = []
    for item in _MOCK_NEWS_POOL:
        text = item["title"]
        # 宽松匹配：任一关键词子串命中即可
        if any(p in text for p in kw_parts if len(p) >= 2) or not kw_parts:
            matched.append(item)

    # 如果没匹配到，返回前几条通用新闻
    if not matched:
        matched = _MOCK_NEWS_POOL[:5]

    # 构造完整结构
    now = datetime.now()
    items = []
    for i, m in enumerate(matched[:max_news]):
        pub_time = (now - timedelta(hours=i * 2)).strftime("%Y-%m-%d %H:%M:%S")
        items.append({
            "title": m["title"],
            "content": m["title"] + "。详细分析内容..." + f"（模拟数据第{i+1}条）",
            "source": m["source"],
            "publish_time": pub_time,
            "url": f"https://example.com/news/{i+1}",
            "sentiment": m["sentiment"],
        })

    elapsed = (time.time() - t0) * 1000
    logger.info(f"[Mock] 新闻搜索 '{keyword}': {len(items)} 条")

    return {
        "keyword": keyword,
        "total": len(items),
        "items": items,
        "elapsed_ms": elapsed,
    }


def mock_fetch_all_news(max_per_source: int = 20) -> List[Dict]:
    """模拟获取全部新闻"""
    now = datetime.now()
    items = []
    for i, m in enumerate(_MOCK_NEWS_POOL[:max_per_source * 3]):
        pub_time = (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        items.append({
            "title": m["title"],
            "content": m["title"] + "。详细分析内容...",
            "source": m["source"],
            "publish_time": pub_time,
            "url": f"https://example.com/news/{i+1}",
            "sentiment": m["sentiment"],
        })

    logger.info(f"[Mock] 全量新闻: {len(items)} 条")
    return items

