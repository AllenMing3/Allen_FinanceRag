"""
Shared test fixtures — AI sector sample data + tool registry setup
"""
import pytest

from financial_rag.tools.core import FunctionRegistry, ToolExecutor, create_financial_registry
from financial_rag.mock_data import MOCK_LONG_ARTICLES


# ===================== AI Sector Sample Texts =====================

SAMPLE_AI_FINANCIAL_REPORT = """
商汤科技集团股份有限公司2024年年度报告

一、经营业绩
2024年度，商汤科技实现营业收入50.3亿元，同比增长36.4%。其中，生成式AI业务收入占比达60%，
成为公司核心增长引擎。全年净亏损42.1亿元，同比收窄28.3%。研发投入18.7亿元，占营收比例37.2%。

二、算力基础设施
公司训练集群规模达4万卡A100，算力利用率提升至85%。推理成本降至0.5元/百万token，
较上年下降65%。日日新大模型API日均调用量突破2000万次，同比增长400%。

三、商业化进展
企业客户数达5800家，同比增长120%。智慧商业、智慧城市、智慧生活三大板块营收占比分别为
45%、30%、25%。年度经常性收入（ARR）达28亿元。

四、模型能力
日日新大模型5.5版本在多项benchmark评测中排名前列，上下文窗口支持128K tokens，
推理延迟优化至200ms以内。
"""

SAMPLE_AI_NEWS = """
英伟达正式发布Blackwell B200 GPU，单卡AI训练性能较上一代H100提升4倍，
推理效率提升30倍。微软Azure已部署10万张B200用于训练GPT-5，
OpenAI表示新一代模型推理成本将显著降低。谷歌同步宣布TPU v6将于Q4量产，
直接对标Blackwell架构。全球AI算力投资预计2025年超3000亿美元。
"""

SAMPLE_AI_FUNDING = """
智谱AI宣布完成B+轮融资，融资金额超50亿元，投后估值超200亿元。
本轮融资由社保基金、北京市政府引导基金等联合投资。
智谱AI是国内领先的大模型创业公司，旗下GLM系列模型累计服务企业客户超3000家，
API日均调用量超500万次。GLM-5系列模型将于2025年Q2正式发布，
参数量达万亿级别，支持多模态理解与生成。
"""

SAMPLE_AI_PRODUCT_LAUNCH = """
科大讯飞重磅发布星火大模型V4.0，新版本在数学推理、代码生成、多模态理解等方面
实现全面升级。星火V4.0数学推理能力超越GPT-4，代码生成准确率提升至92%。
新版本API已开放调用，内测期间免费使用。科大讯飞董事长刘庆峰表示，
星火大模型将全面赋能教育、医疗、办公三大场景。
"""

SAMPLE_AI_TECH_REPORT = """
本文提出了一种基于MoE（Mixture of Experts）架构的高效大模型训练方法。
通过消融实验（ablation study）验证了各组件的有效性。在MMLU、GSM8K、
HumanEval等benchmark评测中，我们的模型（70B参数）在推理任务上达到GPT-4的95%性能，
同时训练成本降低60%。数据集包含1.2万亿token，使用8000卡A100集群训练14天。
模型架构采用Transformer + MoE，每个token仅激活16个专家中的2个。
"""


# ===================== Fixtures =====================

@pytest.fixture
def registry():
    """Create a financial registry without LLM (regex fallback mode)"""
    return create_financial_registry(retriever=None, llm=None)


@pytest.fixture
def executor(registry):
    """Create a tool executor bound to the registry"""
    return ToolExecutor(registry)


@pytest.fixture
def ai_financial_text():
    return SAMPLE_AI_FINANCIAL_REPORT


@pytest.fixture
def ai_news_text():
    return SAMPLE_AI_NEWS


@pytest.fixture
def ai_funding_text():
    return SAMPLE_AI_FUNDING


@pytest.fixture
def ai_product_text():
    return SAMPLE_AI_PRODUCT_LAUNCH


@pytest.fixture
def ai_tech_text():
    return SAMPLE_AI_TECH_REPORT


@pytest.fixture
def long_article_sensetime():
    """商汤科技年报深度解读 — ~2000字"""
    return MOCK_LONG_ARTICLES[0]["content"]


@pytest.fixture
def long_article_nvidia():
    """英伟达 Blackwell 架构解析 — ~2000字"""
    return MOCK_LONG_ARTICLES[1]["content"]


@pytest.fixture
def long_article_funding():
    """AI大模型行业融资盘点 — ~1500字"""
    return MOCK_LONG_ARTICLES[2]["content"]


@pytest.fixture
def all_long_articles():
    return [a["content"] for a in MOCK_LONG_ARTICLES]


@pytest.fixture
def all_sample_texts():
    return [
        SAMPLE_AI_FINANCIAL_REPORT,
        SAMPLE_AI_NEWS,
        SAMPLE_AI_FUNDING,
        SAMPLE_AI_PRODUCT_LAUNCH,
        SAMPLE_AI_TECH_REPORT,
    ]
