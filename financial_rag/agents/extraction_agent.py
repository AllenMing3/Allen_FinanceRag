"""
ExtractionAgent — 关键财务指标抽取

功能:
- 从财报中抽取: 营收、利润、毛利率、现金流等
- 从新闻中抽取: 事件、影响、关联公司
- 结构化输出供后续分析使用
- 【打分】关键词抽取质量 & 查询改写质量评分
"""
import json
import re
from typing import Dict, Any, List, Optional

from financial_rag.config import config
from financial_rag.core.base import BaseAgent, AgentContext, AgentResult
from financial_rag.llm.dashscope_client import get_llm
from financial_rag.prompts import (
    FINANCIAL_METRICS_EXTRACTION_SYSTEM,
    FINANCIAL_METRICS_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_PROMPT,
    FEW_SHOT_EXAMPLES,
)


class ExtractionAgent(BaseAgent):
    """
    Agent 2: 信息抽取

    从原始文档中抽取结构化财务数据
    """

    # 财务指标定义（纯结构，不绑定实现）
    FINANCIAL_METRICS = [
        "revenue",           # 营业收入
        "net_income",        # 净利润
        "gross_margin",      # 毛利率
        "operating_cash_flow",  # 经营现金流
        "total_assets",      # 总资产
        "total_liabilities", # 总负债
        "eps",               # 每股收益
        "roe",               # 净资产收益率
    ]

    NEWS_ENTITIES = [
        "company",           # 涉及公司
        "event_type",        # 事件类型(并购/分红/财报发布/...)
        "impact",            # 影响评估
        "related_companies", # 关联公司
    ]

    def __init__(self):
        super().__init__(
            name="ExtractionAgent",
            description="财务指标与实体抽取"
        )
        self._llm = None

    def _get_llm(self):
        """懒加载 LLM 实例"""
        if self._llm is None:
            try:
                self._llm = get_llm(
                    api_key=config.llm.api_key,
                    model=config.llm.model,
                    temperature=0.0,
                )
            except (ImportError, ValueError):
                self._llm = None
        return self._llm

    def process(self, context: AgentContext) -> AgentResult:
        documents = context.parsed_data or []

        # 提取新闻元数据作为先验知识
        self._news_context = context.metadata.get("news_context", [])

        # 1. 抽取财务指标
        metrics = self._extract_metrics(documents)

        # 2. 抽取实体与事件
        entities = self._extract_entities(documents)

        # 3. 生成多角度查询
        queries = self._generate_queries(documents, metrics, entities)

        # 评估抽取质量
        metric_score = self._evaluate_extraction(metrics, entities)
        query_score = self._evaluate_queries(queries, documents)

        result_data = {
            "metrics": metrics,
            "entities": entities,
            "queries": queries,
            "_scores": {
                "extraction": metric_score,
                "query_rewrite": query_score,
            }
        }

        return AgentResult(
            success=True,
            message=f"抽取 {len(metrics)} 项指标, {len(entities)} 个实体, 生成 {len(queries)} 个查询",
            data=result_data,
            context_updates={
                "extracted_features": result_data,
                "intermediate_findings": [{"stage": "extraction", "metrics": list(metrics.keys()),
                                           "extraction_score": metric_score, "query_score": query_score}]
            }
        )

    def _evaluate_extraction(self, metrics: Dict, entities: List[Dict]) -> float:
        """评估指标和实体抽取的覆盖率"""
        metric_hit = sum(1 for m in self.FINANCIAL_METRICS if m in metrics)
        metric_rate = metric_hit / max(len(self.FINANCIAL_METRICS), 1)

        entity_hit = min(len(entities), len(self.NEWS_ENTITIES))
        entity_rate = entity_hit / max(len(self.NEWS_ENTITIES), 1)

        return 0.6 * metric_rate + 0.4 * entity_rate

    def _evaluate_queries(self, queries: List[str], documents: List[Dict]) -> float:
        """评估生成查询的多样性和覆盖率"""
        if not queries:
            return 0.0
        # 查询数量评分
        count_score = min(1.0, len(queries) / 3)
        # 多样性：基于长度和内容差异的简单评估
        lengths = [len(q) for q in queries]
        diversity = 1.0 if len(set(lengths)) > 1 else 0.6
        return 0.4 * count_score + 0.6 * diversity

    # ===================== 财务指标抽取 =====================

    def _extract_metrics(self, documents: List[Dict]) -> Dict[str, Any]:
        """
        从文档中抽取财务指标。

        策略：
        1. 优先使用 LLM（DashScopeLLM + 结构化 prompt）
        2. 如果 LLM 不可用，使用正则作为 fallback
        """
        if not documents:
            return {}

        # 合并所有文档文本
        combined_text = "\n\n".join(
            d.get("text", "") for d in documents
            if d.get("text")
        )

        if not combined_text:
            return {}

        # 尝试 LLM 抽取
        llm = self._get_llm()
        if llm:
            try:
                return self._llm_extract_metrics(combined_text)
            except Exception as e:
                print(f"[ExtractionAgent] LLM 指标抽取失败，回退正则: {e}")

        # Fallback: 正则抽取
        return self._regex_extract_metrics(combined_text)

    def _llm_extract_metrics(self, text: str) -> Dict[str, Any]:
        """使用 LLM 从文本中抽取财务指标"""
        llm = self._get_llm()
        if llm is None:
            return {}

        # 取前 8000 字符保留完整财务上下文
        prompt_text = text[:8000]

        # 构建 few-shot 增强的 prompt
        user_prompt = FINANCIAL_METRICS_EXTRACTION_PROMPT.format(text=prompt_text)
        system_prompt = FINANCIAL_METRICS_EXTRACTION_SYSTEM

        # 注入新闻元数据作为先验知识
        news_ctx = self._build_news_context()
        if news_ctx:
            system_prompt += f"\n\n以下是近期相关财经动态，可辅助识别文档中涉及的公司和行业背景：\n{news_ctx}"

        # 追加 few-shot 示例到 system prompt
        few_shot = FEW_SHOT_EXAMPLES.get("metrics_extraction", "")
        if few_shot:
            system_prompt += f"\n\n以下是一些示例供参考：\n{few_shot}"

        try:
            response = llm.chat(
                messages=user_prompt,
                system=system_prompt,
                max_tokens=1024,
                temperature=0.0,
            )
            content = response.content.strip()

            # 提取 JSON 对象
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                metrics = json.loads(json_match.group())
                # 标准化字段名：映射中文键到英文键
                return self._normalize_metric_keys(metrics)
        except Exception as e:
            print(f"[ExtractionAgent] LLM 指标解析失败: {e}")

        return {}

    def _build_news_context(self) -> str:
        """将新闻元数据格式化为 LLM 可理解的上下文文本"""
        items = getattr(self, '_news_context', [])
        if not items:
            return ""
        lines = []
        for m in items[:10]:
            parts = []
            if m.get("title"):
                parts.append(m["title"])
            if m.get("keyword"):
                parts.append(f"关键词: {m['keyword']}")
            if m.get("publish_time"):
                parts.append(f"时间: {m['publish_time']}")
            lines.append("- " + " | ".join(parts))
        return "\n".join(lines)

    def _normalize_metric_keys(self, raw_metrics: Dict) -> Dict[str, Any]:
        """将 LLM 返回的中文/混合键名标准化为英文键名"""
        key_map = {
            "revenue": ["revenue", "营业收入", "营收", "营业总收入"],
            "net_income": ["net_income", "净利润", "归属净利润", "归母净利润", "归属于上市公司股东的净利润"],
            "gross_margin": ["gross_margin", "毛利率", "综合毛利率"],
            "operating_cash_flow": ["operating_cash_flow", "经营现金流", "经营活动现金流", "经营活动现金流量净额"],
            "total_assets": ["total_assets", "总资产"],
            "total_liabilities": ["total_liabilities", "总负债"],
            "eps": ["eps", "每股收益", "基本每股收益"],
            "roe": ["roe", "净资产收益率", "ROE"],
        }

        normalized = {}
        for eng_key, aliases in key_map.items():
            for alias in aliases:
                if alias in raw_metrics:
                    normalized[eng_key] = raw_metrics[alias]
                    break

        # 保留未映射的其他指标
        for k, v in raw_metrics.items():
            if k not in normalized and not any(k in aliases for aliases in key_map.values()):
                normalized[k] = v

        return normalized

    def _regex_extract_metrics(self, text: str) -> Dict[str, Any]:
        """正则抽取财务指标（LLM 不可用时的 fallback）"""
        metrics = {}

        # 营业收入
        rev_match = re.search(r'营业(?:总)?收入[约达为]?\s*[\d,.]+\s*亿?元?', text)
        if not rev_match:
            rev_match = re.search(r'营收[约达为]?\s*[\d,.]+\s*亿?元?', text)
        if rev_match:
            num_match = re.search(r'([\d,.]+)\s*(亿|万)?元?', rev_match.group())
            if num_match:
                value = float(num_match.group(1).replace(",", ""))
                unit = num_match.group(2) if num_match.group(2) else ""
                metrics["revenue"] = {"value": value, "unit": unit + "元", "source": "regex"}

        # 净利润
        np_match = re.search(r'(?:归属(?:于(?:上市公司)?股东)?)?净利润[约达为]?\s*[\d,.]+\s*亿?元?', text)
        if np_match:
            num_match = re.search(r'([\d,.]+)\s*(亿|万)?元?', np_match.group())
            if num_match:
                value = float(num_match.group(1).replace(",", ""))
                unit = num_match.group(2) if num_match.group(2) else ""
                metrics["net_income"] = {"value": value, "unit": unit + "元", "source": "regex"}

        # 毛利率
        gm_match = re.search(r'毛利[率率]?\s*[约达为]?\s*[\d.]+\s*%', text)
        if gm_match:
            num_match = re.search(r'([\d.]+)\s*%', gm_match.group())
            if num_match:
                metrics["gross_margin"] = {"value": float(num_match.group(1)), "unit": "%", "source": "regex"}

        # 经营活动现金流
        cf_match = re.search(r'经营活动(?:产生的)?现金流量净额[约达为]?\s*[\d,.]+\s*亿?元?', text)
        if cf_match:
            num_match = re.search(r'([\d,.]+)\s*(亿|万)?元?', cf_match.group())
            if num_match:
                value = float(num_match.group(1).replace(",", ""))
                unit = num_match.group(2) if num_match.group(2) else ""
                metrics["operating_cash_flow"] = {"value": value, "unit": unit + "元", "source": "regex"}

        # EPS
        eps_match = re.search(r'(?:基本)?每股收益[约达为]?\s*[\d.]+\s*元', text)
        if eps_match:
            num_match = re.search(r'([\d.]+)\s*元', eps_match.group())
            if num_match:
                metrics["eps"] = {"value": float(num_match.group(1)), "unit": "元", "source": "regex"}

        # ROE
        roe_match = re.search(r'(?:ROE|净资产收益率)\s*[约达为]?\s*[\d.]+\s*%', text)
        if roe_match:
            num_match = re.search(r'([\d.]+)\s*%', roe_match.group())
            if num_match:
                metrics["roe"] = {"value": float(num_match.group(1)), "unit": "%", "source": "regex"}

        # 总资产
        ta_match = re.search(r'总资产[约达为]?\s*[\d,.]+\s*亿?元?', text)
        if ta_match:
            num_match = re.search(r'([\d,.]+)\s*(亿|万)?元?', ta_match.group())
            if num_match:
                value = float(num_match.group(1).replace(",", ""))
                unit = num_match.group(2) if num_match.group(2) else ""
                metrics["total_assets"] = {"value": value, "unit": unit + "元", "source": "regex"}

        # 总负债
        tl_match = re.search(r'总负债[约达为]?\s*[\d,.]+\s*亿?元?', text)
        if tl_match:
            num_match = re.search(r'([\d,.]+)\s*(亿|万)?元?', tl_match.group())
            if num_match:
                value = float(num_match.group(1).replace(",", ""))
                unit = num_match.group(2) if num_match.group(2) else ""
                metrics["total_liabilities"] = {"value": value, "unit": unit + "元", "source": "regex"}

        return metrics

    # ===================== 实体抽取 =====================

    def _extract_entities(self, documents: List[Dict]) -> List[Dict]:
        """
        抽取实体与事件。

        策略：
        1. 优先使用 LLM（DashScopeLLM + few-shot prompt）
        2. 如果 LLM 不可用，使用正则作为 fallback
        """
        if not documents:
            return []

        combined_text = "\n\n".join(
            d.get("text", "") for d in documents
            if d.get("text")
        )

        if not combined_text:
            return []

        llm = self._get_llm()
        if llm:
            try:
                return self._llm_extract_entities(combined_text)
            except Exception as e:
                print(f"[ExtractionAgent] LLM 实体抽取失败，回退正则: {e}")

        return self._regex_extract_entities(combined_text)

    def _llm_extract_entities(self, text: str) -> List[Dict]:
        """使用 LLM 从文本中抽取实体"""
        llm = self._get_llm()
        if llm is None:
            return []

        prompt_text = text[:8000]
        user_prompt = ENTITY_EXTRACTION_PROMPT.format(text=prompt_text)
        system_prompt = ENTITY_EXTRACTION_SYSTEM

        # 注入新闻元数据作为先验知识
        news_ctx = self._build_news_context()
        if news_ctx:
            system_prompt += f"\n\n以下是近期相关财经动态，可辅助识别相关公司和事件：\n{news_ctx}"

        # 追加 few-shot 示例
        few_shot = FEW_SHOT_EXAMPLES.get("entity_extraction", "")
        if few_shot:
            system_prompt += f"\n\n以下是一些示例供参考：\n{few_shot}"

        try:
            response = llm.chat(
                messages=user_prompt,
                system=system_prompt,
                max_tokens=1024,
                temperature=0.0,
            )
            content = response.content.strip()

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                entities_dict = json.loads(json_match.group())
                # 如果返回的是 dict（按类别组织），转为 list
                return self._normalize_entity_output(entities_dict)
        except Exception as e:
            print(f"[ExtractionAgent] LLM 实体解析失败: {e}")

        return []

    def _normalize_entity_output(self, raw: Dict) -> List[Dict]:
        """
        将 LLM 返回的实体 dict 标准化为 list 格式。
        LLM 返回格式: {companies: [...], persons: [...], event: {...}, ...}
        转为: [{type: "company", data: {...}}, ...]
        """
        if isinstance(raw, list):
            return raw

        entities = []

        # companies
        for c in raw.get("companies", []):
            if isinstance(c, dict):
                entities.append({"type": "company", "data": c})

        # persons
        for p in raw.get("persons", []):
            if isinstance(p, dict):
                entities.append({"type": "person", "data": p})

        # financial figures
        for f in raw.get("financial_figures", []):
            if isinstance(f, dict):
                entities.append({"type": "financial_figure", "data": f})

        # event
        event = raw.get("event", {})
        if isinstance(event, dict) and event:
            entities.append({"type": "event", "data": event})

        # related companies
        for rc in raw.get("related_companies", []):
            entities.append({"type": "related_company", "data": rc if isinstance(rc, dict) else {"name": rc}})

        # industries
        for ind in raw.get("industries", []):
            entities.append({"type": "industry", "data": ind if isinstance(ind, dict) else {"name": ind}})

        # key topics
        for topic in raw.get("key_topics", []):
            entities.append({"type": "topic", "data": topic if isinstance(topic, dict) else {"name": topic}})

        return entities

    def _regex_extract_entities(self, text: str) -> List[Dict]:
        """正则抽取实体（LLM 不可用时的 fallback）"""
        entities = []

        # 抽取公司名
        company_pattern = r'([\u4e00-\u9fa5]{2,6}(?:集团|公司|控股|股份|科技|银行|证券|基金|保险|信托|租赁)(?:有限公司|股份有限公司)?)'
        seen_companies = set()
        for match in re.finditer(company_pattern, text):
            name = match.group(1)
            if name not in seen_companies and len(name) >= 4:
                seen_companies.add(name)
                entities.append({"type": "company", "data": {"name": name, "role": "涉及方"}})

        # 抽取金额
        amount_pattern = r'([\d,.]+)\s*(亿|万|千|百)?\s*(元|美元|港元|人民币)'
        for match in re.finditer(amount_pattern, text):
            entities.append({
                "type": "financial_figure",
                "data": {
                    "label": "金额",
                    "value": float(match.group(1).replace(",", "")),
                    "unit": (match.group(2) or "") + (match.group(3) or "元"),
                }
            })

        # 检测事件类型
        if re.search(r'收购|并购|兼并|入股|控股', text):
            entities.append({"type": "event", "data": {
                "type": "并购", "description": "涉及收购/并购事件", "impact": "中性"
            }})
        elif re.search(r'降准|降息|加息|存款准备金|利率调整|LPR', text):
            entities.append({"type": "event", "data": {
                "type": "政策调整", "description": "货币政策调整", "impact": "正面"
            }})
        elif re.search(r'年报|季报|业绩公告|业绩预告|业绩快报', text):
            entities.append({"type": "event", "data": {
                "type": "财报发布", "description": "财务报告发布", "impact": "中性"
            }})

        return entities

    # ===================== 查询生成 =====================

    def _generate_queries(self, documents: List[Dict], metrics: Dict, entities: List[Dict]) -> List[str]:
        """
        根据抽取结果生成多角度检索查询。

        生成策略：
        1. 基于财务指标生成查询
        2. 基于实体和事件生成查询
        3. 基于文档类型生成特定查询
        """
        queries = []

        # 合并文本用于上下文 — 保留更多原文信息
        combined_text = " ".join(d.get("text", "")[:500] for d in documents if d.get("text"))

        # 基于财务指标的查询
        if "revenue" in metrics:
            rev = metrics["revenue"]
            queries.append(f"营业收入 {rev.get('value', '')}{rev.get('unit', '')} 增长分析")
        if "net_income" in metrics:
            ni = metrics["net_income"]
            queries.append(f"净利润 {ni.get('value', '')}{ni.get('unit', '')} 同比变化")
        if "gross_margin" in metrics:
            gm = metrics["gross_margin"]
            queries.append(f"毛利率 {gm.get('value', '')}% 行业对比")
        if "eps" in metrics:
            eps = metrics["eps"]
            queries.append(f"每股收益 {eps.get('value', '')}元 估值分析")
        if "roe" in metrics:
            roe = metrics["roe"]
            queries.append(f"ROE {roe.get('value', '')}% 盈利质量评估")

        # 基于实体的查询
        companies_in_text = []
        for e in entities:
            if e.get("type") == "company":
                name = e.get("data", {}).get("name", "")
                if name and name not in companies_in_text:
                    companies_in_text.append(name)

        if companies_in_text:
            queries.append(f"{'、'.join(companies_in_text[:2])} 财务状况分析")

        # 基于事件类型的查询
        for e in entities:
            if e.get("type") == "event":
                event_data = e.get("data", {})
                event_type = event_data.get("type", "")
                event_desc = event_data.get("description", "")
                if event_type == "并购":
                    queries.append(f"并购事件影响分析: {event_desc[:30]}")
                elif "政策" in event_type:
                    queries.append(f"政策调整市场影响: {event_desc[:30]}")
                elif "财报" in event_type:
                    queries.append(f"财报核心指标解读")
                break

        # 通用查询（保底）
        if len(queries) < 2:
            # 从文档第一段生成
            first_para = combined_text[:100] if combined_text else "财经信息"
            queries.append(f"{first_para} 关键要点")

        # 去重并限制数量
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries[:5]
