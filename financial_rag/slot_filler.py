"""
槽位填充引擎 — 模板驱动的结构化输出，降低首 token 延迟

核心思路:
- 长文本自由生成 → 拆成 N 个短槽位
- 每个槽位 max_tokens 控制在 30~100，LLM 只需输出非常短的内容
- 首 token 延迟从 2~5s 降至 0.3~0.8s
- 同阶段槽位可并行填充 (ThreadPoolExecutor)
- 每个槽位独立打分，纳入全链路评分卡

用法:
    filler = SlotFiller(llm, scorecard=card)
    results = filler.fill(template, query, context_docs)
    final_output = filler.render(template, results)
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ===================== 填充结果 =====================

@dataclass
class SlotResult:
    """单个槽位的填充结果"""
    key: str                    # 槽位 key
    label: str                  # 显示名
    value: str                  # 填充后的值
    raw_value: str = ""         # LLM 原始输出（未清洗）
    score: float = 1.0          # 该槽位质量分 0~1
    elapsed_ms: float = 0.0     # 总耗时
    ttft_ms: float = 0.0        # 首 token 延迟（近似）
    token_count: int = 0        # 输出 token 数
    success: bool = True        # 是否成功
    error: str = ""             # 错误信息
    phase: int = 0              # 所属阶段
    metrics: Dict = field(default_factory=dict)  # 额外指标

    @property
    def filled(self) -> bool:
        return bool(self.value and self.value.strip() and self.success)


# ===================== 填充统计 =====================

@dataclass
class FillStats:
    """一次完整填充的统计数据"""
    template_name: str = ""
    total_slots: int = 0
    filled_slots: int = 0
    total_elapsed_ms: float = 0.0
    total_ttft_ms: float = 0.0        # 所有槽位首 token 延迟总和
    avg_ttft_ms: float = 0.0          # 平均首 token 延迟
    peak_ttft_ms: float = 0.0         # 最慢的首 token 延迟
    total_tokens: int = 0
    phases: int = 0
    parallel_gain: float = 0.0        # 并行收益（节省的时间比例）
    slot_results: Dict[str, SlotResult] = field(default_factory=dict)


# ===================== 填充引擎 =====================

class SlotFiller:
    """
    槽位填充引擎

    工作流程:
    1. 按 phase 顺序执行（同一 phase 内并行填充无依赖槽位）
    2. 每个槽位: 构建短 prompt → LLM chat → 收集结果 → 质量评分
    3. 前期槽位的结果自动注入后续依赖槽位的 prompt
    4. 与 PipelineScoreCard 集成，记录每个槽位的评分
    """

    # 首 token 延迟预估系数：总耗时 / token 数的经验比例
    TTFT_ESTIMATE_RATIO = 0.15  # 假设首token约占总耗时的15%

    def __init__(
        self,
        llm: Any,                              # DashScopeLLM 实例
        scorecard: Any = None,                 # PipelineScoreCard 实例
        max_workers: int = 6,                  # 并行填充线程数
        system_prompt: str = "",
        verbose: bool = False,
    ):
        self.llm = llm
        self.scorecard = scorecard
        self.max_workers = max_workers
        self.system_prompt = system_prompt or (
            "你是专业的金融分析师。只输出要求的内容，不要额外解释。"
        )
        self.verbose = verbose

    # ===================== 主入口 =====================

    def fill(
        self,
        template: "SlottedTemplate",        # from financial_rag.templates
        query: str,
        context_docs: List[str],            # 检索到的文档文本列表
    ) -> FillStats:
        """
        按模板填充所有槽位

        Args:
            template: 槽位模板
            query: 用户原始查询
            context_docs: 检索到的文档内容列表
        Returns:
            FillStats 包含所有填充结果和统计
        """
        t0 = time.time()
        all_results: Dict[str, SlotResult] = {}
        stats = FillStats(
            template_name=template.name,
            total_slots=len(template.slots),
            phases=len(template.phases),
        )

        # 按 Phase 执行
        for phase_idx, phase_keys in enumerate(template.phases):
            # 构建此 Phase 的上下文（包含前序槽位结果）
            prev_context = self._build_prev_context(all_results)

            # Phase 内并行填充
            phase_results = self._fill_phase(
                phase_idx=phase_idx,
                slot_keys=phase_keys,
                template=template,
                query=query,
                context_docs=context_docs,
                prev_context=prev_context,
            )
            all_results.update(phase_results)

            if self.verbose:
                for key, r in phase_results.items():
                    status = "OK" if r.filled else "FAIL"
                    logger.info(
                        f"[SlotFiller] Phase{phase_idx} {template.name}/{key} "
                        f"= {status} ({r.elapsed_ms:.0f}ms, TTFT~{r.ttft_ms:.0f}ms)"
                    )

        # 汇总统计
        stats.slot_results = all_results
        stats.filled_slots = sum(1 for r in all_results.values() if r.filled)
        stats.total_elapsed_ms = (time.time() - t0) * 1000

        # 计算 TTFT 统计
        ttfts = [r.ttft_ms for r in all_results.values() if r.ttft_ms > 0]
        if ttfts:
            stats.total_ttft_ms = sum(ttfts)
            stats.avg_ttft_ms = sum(ttfts) / len(ttfts)
            stats.peak_ttft_ms = max(ttfts)

        stats.total_tokens = sum(r.token_count for r in all_results.values())

        # 并行收益 = 如果全部串行耗时 - 实际耗时
        serial_estimated = sum(r.elapsed_ms for r in all_results.values())
        stats.parallel_gain = max(0.0, 1.0 - stats.total_elapsed_ms / max(serial_estimated, 1))

        # 记录到打分卡
        if self.scorecard:
            self._record_to_scorecard(template, all_results, stats)

        return stats

    # ===================== Phase 填充 =====================

    def _fill_phase(
        self,
        phase_idx: int,
        slot_keys: List[str],
        template: "SlottedTemplate",
        query: str,
        context_docs: List[str],
        prev_context: str,
    ) -> Dict[str, SlotResult]:
        """并行填充一个 Phase 内的所有槽位"""
        results: Dict[str, SlotResult] = {}

        # 构建并行任务
        tasks = []
        for key in slot_keys:
            slot_def = template.get_slot(key)
            if slot_def is None:
                logger.warning(f"槽位 {key} 不在模板 {template.name} 中")
                continue

            # 检查依赖是否已满足（理论上按 phase 执行不会出问题）
            deps_unmet = [d for d in slot_def.depends_on
                         if d not in results and d not in results]

            if deps_unmet and prev_context:
                # 依赖的前序槽位已填充，但在 prev_context 中
                pass

            tasks.append((key, slot_def))

        # 并行执行（如果只有一个任务，直接执行避免线程开销）
        if len(tasks) == 1:
            key, slot_def = tasks[0]
            result = self._fill_single(
                key=key,
                slot_def=slot_def,
                phase_idx=phase_idx,
                query=query,
                context_docs=context_docs,
                prev_context=prev_context,
            )
            results[key] = result
        elif len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as executor:
                futures = {}
                for key, slot_def in tasks:
                    fut = executor.submit(
                        self._fill_single,
                        key=key,
                        slot_def=slot_def,
                        phase_idx=phase_idx,
                        query=query,
                        context_docs=context_docs,
                        prev_context=prev_context,
                    )
                    futures[fut] = key

                for fut in as_completed(futures):
                    key = futures[fut]
                    try:
                        results[key] = fut.result(timeout=60)
                    except Exception as e:
                        results[key] = SlotResult(
                            key=key,
                            label=template.get_slot(key).label if template.get_slot(key) else key,
                            value="",
                            score=0.0,
                            success=False,
                            error=str(e),
                            phase=phase_idx,
                        )
                        logger.error(f"槽位 {key} 填充异常: {e}")

        return results

    # ===================== 单个槽位填充 =====================

    def _fill_single(
        self,
        key: str,
        slot_def: "SlotDef",
        phase_idx: int,
        query: str,
        context_docs: List[str],
        prev_context: str,
    ) -> SlotResult:
        """填充单个槽位"""
        t_slot = time.time()

        # 构建 Prompt
        prompt = self._build_slot_prompt(
            slot_def=slot_def,
            query=query,
            context_docs=context_docs,
            prev_context=prev_context,
        )

        try:
            # 调用 LLM（使用流式以测量 TTFT）
            raw_value, ttft_ms, token_count = self._call_llm_with_ttft(
                prompt=prompt,
                system=self.system_prompt,
                max_tokens=slot_def.max_tokens,
            )

            # 清洗输出
            clean_value = self._clean_slot_value(raw_value, slot_def)

            elapsed_ms = (time.time() - t_slot) * 1000

            # 质量评分
            score = self._score_slot(slot_def, clean_value)

            return SlotResult(
                key=key,
                label=slot_def.label,
                value=clean_value,
                raw_value=raw_value,
                score=score,
                elapsed_ms=elapsed_ms,
                ttft_ms=ttft_ms,
                token_count=token_count,
                success=bool(clean_value.strip()),
                phase=phase_idx,
            )

        except Exception as e:
            elapsed_ms = (time.time() - t_slot) * 1000
            logger.warning(f"槽位 {key} 填充失败: {e}，使用默认值")
            return SlotResult(
                key=key,
                label=slot_def.label,
                value=slot_def.default,
                score=0.1,
                elapsed_ms=elapsed_ms,
                success=bool(slot_def.default),
                error=str(e),
                phase=phase_idx,
            )

    # ===================== Prompt 构建 =====================

    def _build_slot_prompt(
        self,
        slot_def: "SlotDef",
        query: str,
        context_docs: List[str],
        prev_context: str,
    ) -> str:
        """为单个槽位构建短 prompt"""
        parts = []

        # 上下文（检索到的文档） — 完整内容交给 LLM 深入分析
        if context_docs:
            short_context = "\n---\n".join(doc for doc in context_docs[:5])
            parts.append(f"参考信息:\n{short_context}")

        # 用户问题
        if query:
            parts.append(f"用户问题: {query}")

        # 前序槽位结果（供依赖使用）
        if prev_context:
            parts.append(f"已知信息:\n{prev_context}")

        # 槽位指令
        parts.append(slot_def.prompt)

        return "\n\n".join(parts)

    def _build_prev_context(self, results: Dict[str, SlotResult]) -> str:
        """将已填充的槽位结果构建为上下文字符串"""
        if not results:
            return ""
        lines = []
        for key, r in results.items():
            if r.filled:
                lines.append(f"- {r.label}: {r.value}")
        return "\n".join(lines)

    # ===================== LLM 调用 =====================

    def _call_llm_with_ttft(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> Tuple[str, float, int]:
        """
        调用 LLM 并测量首 token 延迟

        优先用流式 API 精确测量 TTFT，回退到预估
        Returns:
            (content, ttft_ms, token_count)
        """
        # 尝试流式调用
        if hasattr(self.llm, 'chat_stream'):
            try:
                content_parts = []
                ttft_ms = 0.0
                t_call = time.time()

                for delta, finish_reason in self.llm.chat_stream(
                    messages=prompt,
                    system=system,
                    max_tokens=max_tokens,
                ):
                    if ttft_ms == 0.0 and delta:
                        ttft_ms = (time.time() - t_call) * 1000
                    content_parts.append(delta)

                content = "".join(content_parts)
                # 粗略估算 token 数（中文: 1字≈1token, 英文: 1word≈1.3token）
                token_count = max(1, len(content)) if any('\u4e00' <= c <= '\u9fff' for c in content) \
                    else max(1, len(content.split()))
                return content.strip(), ttft_ms, token_count
            except Exception:
                pass  # 流式失败，回退

        # 回退：非流式 + TTFT 预估
        response = self.llm.chat(
            messages=prompt,
            system=system,
            max_tokens=max_tokens,
        )
        content = response.content.strip()
        token_count = response.usage.get('total_tokens', max(1, len(content)))
        # 首 token 延迟预估（总耗时 * 系数）
        ttft_ms = token_count * self.TTFT_ESTIMATE_RATIO * 30  # 经验公式

        return content, ttft_ms, token_count

    # ===================== 清洗 =====================

    def _clean_slot_value(self, raw: str, slot_def: "SlotDef") -> str:
        """清洗 LLM 输出"""
        if not raw:
            return slot_def.default

        value = raw.strip()

        # 去除常见的 LLM 废话前缀
        prefixes_to_strip = [
            "好的，", "根据参考信息，", "综上所述，", "分析如下：",
            "以下是", "回答：", "答案是", "根据以上信息，",
            "OK，", "好的。", "根据您提供的信息，",
        ]
        for pfx in prefixes_to_strip:
            if value.startswith(pfx):
                value = value[len(pfx):].strip()
                break

        # 去除引号包裹
        for q in ['"', '"', '"', "'", "'", "'"]:
            if value.startswith(q) and value.endswith(q) and len(value) > 2:
                value = value[1:-1]

        return value.strip()

    # ===================== 槽位评分 =====================

    def _score_slot(self, slot_def: "SlotDef", value: str) -> float:
        """对填充结果评分 (0~1)"""
        if not value:
            return 0.0

        score = 1.0

        # 检查是否输出了合理的长度
        vlen = len(value)
        if vlen < 3:
            score -= 0.5
        if vlen > slot_def.max_tokens * 3:
            # 输出过长（可能废话多）
            score -= 0.2

        # 检查是否包含拒绝回答的标记
        refuse_markers = ["无法确定", "没有足够信息", "我不知道", "无法判断",
                          "抱歉", "对不起", "信息不足"]
        for m in refuse_markers:
            if m in value:
                score -= 0.3
                break

        return max(0.0, min(1.0, score))

    # ===================== 渲染 =====================

    def render(self, template: "SlottedTemplate", stats: FillStats) -> str:
        """
        将填充结果渲染为最终文本

        处理：
        - {slot_key} → 填充值
        - 未填充的槽位显示为友好占位符
        """
        output = template.render
        for key, result in stats.slot_results.items():
            placeholder = "{" + key + "}"
            if result.filled:
                output = output.replace(placeholder, result.value)
            else:
                slot_def = template.get_slot(key)
                label = slot_def.label if slot_def else key
                output = output.replace(placeholder, f"（{label}暂缺）")

        return output.strip()

    # ===================== 打分卡集成 =====================

    def _record_to_scorecard(
        self,
        template: "SlottedTemplate",
        results: Dict[str, SlotResult],
        stats: FillStats,
    ):
        """将槽位填充结果写入 PipelineScoreCard"""
        if not self.scorecard:
            return

        # 记录每个槽位
        for key, r in results.items():
            score_val = r.score
            elapsed = r.elapsed_ms
            details = {
                "ttft_ms": round(r.ttft_ms, 1),
                "tokens": r.token_count,
                "phase": r.phase,
                "template": template.name,
            }
            diag = "" if r.success else f"槽位 {r.label} 填充失败: {r.error}"

            self.scorecard.record(
                stage=f"slot_{key}",
                display_name=f"[槽位] {r.label}",
                score=score_val,
                elapsed_ms=elapsed,
                details=details,
                diagnosis=diag,
            )

        # 记录整体槽位填充汇总
        from financial_rag.core.scorer import PipelineScoreCard
        self.scorecard.record(
            stage="slot_filling_summary",
            display_name="槽位填充汇总",
            score=stats.filled_slots / max(stats.total_slots, 1),
            elapsed_ms=stats.total_elapsed_ms,
            details={
                "template": template.name,
                "filled": stats.filled_slots,
                "total": stats.total_slots,
                "avg_ttft_ms": round(stats.avg_ttft_ms, 1),
                "peak_ttft_ms": round(stats.peak_ttft_ms, 1),
                "parallel_gain": round(stats.parallel_gain, 2),
            },
        )


# ===================== 便捷函数 =====================

def create_slot_filler(
    llm: Any,
    scorecard: Any = None,
    max_workers: int = 6,
    verbose: bool = False,
) -> SlotFiller:
    """创建槽位填充器"""
    return SlotFiller(
        llm=llm,
        scorecard=scorecard,
        max_workers=max_workers,
        verbose=verbose,
    )
