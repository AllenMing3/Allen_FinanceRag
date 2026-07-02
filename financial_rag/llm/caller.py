"""
LLMCaller — 智能调用层，封装 DashScopeLLM 并添加:

1. 重试机制 (指数退避)
2. 结构化 JSON 输出 (自动解析 + 解析失败自动重试)
3. 输入长度校验 (token 预算警告)
4. 响应缓存 (哈希 + TTL)
5. 默认防幻觉约束 (仅在无 system prompt 时生效)

用法:
    from financial_rag.llm.caller import LLMCaller, get_caller

    caller = LLMCaller(llm)
    # 普通调用（带重试）
    resp = caller.call("分析一下这段文本")

    # 结构化 JSON 调用（自动解析 + 解析失败重试）
    data = caller.call_json("提取指标...", system="你是指标提取专家")
"""
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .dashscope_client import DashScopeLLM, LLMResponse

logger = logging.getLogger(__name__)


# ===================== 默认约束 =====================

DEFAULT_SYSTEM_CONSTRAINTS = (
    "基于提供的信息回答，不要编造任何数据或数字。"
    "如果信息不足，明确说明'数据不足，无法判断'。"
    "不要推测未来走势，只做基于现有信息的客观分析。"
)

JSON_OUTPUT_HINT = "\n\n重要：只输出合法的 JSON，不要添加任何额外文字、解释或 Markdown 标记。"


# ===================== JSON 解析工具 =====================

def parse_json_from_text(text: str) -> Optional[Dict]:
    """从 LLM 响应文本中提取 JSON 对象 — 兼容裸 JSON / Markdown 代码块。

    这是共享工具，替代各 tool 文件中重复的 _parse_json_from_text()。
    使用平衡括号匹配而非贪婪 regex，避免跨多个 JSON 对象误匹配。
    """
    if not text:
        return None

    # 尝试 Markdown 代码块
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 平衡括号匹配：找到第一个 { 然后用计数器找对应的 }
    result = _extract_balanced_json(text)
    if result is not None:
        return result

    return None


def _extract_balanced_json(text: str) -> Optional[Dict]:
    """用平衡括号算法提取第一个完整的 JSON 对象"""
    start = text.find('{')
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    return None

    return None


def parse_json_list_from_text(text: str) -> Optional[List]:
    """从 LLM 响应文本中提取 JSON 数组。"""
    if not text:
        return None

    # 尝试 Markdown 代码块
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_match:
        try:
            result = json.loads(code_match.group(1).strip())
            return result if isinstance(result, list) else [result]
        except (json.JSONDecodeError, ValueError):
            pass

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ===================== 缓存 =====================

@dataclass
class _CacheEntry:
    timestamp: float
    response: LLMResponse


# ===================== LLMCaller =====================

class LLMCaller:
    """智能 LLM 调用器 — 在 DashScopeLLM 之上叠加保护层。

    Args:
        llm: DashScopeLLM 实例
        max_retries: 最大重试次数（API 调用失败时）
        cache_ttl: 缓存有效期（秒），0 表示不缓存
        max_input_chars: 输入字符数警告阈值
    """

    _cache: Dict[str, _CacheEntry] = {}  # shared across instances (process-level cache)

    def __init__(
        self,
        llm: DashScopeLLM,
        max_retries: int = 2,
        cache_ttl: int = 300,
        max_input_chars: int = 30000,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl
        self.max_input_chars = max_input_chars

    # ---------- 普通调用 ----------

    def call(
        self,
        messages: Union[str, List[Dict]],
        system: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """调用 LLM，带重试 + 缓存 + 输入长度校验。

        Args:
            messages: 用户消息（str 或 messages 列表）
            system: system prompt，None 时使用默认防幻觉约束
            use_cache: 是否使用缓存
            **kwargs: 透传给 llm.chat() (temperature, max_tokens, top_p)

        Returns:
            LLMResponse
        """
        # 统一使用 effective_system（None → 默认约束）
        effective_system = system if system is not None else DEFAULT_SYSTEM_CONSTRAINTS

        # 1. 构造完整 messages 列表（用于哈希和长度校验）
        full_messages = self._build_messages(messages, effective_system)

        # 2. 输入长度校验
        self._check_input_length(full_messages)

        # 3. 缓存查询
        if use_cache and self.cache_ttl > 0:
            cache_key = self._hash(full_messages, kwargs)
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug("[LLMCaller] 缓存命中")
                return cached

        # 4. 带重试的 API 调用（始终传 effective_system）
        resp = self._call_with_retry(messages, effective_system, **kwargs)

        # 5. 写入缓存
        if use_cache and self.cache_ttl > 0:
            self._set_cached(cache_key, resp)

        return resp

    # ---------- 结构化 JSON 调用 ----------

    def call_json(
        self,
        messages: Union[str, List[Dict]],
        system: Optional[str] = None,
        max_json_retries: int = 2,
        use_cache: bool = True,
        **kwargs,
    ) -> Optional[Union[Dict, List]]:
        """调用 LLM 并期望 JSON 输出，解析失败自动重试。

        Args:
            messages: 用户消息
            system: system prompt（自动追加 JSON 输出提示）
            max_json_retries: JSON 解析失败时最大重试次数
            use_cache: 是否使用缓存
            **kwargs: 透传给 llm.chat()

        Returns:
            解析后的 Dict/List，或 None（全部重试耗尽仍无法解析）
        """
        # 追加 JSON 输出约束到 system prompt
        json_system = (system or DEFAULT_SYSTEM_CONSTRAINTS) + JSON_OUTPUT_HINT

        for attempt in range(max_json_retries + 1):
            # 每次重试禁用缓存（避免拿到相同的错误响应）
            resp = self.call(
                messages,
                system=json_system,
                use_cache=(use_cache and attempt == 0),
                **kwargs,
            )

            # 尝试解析 dict
            result = parse_json_from_text(resp.content)
            if result is not None:
                return result

            # 尝试解析 list
            list_result = parse_json_list_from_text(resp.content)
            if list_result is not None:
                return list_result

            logger.warning(
                f"[LLMCaller] JSON 解析失败 (attempt {attempt + 1}/{max_json_retries + 1}): "
                f"content={resp.content[:120]!r}"
            )

        logger.error("[LLMCaller] JSON 解析全部重试耗尽，返回 None")
        return None

    # ---------- Function Calling 调用 ----------

    def call_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: str = "auto",
        use_cache: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """Function Calling 调用（带重试，不缓存 — tool call 结果动态性强）。

        Args:
            messages: 对话历史
            tools: 工具定义列表
            tool_choice: "auto" | "required" | "none"
            use_cache: 是否缓存（默认关闭）
            **kwargs: 透传

        Returns:
            LLMResponse（含 tool_calls）
        """
        # 长度校验
        self._check_input_length(messages)

        if use_cache and self.cache_ttl > 0:
            cache_key = self._hash(messages, {"tools": tools, **kwargs})
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        resp = self._call_tools_with_retry(messages, tools, tool_choice, **kwargs)

        if use_cache and self.cache_ttl > 0:
            self._set_cached(cache_key, resp)

        return resp

    # ---------- 内部方法 ----------

    def _build_messages(
        self,
        messages: Union[str, List[Dict]],
        system: Optional[str],
    ) -> List[Dict]:
        """构造完整 messages 列表（含 system）"""
        effective_system = system if system is not None else DEFAULT_SYSTEM_CONSTRAINTS

        if isinstance(messages, str):
            msgs = [{"role": "system", "content": effective_system},
                    {"role": "user", "content": messages}]
        else:
            msgs = list(messages)
            # 如果列表中没有 system 消息，插入
            has_system = any(m.get("role") == "system" for m in msgs)
            if not has_system:
                msgs.insert(0, {"role": "system", "content": effective_system})

        return msgs

    def _check_input_length(self, messages: List[Dict]) -> None:
        """输入长度警告 — 仅记录日志，不阻断调用"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        if total_chars > self.max_input_chars:
            logger.warning(
                f"[LLMCaller] 输入长度 {total_chars} 超过阈值 {self.max_input_chars}，"
                f"可能超出模型上下文窗口"
            )

    def _call_with_retry(
        self,
        messages: Union[str, List[Dict]],
        system: Optional[str],
        **kwargs,
    ) -> LLMResponse:
        """带指数退避重试的 LLM 调用"""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return self.llm.chat(
                    messages=messages,
                    system=system,
                    **kwargs,
                )
            except RuntimeError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s...
                    logger.warning(
                        f"[LLMCaller] API 调用失败 (attempt {attempt + 1}/{self.max_retries + 1}): {e}，"
                        f"{wait:.1f}s 后重试"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"[LLMCaller] API 调用失败且重试耗尽 ({self.max_retries} 次): {last_error}"
        ) from last_error

    def _call_tools_with_retry(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: str,
        **kwargs,
    ) -> LLMResponse:
        """Function Calling 带重试"""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return self.llm.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            except RuntimeError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"[LLMCaller] Function Calling 失败 (attempt {attempt + 1}): {e}，"
                        f"{wait:.1f}s 后重试"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"[LLMCaller] Function Calling 重试耗尽: {last_error}"
        ) from last_error

    # ---------- 缓存实现 ----------

    def _hash(self, messages: List[Dict], extra: dict) -> str:
        """计算请求哈希用于缓存（包含 system prompt 在内已由 _build_messages 纳入 messages）"""
        content = json.dumps(
            {"messages": messages, "extra": {k: v for k, v in extra.items() if k not in ("tools",)}},
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[LLMResponse]:
        """从缓存获取（检查 TTL）"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.timestamp > self.cache_ttl:
            del self._cache[key]
            return None
        return entry.response

    def _set_cached(self, key: str, response: LLMResponse) -> None:
        """写入缓存"""
        self._cache[key] = _CacheEntry(timestamp=time.time(), response=response)
        # 限制缓存大小（简单 LRU：超过 100 条清空一半）
        if len(self._cache) > 100:
            sorted_keys = sorted(self._cache, key=lambda k: self._cache[k].timestamp)
            for k in sorted_keys[:50]:
                del self._cache[k]

    def clear_cache(self) -> int:
        """清空缓存，返回清除条数"""
        count = len(self._cache)
        self._cache.clear()
        return count


# ===================== 便捷工厂 =====================

def get_caller(llm: DashScopeLLM, **kwargs) -> LLMCaller:
    """快速构造 LLMCaller"""
    return LLMCaller(llm, **kwargs)
