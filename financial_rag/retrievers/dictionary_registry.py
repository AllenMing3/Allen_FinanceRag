"""
DictionaryRegistry — 统一业务字典管理

职责:
1. 管理所有领域词表 (股票映射、金融术语、同义词、概念关联等)
2. 从外部 JSON 文件加载并合并 (热扩展，不改源码)
3. 自动注入 jieba 分词器
4. 运行时增词 API
5. 字典覆盖率可视化 (哪里弱一目了然)

使用:
    from financial_rag.retrievers.dictionary_registry import get_registry
    reg = get_registry()
    print(reg.summary())          # 一眼看清覆盖情况
    reg.add_words("stocks", {"寒武纪": ("688256.SH", "寒武纪")})
    reg.save_external()           # 持久化到 JSON
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# 默认外部词典目录
_DEFAULT_DICT_DIR = Path(__file__).parent.parent.parent / "data" / "dictionaries"


class DictionaryRegistry:
    """
    统一业务字典注册中心

    管理 10 种字典类型，支持内置 + 外部 JSON 合并:

    集合类 (set):
      financial_terms  — 金融术语 (QueryParser 关键词, weight=2.0)
      industry_terms   — 行业/主题词 (QueryParser 关键词, weight=1.5)
      action_terms     — 动作词 (查询分类, weight=1.0)
      stop_words       — 停用词

    字典类 (dict):
      stock_map        — {关键词: (ts_code, 名称)}
      concept_map      — {概念: [关联词列表]}
      doc_type_keywords — {文档类型: {关键词集合}}
      doc_type_patterns — {文档类型: [正则列表]}

    查找表 (dict, 自动生成):
      synonym_lookup   — {小写词: frozenset(同义词组)}

    列表类 (list):
      jieba_words      — jieba 专用扩展词

    外部 JSON 格式:
      {
        "stocks": {"茅台": ["600519.SH", "贵州茅台"]},
        "financial_terms": ["营收", "净利润"],
        "industry_terms": ["AI", "大模型"],
        "jieba_words": ["大模型", "算力"],
        "synonym_groups": [["寒武纪", "Cambricon", "688256"]],
        "concept_map": {"算力": ["GPU", "芯片"]},
        "doc_type_keywords": {"news": ["记者"]},
        "doc_type_patterns": {"query": ["\\\\？$"]}
      }
    """

    def __init__(self):
        # 集合类
        self._sets: Dict[str, Set[str]] = {}
        # 字典类
        self._dicts: Dict[str, Dict] = {}
        # 列表类
        self._lists: Dict[str, list] = {}
        # 已加载的外部文件
        self._loaded_files: List[str] = []
        # jieba 引用 (延迟设置)
        self._jieba = None

    # ===================== 内置注册 =====================

    def register_builtins(self, **named_collections):
        """注册内置词表 (来自 dictionaries.py)

        Args:
            financial_terms=FINANCIAL_TERMS (set)
            stock_map=STOCK_MAP (dict)
            jieba_words=JIEBA_FINANCE_WORDS (list)
            ...
        """
        _set_types = {
            "financial_terms", "industry_terms", "action_terms", "stop_words",
        }
        _dict_types = {
            "stock_map", "synonym_lookup", "concept_map",
            "doc_type_keywords", "doc_type_patterns",
        }

        for name, data in named_collections.items():
            if name in _set_types:
                self._sets[name] = set(data) if data else set()
            elif name in _dict_types:
                self._dicts[name] = dict(data) if data else {}
            elif name == "jieba_words":
                self._lists["jieba_words"] = list(data) if data else []
            else:
                # 未知类型: 按实际类型存放
                if isinstance(data, set):
                    self._sets[name] = data
                elif isinstance(data, dict):
                    self._dicts[name] = data
                elif isinstance(data, list):
                    self._lists[name] = data

    # ===================== 外部加载 =====================

    def load_external(self, directory: str = None) -> int:
        """从目录加载所有 .json 文件，合并到现有字典

        Returns:
            成功加载的文件数
        """
        dir_path = Path(directory) if directory else _DEFAULT_DICT_DIR
        if not dir_path.exists():
            logger.debug(f"DictionaryRegistry: 外部词典目录不存在: {dir_path}")
            return 0

        count = 0
        for json_file in sorted(dir_path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._merge_external(data, json_file.name)
                self._loaded_files.append(str(json_file))
                count += 1
                logger.info(f"DictionaryRegistry: 已加载 {json_file.name}")
            except Exception as e:
                logger.warning(f"DictionaryRegistry: 加载失败 {json_file.name}: {e}")

        return count

    def _merge_external(self, data: dict, filename: str):
        """合并单个外部文件的数据"""
        # --- 集合类: 直接 update ---
        for key in ("financial_terms", "industry_terms", "action_terms", "stop_words"):
            if key in data and isinstance(data[key], list):
                self._sets.setdefault(key, set()).update(data[key])

        # --- stock_map: {keyword: [ts_code, name]} → {keyword: (ts_code, name)} ---
        if "stocks" in data and isinstance(data["stocks"], dict):
            for kw, val in data["stocks"].items():
                if isinstance(val, list) and len(val) == 2:
                    self._dicts.setdefault("stock_map", {})[kw] = tuple(val)
                elif isinstance(val, (tuple, list)) and len(val) == 2:
                    self._dicts["stock_map"][kw] = tuple(val)

        # --- jieba_words: 列表，去重追加 ---
        if "jieba_words" in data and isinstance(data["jieba_words"], list):
            existing = self._lists.setdefault("jieba_words", [])
            seen = set(existing)
            for w in data["jieba_words"]:
                if w not in seen:
                    existing.append(w)
                    seen.add(w)

        # --- synonym_groups: [[词1, 词2, ...]] → 构建 synonym_lookup ---
        if "synonym_groups" in data and isinstance(data["synonym_groups"], list):
            lookup = self._dicts.setdefault("synonym_lookup", {})
            for group in data["synonym_groups"]:
                if isinstance(group, list) and len(group) >= 2:
                    frozen = frozenset(group)
                    for term in group:
                        lookup[term.lower()] = frozen

        # --- concept_map: {概念: [关联词]} ---
        if "concept_map" in data and isinstance(data["concept_map"], dict):
            cm = self._dicts.setdefault("concept_map", {})
            for concept, terms in data["concept_map"].items():
                if isinstance(terms, list):
                    cm.setdefault(concept, []).extend(
                        t for t in terms if t not in cm[concept]
                    )

        # --- doc_type_keywords: {类型: [词]} ---
        if "doc_type_keywords" in data and isinstance(data["doc_type_keywords"], dict):
            dtk = self._dicts.setdefault("doc_type_keywords", {})
            for dtype, words in data["doc_type_keywords"].items():
                if isinstance(words, list):
                    dtk.setdefault(dtype, set()).update(words)

        # --- doc_type_patterns: {类型: [正则]} ---
        if "doc_type_patterns" in data and isinstance(data["doc_type_patterns"], dict):
            dtp = self._dicts.setdefault("doc_type_patterns", {})
            for dtype, patterns in data["doc_type_patterns"].items():
                if isinstance(patterns, list):
                    existing = dtp.setdefault(dtype, [])
                    for p in patterns:
                        if p not in existing:
                            existing.append(p)

    # ===================== jieba 注入 =====================

    def set_jieba(self, jieba_module):
        """设置 jieba 引用并注入所有已注册词 (幂等)"""
        if self._jieba is jieba_module:
            return  # 已设置，不重复注入
        self._jieba = jieba_module
        words = self._lists.get("jieba_words", [])
        if words:
            jieba_module.setLogLevel(20)
            for w in words:
                jieba_module.add_word(w)
            logger.info(f"DictionaryRegistry: 注入 {len(words)} 词到 jieba")

    def inject_jieba_word(self, word: str):
        """运行时向 jieba 追加单个词"""
        if self._jieba and word:
            self._jieba.add_word(word)

    # ===================== 运行时增词 API =====================

    def add_words(self, category: str, words, source: str = "runtime") -> str:
        """运行时添加词

        Args:
            category: stocks / financial_terms / industry_terms / jieba_words /
                      synonyms / concepts / action_terms / stop_words
            words: 数据 (格式因类型而异)
            source: 来源标记

        Returns:
            操作描述
        """
        if category == "stocks" and isinstance(words, dict):
            for kw, val in words.items():
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    self._dicts.setdefault("stock_map", {})[kw] = tuple(val)
            self.inject_jieba_word_list(words.keys())
            return f"stocks +{len(words)} from {source}"

        elif category in ("financial_terms", "industry_terms",
                          "action_terms", "stop_words"):
            if isinstance(words, (list, set, tuple)):
                self._sets.setdefault(category, set()).update(words)
                return f"{category} +{len(words)} from {source}"

        elif category == "jieba_words" and isinstance(words, list):
            existing = self._lists.setdefault("jieba_words", [])
            seen = set(existing)
            added = 0
            for w in words:
                if w not in seen:
                    existing.append(w)
                    seen.add(w)
                    self.inject_jieba_word(w)
                    added += 1
            return f"jieba_words +{added} from {source}"

        elif category == "synonyms" and isinstance(words, dict):
            # words = {trigger: [同义词列表]}
            lookup = self._dicts.setdefault("synonym_lookup", {})
            for trigger, syns in words.items():
                frozen = frozenset(syns)
                for t in syns:
                    lookup[t.lower()] = frozen
            return f"synonyms +{len(words)} groups from {source}"

        elif category == "concepts" and isinstance(words, dict):
            cm = self._dicts.setdefault("concept_map", {})
            for concept, terms in words.items():
                cm.setdefault(concept, []).extend(
                    t for t in terms if t not in cm[concept]
                )
            return f"concepts +{len(words)} from {source}"

        return f"unknown category: {category}"

    def inject_jieba_word_list(self, words):
        """批量注入 jieba 词"""
        for w in words:
            self.inject_jieba_word(w)

    # ===================== 持久化 =====================

    def save_external(self, filename: str = "custom.json",
                      directory: str = None):
        """将当前全量数据保存为外部 JSON (可用于备份/迁移)"""
        dir_path = Path(directory) if directory else _DEFAULT_DICT_DIR
        dir_path.mkdir(parents=True, exist_ok=True)

        data = {
            "stocks": {
                k: list(v) for k, v in self._dicts.get("stock_map", {}).items()
            },
            "financial_terms": sorted(self._sets.get("financial_terms", set())),
            "industry_terms": sorted(self._sets.get("industry_terms", set())),
            "action_terms": sorted(self._sets.get("action_terms", set())),
            "stop_words": sorted(self._sets.get("stop_words", set())),
            "jieba_words": self._lists.get("jieba_words", []),
            "synonym_groups": self._export_synonym_groups(),
            "concept_map": self._dicts.get("concept_map", {}),
        }

        path = dir_path / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"DictionaryRegistry: 已保存全量词典到 {path}")

    def _export_synonym_groups(self) -> List[List[str]]:
        """从 synonym_lookup 反推 synonym_groups"""
        seen = set()
        groups = []
        for _, frozen in self._dicts.get("synonym_lookup", {}).items():
            key = id(frozen)
            if key not in seen:
                seen.add(key)
                groups.append(sorted(frozen))
        return groups

    # ===================== 访问器 =====================

    def get(self, name: str):
        """获取字典数据"""
        if name in self._sets:
            return self._sets[name]
        if name in self._dicts:
            return self._dicts[name]
        if name in self._lists:
            return self._lists[name]
        return None

    # ===================== 可视化 =====================

    def stats(self) -> Dict:
        """各字典规模统计 — 一眼看出哪里弱"""
        result = {}
        for name, data in self._sets.items():
            result[name] = {"count": len(data), "type": "set"}
        for name, data in self._dicts.items():
            result[name] = {"count": len(data), "type": "dict"}
        for name, data in self._lists.items():
            result[name] = {"count": len(data), "type": "list"}
        result["_external_files"] = list(self._loaded_files)
        return result

    def summary(self) -> str:
        """人可读的覆盖率摘要"""
        s = self.stats()
        lines = [
            "=== DictionaryRegistry 覆盖率 ===",
            f"  stock_map:          {s.get('stock_map', {}).get('count', 0):>4} 条 "
            f"(关键词→股票代码映射)",
            f"  financial_terms:    {s.get('financial_terms', {}).get('count', 0):>4} 个 "
            f"(高权重, BM25 weight=2.0)",
            f"  industry_terms:     {s.get('industry_terms', {}).get('count', 0):>4} 个 "
            f"(中权重, BM25 weight=1.5)",
            f"  action_terms:       {s.get('action_terms', {}).get('count', 0):>4} 个 "
            f"(查询分类)",
            f"  synonym_lookup:     {s.get('synonym_lookup', {}).get('count', 0):>4} 条 "
            f"(同义词→扩展词)",
            f"  concept_map:        {s.get('concept_map', {}).get('count', 0):>4} 组 "
            f"(概念→关联词)",
            f"  jieba_words:        {s.get('jieba_words', {}).get('count', 0):>4} 个 "
            f"(jieba 分词扩展)",
            f"  doc_type_keywords:  {s.get('doc_type_keywords', {}).get('count', 0):>4} 类 "
            f"(文档分类)",
            f"  stop_words:         {s.get('stop_words', {}).get('count', 0):>4} 个",
            f"  外部文件: {len(self._loaded_files)} 个已加载",
        ]
        return "\n".join(lines)


# ===================== 全局单例 =====================

_registry: Optional[DictionaryRegistry] = None


def get_registry() -> DictionaryRegistry:
    """获取全局 DictionaryRegistry 实例"""
    global _registry
    if _registry is None:
        _registry = DictionaryRegistry()
    return _registry


def initialize_registry(builtins: dict, dict_dir: str = None) -> DictionaryRegistry:
    """初始化全局 registry: 注册内置 + 加载外部 + 合并

    由 dictionaries.py 在模块加载时调用。

    Args:
        builtins: {name: data} 内置词表
        dict_dir: 外部 JSON 目录 (None 用默认路径)

    Returns:
        初始化后的 DictionaryRegistry
    """
    reg = get_registry()
    reg.register_builtins(**builtins)
    reg.load_external(dict_dir)
    return reg
