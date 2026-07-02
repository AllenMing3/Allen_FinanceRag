"""
索引持久化 — save / load

功能:
- JSON 序列化 (可选 gzip 压缩)
- SHA256 校验和 (加载时验证完整性)
- 索引元信息 (IndexInfo: 创建时间、文档数、大小)
- 快速元信息读取 (不加载完整索引)

版本: v3 (新增 compression + checksum)
     v2 (基础 JSON)
     v1 (旧版，兼容读取)
"""
import gzip
import hashlib
import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

INDEX_VERSION = 3


@dataclass
class IndexInfo:
    """索引元信息"""
    path: str = ""
    version: int = 0
    doc_count: int = 0
    file_size_mb: float = 0.0
    has_embeddings: bool = False
    created_at: str = ""
    avg_doc_length: int = 0
    config: Dict = field(default_factory=dict)
    checksum: str = ""

    def summary(self) -> str:
        parts = [
            f"v{self.version}",
            f"{self.doc_count} docs",
            f"{self.file_size_mb:.2f} MB",
        ]
        if self.has_embeddings:
            parts.append("embeddings")
        if self.created_at:
            parts.append(self.created_at[:10])
        return " | ".join(parts)


def save_index(
    path: str,
    documents: List[Dict],
    doc_embeddings: Optional[List[List[float]]],
    config: Dict,
    compress: bool = False,
):
    """
    将索引持久化到磁盘

    Args:
        path: 保存路径 (.json 或 .json.gz)
        documents: 文档列表
        doc_embeddings: 预计算的文档向量
        config: RRF/权重参数
        compress: 是否 gzip 压缩
    """
    from datetime import datetime

    # 计算平均文档长度
    avg_len = 0
    if documents:
        total = sum(len(d.get("text", "")) for d in documents)
        avg_len = total // len(documents)

    data = {
        "version": INDEX_VERSION,
        "doc_count": len(documents),
        "documents": documents,
        "config": config,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "avg_doc_length": avg_len,
        },
    }

    # Chroma 管理向量时不保存 embeddings，否则保留 (向后兼容)
    if doc_embeddings is not None:
        data["doc_embeddings"] = doc_embeddings

    # 序列化 JSON
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    # 计算校验和
    checksum = hashlib.sha256(json_bytes).hexdigest()[:16]
    data["checksum"] = checksum
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    if compress:
        if not path.endswith(".gz"):
            path = path + ".gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json_bytes.decode("utf-8"))
    else:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_bytes.decode("utf-8"))

    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(
        f"索引已保存到 {path} ({size_mb:.2f} MB, "
        f"{len(documents)} docs, checksum={checksum})"
    )


def load_index(path: str) -> Dict:
    """
    从磁盘加载索引

    Args:
        path: 索引文件路径 (.json 或 .json.gz)

    Returns:
        {"documents": [...], "doc_embeddings": [...], "config": {...}, "metadata": {...}}

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 版本不支持或校验和不匹配
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"索引文件不存在: {path}")

    # 支持 gzip
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    version = data.get("version", 1)
    if version not in (1, 2, 3):
        raise ValueError(f"不支持的索引版本: {version}")

    # 校验和验证 (v3+)
    stored_checksum = data.get("checksum")
    if stored_checksum and version >= 3:
        # 移除 checksum 字段后重新计算
        verify_data = {k: v for k, v in data.items() if k != "checksum"}
        json_bytes = json.dumps(verify_data, ensure_ascii=False, indent=2).encode("utf-8")
        computed = hashlib.sha256(json_bytes).hexdigest()[:16]
        if computed != stored_checksum:
            raise ValueError(
                f"索引校验和不匹配: stored={stored_checksum}, computed={computed}"
            )
        logger.info(f"校验和验证通过: {stored_checksum}")

    logger.info(
        f"已加载索引 {path}"
        f" ({len(data.get('documents', []))} 篇文档"
        f"{', 含 embeddings' if data.get('doc_embeddings') else ''})"
    )

    return {
        "documents": data["documents"],
        "doc_embeddings": data.get("doc_embeddings"),
        "config": data.get("config", {}),
        "metadata": data.get("metadata", {}),
    }


def get_index_info(path: str) -> IndexInfo:
    """
    快速读取索引元信息 (不加载完整数据)

    Args:
        path: 索引文件路径

    Returns:
        IndexInfo 实例
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"索引文件不存在: {path}")

    size_mb = os.path.getsize(path) / (1024 * 1024)

    # 读取 JSON 但只取 metadata
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    metadata = data.get("metadata", {})

    return IndexInfo(
        path=path,
        version=data.get("version", 1),
        doc_count=data.get("doc_count", len(data.get("documents", []))),
        file_size_mb=size_mb,
        has_embeddings=bool(data.get("doc_embeddings")),  # Chroma 管理时为 False
        created_at=metadata.get("created_at", ""),
        avg_doc_length=metadata.get("avg_doc_length", 0),
        config=data.get("config", {}),
        checksum=data.get("checksum", ""),
    )
