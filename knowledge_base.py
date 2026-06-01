"""
知识库构建模块 - 基于 LlamaIndex 的向量索引和检索
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import config


class KnowledgeBase:
    """知识库管理器"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or config.rag.vector_store_path
        self.index: Optional[VectorStoreIndex] = None
        self.documents: List[Document] = []
        
        # 初始化 LlamaIndex 设置
        self._setup_settings()
    
    def _setup_settings(self):
        """配置 LlamaIndex 全局设置"""
        # 设置嵌入模型
        Settings.embed_model = OpenAIEmbedding(
            model=config.rag.embedding_model,
            api_key=config.openai_api_key,
            api_base=config.openai_api_base
        )
        
        # 设置 LLM
        Settings.llm = OpenAI(
            model=config.rag.llm_model,
            api_key=config.openai_api_key,
            api_base=config.openai_api_base,
            temperature=0.1
        )
        
        # 设置文本分割器
        Settings.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
    
    def add_documents_from_directory(self, dir_path: str) -> "KnowledgeBase":
        """从目录加载文档"""
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            return self
        
        print(f"从目录加载文档: {dir_path}")
        reader = SimpleDirectoryReader(
            dir_path,
            recursive=True,
            required_exts=[".txt", ".md", ".pdf", ".docx", ".json"]
        )
        docs = reader.load_data()
        self.documents.extend(docs)
        print(f"加载了 {len(docs)} 个文档")
        return self
    
    def add_documents_from_files(self, file_paths: List[str]) -> "KnowledgeBase":
        """从文件列表加载文档"""
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                doc = Document(
                    text=content,
                    metadata={"source": file_path, "filename": os.path.basename(file_path)}
                )
                self.documents.append(doc)
                print(f"加载文档: {file_path}")
        return self
    
    def add_text_documents(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> "KnowledgeBase":
        """添加纯文本文档"""
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            doc = Document(text=text, metadata=metadata)
            self.documents.append(doc)
        print(f"添加了 {len(texts)} 个文本文档")
        return self
    
    def add_log_entries(self, log_entries: List[Any]) -> "KnowledgeBase":
        """添加日志条目（从 log_parser 模块）"""
        from log_parser import LogEntry
        
        documents = []
        for entry in log_entries:
            if isinstance(entry, LogEntry):
                doc = Document(
                    text=entry.to_document_text(),
                    metadata={
                        "source": entry.source or "unknown",
                        "level": entry.level or "unknown",
                        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                        "type": "log_entry"
                    }
                )
                documents.append(doc)
        
        self.documents.extend(documents)
        print(f"添加了 {len(documents)} 个日志条目")
        return self
    
    def build_index(self) -> "KnowledgeBase":
        """构建向量索引"""
        if not self.documents:
            print("没有文档可索引")
            return self
        
        print(f"构建索引，共 {len(self.documents)} 个文档...")
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            show_progress=True
        )
        print("索引构建完成")
        return self
    
    def save_index(self) -> "KnowledgeBase":
        """保存索引到磁盘"""
        if self.index is None:
            print("没有可保存的索引")
            return self
        
        os.makedirs(self.storage_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.storage_path)
        print(f"索引已保存到: {self.storage_path}")
        return self
    
    def load_index(self) -> bool:
        """从磁盘加载索引"""
        if not os.path.exists(self.storage_path):
            print(f"索引目录不存在: {self.storage_path}")
            return False
        
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
            self.index = load_index_from_storage(storage_context)
            print(f"索引已从 {self.storage_path} 加载")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def get_query_engine(self, **kwargs):
        """获取查询引擎"""
        if self.index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        # 合并配置
        query_kwargs = {
            "similarity_top_k": config.rag.similarity_top_k,
            "response_mode": config.rag.response_mode,
        }
        query_kwargs.update(kwargs)
        
        return self.index.as_query_engine(**query_kwargs)
    
    def get_chat_engine(self, **kwargs):
        """获取聊天引擎（支持多轮对话）"""
        if self.index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        from llama_index.core.chat_engine import CondenseQuestionChatEngine
        
        query_engine = self.get_query_engine()
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            **kwargs
        )
        return chat_engine
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相似文档（仅检索，不生成回答）"""
        if self.index is None:
            raise ValueError("索引未构建")
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        results = []
        for node in nodes:
            results.append({
                "text": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata
            })
        
        return results
    
    def clear(self) -> "KnowledgeBase":
        """清空知识库"""
        self.documents = []
        self.index = None
        print("知识库已清空")
        return self
    
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        stats = {
            "total_documents": len(self.documents),
            "index_built": self.index is not None,
            "storage_path": self.storage_path
        }
        
        if self.documents:
            # 按类型统计
            type_counts = {}
            for doc in self.documents:
                doc_type = doc.metadata.get("type", "unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            stats["document_types"] = type_counts
        
        return stats


class LogKnowledgeBase(KnowledgeBase):
    """专门用于日志的知识库"""
    
    def __init__(self, storage_path: Optional[str] = None):
        super().__init__(storage_path)
        self.log_entries = []
    
    def add_logs(self, log_entries: List[Any]) -> "LogKnowledgeBase":
        """添加日志条目并构建文档"""
        from log_parser import LogEntry
        
        self.log_entries.extend(log_entries)
        
        # 按日志级别分组构建文档
        level_groups = {}
        for entry in log_entries:
            if isinstance(entry, LogEntry):
                level = entry.level or "UNKNOWN"
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(entry)
        
        # 为每个级别创建聚合文档
        for level, entries in level_groups.items():
            # 每 10 条日志聚合成一个文档
            batch_size = 10
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i+batch_size]
                text = "\n\n---\n\n".join([e.to_document_text() for e in batch])
                
                doc = Document(
                    text=text,
                    metadata={
                        "type": "log_batch",
                        "level": level,
                        "batch_index": i // batch_size,
                        "entry_count": len(batch)
                    }
                )
                self.documents.append(doc)
        
        print(f"添加了 {len(log_entries)} 条日志，生成 {len(self.documents)} 个文档")
        return self
    
    def get_error_analysis(self) -> str:
        """获取错误分析摘要"""
        from log_parser import LogEntry
        
        errors = [e for e in self.log_entries if isinstance(e, LogEntry) and e.level in ['ERROR', 'CRITICAL', 'FATAL']]
        
        if not errors:
            return "未检测到错误日志"
        
        # 按来源分组
        by_source = {}
        for e in errors:
            source = e.source or "unknown"
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(e.message)
        
        # 生成摘要
        summary = f"错误日志统计:\n"
        summary += f"总错误数: {len(errors)}\n\n"
        
        for source, messages in by_source.items():
            summary += f"来源 [{source}]: {len(messages)} 个错误\n"
            # 显示前3个不同的错误消息
            unique_msgs = list(set(messages))[:3]
            for msg in unique_msgs:
                summary += f"  - {msg[:100]}...\n"
            summary += "\n"
        
        return summary


# 便捷函数
def create_knowledge_base_from_logs(log_entries: List[Any], storage_path: Optional[str] = None) -> KnowledgeBase:
    """从日志条目快速创建知识库"""
    kb = LogKnowledgeBase(storage_path)
    kb.add_logs(log_entries).build_index().save_index()
    return kb


def create_knowledge_base_from_directory(dir_path: str, storage_path: Optional[str] = None) -> KnowledgeBase:
    """从目录快速创建知识库"""
    kb = KnowledgeBase(storage_path)
    kb.add_documents_from_directory(dir_path).build_index().save_index()
    return kb
