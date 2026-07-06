"""Patch retriever.py to add IngestionScoreCard integration."""
import pathlib

p = pathlib.Path('financial_rag/retrievers/retriever.py')
content = p.read_text(encoding='utf-8')

# Change 1: Add import
old1 = 'from financial_rag.retrievers import persistence'
new1 = 'from financial_rag.retrievers import persistence\nfrom financial_rag.core.ingestion_scorer import IngestionScoreCard'
content = content.replace(old1, new1, 1)

# Change 2: Add _last_ingestion_score to __init__
old2 = '        self.doc_embeddings: Optional[List[List[float]]] = None\n\n    # ===================== \u7d22\u5f15 ====================='
new2 = '        self.doc_embeddings: Optional[List[List[float]]] = None\n        self._last_ingestion_score: Optional[IngestionScoreCard] = None\n\n    # ===================== \u7d22\u5f15 ====================='
content = content.replace(old2, new2, 1)

# Change 3: Add ingestion scoring after the existing logger.info in index()
# Find the unique marker: the logger.info block ending with rerank, followed by blank line and def add
old3 = (
    '            f"{\' (\u542b rerank)\' if self.reranker else \'\'}"\n'
    '        )\n'
    '\n'
    '    def add('
)
new3 = (
    '            f"{\' (\u542b rerank)\' if self.reranker else \'\'}"\n'
    '        )\n'
    '\n'
    '        # Ingestion scoring\n'
    '        try:\n'
    '            card = IngestionScoreCard()\n'
    '            card.record_preprocessing(documents)\n'
    '            empty = sum(1 for d in documents if len(d.get("text", "").strip()) < 10)\n'
    '            total_vocab = len(set(t for toks in self._bm25._corpus_tokens for t in toks))\n'
    '            card.record_tokenization(self._bm25._corpus_tokens)\n'
    '            card.record_index(\n'
    '                doc_count=len(documents),\n'
    '                total_vocab=total_vocab,\n'
    '                bm25_built=self._bm25.is_built,\n'
    '                chroma_built=self._vector._collection is not None,\n'
    '                embedding_dim=len(self.doc_embeddings[0]) if self.doc_embeddings else 0,\n'
    '                empty_docs=empty,\n'
    '            )\n'
    '            card.compute()\n'
    '            card.log_summary()\n'
    '            self._last_ingestion_score = card\n'
    '        except Exception as e:\n'
    '            logger.warning(f"IngestionScoreCard failed: {e}")\n'
    '\n'
    '    def get_ingestion_score(self) -> Optional[IngestionScoreCard]:\n'
    '        """Return the last ingestion scorecard (after index() or add())"""\n'
    '        return self._last_ingestion_score\n'
    '\n'
    '    def add('
)
content = content.replace(old3, new3, 1)

p.write_text(content, encoding='utf-8')
print('OK: retriever.py patched')
