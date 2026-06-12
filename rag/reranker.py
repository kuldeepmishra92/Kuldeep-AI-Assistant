"""
Cross-Encoder Re-Ranker
------------------------
Uses sentence-transformers cross-encoder to re-score candidate chunks
against the user query. This dramatically improves precision by catching
semantically relevant chunks that BM25/vector search may rank low.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast (~80MB), runs locally, no API cost
  - Trained on MS MARCO passage ranking
  - Returns relevance score in range ~[-10, 10]
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)

_reranker = None   # lazy-loaded

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder re-ranker model...")
            _reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.info("  → Cross-encoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load cross-encoder: {e}. Re-ranking disabled.")
            _reranker = None
    return _reranker


def rerank(
    query: str,
    documents: List[Document],
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Re-rank documents against query using cross-encoder.

    Returns:
        List of (document, score) tuples sorted by relevance descending.
        Score is a raw float; higher = more relevant.
    """
    if not documents:
        return []

    reranker = _get_reranker()
    if reranker is None:
        # Fallback: return original order with dummy scores
        logger.warning("Re-ranker unavailable — returning original order.")
        return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents[:top_k])]

    pairs = [(query, doc.page_content) for doc in documents]

    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.error(f"Re-ranking prediction failed: {e}")
        return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents[:top_k])]

    scored = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top = scored[:top_k]
    logger.info(
        f"  → Re-ranked {len(documents)} → {len(top)} chunks. "
        f"Top score: {top[0][1]:.3f}" if top else "No results."
    )
    return top
