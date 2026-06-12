"""
Advanced Hybrid Retriever
--------------------------
Pipeline:
  1. Query expansion   — generate multiple query variants via LLM
  2. BM25 search       — keyword-based retrieval (top-K candidates)
  3. Semantic search   — dense vector retrieval (top-K candidates)
  4. RRF fusion        — merge both result sets with Reciprocal Rank Fusion
  5. Cross-encoder     — re-rank fused candidates for final top-K
"""
from __future__ import annotations

from typing import List, Tuple
from collections import defaultdict
from langchain_core.documents import Document

from rag.vector_store import similarity_search, get_vector_store, get_document_count
from rag.reranker import rerank
import config
from utils.logger import get_logger

logger = get_logger(__name__)

# ── BM25 index cache ────────────────────────────────────────
from rank_bm25 import BM25Okapi

_bm25_index = None
_bm25_docs: List[Document] = []

def _build_bm25_index() -> None:
    global _bm25_index, _bm25_docs
    store = get_vector_store()
    total = store._collection.count()
    if total == 0:
        logger.warning("BM25 skipped — no documents in ChromaDB.")
        _bm25_index = None
        _bm25_docs  = []
        return
    logger.info(f"Building BM25 index over {total} chunks...")
    raw = store._collection.get(include=["documents", "metadatas"])
    _bm25_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    tokenised = [doc.page_content.lower().split() for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenised)
    logger.info(f"  → BM25 index built over {len(_bm25_docs)} chunks.")

def _bm25_search(query: str, k: int) -> List[Tuple[Document, float]]:
    global _bm25_index, _bm25_docs
    _build_bm25_index()
    if _bm25_index is None:
        return []
    tokens     = query.lower().split()
    raw_scores = _bm25_index.get_scores(tokens)
    scored = sorted(zip(_bm25_docs, raw_scores), key=lambda x: x[1], reverse=True)
    top_score = scored[0][1] if scored and scored[0][1] > 0 else 1.0
    normalised = [
        (doc, score / top_score)
        for doc, score in scored[:k]
        if score > 0
    ]
    logger.info(f"  → BM25 returned {len(normalised)} results.")
    return normalised

def invalidate_bm25_cache() -> None:
    global _bm25_index, _bm25_docs
    _bm25_index = None
    _bm25_docs  = []
    logger.info("BM25 cache invalidated.")

# ── Query expansion ─────────────────────────────────────────
def _expand_query(query: str) -> List[str]:
    """Generate 2 query variants to improve recall. Falls back to [query] on error."""
    if not config.ENABLE_QUERY_EXPANSION:
        return [query]
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        prompt = (
            f"Generate 2 alternative search queries for the following question. "
            f"Return ONLY the queries, one per line, no numbering, no explanations.\n\n"
            f"Question: {query}"
        )
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=80,
        )
        lines = [
            l.strip()
            for l in resp.choices[0].message.content.strip().split("\n")
            if l.strip()
        ]
        variants = [query] + lines[:2]
        logger.info(f"  → Query expanded to {len(variants)} variants.")
        return variants
    except Exception as e:
        logger.warning(f"Query expansion failed: {e} — using original query.")
        return [query]

# ── RRF fusion ──────────────────────────────────────────────
def _rrf_fuse(
    semantic_lists: List[List[Document]],
    bm25_list:      List[Tuple[Document, float]],
    k_rrf: int = 60,
) -> List[Document]:
    scores: dict  = defaultdict(float)
    doc_map: dict = {}

    for result_list in semantic_lists:
        for rank, doc in enumerate(result_list):
            key = doc.page_content[:120]
            scores[key]  += 1.0 / (rank + 1 + k_rrf)
            doc_map[key]  = doc

    for rank, (doc, _) in enumerate(bm25_list):
        key = doc.page_content[:120]
        scores[key]  += 1.0 / (rank + 1 + k_rrf)
        doc_map[key]  = doc

    ranked_keys   = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in ranked_keys]

# ── Main entry point ────────────────────────────────────────
def hybrid_search(
    query: str,
    k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Full 5-stage retrieval pipeline.
    Returns list of (Document, relevance_score) sorted by score descending.
    """
    final_k      = k or config.TOP_K_RETRIEVAL
    candidate_k  = config.TOP_K_CANDIDATES

    if get_document_count() == 0:
        logger.warning("hybrid_search: ChromaDB is empty.")
        return []

    logger.info(f"Advanced hybrid retrieval for: '{query[:70]}'")

    # Stage 1 — Query expansion
    queries = _expand_query(query)

    # Stage 2 — Semantic search for each query variant
    semantic_results = []
    for q in queries:
        results = similarity_search(q, k=candidate_k)
        semantic_results.append(results)

    # Stage 3 — BM25 keyword search
    bm25_results = _bm25_search(query, k=candidate_k)

    # Stage 4 — RRF fusion
    fused = _rrf_fuse(semantic_results, bm25_results)
    # Deduplicate by content key, keep order
    seen, unique_fused = set(), []
    for doc in fused:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique_fused.append(doc)

    candidates = unique_fused[:candidate_k]
    logger.info(f"  → {len(candidates)} unique candidates after RRF fusion.")

    # Stage 5 — Cross-encoder re-ranking
    if config.ENABLE_RERANKING and candidates:
        ranked = rerank(query, candidates, top_k=final_k)
    else:
        ranked = [(doc, 1.0 - i * 0.01) for i, doc in enumerate(candidates[:final_k])]

    logger.info(f"  → Final: {len(ranked)} chunks ready.")
    return ranked
