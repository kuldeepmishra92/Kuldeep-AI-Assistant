from typing import List, Optional
import sys

# Support for Streamlit Cloud / Linux environments with old SQLite
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_embedding_model: Optional[HuggingFaceEmbeddings] = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("  → Embedding model loaded.")
    return _embedding_model

_vector_store: Optional[Chroma] = None

def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        logger.info(f"Connecting to ChromaDB at: {config.CHROMA_DB_PATH}")
        _vector_store = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=config.CHROMA_DB_PATH,
        )
        count = _vector_store._collection.count()
        logger.info(f"  → ChromaDB ready. Chunks in store: {count}")
    return _vector_store

# ── Write ───────────────────────────────────────────────────
def add_documents(documents: List[Document]) -> int:
    if not documents:
        logger.warning("add_documents: empty list, nothing to index.")
        return 0
    store = get_vector_store()
    logger.info(f"Indexing {len(documents)} chunks into ChromaDB...")
    store.add_documents(documents)
    logger.info(f"  → Done. Total in store: {store._collection.count()}")
    return len(documents)

# ── Read ────────────────────────────────────────────────────
def similarity_search(query: str, k: int = None) -> List[Document]:
    k = k or config.TOP_K_RETRIEVAL
    store = get_vector_store()
    q_preview = query[:60] + ("..." if len(query) > 60 else "")
    logger.info(f"Semantic search (k={k}): '{q_preview}'")
    results = store.similarity_search(query, k=k)
    logger.info(f"  → {len(results)} chunks retrieved.")
    return results

def get_document_count() -> int:
    return get_vector_store()._collection.count()

def source_exists(source_name: str) -> bool:
    """Return True if any chunk with this source name is already indexed."""
    try:
        store = get_vector_store()
        results = store._collection.get(
            where={"source": source_name},
            limit=1,
            include=["metadatas"],
        )
        return bool(results and results.get("ids"))
    except Exception as e:
        logger.warning(f"source_exists check failed for '{source_name}': {e}")
        return False

def get_unique_sources() -> List[str]:
    store = get_vector_store()
    try:
        results = store._collection.get(include=["metadatas"])
        if not results or "metadatas" not in results:
            return []
        sources = {m.get("source") for m in results["metadatas"] if m and m.get("source")}
        return sorted(list(sources))
    except Exception as e:
        logger.error(f"Error fetching unique sources: {e}")
        return []

def get_source_chunk_count(source_name: str) -> int:
    """Return how many chunks a specific source has in ChromaDB."""
    try:
        store = get_vector_store()
        results = store._collection.get(
            where={"source": source_name},
            include=["metadatas"],
        )
        return len(results.get("ids", []))
    except Exception:
        return 0

def get_source_preview(source_name: str, max_chars: int = 3000) -> str:
    """Return concatenated text of all chunks for a source (for View modal)."""
    try:
        store = get_vector_store()
        results = store._collection.get(
            where={"source": source_name},
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        # Sort by chunk_index if available
        paired = list(zip(metas, docs))
        paired.sort(key=lambda x: x[0].get("chunk_index", 0) if x[0] else 0)
        combined = "\n\n".join(d for _, d in paired)
        return combined[:max_chars] + ("…" if len(combined) > max_chars else "")
    except Exception as e:
        logger.error(f"Error getting source preview for '{source_name}': {e}")
        return ""

# ── Delete ──────────────────────────────────────────────────
def delete_source(source_name: str) -> bool:
    store = get_vector_store()
    try:
        logger.info(f"Deleting source: '{source_name}'")
        store._collection.delete(where={"source": source_name})
        logger.info(f"  → '{source_name}' removed from ChromaDB.")
        return True
    except Exception as e:
        logger.error(f"Error deleting source '{source_name}': {e}")
        return False

def clear_store() -> None:
    global _vector_store
    store = get_vector_store()
    store.delete_collection()
    _vector_store = None
    logger.warning("ChromaDB collection cleared.")
