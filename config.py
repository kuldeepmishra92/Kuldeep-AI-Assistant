import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)

def _get_secret(key: str, default: str = "") -> str:
    return os.getenv(key, default)

IS_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") == "true" or os.path.exists("/mount/src")

# ── LLM ────────────────────────────────────────────────────
GROQ_API_KEY: str       = _get_secret("GROQ_API_KEY")
GROQ_MODEL_NAME: str    = _get_secret("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
GROQ_TEMPERATURE: float = 0.2
GROQ_MAX_TOKENS: int    = 1024

# ── Embeddings ─────────────────────────────────────────────
EMBEDDING_MODEL: str = _get_secret("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Storage paths ──────────────────────────────────────────
if IS_CLOUD:
    CHROMA_DB_PATH: str = "/tmp/chroma_db"
    SQLITE_DB_PATH: str = "/tmp/chat_memory.db"
else:
    CHROMA_DB_PATH: str = _get_secret("CHROMA_DB_PATH", "./data/chroma_db")
    SQLITE_DB_PATH: str = _get_secret("SQLITE_DB_PATH", "./data/chat_memory.db")

CHROMA_COLLECTION_NAME: str = "rag_documents"

# ── Knowledge base directories ─────────────────────────────
# Permanent base knowledge (your personal info files — always auto-indexed)
KNOWLEDGE_DIR: str = str(BASE_DIR / "data" / "knowledge")
# Dynamic uploads via admin panel (also persisted on disk + auto-indexed)
UPLOADS_DIR: str   = str(BASE_DIR / "data" / "uploads")

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# ── RAG / Retrieval ────────────────────────────────────────
CHUNK_SIZE: int    = int(_get_secret("CHUNK_SIZE",    "600"))
CHUNK_OVERLAP: int = int(_get_secret("CHUNK_OVERLAP", "80"))
# Candidates fetched before re-ranking
TOP_K_CANDIDATES: int = int(_get_secret("TOP_K_CANDIDATES", "20"))
# Final chunks returned after re-ranking
TOP_K_RETRIEVAL: int  = int(_get_secret("TOP_K_RETRIEVAL",  "5"))
# Cross-encoder relevance threshold — below this = "I don't know"
RAG_CONFIDENCE_THRESHOLD: float = float(_get_secret("RAG_CONFIDENCE_THRESHOLD", "0.05"))
ENABLE_RERANKING: bool       = _get_secret("ENABLE_RERANKING",       "true").lower() == "true"
ENABLE_QUERY_EXPANSION: bool = _get_secret("ENABLE_QUERY_EXPANSION", "false").lower() == "true"

# ── Agent routing labels ───────────────────────────────────
ROUTE_MATH:    str = "math"
ROUTE_RAG:     str = "rag"
ROUTE_MEMORY:  str = "memory"
ROUTE_GENERAL: str = "general"
ROUTE_SEARCH:  str = "search"

# ── Admin panel ────────────────────────────────────────────
ADMIN_SECRET_KEY: str = _get_secret("ADMIN_SECRET_KEY", "kuldeep-admin-2025")

# ── Ensure dirs exist ──────────────────────────────────────
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(KNOWLEDGE_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

def validate_config() -> None:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file locally. "
            "Get a free key at https://console.groq.com"
        )
