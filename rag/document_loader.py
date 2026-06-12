from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Splitter shared for both PDF and TXT ────────────────────
def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )

# ── PDF Loader ──────────────────────────────────────────────
def load_and_chunk_pdf(file_path: str, source_name: str = None) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    source = source_name or path.name
    logger.info(f"Loading PDF: {path.name}")

    loader = PyPDFLoader(str(path))
    pages  = loader.load()
    logger.info(f"  → {len(pages)} pages loaded from '{path.name}'")

    # Inject consistent source metadata
    for doc in pages:
        doc.metadata["source"]    = source
        doc.metadata["file_type"] = "pdf"

    chunks = _get_splitter().split_documents(pages)
    logger.info(f"  → {len(chunks)} chunks created from PDF '{source}'")
    return chunks


# ── TXT Loader ──────────────────────────────────────────────
def load_and_chunk_txt(file_path: str, source_name: str = None) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    source = source_name or path.name
    logger.info(f"Loading TXT: {path.name}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info(f"  → {len(text)} characters loaded from '{path.name}'")

    base_doc = Document(
        page_content=text,
        metadata={
            "source":    source,
            "file_type": "txt",
            "page":      0,
        },
    )

    chunks = _get_splitter().split_documents([base_doc])

    # Enrich each chunk with section hint (first line of chunk as section label)
    for i, chunk in enumerate(chunks):
        first_line = chunk.page_content.strip().split("\n")[0][:60]
        chunk.metadata["section"] = first_line
        chunk.metadata["chunk_index"] = i

    logger.info(f"  → {len(chunks)} chunks created from TXT '{source}'")
    return chunks


# ── Universal entry point ────────────────────────────────────
def load_and_chunk_file(file_path: str, source_name: str = None) -> List[Document]:
    """Route to correct loader based on file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return load_and_chunk_pdf(file_path, source_name)
    elif suffix == ".txt":
        return load_and_chunk_txt(file_path, source_name)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt")
