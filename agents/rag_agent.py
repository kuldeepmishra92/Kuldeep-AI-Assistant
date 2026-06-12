from typing import List, Dict, Tuple
from langchain_core.documents import Document
from groq import Groq
from agents.base_agent import BaseAgent
from rag.retriever import hybrid_search, invalidate_bm25_cache
from rag.vector_store import get_document_count
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_CHUNK_LENGTH = 25

def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "")
        page_str = f" | Page: {page}" if page else ""
        parts.append(
            f"[Chunk {i}] Source: {source}{page_str}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)

def _format_sources(chunks: List[Tuple[Document, float]]) -> str:
    seen, sources = set(), []
    for doc, _ in chunks:
        src  = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        key  = f"{src}:p{page}" if page else src
        if key not in seen:
            seen.add(key)
            label = f"📄 `{src}`" + (f" (page {page})" if page else "")
            sources.append(label)
    return "\n".join(sources) if sources else ""

class RAGAgent(BaseAgent):

    # ── Strict grounding prompt — NO hallucination, but natural tone ──
    RAG_SYSTEM_PROMPT = """You are Kuldeep AI — a personal AI assistant representing Kuldeep Kumar Mishra.

Your job is to answer questions about Kuldeep using ONLY the document context provided below.

Tone & Style:
- Be warm, natural, and conversational — like a personal assistant who knows Kuldeep well
- Answer in first person where appropriate (e.g., "Kuldeep is currently..." or "He works on...")
- Be specific and detailed — use the actual facts from the context
- Format with markdown (bold key info, use bullet points for lists)
- Never be dry or robotic — make it feel personal and genuine

Strict Rules:
- Answer ONLY from the provided context — do NOT use outside knowledge
- If the context does not contain the answer, say naturally: "That's not something I have details on right now. Feel free to reach out to Kuldeep directly!"
- Do NOT say "Based on the document" or "According to the context" — just answer naturally
- Do NOT add "I'm just an AI" disclaimers

Document Context:
{context}"""

    def __init__(self):
        super().__init__(name="RAG Agent")
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("RAGAgent (advanced, anti-hallucination) ready.")

    def run(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        session_id: str = "",
    ) -> str:
        logger.info(f"RAGAgent processing: '{query[:80]}'")

        # Check if KB has any documents
        try:
            doc_count = get_document_count()
        except Exception:
            doc_count = 0

        if doc_count == 0:
            logger.info("  → Knowledge base empty.")
            return (
                "My knowledge base is currently empty. "
                "Please upload documents via the admin panel first."
            )

        # Run advanced hybrid search → (doc, score) pairs
        logger.info(f"  → Running advanced retrieval over {doc_count} chunks...")
        ranked_chunks = hybrid_search(query, k=config.TOP_K_RETRIEVAL)

        if not ranked_chunks:
            return "I don't have specific information about that in my knowledge base."

        # Filter chunks below minimum length
        valid = [
            (doc, score)
            for doc, score in ranked_chunks
            if len(doc.page_content.strip()) >= _MIN_CHUNK_LENGTH
        ]

        if not valid:
            return "That's not something I have details on right now. Feel free to reach out to Kuldeep directly!"

        logger.info(f"  → {len(valid)} relevant chunks found. Generating answer...")
        return self._rag_answer(query, valid, history)

    def _rag_answer(
        self,
        query: str,
        chunks: List[Tuple[Document, float]],
        history: List[Dict[str, str]],
    ) -> str:
        context_block = _format_context(chunks)

        messages = [
            {"role": "system", "content": self.RAG_SYSTEM_PROMPT.format(context=context_block)}
        ]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": query})

        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("  → RAG answer generated.")
            return answer
        except Exception as exc:
            logger.error(f"RAGAgent LLM call failed: {exc}")
            return f"I encountered an error generating the answer. Please try again.\n\nError: {exc}"

