import re
from typing import Literal
from groq import Groq
import config
from utils.logger import get_logger

logger = get_logger(__name__)

Route = Literal["math", "rag", "memory", "general", "search"]

_MATH_REGEX = re.compile(r"\b\d+\s*[\+\-\*\/\^]\s*\d+")

_GREETING_KEYWORDS = {
    "hi", "hii", "hello", "hey", "helo", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "sup", "wassup", "whats up", "what's up",
}

_MATH_KEYWORDS = {
    "calculate", "compute", "solve", "equation", "formula",
    "add", "subtract", "multiply", "divide", "sum", "average",
    "percentage", "derivative", "integral", "square root",
}

_MEMORY_KEYWORDS = {
    "earlier", "before", "previously", "last time", "remember",
    "what did we", "what did i", "you said", "we discussed",
    "recall", "history", "our conversation", "mentioned",
}

_SEARCH_KEYWORDS = {
    "today", "right now", "current events", "latest news", "breaking news",
    "weather", "live score", "stock price", "real time", "this week's",
}

# Keywords strongly hinting the user wants info about Kuldeep from knowledge base
_PERSONAL_KEYWORDS = {
    "kuldeep", "your skills", "your projects", "your experience",
    "your education", "your background", "who are you", "about you",
    "what do you do", "your work", "your portfolio", "contact",
    "github", "linkedin", "hugging face", "resume", "cv",
    "achievements", "certification", "iiit", "lnmu", "iiit lucknow",
    "iit jam", "teaching assistant", "outlier", "skin cancer", "rag chatbot",
    "airline", "youtube", "ai simplified",
}

class RouterAgent:
    ROUTER_PROMPT = """You are a query router for Kuldeep AI — a personal AI assistant for Kuldeep Kumar Mishra.

The knowledge base contains detailed information about Kuldeep: his education, work experience, projects, skills, achievements, social links, and FAQ.

Classify the user query into EXACTLY ONE category:

  general → Greetings (hi, hello, hey, good morning, etc.), casual conversation, general knowledge,
            definitions, science, history, or any topic NOT specifically about Kuldeep.
  rag     → The query is specifically about Kuldeep himself: his background, education, work, projects,
            skills, contact info, achievements, portfolio, resume, or uploaded documents.
  math    → The query requires mathematical calculation or numerical reasoning.
  memory  → The query refers to previous conversation (e.g., "what did we discuss?").
  search  → Requires REAL-TIME information: today's news, live scores, current weather, recent events.

Rules:
- GREETINGS (hi, hello, hey, etc.) MUST always be classified as 'general'
- Reply with ONLY ONE word: general, rag, math, memory, or search
- No explanation, no punctuation

User query: "{query}"
Category:"""

    def __init__(self):
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("RouterAgent initialised.")

    def _llm_classify(self, query: str) -> Route | None:
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=[{"role": "user", "content": self.ROUTER_PROMPT.format(query=query)}],
                temperature=0.0,
                max_tokens=5,
            )
            label = response.choices[0].message.content.strip().lower()
            if label in ("math", "rag", "memory", "general", "search"):
                logger.info(f"Router (LLM): '{query[:50]}' → {label}")
                return label
            logger.warning(f"Router (LLM) unexpected label: '{label}'")
            return None
        except Exception as exc:
            logger.error(f"Router LLM call failed: {exc}. Using rules.")
            return None

    def _rule_classify(self, query: str) -> Route:
        q_lower = query.lower().strip()

        # Greetings always → general
        if q_lower in _GREETING_KEYWORDS or any(q_lower.startswith(kw) for kw in _GREETING_KEYWORDS):
            logger.info(f"Router (rules/greeting): → general")
            return "general"

        if _MATH_REGEX.search(query):
            return config.ROUTE_MATH
        if any(kw in q_lower for kw in _MATH_KEYWORDS):
            return config.ROUTE_MATH
        if any(kw in q_lower for kw in _MEMORY_KEYWORDS):
            return "memory"
        # Personal / Kuldeep keywords → RAG first
        if any(kw in q_lower for kw in _PERSONAL_KEYWORDS):
            return config.ROUTE_RAG
        if any(kw in q_lower for kw in _SEARCH_KEYWORDS):
            return "search"
        return "general"

    def route(self, query: str) -> Route:
        q_lower = query.lower().strip()

        # ── Fast path: greetings always → general, skip LLM entirely ──
        if q_lower in _GREETING_KEYWORDS or any(
            q_lower.startswith(kw) for kw in _GREETING_KEYWORDS
        ):
            logger.info(f"Router (greeting fast-path): '{query[:30]}' → general")
            return "general"

        # ── LLM classification ──
        llm_route = self._llm_classify(query)
        return llm_route if llm_route is not None else self._rule_classify(query)
