from typing import List, Dict
from groq import Groq
from ddgs import DDGS
from agents.base_agent import BaseAgent
import config
from utils.logger import get_logger

logger = get_logger(__name__)

class SearchAgent(BaseAgent):
    SYSTEM_PROMPT = """You are a real-time information retrieval assistant.
Your job is to provide accurate, up-to-date answers using the provided search results.

Rules:
- Synthesize a coherent answer from the search snippets.
- If the snippets provide a definitive answer (e.g., current date, president name), state it clearly.
- Always include a short "Sources" section at the end if links are available.
- Be concise but thorough.
- Format your answer with markdown. Always include a space after bolding (e.g., **Event:** description) for correct UI rendering.

Search Results:
{context}"""

    QUERY_REWRITE_PROMPT = """You are a search query optimizer.
Convert the user's natural-language question into a short, precise web search query (5 words or fewer).
Remove filler words like "who is", "what is", "tell me about", "can you".
Expand abbreviations: "cm" -> "Chief Minister", "pm" -> "Prime Minister", "up" -> "Uttar Pradesh", "uk" -> "Uttarakhand".
Output ONLY the search query, no explanation, no punctuation at the end.

Examples:
  "who is the current cm of uttar pradesh" -> "Chief Minister Uttar Pradesh 2025"
  "what is today's weather in delhi" -> "Delhi weather today"
  "latest iphone price" -> "iPhone latest model price 2025"
  "who is the president of usa" -> "current US President 2025"

User question: "{query}"
Search query:"""

    def __init__(self):
        super().__init__(name="Web Search Agent")
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("SearchAgent ready.")

    def _reformulate_query(self, query: str) -> str:
        """Use LLM to convert a natural-language question into a clean search query."""
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=[
                    {"role": "user", "content": self.QUERY_REWRITE_PROMPT.format(query=query)}
                ],
                temperature=0.0,
                max_tokens=20,
            )
            reformulated = response.choices[0].message.content.strip().strip('"').strip("'")
            logger.info(f"  -> Query reformulated: '{query[:60]}' -> '{reformulated}'")
            return reformulated
        except Exception as exc:
            logger.warning(f"  -> Query reformulation failed ({exc}), using original query.")
            return query

    def _do_search(self, search_query: str, max_results: int = 8) -> str:
        """Run DuckDuckGo search and return formatted results string."""
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(search_query, max_results=max_results)]
        if not results:
            return ""
        return "\n\n".join([
            f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nLink: {r.get('href')}"
            for r in results
        ])

    def run(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        session_id: str = "",
    ) -> str:
        logger.info(f"SearchAgent processing: '{query[:80]}'")

        try:
            # Step 1: Reformulate the query into a clean, unambiguous search query
            search_query = self._reformulate_query(query)

            # Step 2: Perform Web Search with the clean query
            logger.info(f"  -> Searching DuckDuckGo for: '{search_query}'")
            search_results = self._do_search(search_query, max_results=8)

            # Step 3: Fallback — retry with the original query if nothing found
            if not search_results:
                logger.warning(f"  -> No results for reformulated query. Retrying with original.")
                search_results = self._do_search(query, max_results=8)

            if not search_results:
                return (
                    "I searched the web for your question but couldn't find any relevant "
                    "recent information. I'll try to answer based on my general knowledge instead."
                )

            # Step 4: Synthesize answer with LLM using ORIGINAL user query for correct intent
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT.format(context=search_results)},
                {"role": "user", "content": query}
            ]

            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=config.GROQ_MAX_TOKENS,
            )

            answer = response.choices[0].message.content.strip()
            logger.info("  -> Web search response synthesized.")

            return answer

        except Exception as exc:
            logger.error(f"SearchAgent failed: {exc}")
            return (
                f"I encountered an error while searching the web for your request. "
                f"Please try again later.\n\nError: {exc}"
            )
