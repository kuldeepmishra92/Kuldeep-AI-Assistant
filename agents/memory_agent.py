from typing import List, Dict
from groq import Groq
from agents.base_agent import BaseAgent
from memory.sqlite_memory import load_history, get_session_summary, count_turns
import config
from utils.logger import get_logger

logger = get_logger(__name__)

class MemoryAgent(BaseAgent):
    SYSTEM_PROMPT = """You are a helpful assistant with access to the conversation history below.
The user is asking about something that was previously discussed.

Your job:
- Recall relevant details from the conversation history
- Provide a clear, accurate summary or answer based on what was discussed
- If a topic was NOT discussed, say so clearly: "We haven't discussed that yet."
- Be conversational and natural — you remember the user personally
- Format your answer with markdown. IMPORTANT: Always include a space after bolding (e.g., **Hello!** nice) to ensure the UI renders correctly.

Conversation History:
{history}"""

    def __init__(self):
        super().__init__(name="Memory Agent")
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("MemoryAgent ready.")

    def run(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        session_id: str = "",
    ) -> str:
        logger.info(f"MemoryAgent processing: '{query[:80]}'")

        total_turns = count_turns(session_id) if session_id else 0

        if total_turns == 0 and not history:
            logger.warning("MemoryAgent: No conversation history found.")
            return (
                "🧠 **No conversation history yet.**\n\n"
                "We haven't talked about anything yet in this session. "
                "Ask me a question and I'll remember it for you!"
            )

        db_history_text = get_session_summary(session_id) if session_id else ""
        in_memory_text = ""
        if history:
            lines = [f"{t['role'].capitalize()}: {t['content']}" for t in history]
            in_memory_text = "\n".join(lines)

        full_history = db_history_text or in_memory_text
        if db_history_text and in_memory_text and db_history_text != in_memory_text:
            full_history = f"{db_history_text}\n\n[Recent turns:]\n{in_memory_text}"

        logger.info(f"  → Loaded {total_turns} turns from SQLite for session '{session_id}'.")

        system_msg = self.SYSTEM_PROMPT.format(history=full_history)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": query},
        ]

        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("  → Memory recall response generated.")

            answer += (
                f"\n\n---\n"
                f"*🧠 I remember {total_turns} message(s) from our conversation.*"
            )
            return answer

        except Exception as exc:
            logger.error(f"MemoryAgent LLM call failed: {exc}")
            return (
                f"I encountered an error recalling our conversation. "
                f"Please try again.\n\nError: {exc}"
            )
