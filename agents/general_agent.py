from typing import List, Dict
from groq import Groq
from agents.base_agent import BaseAgent
import config
from utils.logger import get_logger

logger = get_logger(__name__)

class GeneralAgent(BaseAgent):
    SYSTEM_PROMPT = """You are Kuldeep AI — a personal AI assistant created for Kuldeep Kumar Mishra.

You can answer general knowledge questions, have friendly conversations, help with explanations, and discuss any topic.

Personality:
- Warm, friendly, and conversational
- When greeted (hi, hello, hey, etc.), introduce yourself naturally: "Hi! I'm Kuldeep AI. How can I help you today?"
- Be concise but thorough
- Format responses with markdown when helpful
- IMPORTANT: Always include a space after bold text (e.g., **Hello!** nice) for correct rendering

Note: For questions specifically about Kuldeep Kumar Mishra (his background, projects, skills, experience), I have a dedicated knowledge base for that."""

    def __init__(self):
        super().__init__(name="General Agent")
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("GeneralAgent ready.")

    def run(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        session_id: str = "",
    ) -> str:
        logger.info(f"GeneralAgent processing: '{query[:80]}'")

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": query})

        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.5,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("  → General knowledge response generated.")
            return answer

        except Exception as exc:
            logger.error(f"GeneralAgent LLM call failed: {exc}")
            return f"I encountered an error answering your question. Please try again.\n\nError: {exc}"
