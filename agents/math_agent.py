import re
import ast
import operator
from typing import List, Dict
from groq import Groq
from agents.base_agent import BaseAgent
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_PURE_MATH_RE = re.compile(
    r"^\s*[\d\s\.\+\-\*\/\(\)\^%]+\s*$"
)

def _safe_eval(expr: str) -> float | None:
    expr = expr.replace("^", "**")

    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_func = _SAFE_OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op_func(_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_func = _SAFE_OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op_func(_eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
        result = _eval_node(tree.body)
        return result
    except Exception:
        return None

def _extract_expression(query: str) -> str | None:
    cleaned = re.sub(
        r"(?i)^(what\s+is|calculate|compute|evaluate|solve|find)\s*:?\s*", "", query.strip()
    )
    cleaned = re.sub(r"[?!]+$", "", cleaned).strip()

    if _PURE_MATH_RE.match(cleaned):
        return cleaned

    match = re.search(r"[\d]+\s*[\+\-\*\/\^%]\s*[\d\.\s\+\-\*\/\^\(\)%]+", query)
    if match:
        return match.group(0).strip()

    return None

class MathAgent(BaseAgent):
    SYSTEM_PROMPT = """You are a precise mathematical reasoning assistant.
Solve the given problem step by step using chain-of-thought reasoning.
Always:
  1. Identify the mathematical concept involved
  2. Show your working step by step
  3. State your final answer clearly on a new line starting with "Answer:"
Be concise but thorough. Always include a space after bolding (e.g., **Calculation:** result) for correct UI rendering."""

    def __init__(self):
        super().__init__(name="Math Agent")
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("MathAgent ready.")

    def run(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        session_id: str = "",
    ) -> str:
        logger.info(f"MathAgent processing: '{query[:80]}'")

        expr = _extract_expression(query)
        if expr is not None:
            result = _safe_eval(expr)
            if result is not None:
                formatted = int(result) if result == int(result) else round(result, 6)
                answer = (
                    f"**Calculation:** `{expr.strip()} = {formatted}`\n\n"
                    f"**Answer: {formatted}**\n\n"
                    f"*(Computed via Python arithmetic — 100% accurate)*"
                )
                logger.info(f"  → Computed via Python eval: {formatted}")
                return answer

        logger.info("  → Falling back to LLM chain-of-thought reasoning.")

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": query})

        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("  → LLM math response received.")
            return answer

        except Exception as exc:
            logger.error(f"MathAgent LLM call failed: {exc}")
            return (
                f"I encountered an error processing your math query. "
                f"Please try rephrasing it.\n\nError: {exc}"
            )
