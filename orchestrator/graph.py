from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from agents.router_agent import RouterAgent
from agents.math_agent   import MathAgent
from agents.rag_agent    import RAGAgent
from agents.memory_agent import MemoryAgent
from agents.general_agent import GeneralAgent
from agents.search_agent import SearchAgent
from memory.sqlite_memory import load_history, save_turn
import config
from utils.logger import get_logger

logger = get_logger(__name__)

class ChatState(TypedDict):
    query:      str
    session_id: str
    history:    List[Dict[str, str]]
    route:      Optional[str]
    context:    str
    response:   str
    agent_used: str

_router  = RouterAgent()
_math    = MathAgent()
_rag     = RAGAgent()
_memory  = MemoryAgent()
_general = GeneralAgent()
_search  = SearchAgent()

def router_node(state: ChatState) -> ChatState:
    query      = state["query"]
    session_id = state["session_id"]
    history = load_history(session_id, limit=10)
    route = _router.route(query)
    logger.info(f"[Graph] Routed '{query[:50]}' → {route}")

    return {
        **state,
        "route":   route,
        "history": history,
    }

def math_node(state: ChatState) -> ChatState:
    response = _math.run(
        query=state["query"],
        history=state["history"],
        session_id=state["session_id"],
    )
    return {**state, "response": response, "agent_used": "Math Agent"}

def rag_node(state: ChatState) -> ChatState:
    response = _rag.run(
        query=state["query"],
        history=state["history"],
        session_id=state["session_id"],
    )
    return {**state, "response": response, "agent_used": "RAG Agent"}

def memory_node(state: ChatState) -> ChatState:
    response = _memory.run(
        query=state["query"],
        history=state["history"],
        session_id=state["session_id"],
    )
    return {**state, "response": response, "agent_used": "Memory Agent"}
def general_node(state: ChatState) -> ChatState:
    response = _general.run(
        query=state["query"],
        history=state["history"],
        session_id=state["session_id"],
    )
    return {**state, "response": response, "agent_used": "General Agent"}

def search_node(state: ChatState) -> ChatState:
    response = _search.run(
        query=state["query"],
        history=state["history"],
        session_id=state["session_id"],
    )
    return {**state, "response": response, "agent_used": "Web Search Agent"}

def save_memory_node(state: ChatState) -> ChatState:
    session_id = state["session_id"]
    try:
        save_turn(session_id, "user",      state["query"])
        save_turn(session_id, "assistant", state["response"])
        logger.info(f"[Graph] Turn saved to SQLite for session '{session_id}'.")
    except Exception as exc:
        logger.error(f"[Graph] Failed to save turn: {exc}")
    return state

def route_decision(state: ChatState) -> str:
    route = state.get("route", config.ROUTE_GENERAL)
    if route == config.ROUTE_MATH:
        return "math_node"
    elif route == config.ROUTE_MEMORY:
        return "memory_node"
    elif route == config.ROUTE_RAG:
        return "rag_node"
    elif route == "search":
        return "search_node"
    else:
        return "general_node"

def build_graph() -> StateGraph:
    graph = StateGraph(ChatState)

    graph.add_node("router_node",      router_node)
    graph.add_node("math_node",        math_node)
    graph.add_node("rag_node",         rag_node)
    graph.add_node("memory_node",      memory_node)
    graph.add_node("general_node",     general_node)
    graph.add_node("search_node",      search_node)
    graph.add_node("save_memory_node", save_memory_node)

    graph.set_entry_point("router_node")

    graph.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "math_node":    "math_node",
            "rag_node":     "rag_node",
            "memory_node":  "memory_node",
            "general_node": "general_node",
            "search_node":  "search_node",
        },
    )

    graph.add_edge("math_node",   "save_memory_node")
    graph.add_edge("rag_node",    "save_memory_node")
    graph.add_edge("memory_node",  "save_memory_node")
    graph.add_edge("general_node", "save_memory_node")
    graph.add_edge("search_node",  "save_memory_node")
    graph.add_edge("save_memory_node", END)

    compiled = graph.compile()
    logger.info("[Graph] LangGraph compiled successfully.")
    return compiled

_compiled_graph = build_graph()

def run_chat(query: str, session_id: str) -> Dict:
    initial_state: ChatState = {
        "query":      query,
        "session_id": session_id,
        "history":    [],
        "route":      None,
        "context":    "",
        "response":   "",
        "agent_used": "",
    }

    final_state = _compiled_graph.invoke(initial_state)

    return {
        "response":   final_state["response"],
        "agent_used": final_state["agent_used"],
        "route":      final_state["route"],
    }
