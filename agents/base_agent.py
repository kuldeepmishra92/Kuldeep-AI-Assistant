from abc import ABC, abstractmethod
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        logger.debug(f"Agent initialised: {self.name}")

    @abstractmethod
    def run(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        session_id: str = "",
    ) -> str:
        ...

    def __repr__(self) -> str:
        return f"<Agent: {self.name}>"
