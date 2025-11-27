from abc import ABC, abstractmethod
from typing import Optional


class IChatService(ABC):
    """Interface for chat/RAG service."""
    
    @abstractmethod
    async def evaluate_question(self, question: str, question_uid: str) -> str:
        """Evaluate a question using RAG."""
        pass
    
    @abstractmethod
    async def get_learning_path(self, topics: str, level: Optional[str], questions: str) -> dict:
        """Generate a learning path based on topics and requirements."""
        pass
