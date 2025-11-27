from abc import ABC, abstractmethod


class IRAGService(ABC):
    """Interface for RAG (Retrieval-Augmented Generation) operations."""
    
    @abstractmethod
    async def retrieve_documents(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant documents from vector store."""
        pass
    
    @abstractmethod
    async def generate_response(self, context: str, question: str) -> str:
        """Generate response using LLM with context."""
        pass
    
    @abstractmethod
    async def create_chain(self) -> object:
        """Create a conversational chain for Q&A."""
        pass
