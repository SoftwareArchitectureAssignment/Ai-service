from abc import ABC, abstractmethod
from typing import Optional


class IEmbeddingRepository(ABC):
    """Interface for embedding data access layer."""
    
    @abstractmethod
    async def save_embeddings(self, texts: list[str], metadatas: list[dict]) -> bool:
        """Save embeddings to vector store."""
        pass
    
    @abstractmethod
    async def load_embeddings(self) -> Optional[object]:
        """Load embeddings from vector store."""
        pass
    
    @abstractmethod
    async def search_similar(self, query: str, k: int = 5) -> list[object]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, course_id: int) -> bool:
        """Delete embeddings for a course."""
        pass
    
    @abstractmethod
    async def exists(self) -> bool:
        """Check if embeddings exist."""
        pass
