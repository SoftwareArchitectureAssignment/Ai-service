"""
Dependency Injection Container for managing service dependencies.

This follows the Dependency Inversion Principle (DIP) by providing
a centralized location for service instantiation and management.
"""

import logging
from app.repositories.faiss_embedding_repository import FAISSEmbeddingRepository
from app.services.embedding_service import EmbeddingService
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)


class Container:
    """IoC Container for dependency injection."""
    
    _instance = None
    _repositories = {}
    _services = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize all dependencies."""
        logger.info("Initializing dependency injection container")
        
        # Initialize repositories
        self._repositories['embedding'] = FAISSEmbeddingRepository()
        
        # Initialize services
        self._services['embedding'] = EmbeddingService(
            self._repositories['embedding']
        )
        self._services['chat'] = ChatService(
            self._repositories['embedding']
        )
        
        logger.info("Dependency injection container initialized")
    
    def get_embedding_repository(self) -> FAISSEmbeddingRepository:
        """Get embedding repository instance."""
        return self._repositories['embedding']
    
    def get_embedding_service(self) -> EmbeddingService:
        """Get embedding service instance."""
        return self._services['embedding']
    
    def get_chat_service(self) -> ChatService:
        """Get chat service instance."""
        return self._services['chat']


# Global container instance
container = Container()
