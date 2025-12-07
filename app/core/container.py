import logging
from app.repositories.faiss_embedding_repository import FAISSEmbeddingRepository
from app.services.embedding_service import EmbeddingService
from app.services.chat_service import ChatService
from app.services.providers.google_ai_provider import GoogleAIModelProvider
from app.services.prompt_builder import PromptBuilder
from app.services.response_parser import ResponseParser
from app.services.context_builder import ContextBuilder
from app.core.config import settings

logger = logging.getLogger(__name__)


class Container:
   
    _instance = None
    _repositories = {}
    _services = {}
    _providers = {}
    _builders = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        logger.info("Initializing dependency injection container")
        
        self._repositories['embedding'] = FAISSEmbeddingRepository()
        
        self._providers['google_ai'] = GoogleAIModelProvider(
            api_key=settings.API_KEY,
            model_name="gemini-2.5-flash-lite"
        )
        
        self._builders['prompt'] = PromptBuilder()
        self._builders['context'] = ContextBuilder()
        self._builders['response'] = ResponseParser()
        
        self._services['embedding'] = EmbeddingService(
            self._repositories['embedding']
        )
        self._services['chat'] = ChatService(
            repository=self._repositories['embedding'],
            ai_provider=self._providers['google_ai'],
            prompt_builder=self._builders['prompt'],
            response_parser=self._builders['response'],
            context_builder=self._builders['context']
        )
        
        logger.info("Dependency injection container initialized")
    
    def get_embedding_repository(self) -> FAISSEmbeddingRepository:
        return self._repositories['embedding']
    
    def get_embedding_service(self) -> EmbeddingService:
        return self._services['embedding']
    
    def get_chat_service(self) -> ChatService:
        return self._services['chat']
    
    def get_ai_provider(self) -> GoogleAIModelProvider:
        return self._providers['google_ai']
    
    def get_prompt_builder(self) -> PromptBuilder:
        return self._builders['prompt']
    
    def get_context_builder(self) -> ContextBuilder:
        return self._builders['context']
    
    def get_response_parser(self) -> ResponseParser:
        return self._builders['response']


container = Container()
