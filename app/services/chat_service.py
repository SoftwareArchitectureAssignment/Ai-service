import logging
from typing import Optional
from fastapi import HTTPException
from app.interfaces.embedding_repository import IEmbeddingRepository
from app.interfaces.ai_model_provider import IAIModelProvider
from app.interfaces.prompt_builder import IPromptBuilder
from app.interfaces.response_parser import IResponseParser
from app.interfaces.context_builder import IContextBuilder

logger = logging.getLogger(__name__)


class ChatService:
    
    def __init__(
        self,
        repository: IEmbeddingRepository,
        ai_provider: IAIModelProvider,
        prompt_builder: IPromptBuilder,
        response_parser: IResponseParser,
        context_builder: IContextBuilder
    ):
        self.repository = repository
        self.ai_provider = ai_provider
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.context_builder = context_builder
    
    async def evaluate_question(self, question: str, question_uid: str) -> str:
        try:
            if not await self.repository.exists():
                raise HTTPException(
                    status_code=404,
                    detail="No course embeddings found. Please upload courses first."
                )
            
            docs = await self.repository.search_similar(question, k=5)
            if not docs:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant courses found for this question."
                )
            
            context = self.context_builder.build_rag_context(docs)
            
            prompt_template, variables = self.prompt_builder.build_rag_prompt(
                context, question
            )
            response = await self.ai_provider.generate_response(
                prompt_template,
                temperature=0.3,
                variables=variables
            )
            
            return self.response_parser.parse_text_response(response)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error evaluating question: {e}", exc_info=True)
            raise
    
    
    async def get_learning_path(
        self,
        topics: str,
        level: Optional[str],
        questions: str
    ) -> dict:
        try:
            logger.info(f"Generating learning path for topics: {topics}")
            
            if not await self.repository.exists():
                raise HTTPException(
                    status_code=404,
                    detail="No course embeddings found. Please upload courses first."
                )
            
            docs = await self.repository.search_similar(topics, k=5)
            if not docs:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant courses found for these topics."
                )
            
            context = self.context_builder.build_context_with_metadata(docs)
            
            prompt_template, variables = self.prompt_builder.build_learning_path_prompt(
                context,
                topics,
                level or "beginner",
                questions
            )
            response = await self.ai_provider.generate_response(
                prompt_template,
                temperature=0.3,
                variables=variables
            )
            
            return self.response_parser.parse_learning_path_response(response)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating learning path: {e}", exc_info=True)
            raise
    
    
    async def chat_free(self, message: str) -> str:
        try:
            prompt_template, variables = self.prompt_builder.build_free_chat_prompt(
                message
            )
            response = await self.ai_provider.generate_response(
                prompt_template,
                temperature=0.7,
                variables=variables
            )
            
            return self.response_parser.parse_text_response(response)
            
        except Exception as e:
            logger.error(f"Error in free chat: {e}", exc_info=True)
            raise

