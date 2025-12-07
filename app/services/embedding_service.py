import logging
from typing import Optional
from app.interfaces.embedding_repository import IEmbeddingRepository
from app.services.rag import get_text_chunks
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    
    def __init__(self, repository: IEmbeddingRepository):
        self.repository = repository
    
    async def ingest_course(
        self,
        course_id: int,
        course_name: str,
        description: Optional[str] = None,
        topic: Optional[str] = None,
        course_uid: Optional[str] = None
    ) -> bool:
        
        try:
            course_text = self._build_course_text(
                course_name, description, topic
            )
            
            text_chunks = get_text_chunks(course_text, settings.MODEL_NAME)
            
            if not text_chunks:
                logger.warning(f"No text chunks generated for course {course_id}")
                return False
            
            metadatas = self._build_metadatas(
                course_id, course_name, topic, course_uid, len(text_chunks)
            )
            
            success = await self.repository.save_embeddings(text_chunks, metadatas)
            if not success:
                logger.error(f"Failed to ingest course {course_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error ingesting course {course_id}: {e}", exc_info=True)
            return False
    
    async def update_course(
        self,
        course_id: int,
        course_name: str,
        description: Optional[str] = None,
        topic: Optional[str] = None,
        course_uid: Optional[str] = None
    ) -> bool:
        try:
            await self.repository.delete_embeddings(course_id)
            success = await self.ingest_course(
                course_id, course_name, description, topic, course_uid
            )
            
            if not success:
                logger.error(f"Failed to update embeddings for course {course_id}")
            return success
        except Exception as e:
            logger.error(f"Error updating course {course_id}: {e}", exc_info=True)
            return False
    
    async def delete_course(self, course_id: int) -> bool:
        try:
            success = await self.repository.delete_embeddings(course_id)
            if not success:
                logger.error(f"Failed to delete embeddings for course {course_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting course {course_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _build_course_text(
        course_name: str,
        description: Optional[str],
        topic: Optional[str]
    ) -> str:
        course_text = f"Course: {course_name}\n"
        course_text += f"Description: {description or 'No description provided'}\n"
        if topic:
            course_text += f"Topic: {topic}"
        return course_text
    
    @staticmethod
    def _build_metadatas(
        course_id: int,
        course_name: str,
        topic: Optional[str],
        course_uid: Optional[str],
        chunk_count: int
    ) -> list[dict]:
        return [
            {
                "course_id": str(course_id),
                "course_uid": course_uid or str(course_id),
                "course_name": course_name,
                "topic": topic or "unknown",
                "chunk_index": i
            }
            for i in range(chunk_count)
        ]
