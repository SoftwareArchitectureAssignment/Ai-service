import logging
import os
from typing import Optional
from app.interfaces.embedding_repository import IEmbeddingRepository
from app.services.rag import get_text_chunks
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for managing course embeddings."""
    
    def __init__(self, repository: IEmbeddingRepository):
        self.repository = repository
    
    async def ingest_course(self, course_id: int, course_name: str, description: Optional[str] = None) -> bool:
        """Ingest a course into embeddings."""
        try:
            logger.info(f"Ingesting course {course_id}: {course_name}")
            
            # Build course text from event data only
            course_text = f"Course: {course_name}\n"
            if description:
                course_text += f"Description: {description}"
            else:
                course_text += "Description: No description provided"
            
            logger.debug(f"Course text: {course_text}")
            
            # Split text into chunks
            text_chunks = get_text_chunks(course_text, settings.MODEL_NAME)
            
            if not text_chunks:
                logger.warning(f"No text chunks generated for course {course_id}")
                return False
            
            # Create metadata for each chunk
            metadatas = [
                {
                    "course_id": str(course_id),
                    "course_name": course_name,
                    "chunk_index": i
                }
                for i in range(len(text_chunks))
            ]
            
            # Save to repository
            success = await self.repository.save_embeddings(text_chunks, metadatas)
            
            if success:
                logger.info(f"✅ Successfully ingested course {course_id} with {len(text_chunks)} chunks")
            else:
                logger.error(f"❌ Failed to ingest course {course_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error ingesting course {course_id}: {e}", exc_info=True)
            return False
    
    async def update_course(self, course_id: int, course_name: str, description: Optional[str] = None) -> bool:
        """Update embeddings for a course."""
        try:
            logger.info(f"Updating embeddings for course {course_id}")
            
            # Delete old embeddings
            await self.repository.delete_embeddings(course_id)
            
            # Re-ingest course
            success = await self.ingest_course(course_id, course_name, description)
            
            if success:
                logger.info(f"✅ Updated embeddings for course {course_id}")
            else:
                logger.error(f"Failed to update embeddings for course {course_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error updating course {course_id}: {e}", exc_info=True)
            return False
    
    async def delete_course(self, course_id: int) -> bool:
        """Delete embeddings for a course."""
        try:
            logger.info(f"Deleting embeddings for course {course_id}")
            success = await self.repository.delete_embeddings(course_id)
            
            if success:
                logger.info(f"✅ Deleted embeddings for course {course_id}")
            else:
                logger.error(f"Failed to delete embeddings for course {course_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error deleting course {course_id}: {e}", exc_info=True)
            return False
