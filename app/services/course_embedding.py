import logging
from typing import Optional
from app.services.rag import get_text_chunks
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

logger = logging.getLogger(__name__)


class CourseEmbeddingService:
    """
    Manages course embeddings in FAISS vector store.
    
    Data source: Exclusively from Redis Stream events published by Course Service
    - Ensures embeddings use EXACT course data from Course Service
    - Keeps AI responses aligned with course catalog
    - No external data fetching or enrichment
    """
    
    @staticmethod
    async def ingest_course(course_id: int, course_name: str, description: Optional[str] = None) -> bool:
        try:
            logger.info(f"Ingesting course {course_id}: {course_name}")
            
            # Build course text ONLY from the event data (course_name + description)
            # This ensures we use EXACT data from Course Service, not fetched from elsewhere
            course_text = f"Course: {course_name}\n"
            if description:
                course_text += f"Description: {description}"
            else:
                course_text += "Description: No description provided"
            
            logger.debug(f"Course text for embedding: {course_text}")
            
            # Split text into chunks for embedding
            text_chunks = get_text_chunks(course_text, settings.MODEL_NAME)
            
            if not text_chunks:
                logger.warning(f"No text chunks generated for course {course_id}")
                return False
            
            # Convert string chunks to text list with metadata
            # FAISS.add_texts accepts: texts (list of strings) and metadatas (list of dicts)
            metadatas = []
            for i in range(len(text_chunks)):
                metadatas.append({
                    "course_id": str(course_id),
                    "course_name": course_name,
                    "chunk_index": i
                })
            
            # Update or create FAISS vector store
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.API_KEY
            )
            
            os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
            index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
            
            # Check if existing index exists
            if os.path.exists(index_path):
                try:
                    # Load existing index and add new chunks with metadata
                    vector_store = FAISS.load_local(
                        settings.FAISS_INDEX_DIR,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    # Add new texts with metadata
                    vector_store.add_texts(text_chunks, metadatas=metadatas)
                    vector_store.save_local(settings.FAISS_INDEX_DIR)
                    logger.info(f"Updated existing FAISS index with {len(text_chunks)} chunks for course {course_id}")
                except Exception as e:
                    logger.warning(f"Error updating existing index, creating new one: {e}")
                    # Create new vector store
                    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
                    vector_store.save_local(settings.FAISS_INDEX_DIR)
                    logger.info(f"Created new FAISS index with {len(text_chunks)} chunks for course {course_id}")
            else:
                # Create new vector store
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
                vector_store.save_local(settings.FAISS_INDEX_DIR)
                logger.info(f"Created new FAISS index with {len(text_chunks)} chunks for course {course_id}")
            
            logger.info(f"✅ Successfully ingested course {course_id} with {len(text_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting course {course_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def update_course_embedding(course_id: int, course_name: str, description: Optional[str] = None) -> bool:
        try:
            logger.info(f"Updating embedding for course {course_id}: {course_name}")
            
            # First delete old embeddings for this course
            await CourseEmbeddingService.delete_course_embedding(course_id)
            
            # Then re-ingest the course
            success = await CourseEmbeddingService.ingest_course(course_id, course_name, description)
            
            if success:
                logger.info(f"✅ Successfully updated embedding for course {course_id}")
            else:
                logger.error(f"Failed to update embedding for course {course_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating course embedding {course_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def delete_course_embedding(course_id: int) -> bool:
        try:
            logger.info(f"Deleting embedding for course {course_id}")
            
            index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
            
            if not os.path.exists(index_path):
                logger.warning(f"No FAISS index found, nothing to delete for course {course_id}")
                return True
            
            # Load embeddings for reconstruction
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.API_KEY
            )
            
            # Load the vector store
            vector_store = FAISS.load_local(
                settings.FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Get the docstore
            docstore = vector_store.docstore._dict if hasattr(vector_store.docstore, '_dict') else {}
            
            # Collect remaining documents (exclude those for this course)
            remaining_docs = []
            remaining_metadatas = []
            remaining_texts = []
            deleted_count = 0
            
            for doc_id, doc in docstore.items():
                # Extract metadata from document
                metadata = getattr(doc, 'metadata', {}) or {}
                
                if str(metadata.get('course_id')) != str(course_id):
                    # Keep this document
                    remaining_texts.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
                    remaining_metadatas.append(metadata)
                    remaining_docs.append(doc)
                else:
                    deleted_count += 1
            
            if deleted_count == 0:
                logger.warning(f"No vectors found for course {course_id}")
                return True
            
            if not remaining_texts:
                # No remaining documents, delete the index
                index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
                pkl_path = os.path.join(settings.FAISS_INDEX_DIR, "index.pkl")
                
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
                    
                logger.info(f"✅ Cleared FAISS index (deleted {deleted_count} vectors for course {course_id})")
                return True
            
            # Rebuild FAISS index with remaining documents
            new_vector_store = FAISS.from_texts(remaining_texts, embedding=embeddings, metadatas=remaining_metadatas)
            new_vector_store.save_local(settings.FAISS_INDEX_DIR)
            
            logger.info(f"✅ Deleted {deleted_count} vectors for course {course_id}, rebuilt index with {len(remaining_texts)} remaining vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting course embedding {course_id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _build_course_text(course: dict) -> str:
        """
        DEPRECATED: This method is no longer used.
        
        We now build course text ONLY from Redis Stream event data (course_name + description)
        to ensure we use EXACT data from Course Service without fetching additional data.
        
        This keeps embeddings synchronized with the exact course information from Course Service.
        """
        parts = []
        if course.get('name'):
            parts.append(f"Course: {course['name']}")
        if course.get('description'):
            parts.append(f"Description: {course['description']}")
        
        return "\n".join(parts) if parts else "No course information"
