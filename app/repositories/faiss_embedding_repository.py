import logging
import os
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings
from app.interfaces.embedding_repository import IEmbeddingRepository

logger = logging.getLogger(__name__)


class FAISSEmbeddingRepository(IEmbeddingRepository):
    """FAISS vector store implementation for embeddings."""
    
    def __init__(self):
        self.index_dir = settings.FAISS_INDEX_DIR
        self.api_key = settings.API_KEY
    
    def _get_embeddings_model(self):
        """Get embeddings model instance."""
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
    
    async def save_embeddings(self, texts: list[str], metadatas: list[dict]) -> bool:
        """Save embeddings to vector store."""
        try:
            logger.info(f"Saving {len(texts)} embeddings to FAISS")
            
            embeddings = self._get_embeddings_model()
            os.makedirs(self.index_dir, exist_ok=True)
            
            index_path = os.path.join(self.index_dir, "index.faiss")
            
            # Check if index exists
            if os.path.exists(index_path):
                try:
                    # Load and update existing index
                    vector_store = FAISS.load_local(
                        self.index_dir,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    vector_store.add_texts(texts, metadatas=metadatas)
                    vector_store.save_local(self.index_dir)
                    logger.info(f"Updated existing FAISS index with {len(texts)} texts")
                except Exception as e:
                    logger.warning(f"Error updating index, creating new: {e}")
                    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
                    vector_store.save_local(self.index_dir)
                    logger.info(f"Created new FAISS index with {len(texts)} texts")
            else:
                # Create new index
                vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
                vector_store.save_local(self.index_dir)
                logger.info(f"Created new FAISS index with {len(texts)} texts")
            
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}", exc_info=True)
            return False
    
    async def load_embeddings(self) -> Optional[object]:
        """Load embeddings from vector store."""
        try:
            embeddings = self._get_embeddings_model()
            index_path = os.path.join(self.index_dir, "index.faiss")
            
            if not os.path.exists(index_path):
                logger.warning("FAISS index not found")
                return None
            
            vector_store = FAISS.load_local(
                self.index_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded FAISS index successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}", exc_info=True)
            return None
    
    async def search_similar(self, query: str, k: int = 5) -> list[object]:
        """Search for similar documents."""
        try:
            vector_store = await self.load_embeddings()
            if not vector_store:
                logger.warning("No vector store available for search")
                return []
            
            docs = vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} similar documents for query")
            return docs
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}", exc_info=True)
            return []
    
    async def delete_embeddings(self, course_id: int) -> bool:
        """Delete embeddings for a course."""
        try:
            logger.info(f"Deleting embeddings for course {course_id}")
            
            index_path = os.path.join(self.index_dir, "index.faiss")
            
            if not os.path.exists(index_path):
                logger.warning(f"No FAISS index found")
                return True
            
            embeddings = self._get_embeddings_model()
            vector_store = FAISS.load_local(
                self.index_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Get docstore
            docstore = vector_store.docstore._dict if hasattr(vector_store.docstore, '_dict') else {}
            
            # Collect remaining documents
            remaining_texts = []
            remaining_metadatas = []
            deleted_count = 0
            
            for doc_id, doc in docstore.items():
                metadata = getattr(doc, 'metadata', {}) or {}
                
                if str(metadata.get('course_id')) != str(course_id):
                    # Keep this document
                    remaining_texts.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
                    remaining_metadatas.append(metadata)
                else:
                    deleted_count += 1
            
            if deleted_count == 0:
                logger.warning(f"No embeddings found for course {course_id}")
                return True
            
            if not remaining_texts:
                # Delete index files
                index_path = os.path.join(self.index_dir, "index.faiss")
                pkl_path = os.path.join(self.index_dir, "index.pkl")
                
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
                
                logger.info(f"Cleared FAISS index (deleted {deleted_count} vectors)")
                return True
            
            # Rebuild index
            new_vector_store = FAISS.from_texts(remaining_texts, embedding=embeddings, metadatas=remaining_metadatas)
            new_vector_store.save_local(self.index_dir)
            
            logger.info(f"Deleted {deleted_count} vectors, rebuilt index with {len(remaining_texts)} remaining")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}", exc_info=True)
            return False
    
    async def exists(self) -> bool:
        """Check if embeddings exist."""
        index_path = os.path.join(self.index_dir, "index.faiss")
        return os.path.exists(index_path)
