import logging
import hashlib
from datetime import datetime
from app.schemas.file_event import FileUpdateEvent
from app.services.pdf import process_files_from_urls
from app.services.rag import delete_vectors_by_file_id
from app.core.mongodb import db
from app.core.config import settings

logger = logging.getLogger(__name__)


class FileEventService:
    
    @staticmethod
    def _generate_url_hash(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()
    
    @staticmethod
    async def handle_file_event(event: FileUpdateEvent) -> bool:
        try:
            logger.info(f"Processing {event.action} event for file {event.file_id}: {event.filename}")

            if event.action == "CREATE":
                return await FileEventService._handle_file_create(event)
            elif event.action == "UPDATE":
                return await FileEventService._handle_file_update(event)
            elif event.action == "DELETE":
                return await FileEventService._handle_file_delete(event)
            else:
                logger.warning(f"Unknown file action: {event.action}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling file event: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def _handle_file_create(event: FileUpdateEvent) -> bool:
        try:
            url_hash = FileEventService._generate_url_hash(event.download_url)
            
            existing = db.files.find_one({"url_hash": url_hash})
            if existing:
                logger.info(f"File {event.file_id} already exists, skipping creation")
                return True
            
            file_doc = {
                "file_id": event.file_id,
                "filename": event.filename,
                "download_url": event.download_url,
                "url_hash": url_hash,
                "user_id": event.user_id,
                "size": event.size,
                "content_type": event.content_type,
                "upload_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "embedding_created": False,
                "processed_date": None
            }
            
            result = db.files.insert_one(file_doc)
            logger.info(f"Inserted file document: {result.inserted_id}")
            
            success = await FileEventService._process_file_embeddings(
                event.download_url,
                url_hash
            )
            
            if success:
                db.files.update_one(
                    {"_id": result.inserted_id},
                    {
                        "$set": {
                            "embedding_created": True,
                            "processed_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                )
                logger.info(f"Successfully created embeddings for file {event.file_id}")
            else:
                logger.warning(f"Failed to create embeddings for file {event.file_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling file create event: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def _handle_file_update(event: FileUpdateEvent) -> bool:
        try:
            
            result = db.files.update_one(
                {"file_id": event.file_id},
                {
                    "$set": {
                        "filename": event.filename,
                        "download_url": event.download_url,
                        "size": event.size,
                        "content_type": event.content_type
                    }
                }
            )
            _id_ai_service=db.files.find_one({"file_id": event.file_id}).get("_id")
            if result.matched_count == 0:
                logger.warning(f"File {event.file_id} not found during update")
                return False
            
            
            url_hash = FileEventService._generate_url_hash(event.download_url)
            delete_vectors_by_file_id(event.file_id, file_id_AI_service=_id_ai_service , api_key=settings.API_KEY)
            
            success = await FileEventService._process_file_embeddings(
                event.download_url,
                url_hash
            )
            
            if success:
                db.files.update_one(
                    {"file_id": event.file_id},
                    {
                        "$set": {
                            "embedding_created": True,
                            "processed_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                )
                logger.info(f"Successfully updated embeddings for file {event.file_id}")
            else:
                logger.warning(f"Failed to update embeddings for file {event.file_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error handling file update event: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def _handle_file_delete(event: FileUpdateEvent) -> bool:
        try:
            
            _id_ai_service=db.files.find_one({"file_id": event.file_id}).get("_id")
            
            result = db.files.delete_one({"file_id": event.file_id})
            
            if result.deleted_count == 0:
                logger.warning(f"File {event.file_id} not found during deletion")
                return False
            db.processed_files.delete_one({"file_id": str(_id_ai_service)})
            delete_vectors_by_file_id(event.file_id, file_id_AI_service=_id_ai_service, api_key=settings.API_KEY)
            
            logger.info(f"Successfully deleted file {event.file_id} and its embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error handling file delete event: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def _process_file_embeddings(download_url: str, url_hash: str) -> bool:
        try:
            processed_count = await process_files_from_urls(
                [(download_url, url_hash)],
                db
            )
            return processed_count > 0
        except Exception as e:
            logger.error(f"Error processing file embeddings: {e}", exc_info=True)
            return False
