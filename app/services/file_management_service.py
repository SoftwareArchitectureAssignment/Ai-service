import logging
import hashlib
from datetime import datetime
from typing import List
from bson import ObjectId
from app.schemas.files import FileMetadata, ProcessFilesResponse
from app.services.pdf import process_files_from_urls
from app.services.rag import delete_vectors_by_file_id
from app.core.mongodb import db

logger = logging.getLogger(__name__)


class FileManagementService:
    @staticmethod
    def _generate_url_hash(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()
    
    @staticmethod
    def _check_existing_file(url_hash: str) -> dict:
        return db.processed_files.find_one({"url_hash": url_hash})
    
    @staticmethod
    def register_files(files_metadata: List[FileMetadata]) -> List[dict]:
        try:
            file_infos = []
            for file_meta in files_metadata:
                url_hash = FileManagementService._generate_url_hash(file_meta.download_url)
                existing = FileManagementService._check_existing_file(url_hash)
                
                file_info = {
                    "filename": file_meta.filename,
                    "download_url": file_meta.download_url,
                    "url_hash": url_hash,
                    "user_id": file_meta.user_id,
                    "size": file_meta.size,
                    "content_type": file_meta.content_type,
                    "upload_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "embedding_created": existing is not None,
                    "processed_date": existing.get("processed_date") if existing else None
                }
                
                result = db.files.insert_one(file_info)
                file_info["_id"] = str(result.inserted_id)
                file_infos.append(file_info)
            
            return file_infos
        except Exception as e:
            logger.error(f"Error registering files: {e}")
            raise ValueError(f"Failed to register files: {str(e)}")
    
    @staticmethod
    def list_pdf_files(user_id: str = None) -> List[dict]:
        try:
            query = {"user_id": user_id} if user_id else {}
            files = list(db.files.find(query))
            
            result = []
            for doc in files:
                doc["file_id"] = str(doc["_id"])
                doc.pop("_id", None)
                result.append(doc)
            
            return result
        except Exception as e:
            logger.error(f"Error listing PDF files: {e}")
            raise ValueError(f"Failed to list PDF files: {str(e)}")
    
    @staticmethod
    def delete_pdf_file(file_id: str) -> bool:
        try:
            result = db.files.delete_one({"_id": ObjectId(file_id)})
            
            if result.deleted_count == 0:
                raise ValueError(f"PDF file with ID {file_id} not found")
            
            delete_vectors_by_file_id(file_id)
            
            return True
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error deleting PDF file: {e}")
            raise ValueError(f"Failed to delete PDF file: {str(e)}")
    
    @staticmethod
    async def process_unprocessed_files() -> ProcessFilesResponse:
        try:
            query = {"embedding_created": False}
            unprocessed_files = list(db.files.find(query))
            
            if not unprocessed_files:
                logger.info("No files to process")
                return ProcessFilesResponse(processed_count=0, message="No files to process")
            
            download_urls = [(f["download_url"], f["url_hash"]) for f in unprocessed_files]
            processed_count = await process_files_from_urls(download_urls, db)
            
            return ProcessFilesResponse(processed_count=processed_count)
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            raise ValueError(f"Failed to process files: {str(e)}")
