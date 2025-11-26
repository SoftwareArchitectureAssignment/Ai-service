from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
import hashlib
from app.schemas.files import FileMetadata
from app.services.pdf import process_files_from_urls
from app.services.rag import delete_vectors_by_file_id

from app.db.mongodb import db
from bson import ObjectId

router = APIRouter()

@router.post("/register-files")
async def register_files(files_metadata: List[FileMetadata]):
    try:
        file_infos = []
        for file_meta in files_metadata:
            url_hash = hashlib.sha256(file_meta.download_url.encode()).hexdigest()
            
            existing = db.processed_files.find_one({"url_hash": url_hash})
            
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

        return {"message": "Files registered successfully", "files": file_infos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pdf-files")
async def list_pdf_files():
    try:
        query = {}

        files = list(db.files.find(query))
        for file in files:
            file["_id"] = str(file["_id"])

        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pdf-files/{file_id}")
async def delete_pdf_file(file_id: str):
    try:

        result = db.files.delete_one({"_id": ObjectId(file_id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="PDF file not found")

        delete_vectors_by_file_id(file_id)
        
        return {"message": "PDF file and its vectors deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-files")
async def process_files():
    try:
        query = {"embedding_created": False}
        
        unprocessed_files = list(db.files.find(query))
        
        if not unprocessed_files:
            return {"message": "No files to process", "processed_count": 0}
        
        download_urls = [(f["download_url"], f["url_hash"]) for f in unprocessed_files]
        processed_count = await process_files_from_urls(download_urls, db)
        
        return {
            "message": f"Successfully processed {processed_count} files",
            "processed_count": processed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
