from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from gridfs import GridFS

from app.db.mongodb import db

router = APIRouter()


@router.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...), user_id: str = Form(None)):
    try:
        fs = GridFS(db)

        file_infos = []
        for file in files:
            content = await file.read()

            file_id = fs.put(content, filename=file.filename, content_type=file.content_type)

            file_info = {
                "filename": file.filename,
                "upload_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "user_id": user_id,
                "file_id": str(file_id),
                "size": len(content),
                "metadata": {
                    "content_type": file.content_type
                }
            }

            result = db.files.insert_one(file_info)
            file_info["_id"] = str(result.inserted_id)
            file_infos.append(file_info)

        return {"message": "PDF files uploaded successfully", "files": file_infos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pdf-files")
async def list_pdf_files(user_id: Optional[str] = None):
    try:
        query = {}
        if user_id:
            query["user_id"] = user_id

        files = list(db.files.find(query))
        for file in files:
            file["_id"] = str(file["_id"])
            file["file_id"] = str(file["file_id"])

        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pdf-files/{file_id}")
async def delete_pdf_file(file_id: str):
    try:
        fs = GridFS(db)

        try:
            fs.delete(ObjectId(file_id))
        except Exception:
            pass

        result = db.files.delete_one({"file_id": file_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="PDF file not found")

        return {"message": "PDF file deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
