import logging
from fastapi import APIRouter
from typing import List, Optional

from app.schemas.files import FileMetadata, FileInfoResponse, FileListResponse, ProcessFilesResponse
from app.services.file_management_service import FileManagementService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/register-files", response_model=List[FileInfoResponse])
async def register_files(files_metadata: List[FileMetadata]):
   
    return FileManagementService.register_files(files_metadata)
  
@router.get("/pdf-files", response_model=List[FileListResponse])
async def list_pdf_files(user_id: Optional[str] = None):
        return FileManagementService.list_pdf_files(user_id)


@router.delete("/pdf-files/{file_id}", response_model=bool)
async def delete_pdf_file(file_id: str):
        return FileManagementService.delete_pdf_file(file_id)
@router.post("/process-files", response_model=ProcessFilesResponse)
async def process_files():
    return await FileManagementService.process_unprocessed_files()
