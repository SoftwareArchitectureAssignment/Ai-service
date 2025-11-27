from pydantic import BaseModel, Field
from typing import Optional

class FileMetadata(BaseModel):
    filename: str
    download_url: str
    user_id: Optional[str] = None
    size: Optional[int] = None
    content_type: Optional[str] = "application/pdf"
    
class FileInfoResponse(BaseModel):
    file_id: str = Field(..., alias="_id")
    filename: str
    download_url: str
    url_hash: str
    user_id: Optional[str] = None
    size: Optional[int] = None
    content_type: Optional[str] = "application/pdf"
    upload_date: str
    embedding_created: bool
    processed_date: Optional[str] = None
    
class FileListResponse(BaseModel):
    file_id: str = Field(..., alias="_id")
    filename: str
    download_url: str
    embedding_created: bool
    class Config:
        populate_by_name = True
    
class ProcessFilesResponse(BaseModel):
    processed_count: int