from pydantic import BaseModel
from typing import Optional


class FileUpdateEvent(BaseModel):
    """Event schema for file updates from course-domain."""
    
    file_id: str
    filename: str
    download_url: str
    action: str  # CREATE, UPDATE, DELETE
    user_id: Optional[str] = None
    size: Optional[int] = None
    content_type: Optional[str] = "application/pdf"
    timestamp: int
    
    class Config:
        from_attributes = True
