from pydantic import BaseModel
from typing import Optional

class FileMetadata(BaseModel):
    filename: str
    download_url: str
    user_id: Optional[str] = None
    size: Optional[int] = None
    content_type: Optional[str] = "application/pdf"