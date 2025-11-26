from typing import Optional
from pydantic import BaseModel
from enum import Enum

class CourseAction(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"

class CourseUpdateEvent(BaseModel):
    courseId: int
    courseName: str
    courseDescription: Optional[str] = None
    action: CourseAction 
    timestamp: int
