from typing import List
from pydantic import BaseModel
from bson import ObjectId


class ConversationHistory(BaseModel):
    user_id: str
    question: str
    answer: str
    timestamp: str
    model_name: str

    class Config:
        json_encoders = {
            ObjectId: lambda v: str(v)
        }


class ConversationHistoryResponse(BaseModel):
    conversations: List[ConversationHistory]
    total_count: int
