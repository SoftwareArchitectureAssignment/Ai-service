from pydantic import BaseModel
import os
from datetime import datetime


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    timestamp: str
    model_name: str
