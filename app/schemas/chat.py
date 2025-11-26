from pydantic import BaseModel
from typing import Optional
from typing import List

class ChatRequest(BaseModel):
    question: str
    question_uid: str

class ChatResponse(BaseModel):
    answer: str
    question_uid: str
    timestamp: str
    model_name: str
    
    
class LearningPathRequest(BaseModel):
    topics: str
    level: Optional[str]
    questions: str

class CourseRecommendation(BaseModel):
    course_name: str
    course_uid: str
    description: str
class LearningPathResponse(BaseModel):
    advice: str
    recommendedLearningPaths: List[CourseRecommendation]
    explanation: str