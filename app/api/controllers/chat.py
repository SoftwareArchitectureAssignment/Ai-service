import logging
from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse, LearningPathRequest, LearningPathResponse
from app.core.config import settings
from app.core.container import container

logger = logging.getLogger(__name__)
router = APIRouter()

chat_service = container.get_chat_service()


@router.post("/evaluate", response_model=ChatResponse)
async def evaluate(request: ChatRequest) -> ChatResponse:
    answer = await chat_service.evaluate_question(
        request.question,
        request.question_uid
    )
    return ChatResponse(
        answer=answer,
        question_uid=request.question_uid,
        timestamp=settings.now_string(),
        model_name=settings.MODEL_NAME
    )
    


@router.post("/learning-path", response_model=LearningPathResponse)
async def get_learning_path(request: LearningPathRequest) -> LearningPathResponse:
        learning_path = await chat_service.get_learning_path(
            request.topics,
            request.level,
            request.questions
        )
        return LearningPathResponse(
            advice=learning_path.get("advice", ""),
            recommendedLearningPaths=learning_path.get("recommendedLearningPaths", []),
            explanation=learning_path.get("explanation", "")
        )