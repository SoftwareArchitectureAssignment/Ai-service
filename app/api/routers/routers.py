from fastapi import APIRouter
from app.api.controllers.chat import router as chat_router
from app.api.controllers.files import router as files_router
from app.api.controllers.health import router as health_router

# Root router
api_router = APIRouter()

# Include all routers
api_router.include_router(chat_router, prefix="", tags=["Chat"])
api_router.include_router(files_router, prefix="", tags=["Files"])
api_router.include_router(health_router, prefix="", tags=["Health"])