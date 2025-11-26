import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.files import router as files_router
from app.api.routes.health import router as health_router
from app.services.course_event_consumer import start_consumer, stop_consumer

logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files_router)
app.include_router(chat_router)
app.include_router(health_router)

@app.on_event("startup")
async def startup_event():
    """Initialize Redis Stream consumer on startup."""
    try:
        await start_consumer()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown Redis Stream consumer."""
    try:
        await stop_consumer()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
