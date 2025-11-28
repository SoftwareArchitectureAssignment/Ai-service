import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.routers import api_router
from app.services.course_event_consumer import start_consumer, stop_consumer
from app.services.file_event_consumer import start_file_consumer, stop_file_consumer

logger = logging.getLogger(__name__)

app = FastAPI(title="Ai Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/ai")

@app.on_event("startup")
async def startup_event():
    """Initialize Redis Stream consumers on startup."""
    try:
        await start_consumer()
        await start_file_consumer()
        logger.info("Application startup completed")
    except Exception as e:
        
        logger.error(f"Failed to start application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown Redis Stream consumers."""
    try:
        await stop_consumer()
        await stop_file_consumer()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
