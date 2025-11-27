import os
import logging
import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.course_event_consumer import consumer
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str  # "healthy" or "unhealthy"
    redis_consumer: dict
    faiss_index: dict
    overall_status: str


class ConsumerStatusResponse(BaseModel):
    """Consumer status response"""
    is_running: bool
    is_connected: bool
    stream_key: str
    consumer_group: str
    consumer_name: str


class FAISSStatusResponse(BaseModel):
    """FAISS index status response"""
    index_exists: bool
    index_path: str
    index_size_mb: Optional[float] = None
    last_modified: Optional[str] = None


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Get overall health status of the AI service.
    
    Returns:
        HealthResponse with status of Redis consumer and FAISS index
    """
    try:
        # Check Redis consumer status
        consumer_status = {
            "is_running": consumer.running,
            "is_connected": consumer.redis is not None,
            "stream_key": consumer.stream_key,
            "consumer_group": consumer.consumer_group
        }
        
        # Check FAISS index status
        index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
        faiss_status = {
            "index_exists": os.path.exists(index_path),
            "index_path": index_path
        }
        
        if faiss_status["index_exists"]:
            # Get file size
            file_size = os.path.getsize(index_path)
            faiss_status["index_size_mb"] = round(file_size / (1024 * 1024), 2)
            
            # Get last modified time
            mod_time = os.path.getmtime(index_path)
            faiss_status["last_modified"] = datetime.datetime.fromtimestamp(mod_time).isoformat()
        
        overall_status = "healthy" if consumer.is_connected and faiss_status["index_exists"] else "degraded"
        
        return HealthResponse(
            status=overall_status,
            redis_consumer=consumer_status,
            faiss_index=faiss_status,
            overall_status=overall_status
        )
        
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return HealthResponse(
            status="unhealthy",
            redis_consumer={
                "is_running": False,
                "is_connected": False,
                "stream_key": consumer.stream_key,
                "consumer_group": consumer.consumer_group
            },
            faiss_index={
                "index_exists": False,
                "index_path": os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
            },
            overall_status="unhealthy"
        )


@router.get("/redis", response_model=ConsumerStatusResponse)
async def redis_status():
    return ConsumerStatusResponse(
        is_running=consumer.running,
        is_connected=consumer.redis is not None,
        stream_key=consumer.stream_key,
        consumer_group=consumer.consumer_group,
        consumer_name=consumer.consumer_name
    )


@router.get("/faiss", response_model=FAISSStatusResponse)
async def faiss_status():
    index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
    
    response = FAISSStatusResponse(
        index_exists=os.path.exists(index_path),
        index_path=index_path
    )
    
    if response.index_exists:
        # Get file size
        file_size = os.path.getsize(index_path)
        response.index_size_mb = round(file_size / (1024 * 1024), 2)
        
        # Get last modified time
        mod_time = os.path.getmtime(index_path)
        response.last_modified = datetime.datetime.fromtimestamp(mod_time).isoformat()
    
    return response
