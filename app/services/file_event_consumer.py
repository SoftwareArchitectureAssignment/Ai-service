import asyncio
import logging
from typing import Optional
from redis.asyncio import Redis
from app.schemas.file_event import FileUpdateEvent
from app.services.file_event_service import FileEventService
from app.core.config import settings

logger = logging.getLogger(__name__)


class FileEventConsumer:
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.stream_key = "file-updates"
        self.consumer_group = "ai-service-group"
        self.consumer_name = "ai-service-file-consumer-1"
        self.running = False

    async def connect(self) -> None:
        try:
            if settings.REDIS_URL:
                logger.info(f"Connecting to Redis via URL for file events")
                self.redis = await Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                )
                connection_info = "Render Redis Cloud"
            else:
                logger.info(f"Connecting to Redis via host/port for file events")
                self.redis = await Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    socket_keepalive_options={
                        1: (9, 3, 3),
                    }
                )
                connection_info = f"{settings.REDIS_HOST}:{settings.REDIS_PORT}"
            
            # Test connection
            await self.redis.ping()
            logger.info(f"Connected to Redis ({connection_info}) for file events")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise e

    async def disconnect(self) -> None:
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis (file consumer)")

    async def create_consumer_group(self) -> None:
        try:
            await self.redis.xgroup_create(
                self.stream_key,
                self.consumer_group,
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group '{self.consumer_group}' for stream '{self.stream_key}'")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{self.consumer_group}' already exists for stream '{self.stream_key}'")
            else:
                logger.error(f"Error creating consumer group: {e}")
                raise

    async def consume(self) -> None:
        """Consume file events from Redis Stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected. Call connect() first.")
        
        await self.create_consumer_group()
        self.running = True
        
        logger.info(f"Starting file event consumer for stream '{self.stream_key}'")
        
        while self.running:
            try:
                pending = await self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    streams={self.stream_key: ">"},
                    count=10,
                    block=1000
                )
                
                if pending:
                    for stream_key, messages in pending:
                        for message_id, data in messages:
                            await self._process_and_ack_file(message_id, data)
                
            except Exception as e:
                logger.error(f"Error in file consumer loop: {e}")
                try:
                    await self.redis.ping()
                except:
                    logger.warning("Redis connection lost (file consumer), attempting to reconnect...")
                    try:
                        await self.connect()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect: {reconnect_error}")
                        await asyncio.sleep(5)  # Wait before retry

    async def _process_and_ack_file(self, message_id: str, data: dict) -> None:
        try:
            event = FileUpdateEvent(
                file_id=data.get("fileId", ""),
                filename=data.get("filename", ""),
                download_url=data.get("downloadUrl", ""),
                action=data.get("action", ""),
                user_id=data.get("userId"),
                size=int(data.get("size", 0)) if data.get("size") else None,
                content_type=data.get("contentType", "application/pdf"),
                timestamp=int(data.get("timestamp", 0))
            )
            
            logger.info(f"Processing file event: {event.action} - {event.file_id}")
            success = await FileEventService.handle_file_event(event)
            
            if success:
                await self.redis.xack(self.stream_key, self.consumer_group, message_id)
                logger.info(f"Acknowledged file event message {message_id}")
            else:
                logger.warning(f"File event processing failed for message {message_id}, will retry")
        
        except Exception as e:
            logger.error(f"Error processing file message {message_id}: {e}", exc_info=True)

    def stop(self) -> None:
        self.running = False
        logger.info("Stopping file event consumer")


# Global file consumer instance
file_consumer = FileEventConsumer()


async def start_file_consumer():
    try:
        await file_consumer.connect()
        asyncio.create_task(file_consumer.consume())
        logger.info("File event consumer started")
    except Exception as e:
        logger.error(f"Failed to start file consumer: {e}")
        raise


async def stop_file_consumer():
    try:
        file_consumer.stop()
        await asyncio.sleep(1)
        await file_consumer.disconnect()
        logger.info("File event consumer stopped")
    except Exception as e:
        logger.error(f"Error stopping file consumer: {e}")
