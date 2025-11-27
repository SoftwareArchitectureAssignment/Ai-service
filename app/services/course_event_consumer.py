import asyncio
import logging
from typing import Optional
from redis.asyncio import Redis
from app.schemas.course_event import CourseUpdateEvent
from app.core.config import settings
from app.core.container import container

logger = logging.getLogger(__name__)

class CourseEventConsumer:
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.stream_key = "course-updates"
        self.consumer_group = "ai-service-group"
        self.consumer_name = "ai-service-consumer-1"
        self.running = False
        # Initialize embedding service from container
        self.embedding_service = container.get_embedding_service()

    async def connect(self) -> None:
        """Initialize Redis connection."""
        try:
            # Support both URL format (Render Redis Cloud) and individual host/port/password
            if settings.REDIS_URL:
                logger.info(f"Connecting to Redis via URL")
                self.redis = await Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                )
                connection_info = "Render Redis Cloud"
            else:
                logger.info(f"Connecting to Redis via host/port")
                self.redis = await Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    socket_keepalive_options={
                        1: (9, 3, 3),  # TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT
                    }
                )
                connection_info = f"{settings.REDIS_HOST}:{settings.REDIS_PORT}"
            
            # Test connection
            await self.redis.ping()
            logger.info(f"Connected to Redis at {connection_info}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")

    async def create_consumer_group(self) -> None:
        """Create consumer group if it doesn't exist."""
        try:
            await self.redis.xgroup_create(
                self.stream_key,
                self.consumer_group,
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group '{self.consumer_group}'")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{self.consumer_group}' already exists")
            else:
                logger.error(f"Error creating consumer group: {e}")
                raise

    async def process_event(self, event: CourseUpdateEvent) -> bool:
        """
        Process a course update event.
        Updates FAISS embeddings based on the action (CREATE, UPDATE, DELETE).
        """
        try:
            logger.info(f"Processing {event.action} event for course {event.courseId}: {event.courseName}")
            
            if event.action == "CREATE":
                # Ingest new course into FAISS
                success = await self.embedding_service.ingest_course(
                    course_id=event.courseId,
                    course_name=event.courseName,
                    description=event.courseDescription
                )
                
                if success:
                    logger.info(f"✅ Successfully created embeddings for course {event.courseId}")
                else:
                    logger.error(f"❌ Failed to create embeddings for course {event.courseId}")
                return success
                
            elif event.action == "UPDATE":
                # Update existing course embeddings
                success = await self.embedding_service.update_course(
                    course_id=event.courseId,
                    course_name=event.courseName,
                    description=event.courseDescription
                )
                
                if success:
                    logger.info(f"✅ Successfully updated embeddings for course {event.courseId}")
                else:
                    logger.error(f"❌ Failed to update embeddings for course {event.courseId}")
                return success
                
            elif event.action == "DELETE":
                # Remove course embeddings from FAISS
                success = await self.embedding_service.delete_course(
                    course_id=event.courseId
                )
                
                if success:
                    logger.info(f"✅ Successfully deleted embeddings for course {event.courseId}")
                else:
                    logger.error(f"❌ Failed to delete embeddings for course {event.courseId}")
                return success
            else:
                logger.warning(f"Unknown action: {event.action}")
                return False
        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error processing event for course {event.courseId}: {e}", exc_info=True)
            return False

    async def consume(self) -> None:
        """
        Main consumer loop.
        Reads from Redis Stream and processes course update events.
        """
        if not self.redis:
            raise RuntimeError("Redis not connected. Call connect() first.")
        
        await self.create_consumer_group()
        self.running = True
        
        logger.info(f"Starting consumer loop for stream '{self.stream_key}'")
        
        while self.running:
            try:
                # Read pending messages first (in case of previous failures)
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
                            await self._process_and_ack(message_id, data)
                
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                # Reconnect on error
                try:
                    await self.redis.ping()
                except:
                    logger.warning("Redis connection lost, attempting to reconnect...")
                    try:
                        await self.connect()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect: {reconnect_error}")
                        await asyncio.sleep(5)  # Wait before retry

    async def _process_and_ack(self, message_id: str, data: dict) -> None:
        """
        Process a single message and acknowledge it.
        
        Args:
            message_id: Redis stream message ID
            data: Message data as dict
        """
        try:
            # Parse message data
            event = CourseUpdateEvent(
                courseId=int(data.get("courseId", 0)),
                courseName=data.get("courseName", ""),
                courseDescription=data.get("courseDescription"),
                action=data.get("action", ""),
                timestamp=int(data.get("timestamp", 0))
            )
            
            # Process the event
            success = await self.process_event(event)
            
            if success:
                # Acknowledge the message
                await self.redis.xack(self.stream_key, self.consumer_group, message_id)
                logger.info(f"Acknowledged message {message_id}")
            else:
                logger.warning(f"Event processing failed for message {message_id}, will retry")
        
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            # Don't acknowledge on error - will be retried

    def stop(self) -> None:
        """Signal consumer to stop."""
        self.running = False
        logger.info("Stopping consumer loop")


# Global consumer instance
consumer = CourseEventConsumer()


async def start_consumer():
    """Startup event handler - initialize and start consumer."""
    try:
        await consumer.connect()
        asyncio.create_task(consumer.consume())
        logger.info("Course event consumer started")
    except Exception as e:
        logger.error(f"Failed to start consumer: {e}")
        raise


async def stop_consumer():
    """Shutdown event handler - gracefully stop consumer."""
    try:
        consumer.stop()
        await asyncio.sleep(1)  # Allow current message to finish
        await consumer.disconnect()
        logger.info("Course event consumer stopped")
    except Exception as e:
        logger.error(f"Error stopping consumer: {e}")
