from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.db.mongodb import db

router = APIRouter()


@router.get("/conversations")
async def get_conversations(user_id: str, limit: int = 10, page: int = 1):
    try:
        skip = (page - 1) * limit

        conversations = list(
            db.conversations
            .find({"user_id": user_id})
            .sort("timestamp", -1)
            .skip(skip)
            .limit(limit)
        )

        total_count = db.conversations.count_documents({"user_id": user_id})

        formatted_conversations = []
        for conv in conversations:
            conv["_id"] = str(conv["_id"])
            formatted_conversations.append(conv)

        return {
            "conversations": formatted_conversations,
            "total_count": total_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
