from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.db.mongodb import db
from app.core.config import settings
from app.services.pdf import read_all_pdfs_text
from app.services.rag import (
    get_text_chunks,
    get_vector_store,
    get_conversational_chain,
    GoogleGenerativeAIEmbeddings,
    FAISS,
)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        api_key = settings.API_KEY
        model_name = settings.MODEL_NAME
        if not api_key:
            raise HTTPException(status_code=500, detail="API key is not configured in environment variables")

        text = read_all_pdfs_text(db)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Không thể đọc nội dung từ file PDF")

        text_chunks = get_text_chunks(text, model_name)
        _ = get_vector_store(text_chunks, model_name, api_key)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )

        new_db = FAISS.load_local(settings.FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(request.question)
        chain = get_conversational_chain(model_name, api_key=api_key)

        # chain expects context variable; create_stuff_documents_chain returns string resp
        response = chain.invoke({"context": docs, "question": request.question})

        return ChatResponse(
            answer=response,
            timestamp=settings.now_string(),
            model_name=model_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
