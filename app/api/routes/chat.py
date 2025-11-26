from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse, LearningPathRequest, LearningPathResponse, CourseRecommendation
from app.core.config import settings
from app.services.rag import (
    get_conversational_chain,
    load_vector_store,
)
from langchain_core.prompts import PromptTemplate as LCPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import json

router = APIRouter()


@router.post("/evaluate", response_model=ChatResponse)
async def evaluate(request: ChatRequest):
    try:
        api_key = settings.API_KEY
        model_name = settings.MODEL_NAME
        if not api_key:
            raise HTTPException(status_code=500, detail="API key is not configured in environment variables")

        vector_store = load_vector_store(api_key)
        
        if vector_store is None:
            raise HTTPException(
                status_code=400, 
                detail="No embeddings found. Please register and process files first."
            )

        docs = vector_store.similarity_search(request.question)
        
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No relevant information found in the knowledge base."
            )

        chain = get_conversational_chain(model_name, api_key=api_key)
        response = chain.invoke({"context": docs, "question": request.question})

        return ChatResponse(
            answer=response,
            question_uid=request.question_uid,
            timestamp=settings.now_string(),
            model_name=model_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning-path", response_model=LearningPathResponse)
async def get_learning_path(request: LearningPathRequest):
    try:
        api_key = settings.API_KEY
        if not api_key:
            raise HTTPException(status_code=500, detail="API key is not configured in environment variables")
        
        # Load vector store to query relevant documents
        vector_store = load_vector_store(api_key)
        
        if vector_store is None:
            raise HTTPException(
                status_code=400, 
                detail="No embeddings found. Please register and process files first."
            )
        
        # Search for relevant documents based on topics
        docs = vector_store.similarity_search(request.topics, k=5)
        
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No relevant information found in the knowledge base for the given topics."
            )
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt to query Gemini AI with all request parameters
        prompt_template = """
        Based on the provided context and learning requirements, create a comprehensive learning path.
        
        Context (from knowledge base):\n{context}\n
        
        Topics to learn: {topics}
        Current level: {level}
        Questions/Requirements: {questions}
        
        Please provide:
        1. General advice for the learning path
        2. Recommended learning paths with courses (provide course name, UID, and description)
        3. Explanation of why these courses are recommended
        
        Format your response as JSON with the following structure:
        {{
            "advice": "General advice text",
            "recommendedLearningPaths": [
                {{
                    "course_name": "Course Name",
                    "course_uid": "unique_identifier",
                    "description": "Course description"
                }}
            ],
            "explanation": "Explanation text"
        }}
        """
        
        # Initialize Gemini model
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            google_api_key=api_key,
        )
        
        # Create prompt
        prompt = LCPromptTemplate(
            template=prompt_template,
            input_variables=["context", "topics", "level", "questions"],
        )
        
        # Create chain
        chain = prompt | model | StrOutputParser()
        
        # Invoke chain with all request parameters
        response_text = chain.invoke({
            "context": context,
            "topics": request.topics,
            "level": request.level or "beginner",
            "questions": request.questions
        })
        
        # Parse JSON response from Gemini
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed_response = json.loads(json_str)
            else:
                parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from the text
            parsed_response = {
                "advice": response_text,
                "recommendedLearningPaths": [
                    {
                        "course_name": "General Learning Path",
                        "course_uid": "general_path_001",
                        "description": "Based on your topics and requirements"
                    }
                ],
                "explanation": "Refer to the advice section for detailed explanation."
            }
        
        return LearningPathResponse(
            advice=parsed_response.get("advice", ""),
            recommendedLearningPaths=[
                CourseRecommendation(
                    course_name=course.get("course_name", ""),
                    course_uid=course.get("course_uid", ""),
                    description=course.get("description", "")
                )
                for course in parsed_response.get("recommendedLearningPaths", [])
            ],
            explanation=parsed_response.get("explanation", "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))