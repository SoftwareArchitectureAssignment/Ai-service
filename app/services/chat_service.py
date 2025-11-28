import logging
import json
from typing import Optional
from langchain_core.prompts import PromptTemplate as LCPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.interfaces.embedding_repository import IEmbeddingRepository
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class ChatService:
    
    def __init__(self, repository: IEmbeddingRepository):
        self.repository = repository
        self.api_key = settings.API_KEY
        self.model_name = settings.MODEL_NAME
    
    async def evaluate_question(self, question: str, question_uid: str) -> str:
        try:
            
            if not self.api_key:
                raise HTTPException(status_code=400, detail="API key not configured")
            if not await self.repository.exists():
                raise HTTPException(status_code=404, detail="No course embeddings found. Please upload courses first.")
            
            docs = await self.repository.search_similar(question, k=5)
            
            if not docs:
                raise HTTPException(status_code=404, detail="No relevant courses found for this question.")
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt_template = """
            Based on the provided course information, answer the following question.
            
            Course Information:
            {context}
            
            Question: {question}
            
            Please provide a helpful answer based on the available course information.
            """
            
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.3,
                google_api_key=self.api_key
            )
            
            prompt = LCPromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = prompt | model | StrOutputParser()
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return response
        except Exception as e:
            logger.error(f"Error evaluating question: {e}", exc_info=True)
            raise
    
    async def get_learning_path(self, topics: str, level: Optional[str], questions: str) -> dict:
        try:
            logger.info(f"Generating learning path for topics: {topics}")
            
            if not self.api_key:
                raise HTTPException(status_code=400, detail="API key not configured")
            
            if not await self.repository.exists():
                raise HTTPException(status_code=404, detail="No course embeddings found. Please upload courses first.")
            
            docs = await self.repository.search_similar(topics, k=5)
            
            if not docs:
                raise HTTPException(status_code=404, detail="No relevant courses found for these topics.")
            
            # Build context with course metadata including uid
            context_parts = []
            for doc in docs:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                course_uid = metadata.get('course_uid', 'unknown')
                course_name = metadata.get('course_name', 'Unknown Course')
                context_parts.append(f"{doc.page_content}\nCourse UID: {course_uid}")
            
            context = "\n\n".join(context_parts)
            
            prompt_template = """
            Based on the provided course catalog and learning requirements, create a comprehensive learning path.
            
            Available Courses:
            {context}
            
            Topics to learn: {topics}
            Current level: {level}
            Specific requirements: {questions}
            
            Please provide:
            1. General advice for this learning path
            2. Recommended courses with their details and exact Course UID from the available courses
            3. Explanation of why these courses are recommended
            
            Format your response as JSON with this structure:
            {{
                "advice": "General learning advice",
                "recommendedLearningPaths": [
                    {{
                        "course_name": "Course Name",
                        "course_uid": "exact_uid_from_available_courses",
                        "description": "Course description"
                    }}
                ],
                "explanation": "Why these courses are recommended"
            }}
            
            IMPORTANT: The course_uid MUST be taken directly from the "Course UID:" field in the available courses above.
            """
            
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.3,
                google_api_key=self.api_key
            )
            
            prompt = LCPromptTemplate(
                template=prompt_template,
                input_variables=["context", "topics", "level", "questions"]
            )
            
            chain = prompt | model | StrOutputParser()
            
            response_text = chain.invoke({
                "context": context,
                "topics": topics,
                "level": level or "beginner",
                "questions": questions
            })
            
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    parsed_response = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw response")
                parsed_response = {
                    "advice": response_text,
                    "recommendedLearningPaths": [],
                    "explanation": "Could not parse structured response"
                }
            
            return parsed_response
        except Exception as e:
            logger.error(f"Error generating learning path: {e}", exc_info=True)
            raise
