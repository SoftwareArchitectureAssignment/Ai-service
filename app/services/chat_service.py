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
            logger.info(f"Evaluating question: {question}")
            
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
            
            logger.info(f"✅ Generated response for question {question_uid}")
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
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt_template = """
            Based on the provided course catalog and learning requirements, create a comprehensive learning path.
            
            Available Courses:
            {context}
            
            Topics to learn: {topics}
            Current level: {level}
            Specific requirements: {questions}
            
            Please provide:
            1. General advice for this learning path
            2. Recommended courses with their details
            3. Explanation of why these courses are recommended
            
            Format your response as JSON with this structure:
            {{
                "advice": "General learning advice",
                "recommendedLearningPaths": [
                    {{
                        "course_name": "Course Name",
                        "course_uid": "unique_identifier",
                        "description": "Course description"
                    }}
                ],
                "explanation": "Why these courses are recommended"
            }}
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
            
            logger.info(f"✅ Generated learning path successfully")
            return parsed_response
        except Exception as e:
            logger.error(f"Error generating learning path: {e}", exc_info=True)
            raise
