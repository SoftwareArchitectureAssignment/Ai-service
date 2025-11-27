import logging
import json
from typing import Optional
from langchain_core.prompts import PromptTemplate as LCPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.interfaces.embedding_repository import IEmbeddingRepository

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat and learning path queries."""
    
    def __init__(self, repository: IEmbeddingRepository):
        self.repository = repository
        self.api_key = settings.API_KEY
        self.model_name = settings.MODEL_NAME
    
    async def evaluate_question(self, question: str, question_uid: str) -> str:
        """Evaluate a question using RAG with course embeddings."""
        try:
            logger.info(f"Evaluating question: {question}")
            
            if not self.api_key:
                raise ValueError("API key not configured")
            
            # Check if embeddings exist
            if not await self.repository.exists():
                raise ValueError("No course embeddings found. Please upload courses first.")
            
            # Search for similar documents
            docs = await self.repository.search_similar(question, k=5)
            
            if not docs:
                raise ValueError("No relevant courses found for this question.")
            
            # Format context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt_template = """
            Based on the provided course information, answer the following question.
            
            Course Information:
            {context}
            
            Question: {question}
            
            Please provide a helpful answer based on the available course information.
            """
            
            # Initialize model and create chain
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
            
            # Generate response
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
        """Generate learning path based on topics and requirements."""
        try:
            logger.info(f"Generating learning path for topics: {topics}")
            
            if not self.api_key:
                raise ValueError("API key not configured")
            
            # Check if embeddings exist
            if not await self.repository.exists():
                raise ValueError("No course embeddings found. Please upload courses first.")
            
            # Search for relevant courses
            docs = await self.repository.search_similar(topics, k=5)
            
            if not docs:
                raise ValueError("No relevant courses found for these topics.")
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt for learning path
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
            
            # Initialize model and create chain
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
            
            # Generate response
            response_text = chain.invoke({
                "context": context,
                "topics": topics,
                "level": level or "beginner",
                "questions": questions
            })
            
            # Parse JSON response
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
