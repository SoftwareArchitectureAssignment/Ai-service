import logging
from app.interfaces.prompt_builder import IPromptBuilder

logger = logging.getLogger(__name__)


class PromptBuilder(IPromptBuilder):
    
    def build_rag_prompt(self, context: str, question: str) -> tuple[str, dict]:
        prompt_template = """
        Based on the provided course information, answer the following question.
        
        Course Information:
        {context}
        
        Question: {question}
        
        Please provide a helpful answer based on the available course information.
        """
        return prompt_template, {"context": context, "question": question}
    
    def build_learning_path_prompt(
        self,
        topics: str,
        level: str,
        questions: str
    ) -> tuple[str, dict]:
        prompt_template = """
        Based on the provided course catalog and learning requirements, create a comprehensive learning path.
        
        
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
        return prompt_template, {
            "topics": topics,
            "level": level,
            "questions": questions
        }
    
    def build_free_chat_prompt(self, message: str) -> tuple[str, dict]:
        prompt_template = """
        You are a helpful AI assistant. Answer the user's question or respond to their message in a clear and helpful manner.
        
        User Message: {message}
        
        Please provide a helpful response.
        """
        return prompt_template, {"message": message}
