import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate as LCPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import HTTPException
from app.interfaces.ai_model_provider import IAIModelProvider

logger = logging.getLogger(__name__)


class GoogleAIModelProvider(IAIModelProvider):
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name
    
    def validate_configuration(self) -> bool:
        if not self.api_key:
            raise HTTPException(status_code=400, detail="API key not configured")
        return True
    
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        variables: dict = None
    ) -> str:
        try:
            self.validate_configuration()
            
            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=temperature,
                google_api_key=self.api_key
            )
            
            lc_prompt = LCPromptTemplate(
                template=prompt,
                input_variables=list(variables.keys()) if variables else []
            )
            
            chain = lc_prompt | model | StrOutputParser()
            
            response = chain.invoke(variables or {})
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise
