import logging
import json
from app.interfaces.response_parser import IResponseParser

logger = logging.getLogger(__name__)


class ResponseParser(IResponseParser):
    
    def parse_text_response(self, response: str) -> str:
        if not response:
            return ""
        return response.strip()
    
    def parse_json_response(self, response: str) -> dict:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            raise
    
    def parse_learning_path_response(self, response: str) -> dict:
        try:
            return self.parse_json_response(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, returning fallback")
            return {
                "advice": response,
                "recommendedLearningPaths": [],
                "explanation": "Could not parse structured response"
            }
