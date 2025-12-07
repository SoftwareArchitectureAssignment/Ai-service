from abc import ABC, abstractmethod


class IResponseParser(ABC):
    
    @abstractmethod
    def parse_text_response(self, response: str) -> str:
        pass
    
    @abstractmethod
    def parse_json_response(self, response: str) -> dict:
        pass
    
    @abstractmethod
    def parse_learning_path_response(self, response: str) -> dict:
        pass
