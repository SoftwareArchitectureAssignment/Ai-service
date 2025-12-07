from abc import ABC, abstractmethod


class IPromptBuilder(ABC):
   
    @abstractmethod
    def build_rag_prompt(self, context: str, question: str) -> tuple[str, dict]:
        pass
    
    @abstractmethod
    def build_learning_path_prompt(
        self,
        context: str,
        topics: str,
        level: str,
        questions: str
    ) -> tuple[str, dict]:
        
        pass
    
    @abstractmethod
    def build_free_chat_prompt(self, message: str) -> tuple[str, dict]:
       
        pass
