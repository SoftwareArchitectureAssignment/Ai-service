from abc import ABC, abstractmethod

class IContextBuilder(ABC):
  
    @abstractmethod
    def build_rag_context(self, docs: list) -> str:
       
        pass
    
    @abstractmethod
    def build_context_with_metadata(self, docs: list) -> str:
       
        pass
