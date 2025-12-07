from abc import ABC, abstractmethod
from typing import Optional


class IAIModelProvider(ABC):
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        variables: Optional[dict] = None
    ) -> str:
       
        pass
    
    @abstractmethod
    def validate_configuration(self) -> bool:
       
        pass
