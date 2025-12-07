from abc import ABC, abstractmethod


class IEventConsumer(ABC):
  
    @abstractmethod
    async def start(self) -> None:
       
        pass
    
    @abstractmethod
    async def stop(self) -> None:
       
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
       
        pass
