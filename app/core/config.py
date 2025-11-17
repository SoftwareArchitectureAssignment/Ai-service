from pydantic_settings import BaseSettings
from datetime import datetime


class Settings(BaseSettings):
    MONGODB_URI: str
    DB_NAME: str = "pdf_chatbot"
    COLLECTION_NAME: str = "pdf_files"
    API_KEY: str | None = None
    MODEL_NAME: str = "Google AI"
    DATA_DIR: str = "data"
    FAISS_INDEX_DIR: str = "data/faiss_index"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @staticmethod
    def now_string() -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


settings = Settings()
