from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Path to store the FAISS vector store
    VECTOR_STORE_PATH: str = "data/vector_store"
    
    # Model name for sentence-transformers
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Number of results to return by default
    DEFAULT_TOP_K: int = 5
    
    class Config:
        env_file = ".env"
        extra = "ignore"

# Create instance
settings = Settings()

# Ensure vector store directory exists
Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
