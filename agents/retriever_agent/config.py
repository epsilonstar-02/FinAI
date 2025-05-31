# agents/retriever_agent/config.py
# No changes needed, seems fine. Original content preserved.

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
import logging # Added for log level validation

# Define enum types for vector store and embedding model
class VectorStoreType(str, Enum):
    """Supported vector store types."""
    FAISS = "faiss"            # Facebook AI Similarity Search
    CHROMA = "chroma"          # ChromaDB
    QDRANT = "qdrant"          # Qdrant vector database
    WEAVIATE = "weaviate"      # Weaviate vector database

class EmbeddingModelType(str, Enum):
    """Supported embedding model types."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # HuggingFace Sentence Transformers
    OPENAI = "openai"                              # OpenAI embeddings
    HUGGINGFACE = "huggingface"                    # HuggingFace models (can be same as sentence_transformers but for clarity)

class Settings(BaseSettings):
    # Vector store configuration
    VECTOR_STORE_PATH: str = "data/vector_store"
    VECTOR_STORE_TYPE: VectorStoreType = VectorStoreType.FAISS
    COLLECTION_NAME: str = "finai_documents"
    
    # Embedding model configuration
    EMBEDDING_MODEL_TYPE: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_DIMENSIONS: Optional[int] = None
    
    # API keys
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None # Typically for inference API, not local sentence-transformers
    
    # Query settings
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7 
    MAX_TOKENS_PER_DOC: int = 1000 
    
    # Performance settings
    CACHE_EMBEDDINGS: bool = True 
    CACHE_TTL_SECONDS: int = 3600
    ENABLE_HYBRID_SEARCH: bool = True # Note: FAISS doesn't natively support hybrid search well. Chroma/Qdrant/Weaviate do.
    
    # Advanced features
    ENABLE_RERANKING: bool = False
    RERANKING_MODEL: Optional[str] = None 
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    LOG_LEVEL: str = "INFO" # Added LOG_LEVEL

    # Pydantic v2+ configuration
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()

# Ensure vector store directory exists
Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)

# Configure logging
log_level_to_set = settings.LOG_LEVEL.upper()
if not hasattr(logging, log_level_to_set):
    logging.warning(f"Invalid LOG_LEVEL '{log_level_to_set}' in Retriever Agent settings. Defaulting to INFO.")
    log_level_to_set = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_to_set),
    format="%(asctime)s - %(name)s (RETRIEVER_AGENT) - %(levelname)s - %(message)s"
)