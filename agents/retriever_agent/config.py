from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

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
    HUGGINGFACE = "huggingface"                    # HuggingFace models

class Settings(BaseSettings):
    # Vector store configuration
    VECTOR_STORE_PATH: str = "data/vector_store"               # Path to store vector databases
    VECTOR_STORE_TYPE: VectorStoreType = VectorStoreType.FAISS  # Type of vector store to use (FAISS is free and open source)
    COLLECTION_NAME: str = "finai_documents"                   # Collection/index name
    
    # Embedding model configuration - Prioritizing free and open-source options
    EMBEDDING_MODEL_TYPE: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS  # Free and open source
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"                 # Default embedding model (free and open source)
    EMBEDDING_BATCH_SIZE: int = 16                            # Batch size for embeddings
    EMBEDDING_DIMENSIONS: Optional[int] = None                # Dimensions (None = auto)
    
    # NOTE: The following API keys are for paid services and are OPTIONAL
    # If not provided, system will use free and open-source alternatives only
    # API keys (optional, can be set in .env)
    OPENAI_API_KEY: Optional[str] = None                      # For OpenAI embeddings (paid service)
    HUGGINGFACE_API_KEY: Optional[str] = None                # For HuggingFace Hub (free tier available)
    
    # Query settings
    DEFAULT_TOP_K: int = 5                                    # Default number of results
    SIMILARITY_THRESHOLD: float = 0.7                         # Minimum similarity score
    MAX_TOKENS_PER_DOC: int = 1000                          # Max tokens per document
    
    # Performance settings
    CACHE_EMBEDDINGS: bool = True                            # Cache embeddings in memory
    CACHE_TTL_SECONDS: int = 3600                           # Cache TTL in seconds
    ENABLE_HYBRID_SEARCH: bool = True                        # Use hybrid search when available
    
    # Advanced features
    ENABLE_RERANKING: bool = False                           # Use reranking model
    RERANKING_MODEL: Optional[str] = None                    # Reranking model name
    CHUNK_SIZE: int = 512                                    # Document chunking size
    CHUNK_OVERLAP: int = 50                                  # Overlap between chunks
    
    # Pydantic v2+ configuration
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields from env vars meant for other services
    }

# Create instance
settings = Settings()

# Ensure vector store directory exists
Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
