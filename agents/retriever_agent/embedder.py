# agents/retriever_agent/embedder.py
# No significant changes needed. It's a functional singleton for SentenceTransformer.
# Small type hint improvement.

from typing import List, Optional # Optional not used, but fine.
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import settings
import logging

logger = logging.getLogger(__name__)

class Embedder:
    _instance: Optional['Embedder'] = None # Type hint for _instance
    _model: Optional[SentenceTransformer] = None # Type hint for _model

    def __new__(cls) -> 'Embedder': # Return type hint
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            try:
                logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                # TODO: Consider adding model_kwargs for device mapping ('cuda', 'mps', 'cpu')
                # device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._model = SentenceTransformer(settings.EMBEDDING_MODEL) 
                logger.info(f"Embedding model '{settings.EMBEDDING_MODEL}' loaded successfully.")
                if settings.EMBEDDING_DIMENSIONS is None and cls._model is not None:
                    # Auto-set embedding dimensions from the loaded model
                    # This assumes the SentenceTransformer model object has a way to expose its dimension.
                    # Common way is model.get_sentence_embedding_dimension()
                    try:
                        dim = cls._model.get_sentence_embedding_dimension()
                        # Mutating settings like this is generally not ideal after init,
                        # but for auto-detection it can be practical.
                        # A better way would be to have the embedder instance store this.
                        # For now, let's log it.
                        logger.info(f"Detected embedding dimension: {dim} for model {settings.EMBEDDING_MODEL}")
                        # settings.EMBEDDING_DIMENSIONS = dim # Avoid mutating global settings here
                    except Exception as e_dim:
                        logger.warning(f"Could not auto-detect embedding dimension: {e_dim}")

            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{settings.EMBEDDING_MODEL}': {e}", exc_info=True)
                # Application might not be viable without an embedder.
                # Depending on desired behavior, could raise an error here to halt startup.
                # For now, it will fail later when embed_documents/query is called.
                cls._model = None # Ensure model is None if loading fails
        return cls._instance
    
    def get_embedding_dimension(self) -> Optional[int]:
        if self._model:
            try:
                return self._model.get_sentence_embedding_dimension()
            except Exception:
                return None
        return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents."""
        if self._model is None:
            logger.error("Embedding model not loaded. Cannot embed documents.")
            # Return list of empty lists or raise error, matching embed_query behavior
            return [[] for _ in texts] if texts else []
        if not texts:
            return []
        # SentenceTransformer can take a batch_size argument in encode.
        embeddings_np = self._model.encode(
            texts, 
            convert_to_numpy=True, 
            batch_size=settings.EMBEDDING_BATCH_SIZE
        )
        return embeddings_np.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if self._model is None:
            logger.error("Embedding model not loaded. Cannot embed query.")
            return [] # Return empty list if model not loaded
        if not text:
            return []
        # embed_documents handles list, so wrap and unwrap for single query
        embedded_query = self.embed_documents([text])
        return embedded_query[0] if embedded_query else []

embedder_instance = Embedder() # Renamed for clarity vs. class name

def get_embedder() -> Embedder: # Getter for the singleton
    return embedder_instance