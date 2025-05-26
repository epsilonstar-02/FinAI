from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import settings

class Embedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._instance
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents."""
        if not texts:
            return []
        return self._model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if not text:
            return []
        return self.embed_documents([text])[0]

# Singleton instance
embedder = Embedder()
