import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from pathlib import Path
from .models import Document, SearchResult
from .config import settings
from .embedder import embedder

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = {}
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.namespace = "default"
        self._ensure_vector_store_dir()
        
    def _ensure_vector_store_dir(self):
        """Ensure the vector store directory exists."""
        Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
    
    def _get_index_path(self, namespace: Optional[str] = None) -> str:
        """Get the path to the FAISS index file."""
        ns = namespace or self.namespace
        return os.path.join(settings.VECTOR_STORE_PATH, f"{ns}.index")
    
    def _get_documents_path(self, namespace: Optional[str] = None) -> str:
        """Get the path to the documents JSON file."""
        ns = namespace or self.namespace
        return os.path.join(settings.VECTOR_STORE_PATH, f"{ns}_documents.json")
    
    def save(self, namespace: Optional[str] = None):
        """Save the index and documents to disk."""
        if self.index is None:
            return
            
        ns = namespace or self.namespace
        faiss.write_index(self.index, self._get_index_path(ns))
        
        # Save documents as JSON
        with open(self._get_documents_path(ns), 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def load(self, namespace: Optional[str] = None) -> bool:
        """Load the index and documents from disk."""
        ns = namespace or self.namespace
        index_path = self._get_index_path(ns)
        docs_path = self._get_documents_path(ns)
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            return False
            
        try:
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            self.dimension = self.index.d
            self.namespace = ns
            return True
        except Exception:
            return False
    
    def add_documents(self, documents: List[Document], namespace: Optional[str] = None):
        """Add documents to the vector store."""
        if not documents:
            return
            
        ns = namespace or self.namespace
        
        # Load existing index if it exists
        if not self.load(ns):
            # Create new index if it doesn't exist
            self.documents = {}
            self.dimension = len(embedder.embed_query("test"))  # Get dimension from embedder
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add documents to index
        texts = [doc.page_content for doc in documents]
        embeddings = embedder.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_np)
        
        # Update documents dictionary
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            self.documents[str(start_idx + i)] = {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
        
        # Save the updated index and documents
        self.save(ns)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if not self.load(namespace):
            return []
        
        # Get query embedding
        query_embedding = embedder.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Convert to SearchResult objects
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No more results
                continue
                
            doc_id = str(idx)
            if doc_id not in self.documents:
                continue
                
            doc_data = self.documents[doc_id]
            doc = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            
            # Apply filters if provided
            if filter and not self._matches_filter(doc.metadata, filter):
                continue
                
            results.append(SearchResult(
                document=doc,
                score=float(distances[0][i])
            ))
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.load(namespace):
            return {"document_count": 0, "dimension": 0}
        
        return {
            "document_count": len(self.documents),
            "dimension": self.dimension,
            "namespace": namespace or self.namespace
        }

# Singleton instance
vector_store = VectorStore()
