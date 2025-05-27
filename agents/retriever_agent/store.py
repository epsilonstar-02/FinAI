import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Set, Tuple
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from .models import Document, SearchResult
from .config import settings
from .embedder import embedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = {}
        self.document_ids = []  # Keep track of document order for FAISS index
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.namespace = "default"
        self.last_updated = datetime.utcnow()
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
        if self.index is None or len(self.documents) == 0:
            logger.warning(f"Cannot save empty index for namespace: {namespace or self.namespace}")
            return
            
        ns = namespace or self.namespace
        try:
            # Update timestamp
            self.last_updated = datetime.utcnow()
            
            # Save FAISS index
            faiss_path = self._get_index_path(ns)
            faiss.write_index(self.index, faiss_path)
            logger.info(f"Saved FAISS index to {faiss_path}")
            
            # Save documents as JSON
            docs_path = self._get_documents_path(ns)
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.documents,
                    "document_ids": self.document_ids,
                    "metadata": {
                        "last_updated": self.last_updated.isoformat(),
                        "document_count": len(self.documents),
                        "dimension": self.dimension,
                        "namespace": ns
                    }
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.documents)} documents to {docs_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise RuntimeError(f"Failed to save vector store: {str(e)}")
    
    def load(self, namespace: Optional[str] = None) -> bool:
        """Load the index and documents from disk."""
        ns = namespace or self.namespace
        index_path = self._get_index_path(ns)
        docs_path = self._get_documents_path(ns)
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            logger.info(f"Vector store files not found for namespace: {ns}")
            return False
            
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            self.dimension = self.index.d
            logger.info(f"Loaded FAISS index from {index_path} with dimension {self.dimension}")
            
            # Load documents
            try:
                with open(docs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle both formats (old format compatibility)
                if isinstance(data, dict) and "documents" in data:
                    self.documents = data["documents"]
                    self.document_ids = data.get("document_ids", list(self.documents.keys()))
                    # Try to get last_updated from metadata
                    meta = data.get("metadata", {})
                    if "last_updated" in meta:
                        try:
                            self.last_updated = datetime.fromisoformat(meta["last_updated"])
                        except (ValueError, TypeError):
                            self.last_updated = datetime.utcnow()
                else:
                    self.documents = data
                    self.document_ids = list(self.documents.keys())
                    self.last_updated = datetime.utcnow()
                    
                logger.info(f"Loaded {len(self.documents)} documents from {docs_path}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error loading documents: {str(e)}")
                return False
                
            self.namespace = ns
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document], namespace: Optional[str] = None):
        """Add documents to the vector store."""
        if not documents:
            logger.info("No documents to add")
            return
            
        ns = namespace or self.namespace
        
        # Load existing index if it exists
        if not self.load(ns):
            # Create new index if it doesn't exist
            self.documents = {}
            self.document_ids = []
            self.dimension = len(embedder.embed_query("test"))  # Get dimension from embedder
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
        
        try:
            # Add documents to index
            texts = [doc.page_content for doc in documents]
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = embedder.embed_documents(texts)
            
            # Convert to numpy array
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_np)
            
            # Update documents dictionary using UUIDs as keys for better management
            for i, doc in enumerate(documents):
                doc_id = str(uuid.uuid4())
                self.documents[doc_id] = {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata or {},
                    'added_at': datetime.utcnow().isoformat()
                }
                self.document_ids.append(doc_id)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
            # Save the updated index and documents
            self.save(ns)
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise RuntimeError(f"Failed to add documents: {str(e)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        if not self.load(namespace):
            logger.info("No documents in vector store for search")
            return []
        
        if not self.documents:
            logger.info("No documents available for search")
            return []
        
        try:
            # Get query embedding
            query_embedding = embedder.embed_query(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
                
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding_np, min(k, len(self.document_ids)))
            
            # Convert to SearchResult objects
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1 or idx >= len(self.document_ids):  # No more results or invalid index
                    continue
                doc_id = self.document_ids[idx]
                if doc_id not in self.documents:
                    continue
                doc_data = self.documents[doc_id]
                doc = Document(
                    page_content=doc_data['page_content'],
                    metadata=doc_data.get('metadata', {})
                )
                # Apply filters if provided
                if filter and not self._matches_filter(doc.metadata, filter):
                    continue
                results.append(SearchResult(
                    document=doc,
                    score=float(distances[0][i])
                ))
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter criteria."""
        if not filter_dict:
            return True
            
        for key, value in filter_dict.items():
            # Handle special case for list/array values (OR logic)
            if isinstance(value, list):
                if key not in metadata or metadata[key] not in value:
                    return False
            # Handle special case for range values (for dates or numbers)
            elif isinstance(value, dict) and any(op in value for op in ["$gt", "$lt", "$gte", "$lte"]):
                if key not in metadata:
                    return False
                    
                metadata_value = metadata[key]
                # Greater than
                if "$gt" in value and not (metadata_value > value["$gt"]):
                    return False
                # Less than
                if "$lt" in value and not (metadata_value < value["$lt"]):
                    return False
                # Greater than or equal
                if "$gte" in value and not (metadata_value >= value["$gte"]):
                    return False
                # Less than or equal
                if "$lte" in value and not (metadata_value <= value["$lte"]):
                    return False
            # Standard equality check
            elif key not in metadata or metadata[key] != value:
                return False
                
        return True
    
    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            if not self.load(namespace):
                return {
                    "document_count": 0, 
                    "dimension": self.dimension,
                    "namespace": namespace or self.namespace
                }
            
            return {
                "document_count": len(self.documents),
                "dimension": self.dimension,
                "namespace": namespace or self.namespace,
                "last_updated": self.last_updated.isoformat() if self.last_updated else None
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "document_count": 0, 
                "dimension": 0,
                "namespace": namespace or self.namespace,
                "error": str(e)
            }
    
    def add_documents_batched(self, documents: List[Document], batch_size: int = 100, namespace: Optional[str] = None):
        """Add documents to the vector store in batches."""
        if not documents:
            logger.info("No documents to add")
            return
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.add_documents(batch, namespace)
    
    def delete_documents(self, document_ids: List[str], namespace: Optional[str] = None) -> int:
        """Delete documents from the vector store."""
        if not self.load(namespace):
            logger.warning(f"No documents found in namespace: {namespace or self.namespace}")
            return 0
            
        if not document_ids:
            return 0
            
        try:
            deleted_count = 0
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    deleted_count += 1
            
            if deleted_count > 0:
                # Rebuild index after deletion
                self._rebuild_index()
                # Save changes
                self.save(namespace)
                
            logger.info(f"Deleted {deleted_count} documents from namespace: {namespace or self.namespace}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}")
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from the current documents."""
        if not self.documents:
            logger.warning("No documents to rebuild index")
            return
            
        try:
            # Get all document texts in order and update document_ids
            self.document_ids = list(self.documents.keys())
            texts = [self.documents[doc_id]['page_content'] for doc_id in self.document_ids]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = embedder.embed_documents(texts)
            
            # Convert to numpy array
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings_np)
            
            logger.info(f"Rebuilt FAISS index with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            raise RuntimeError(f"Failed to rebuild index: {str(e)}")
    
    def clear_vector_store(self, namespace: Optional[str] = None):
        """Clear all documents from the vector store."""
        ns = namespace or self.namespace
        try:
            self.documents = {}
            self.document_ids = []
            self.index = faiss.IndexFlatL2(self.dimension)
            self.save(ns)
            logger.info(f"Cleared all documents from namespace: {ns}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise RuntimeError(f"Failed to clear vector store: {str(e)}")
    
    def update_document(self, document_id: str, document: Document, namespace: Optional[str] = None) -> bool:
        """Update a document in the vector store."""
        if not self.load(namespace):
            logger.warning(f"No documents found in namespace: {namespace or self.namespace}")
            return False
            
        if document_id not in self.documents:
            logger.warning(f"Document {document_id} not found")
            return False
            
        try:
            # Update document
            self.documents[document_id] = {
                'page_content': document.page_content,
                'metadata': document.metadata or {},
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Rebuild index to update embeddings
            self._rebuild_index()
            
            # Save changes
            self.save(namespace)
            
            logger.info(f"Updated document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            raise RuntimeError(f"Failed to update document: {str(e)}")

# Singleton instance
vector_store = VectorStore()