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
        if self.index is None:
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
                    "metadata": {
                        "last_updated": self.last_updated.isoformat(),
                        "document_count": len(self.documents),
                        "dimension": self.dimension
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
                    # Try to get last_updated from metadata
                    meta = data.get("metadata", {})
                    if "last_updated" in meta:
                        try:
                            self.last_updated = datetime.fromisoformat(meta["last_updated"])
                        except (ValueError, TypeError):
                            self.last_updated = datetime.utcnow()
                else:
                    self.documents = data
                    self.last_updated = datetime.utcnow()
                    
                logger.info(f"Loaded {len(self.documents)} documents from {docs_path}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error loading documents: {str(e)}")
                return False
                
            self.namespace = ns
            return True
        except IOError as e:
            logger.error(f"IO error loading vector store: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading vector store: {str(e)}")
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
                    'metadata': doc.metadata,
                    'added_at': datetime.utcnow().isoformat()
                }
            
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
        if not filter_dict:
            return True
            
        for key, value in filter_dict.items():
            # Handle special case for list/array values (OR logic)
            if isinstance(value, list):
                if key not in metadata or metadata[key] not in value:
                    return False
            # Handle special case for range values (for dates or numbers)
            elif isinstance(value, dict) and ("$gt" in value or "$lt" in value or "$gte" in value or "$lte" in value):
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
        if not self.load(namespace):
            return {"document_count": 0, "dimension": 0}
        
        return {
            "document_count": len(self.documents),
            "dimension": self.dimension,
            "namespace": namespace or self.namespace
        }
    def add_documents_batched(self, documents: List[Document], batch_size: int = 100, namespace: Optional[str] = None):
        """Add documents to the vector store in batches to avoid memory issues."""
        if not documents:
            return
            
        logger.info(f"Adding {len(documents)} documents in batches of {batch_size}")
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
            self.add_documents(batch, namespace)
            
        logger.info(f"Completed adding {len(documents)} documents in {(len(documents) - 1) // batch_size + 1} batches")
    
    def delete_documents(self, document_ids: List[str], namespace: Optional[str] = None):
        """Delete documents from the vector store."""
        if not document_ids:
            logger.info("No document IDs provided for deletion")
            return
            
        if not self.load(namespace):
            logger.warning("No documents to delete (store is empty)")
            return
            
        ns = namespace or self.namespace
        deleted_count = 0
        
        try:
            # Remove from documents dictionary
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    deleted_count += 1
            
            if deleted_count > 0:
                # We need to rebuild the index when deleting
                logger.info(f"Deleted {deleted_count} documents, rebuilding index")
                self._rebuild_index()
                self.save(ns)
            else:
                logger.info("No matching documents found for deletion")
                
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}")
    
    def _rebuild_index(self):
        """Rebuild the index from the documents dictionary."""
        if not self.documents:
            logger.info("No documents available to rebuild index")
            self.index = faiss.IndexFlatL2(self.dimension)
            return
            
        try:
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Extract document contents and create embeddings
            texts = [doc_data['page_content'] for doc_data in self.documents.values()]
            logger.info(f"Rebuilding index with {len(texts)} documents")
            
            embeddings = embedder.embed_documents(texts)
            
            # Convert to numpy array and add to index
            embeddings_np = np.array(embeddings).astype('float32')
            self.index.add(embeddings_np)
            
            logger.info(f"Index rebuilt with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            raise RuntimeError(f"Failed to rebuild index: {str(e)}")
    
    def clear_vector_store(self, namespace: Optional[str] = None):
        """Clear all documents from the vector store."""
        ns = namespace or self.namespace
        try:
            self.documents = {}
            self.index = faiss.IndexFlatL2(self.dimension)
            self.save(ns)
            logger.info(f"Cleared all documents from namespace: {ns}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise RuntimeError(f"Failed to clear vector store: {str(e)}")
    
    def update_document(self, document_id: str, document: Document, namespace: Optional[str] = None):
        """Update a document in the vector store."""
        if not self.load(namespace):
            logger.warning("Vector store is empty, nothing to update")
            return False
            
        ns = namespace or self.namespace
        
        try:
            # Check if document exists
            if document_id not in self.documents:
                logger.warning(f"Document ID {document_id} not found in vector store")
                return False
                
            # Update document
            self.documents[document_id] = {
                'page_content': document.page_content,
                'metadata': document.metadata,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Rebuild the index
            self._rebuild_index()
            self.save(ns)
            
            logger.info(f"Updated document {document_id} in namespace {ns}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            raise RuntimeError(f"Failed to update document: {str(e)}")

# Singleton instance
vector_store = VectorStore()
