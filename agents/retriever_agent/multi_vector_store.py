"""
Multi-backend vector store implementation that supports multiple vector databases.
Provides a unified interface for document storage and retrieval with fallback mechanisms.
"""
import os
import time
import logging
import asyncio
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import json
import uuid
from pathlib import Path

# Vector stores
from langchain.vectorstores import FAISS, Chroma, Qdrant, Weaviate
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

# Import local models
from .models import Document, SearchResult
from .config import settings

# Setup logging
logger = logging.getLogger(__name__)

class VectorStoreType(str, Enum):
    """Supported vector store types."""
    FAISS = "faiss"
    CHROMA = "chroma"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"


class EmbeddingModelType(str, Enum):
    """Supported embedding model types."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class MultiVectorStore:
    """
    Vector store implementation that supports multiple backends.
    Provides fallback mechanisms and unified interface for document storage and retrieval.
    """
    
    def __init__(self, 
                 vector_store_type: VectorStoreType = VectorStoreType.FAISS,
                 embedding_model_type: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS,
                 embedding_model_name: str = settings.EMBEDDING_MODEL,
                 persist_directory: str = settings.VECTOR_STORE_PATH,
                 openai_api_key: Optional[str] = None,
                 collection_name: str = "finai_documents"):
        """
        Initialize the multi vector store.
        
        Args:
            vector_store_type: Type of vector store to use
            embedding_model_type: Type of embedding model to use
            embedding_model_name: Name of the embedding model
            persist_directory: Directory to persist vector store
            openai_api_key: OpenAI API key (if using OpenAI embeddings)
            collection_name: Name of the collection/index
        """
        self.vector_store_type = vector_store_type
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name = embedding_model_name
        self.persist_directory = Path(persist_directory)
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.document_count = 0
        self.last_updated = datetime.now()
        
        # Setup the embedding model
        self.embeddings = self._init_embedding_model()
        
        # Setup the vector store
        self.vector_store = self._init_vector_store()
        
        # Cache for document lookup
        self.document_cache = {}
        
        logger.info(f"Initialized {vector_store_type} vector store with {embedding_model_type} embeddings")
        
    def _init_embedding_model(self) -> Embeddings:
        """Initialize the embedding model."""
        if self.embedding_model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        elif self.embedding_model_type == EmbeddingModelType.OPENAI:
            if not self.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            return OpenAIEmbeddings(
                openai_api_key=self.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
        elif self.embedding_model_type == EmbeddingModelType.HUGGINGFACE:
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        else:
            raise ValueError(f"Unsupported embedding model type: {self.embedding_model_type}")

    def _init_vector_store(self) -> Any:
        """Initialize the vector store backend."""
        try:
            if self.vector_store_type == VectorStoreType.FAISS:
                # Check if FAISS index exists
                if (self.persist_directory / "index.faiss").exists():
                    return FAISS.load_local(
                        folder_path=str(self.persist_directory),
                        embeddings=self.embeddings,
                        index_name=self.collection_name
                    )
                else:
                    # Create a new FAISS index
                    return FAISS.from_documents(
                        documents=[],  # Empty initial documents
                        embedding=self.embeddings,
                        persist_directory=str(self.persist_directory)
                    )
            
            elif self.vector_store_type == VectorStoreType.CHROMA:
                return Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory)
                )
            
            elif self.vector_store_type == VectorStoreType.QDRANT:
                # For Qdrant, use a local file-based storage
                qdrant_path = self.persist_directory / "qdrant"
                qdrant_path.mkdir(exist_ok=True)
                
                from qdrant_client import QdrantClient
                from langchain.vectorstores import Qdrant
                
                client = QdrantClient(path=str(qdrant_path))
                
                return Qdrant(
                    client=client,
                    collection_name=self.collection_name,
                    embeddings=self.embeddings
                )
            
            elif self.vector_store_type == VectorStoreType.WEAVIATE:
                # For Weaviate, we'll use the HTTP client
                import weaviate
                from weaviate.embedded import EmbeddedOptions
                
                client = weaviate.Client(
                    embedded_options=EmbeddedOptions(
                        persistence_data_path=str(self.persist_directory / "weaviate")
                    )
                )
                
                return Weaviate(
                    client=client,
                    index_name=self.collection_name,
                    text_key="content",
                    embedding=self.embeddings
                )
            
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            # Fallback to FAISS if other vector stores fail
            logger.info("Falling back to FAISS vector store")
            return FAISS.from_documents(
                documents=[],  # Empty initial documents
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )

    def add_documents(self, documents: List[Document], namespace: Optional[str] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            namespace: Optional namespace for the documents
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Convert to LangChain document format
        langchain_docs = []
        doc_ids = []
        
        for doc in documents:
            # Generate a document ID if not in metadata
            doc_id = doc.metadata.get("id", str(uuid.uuid4()))
            doc.metadata["id"] = doc_id
            doc_ids.append(doc_id)
            
            # Add namespace to metadata if provided
            if namespace:
                doc.metadata["namespace"] = namespace
            
            # Convert to LangChain document
            from langchain.schema import Document as LCDocument
            langchain_docs.append(
                LCDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
            )
            
            # Update cache
            self.document_cache[doc_id] = doc
        
        # Add to vector store
        try:
            self.vector_store.add_documents(langchain_docs)
            
            # Update stats
            self.document_count += len(documents)
            self.last_updated = datetime.now()
            
            # Save vector store if it's FAISS
            if self.vector_store_type == VectorStoreType.FAISS:
                self.vector_store.save_local(str(self.persist_directory))
            
            return doc_ids
        
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise ValueError(f"Failed to add documents: {str(e)}")

    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            namespace: Optional namespace to search in
            
        Returns:
            List of search results
        """
        if not query:
            return []
        
        # Combine namespace filter with other filters
        if namespace:
            if filter is None:
                filter = {}
            filter["namespace"] = namespace
        
        try:
            # Perform similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Convert to search results
            results = []
            for doc, score in docs_with_scores:
                # Convert LangChain document to our Document model
                document = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
                
                # Add to search results
                results.append(
                    SearchResult(
                        document=document,
                        score=float(score)
                    )
                )
            
            return results
        
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise ValueError(f"Failed to perform similarity search: {str(e)}")

    def delete_documents(self, document_ids: List[str], namespace: Optional[str] = None) -> int:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            namespace: Optional namespace the documents are in
            
        Returns:
            Number of documents deleted
        """
        if not document_ids:
            return 0
        
        try:
            # Delete from vector store
            # Note: Implementation depends on the specific vector store
            if self.vector_store_type == VectorStoreType.FAISS:
                # FAISS doesn't support direct deletion, so we need to rebuild the index
                # First, get all documents
                all_docs = self.vector_store.index_to_docstore
                
                # Filter out the documents to delete
                remaining_docs = {
                    idx: doc for idx, doc in all_docs.items()
                    if doc.metadata.get("id") not in document_ids
                }
                
                if namespace:
                    remaining_docs = {
                        idx: doc for idx, doc in remaining_docs.items()
                        if doc.metadata.get("namespace") != namespace or doc.metadata.get("id") not in document_ids
                    }
                
                # Convert to list of documents
                docs_to_keep = list(remaining_docs.values())
                
                # Create a new FAISS index
                self.vector_store = FAISS.from_documents(
                    documents=docs_to_keep,
                    embedding=self.embeddings,
                    persist_directory=str(self.persist_directory)
                )
                
                # Update stats
                deleted_count = len(all_docs) - len(remaining_docs)
                self.document_count -= deleted_count
                self.last_updated = datetime.now()
                
                # Remove from cache
                for doc_id in document_ids:
                    if doc_id in self.document_cache:
                        del self.document_cache[doc_id]
                
                return deleted_count
            
            else:
                # For other vector stores, use their delete methods
                # This is a simplified approach and may need to be adapted
                delete_count = 0
                for doc_id in document_ids:
                    try:
                        filter_dict = {"id": doc_id}
                        if namespace:
                            filter_dict["namespace"] = namespace
                        
                        self.vector_store.delete(filter=filter_dict)
                        delete_count += 1
                        
                        # Remove from cache
                        if doc_id in self.document_cache:
                            del self.document_cache[doc_id]
                    except Exception as e:
                        logger.warning(f"Error deleting document {doc_id}: {str(e)}")
                
                # Update stats
                self.document_count -= delete_count
                self.last_updated = datetime.now()
                
                return delete_count
        
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise ValueError(f"Failed to delete documents: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "document_count": self.document_count,
            "last_updated": self.last_updated.isoformat(),
            "vector_store_type": self.vector_store_type,
            "embedding_model_type": self.embedding_model_type,
            "embedding_model_name": self.embedding_model_name
        }


# Create a singleton instance
_multi_vector_store = None

def get_multi_vector_store() -> MultiVectorStore:
    """Get the multi vector store singleton instance."""
    global _multi_vector_store
    if _multi_vector_store is None:
        _multi_vector_store = MultiVectorStore(
            vector_store_type=VectorStoreType(settings.VECTOR_STORE_TYPE),
            embedding_model_type=EmbeddingModelType(settings.EMBEDDING_MODEL_TYPE),
            embedding_model_name=settings.EMBEDDING_MODEL,
            persist_directory=settings.VECTOR_STORE_PATH,
            openai_api_key=settings.OPENAI_API_KEY,
            collection_name=settings.COLLECTION_NAME
        )
    return _multi_vector_store
