# agents/retriever_agent/multi_vector_store.py
# This will be the primary vector store implementation.
# `store.py` will be considered deprecated/removed.

import os
import time
import logging
import asyncio # Not heavily used here yet, but good for future async store operations
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import json # For potential metadata persistence if not handled by Langchain store
import uuid
from pathlib import Path
import shutil # For directory operations like deleting a namespace

# Langchain vector stores and embeddings
from langchain_community.vectorstores import FAISS, Chroma # Qdrant, Weaviate can be added if needed
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings as LangchainEmbeddingsBase # Correct base class
from langchain_core.documents import Document as LangchainDocument

from .models import Document as AppDocument, SearchResult # AppDocument is our Pydantic model
from .config import settings, VectorStoreType, EmbeddingModelType
from .embedder import get_embedder # Use our singleton embedder

logger = logging.getLogger(__name__)

# LangChain vector stores can be type hinted more specifically if needed
LangchainVectorStore = Union[FAISS, Chroma] # Add Qdrant, Weaviate etc. if supported

class MultiVectorStore:
    """
    Vector store implementation supporting multiple Langchain backends.
    Manages persistence and provides a unified interface.
    """
    
    def __init__(self):
        self.vector_store_type: VectorStoreType = settings.VECTOR_STORE_TYPE
        self.embedding_model_type: EmbeddingModelType = settings.EMBEDDING_MODEL_TYPE
        self.embedding_model_name: str = settings.EMBEDDING_MODEL
        self.base_persist_directory: Path = Path(settings.VECTOR_STORE_PATH)
        self.collection_name_template: str = settings.COLLECTION_NAME # e.g., "finai_docs_{namespace}"
        
        self.embeddings: LangchainEmbeddingsBase = self._init_embedding_model()
        
        # Stores are loaded/initialized on demand per namespace
        self._stores: Dict[str, LangchainVectorStore] = {}
        self._store_metadata: Dict[str, Dict[str, Any]] = {} # To store last_updated, doc_count per namespace

        self.base_persist_directory.mkdir(parents=True, exist_ok=True)
        self._load_all_store_metadata() # Load metadata for existing namespaces
        logger.info(
            f"MultiVectorStore initialized. Base path: {self.base_persist_directory}. "
            f"Default store type: {self.vector_store_type.value}. "
            f"Embedding model: {self.embedding_model_name} ({self.embedding_model_type.value})."
        )

    def _get_namespace_persist_path(self, namespace: str) -> Path:
        return self.base_persist_directory / namespace

    def _get_metadata_file_path(self, namespace: str) -> Path:
        return self._get_namespace_persist_path(namespace) / "store_metadata.json"

    def _load_store_metadata(self, namespace: str):
        meta_file = self._get_metadata_file_path(namespace)
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    self._store_metadata[namespace] = json.load(f)
                    if "last_updated" in self._store_metadata[namespace]:
                        self._store_metadata[namespace]["last_updated"] = datetime.fromisoformat(
                            self._store_metadata[namespace]["last_updated"]
                        )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading metadata for namespace '{namespace}': {e}. Initializing empty.")
                self._store_metadata[namespace] = {"doc_count": 0, "last_updated": datetime.utcnow()}
        else:
            self._store_metadata[namespace] = {"doc_count": 0, "last_updated": datetime.utcnow()}

    def _save_store_metadata(self, namespace: str):
        meta_file = self._get_metadata_file_path(namespace)
        meta_data_to_save = self._store_metadata.get(namespace, {}).copy()
        if "last_updated" in meta_data_to_save and isinstance(meta_data_to_save["last_updated"], datetime):
            meta_data_to_save["last_updated"] = meta_data_to_save["last_updated"].isoformat()
        
        try:
            with open(meta_file, 'w') as f:
                json.dump(meta_data_to_save, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving metadata for namespace '{namespace}': {e}")

    def _load_all_store_metadata(self):
        for ns_dir in self.base_persist_directory.iterdir():
            if ns_dir.is_dir():
                self._load_store_metadata(ns_dir.name)

    def _init_embedding_model(self) -> LangchainEmbeddingsBase:
        if self.embedding_model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            # Uses the embedder.py singleton which initializes SentenceTransformer
            # Need a wrapper to fit LangchainEmbeddingsBase interface
            app_embedder = get_embedder()
            
            class CustomLangchainEmbedder(LangchainEmbeddingsBase):
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return app_embedder.embed_documents(texts)
                def embed_query(self, text: str) -> List[float]:
                    return app_embedder.embed_query(text)
            return CustomLangchainEmbedder()
        
        elif self.embedding_model_type == EmbeddingModelType.OPENAI:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required for OpenAI embeddings but not found in settings.")
            return OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL) # OpenAI model can be specified here
        
        elif self.embedding_model_type == EmbeddingModelType.HUGGINGFACE:
             # This is essentially same as SentenceTransformers for local models via Langchain
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}, # Example device selection
                encode_kwargs={'normalize_embeddings': True} # Example encode kwarg
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {self.embedding_model_type}")

    def _get_or_create_store(self, namespace: str) -> LangchainVectorStore:
        if namespace in self._stores:
            return self._stores[namespace]

        persist_path = self._get_namespace_persist_path(namespace)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Collection name specific to namespace for stores that use it (like Chroma)
        # For FAISS, persist_path is the folder.
        namespace_collection_name = f"{self.collection_name_template}_{namespace}"

        store: LangchainVectorStore
        try:
            if self.vector_store_type == VectorStoreType.FAISS:
                faiss_index_path = persist_path / "index.faiss"
                if faiss_index_path.exists():
                    logger.info(f"Loading existing FAISS index for namespace '{namespace}' from {persist_path}")
                    store = FAISS.load_local(
                        folder_path=str(persist_path),
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True # Required for FAISS with custom embeddings
                    )
                else:
                    logger.info(f"Creating new FAISS index for namespace '{namespace}' at {persist_path}")
                    # FAISS needs at least one document to be initialized.
                    # We can't create an empty one directly via `from_texts` or `from_documents`.
                    # So, we handle this typically during the first `add_documents` call.
                    # For now, if no index, we'll let add_documents create it.
                    # This means _get_or_create_store might return None or raise if no docs added yet
                    # Let's create with a dummy doc if it must be initialized here.
                    # This is tricky. Langchain's FAISS.from_documents creates and returns.
                    # If we need an empty store, it's often done by not loading.
                    # For simplicity, let's assume an empty FAISS is okay to be "non-existent" until first add.
                    # However, most operations would fail.
                    # A common pattern:
                    # Create a dummy document to initialize if no path exists
                    dummy_lc_doc = LangchainDocument(page_content="init", metadata={"id": "dummy"})
                    store = FAISS.from_documents([dummy_lc_doc], self.embeddings)
                    store.delete([store.index_to_docstore_id[0]]) # Delete the dummy doc
                    store.save_local(str(persist_path))
            
            elif self.vector_store_type == VectorStoreType.CHROMA:
                logger.info(f"Initializing ChromaDB for namespace '{namespace}' at {persist_path} with collection '{namespace_collection_name}'")
                store = Chroma(
                    collection_name=namespace_collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_path)
                )
            # Add Qdrant, Weaviate initializations here if needed
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
            
            self._stores[namespace] = store
            if namespace not in self._store_metadata: # Load or init metadata
                self._load_store_metadata(namespace)
            return store
        except Exception as e:
            logger.error(f"Error initializing vector store for namespace '{namespace}': {e}", exc_info=True)
            # Fallback to FAISS (in-memory for this instance if persist fails hard)
            logger.warning(f"Falling back to in-memory FAISS for namespace '{namespace}' due to error.")
            dummy_lc_doc = LangchainDocument(page_content="init_fallback", metadata={"id": "dummy_fb"})
            store = FAISS.from_documents([dummy_lc_doc], self.embeddings)
            store.delete([store.index_to_docstore_id[0]]) # Delete dummy
            self._stores[namespace] = store # Store in-memory version
            if namespace not in self._store_metadata: self._load_store_metadata(namespace) # Still try to manage metadata
            return store


    def add_documents(self, documents: List[AppDocument], namespace: str = "default") -> Tuple[List[str], List[str]]:
        if not documents:
            return [], []
        
        store = self._get_or_create_store(namespace)
        langchain_docs: List[LangchainDocument] = []
        doc_ids_to_add: List[str] = []
        failed_doc_ids: List[str] = []
        
        for app_doc in documents:
            # Ensure ID in metadata
            doc_id = app_doc.metadata.get("id", str(uuid.uuid4()))
            app_doc.metadata["id"] = doc_id
            
            # Add/update namespace in AppDocument metadata (redundant if IngestRequest validator did it)
            app_doc.metadata["namespace"] = namespace 
            
            # Convert AppDocument to LangchainDocument
            lc_doc = LangchainDocument(page_content=app_doc.page_content, metadata=app_doc.metadata)
            langchain_docs.append(lc_doc)
            doc_ids_to_add.append(doc_id)

        try:
            # Langchain's add_documents typically returns list of added IDs if store supports it.
            # For FAISS, it doesn't return IDs in the same way as Chroma.
            # We are managing IDs via metadata.
            if self.vector_store_type == VectorStoreType.FAISS and not store.index_to_docstore_id: # FAISS needs init
                 if not langchain_docs: # Should not happen if documents list is not empty
                     logger.warning(f"No documents to initialize FAISS for namespace '{namespace}'")
                     return [], []
                 # Re-init store with actual docs for FAISS if it was empty
                 logger.info(f"Initializing FAISS for '{namespace}' with first batch of documents.")
                 self._stores[namespace] = FAISS.from_documents(langchain_docs, self.embeddings)
                 # save after this first add
            else:
                 store.add_documents(langchain_docs) # Returns list of IDs for some stores

            # Update metadata (doc_count needs to be accurate)
            # A more robust doc_count would be to query the store, but can be slow.
            # For now, increment. This assumes no overwrites based on ID.
            # If overwrite_duplicates is true in request, logic here needs to be smarter (delete then add).
            current_meta = self._store_metadata.setdefault(namespace, {"doc_count": 0, "last_updated": datetime.utcnow()})
            current_meta["doc_count"] = current_meta.get("doc_count", 0) + len(doc_ids_to_add)
            current_meta["last_updated"] = datetime.utcnow()
            
            if self.vector_store_type == VectorStoreType.FAISS:
                store.save_local(str(self._get_namespace_persist_path(namespace)))
            elif self.vector_store_type == VectorStoreType.CHROMA:
                store.persist() # Chroma specific
            
            self._save_store_metadata(namespace)
            logger.info(f"Added {len(doc_ids_to_add)} documents to namespace '{namespace}'.")
            return doc_ids_to_add, [] # Return added IDs, no failures here
        
        except Exception as e:
            logger.error(f"Error adding documents to namespace '{namespace}': {e}", exc_info=True)
            # If add_documents fails, all in this batch are considered failed.
            return [], doc_ids_to_add 


    def similarity_search(
        self, query: str, k: int = 5, namespace: str = "default", 
        filter_dict: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        if not query: return []
        
        store = self._get_or_create_store(namespace)
        if not hasattr(store, 'similarity_search_with_score'):
            logger.error(f"Store for namespace '{namespace}' (type: {type(store)}) does not support similarity_search_with_score.")
            return []

        # Langchain filters vary by backend. FAISS supports a custom filter function.
        # Chroma, Qdrant, Weaviate support metadata filtering via `where` clauses or similar.
        # For FAISS, the `filter` kwarg in `similarity_search_with_score` expects a callable.
        # For others, it might be a dict. This needs to be harmonized or backend-specific.
        
        # Let's assume for now the 'filter' in Langchain's methods will handle it if it's a dict.
        # This is true for Chroma. For FAISS, if `filter` is a dict, it's often ignored or needs a custom func.
        # A simple dict filter for FAISS would require post-filtering.
        
        # The `k` for search might need to be larger if post-filtering is aggressive.
        search_k = k * 5 if filter_dict and self.vector_store_type == VectorStoreType.FAISS else k

        try:
            # Using similarity_search_with_relevance_scores if available, else similarity_search_with_score
            if hasattr(store, 'similarity_search_with_relevance_scores'):
                 docs_with_scores = store.similarity_search_with_relevance_scores(query, k=search_k, filter=filter_dict)
            else:
                 docs_with_scores = store.similarity_search_with_score(query, k=search_k, filter=filter_dict)
        except NotImplementedError: # Some stores might not implement filter for similarity_search_with_score
             logger.warning(f"Filter not implemented for similarity_search_with_score in {self.vector_store_type} via Langchain. Fetching without filter.")
             docs_with_scores = store.similarity_search_with_score(query, k=search_k) # Try without filter
        except Exception as e:
            logger.error(f"Similarity search error in namespace '{namespace}': {e}", exc_info=True)
            return []

        results: List[SearchResult] = []
        for lc_doc, score in docs_with_scores:
            app_doc = AppDocument(page_content=lc_doc.page_content, metadata=lc_doc.metadata)
            
            # Manual post-filtering if FAISS and filter_dict provided (FAISS filter arg is complex)
            if self.vector_store_type == VectorStoreType.FAISS and filter_dict:
                if not self._matches_filter(app_doc.metadata, filter_dict):
                    continue
            
            # Filter by min_score (Note: FAISS score is L2 distance (lower is better), others are similarity (higher is better))
            # This needs to be normalized or handled per-store.
            # For now, assuming score is similarity (0-1, higher is better) for min_score.
            # FAISS L2 distance needs conversion to similarity: score = 1 / (1 + distance) or exp(-distance)
            effective_score = score
            if self.vector_store_type == VectorStoreType.FAISS: # L2 distance
                 effective_score = 1 / (1 + score) if score >= 0 else 0 # Normalize L2 to similarity-like

            if min_score is not None and effective_score < min_score:
                continue

            results.append(SearchResult(document=app_doc, score=effective_score))
            if len(results) >= k:
                break 
        return results

    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Checks if a document's metadata matches all filter criteria."""
        for key, expected_value in filter_criteria.items():
            if key not in metadata or metadata[key] != expected_value:
                return False
        return True

    def delete_documents(self, document_ids: List[str], namespace: str = "default") -> int:
        if not document_ids: return 0
        
        store = self._get_or_create_store(namespace)
        # Langchain vector stores often take `ids` argument for deletion.
        # Need to ensure our app_doc.id corresponds to what the store uses.
        # FAISS deletion is complex (rebuild). Chroma has `delete(ids=[...])`.
        deleted_count = 0
        if self.vector_store_type == VectorStoreType.FAISS:
            # FAISS requires finding internal indices then rebuilding.
            # This is simplified; real FAISS deletion in Langchain is tricky.
            # Often, you re-create the index from remaining docs.
            # Langchain's FAISS.delete(ids) does this.
            try:
                # This assumes document_ids are the ones FAISS uses (docstore_id)
                # Our 'id' in metadata needs to be mapped/used by FAISS correctly.
                # If FAISS uses its own sequential IDs, we need a map.
                # For now, assuming 'id' in metadata *is* the key.
                # This is a placeholder, FAISS specific ID logic in LangChain is complex.
                # Langchain FAISS delete method might be what we need IF ids are managed by it.
                # If we pass our own UUIDs as "id" metadata, this needs to be reconciled.
                # A common way: retrieve doc by UUID, get its internal FAISS index, then mark for deletion.
                # Let's assume a simpler path for now, or that underlying store handles string IDs.
                
                # The FAISS.delete() method expects the internal docstore IDs.
                # We need to fetch these first if we only have our custom metadata IDs.
                # This is a simplified representation:
                if hasattr(store, 'delete') and callable(getattr(store, 'delete')):
                    store.delete(document_ids) # This assumes `document_ids` are what the store expects.
                    deleted_count = len(document_ids) # Approximation
                    store.save_local(str(self._get_namespace_persist_path(namespace))) # Re-save
                else:
                     logger.warning(f"FAISS store for namespace '{namespace}' via Langchain does not have a direct robust delete method based on custom IDs. Deletion might be partial or ineffective.")
                     # Manual rebuild would be required here based on remaining documents. This is costly.

            except Exception as e:
                logger.error(f"Error deleting from FAISS namespace '{namespace}': {e}", exc_info=True)
        elif hasattr(store, 'delete') and callable(getattr(store, 'delete')): # Chroma, Qdrant, Weaviate often have this
            try:
                store.delete(ids=document_ids) # Pass our IDs
                deleted_count = len(document_ids) # Assume all were found and deleted.
                if self.vector_store_type == VectorStoreType.CHROMA:
                    store.persist()
            except Exception as e:
                logger.error(f"Error deleting documents from {self.vector_store_type} namespace '{namespace}': {e}", exc_info=True)
        else:
            logger.warning(f"Deletion not directly supported or implemented for store type {self.vector_store_type} in namespace '{namespace}'.")

        if deleted_count > 0:
            current_meta = self._store_metadata.setdefault(namespace, {"doc_count": 0, "last_updated": datetime.utcnow()})
            current_meta["doc_count"] = max(0, current_meta.get("doc_count", 0) - deleted_count)
            current_meta["last_updated"] = datetime.utcnow()
            self._save_store_metadata(namespace)
        return deleted_count


    def update_document(self, document_id: str, document: AppDocument, namespace: str = "default") -> bool:
        # Most vector stores handle updates as delete + add.
        # Ensure the new document has the same ID in its metadata.
        document.metadata["id"] = document_id
        document.metadata["namespace"] = namespace # Ensure namespace consistency
        
        # Attempt deletion first. This also depends on how underlying store handles non-existent IDs.
        self.delete_documents([document_id], namespace=namespace) # Ignore result of deletion for now
        
        # Add the updated document
        added_ids, failed_ids = self.add_documents([document], namespace=namespace)
        
        return document_id in added_ids


    def clear_vector_store(self, namespace: str = "default"):
        ns_path = self._get_namespace_persist_path(namespace)
        if ns_path.exists():
            try:
                shutil.rmtree(ns_path)
                logger.info(f"Cleared vector store for namespace '{namespace}' at {ns_path}.")
            except OSError as e:
                logger.error(f"Error clearing vector store directory for namespace '{namespace}': {e}", exc_info=True)
                raise
        
        if namespace in self._stores:
            del self._stores[namespace]
        if namespace in self._store_metadata:
            del self._store_metadata[namespace] # Remove its metadata entry
        
        # Re-create empty directory and metadata for future use if needed
        ns_path.mkdir(parents=True, exist_ok=True)
        self._store_metadata[namespace] = {"doc_count": 0, "last_updated": datetime.utcnow()}
        self._save_store_metadata(namespace)


    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        # If specific namespace, load/return its stats. If None, aggregate.
        if namespace:
            self._get_or_create_store(namespace) # Ensure it's loaded/meta exists
            meta = self._store_metadata.get(namespace, {})
            # Try to get vector_count from the store if possible (some stores expose this)
            # For now, doc_count from metadata is the primary count.
            # Disk usage is hard to get generically from Langchain stores.
            return {
                "document_count": meta.get("doc_count", 0),
                "vector_count": meta.get("doc_count", 0), # Approximation
                "last_updated": meta.get("last_updated", datetime.utcnow()),
                "namespaces": {namespace: meta.get("doc_count", 0)}, # Stats for this namespace
                "vector_store_type": self.vector_store_type.value,
                "embedding_model_name": self.embedding_model_name
            }
        else: # Aggregate stats
            total_docs = 0
            all_ns_counts = {}
            latest_update = datetime.min
            for ns, meta in self._store_metadata.items():
                count = meta.get("doc_count", 0)
                total_docs += count
                all_ns_counts[ns] = count
                ns_updated_at = meta.get("last_updated", datetime.min)
                if isinstance(ns_updated_at, str): # If loaded from JSON as string
                    try: ns_updated_at = datetime.fromisoformat(ns_updated_at)
                    except ValueError: ns_updated_at = datetime.min
                if ns_updated_at > latest_update:
                    latest_update = ns_updated_at
            
            return {
                "document_count": total_docs, "vector_count": total_docs,
                "last_updated": latest_update if latest_update != datetime.min else None,
                "namespaces": all_ns_counts,
                "vector_store_type": self.vector_store_type.value,
                "embedding_model_name": self.embedding_model_name
            }
    
    def list_namespaces(self) -> List[str]:
        return list(self._store_metadata.keys())


_vector_store_instance: Optional[MultiVectorStore] = None

def get_multi_vector_store() -> MultiVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = MultiVectorStore()
    return _vector_store_instance