"""
Document indexer module for Zed-KB.
Handles indexing of documents in AstraDB vector store with security awareness.
"""

from typing import List, Dict, Any, Optional, Callable
import uuid
import datetime

from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Import our custom vector store and embedding models
from ..vector_store.gemini_embeddings import GeminiEmbeddings
from ..vector_store.openai_embeddings import OpenAIEmbeddings
from ..vector_store.astra_db import AstraDBStore


class DocumentIndexer:
    """Handles indexing of documents in AstraDB vector store with security filtering."""

    EMBEDDING_MODELS = {
        "openai": lambda **kwargs: OpenAIEmbeddings(
            model=kwargs.get("model", "text-embedding-ada-002")
        ),
        "gemini": lambda **kwargs: GeminiEmbeddings(
            model=kwargs.get("model", "embedding-001"),
            task_type=kwargs.get("task_type", "RETRIEVAL_DOCUMENT")
        ),
    }

    def __init__(
        self,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
        collection_name: str = "zed_kb_documents",
        hybrid_search: bool = True,
        astra_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the document indexer for AstraDB.

        Args:
            embedding_provider: Provider of embedding model ('openai' or 'gemini')
            embedding_model: Name of embedding model to use
            collection_name: Name of the AstraDB collection to use
            hybrid_search: Whether to enable hybrid search
            astra_config: Configuration for AstraDB connection
        """
        if embedding_provider not in self.EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding provider: {embedding_provider}. "
                f"Choose from: {', '.join(self.EMBEDDING_MODELS.keys())}"
            )

        self.hybrid_search = hybrid_search
        self.astra_config = astra_config or {}

        # Get the collection name from config or use default
        self.collection_name = self.astra_config.get(
            "collection_name", collection_name)

        # Initialize embedding model
        self.embedding_model = self.EMBEDDING_MODELS[embedding_provider](
            model=embedding_model
        )

        # Vector store will be initialized when needed
        self.vector_store = None

    def _initialize_vector_store(self, documents: Optional[List[Document]] = None):
        """
        Initialize the AstraDB vector store.

        Args:
            documents: Optional list of documents to initialize with
        """
        if self.vector_store is not None:
            return

        # Create AstraDB store with hybrid search if enabled
        self.vector_store = AstraDBStore(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            token=self.astra_config.get("token"),
            api_endpoint=self.astra_config.get("api_endpoint"),
            astra_db_id=self.astra_config.get("astra_db_id"),
            astra_db_region=self.astra_config.get("astra_db_region"),
            namespace=self.astra_config.get("namespace"),
            hybrid_search=self.hybrid_search,
        )

        # Add initial documents if provided
        if documents:
            self.vector_store.add_documents(documents)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        # Add document IDs if they don't exist
        for doc in documents:
            if "document_id" not in doc.metadata:
                doc.metadata["document_id"] = str(uuid.uuid4())

            # Add indexing timestamp
            doc.metadata["indexed_at"] = datetime.datetime.now().isoformat()

        # Initialize vector store if needed
        if self.vector_store is None:
            self._initialize_vector_store(documents)
            # Documents are added during initialization
            return [doc.metadata["document_id"] for doc in documents]
        else:
            # Add documents to existing store
            return self.vector_store.add_documents(documents)

    def search(
        self,
        query: str,
        user_info: Dict[str, Any] = None,
        k: int = 5,
        filter_metadata: Dict[str, Any] = None,
        hybrid_alpha: float = 0.5,  # Balance between vector and keyword search
    ) -> List[Document]:
        """
        Search for documents in the vector store with security filtering.

        Args:
            query: Search query
            user_info: User information for security filtering
            k: Number of documents to return
            filter_metadata: Additional metadata filter
            hybrid_alpha: Ratio between vector and keyword search (0 = only keywords, 1 = only vector)

        Returns:
            List of matching documents
        """
        if self.vector_store is None:
            # Initialize an empty vector store if needed
            self._initialize_vector_store()
            return []

        # Create metadata filter
        metadata_filter = filter_metadata or {}

        # Apply security filtering if user info is provided
        if user_info:
            # Use the specialized security-aware search method if available
            if hasattr(self.vector_store, "similarity_search_with_security"):
                return self.vector_store.similarity_search_with_security(
                    query=query,
                    user_info=user_info,
                    k=k,
                    filter=metadata_filter,
                    hybrid_alpha=hybrid_alpha if self.hybrid_search else None
                )

            # Otherwise, enhance the filter with security constraints
            if user_info.get("roles"):
                metadata_filter["allowed_roles"] = {
                    "$in": user_info["roles"]
                }

            # Add clearance level filtering
            if user_info.get("clearance"):
                user_clearance = user_info["clearance"]
                # Convert to numeric levels for comparison
                clearance_levels = {
                    "public": 0,
                    "internal": 1,
                    "confidential": 2,
                    "restricted": 3,
                    "top_secret": 4
                }
                user_clearance_level = clearance_levels.get(user_clearance, 0)

                # Filter to only include documents with equal or lower security level
                eligible_levels = [
                    k for k, v in clearance_levels.items() if v <= user_clearance_level
                ]
                metadata_filter["security_level"] = {
                    "$in": eligible_levels
                }

        # Use hybrid search if enabled
        results = self.vector_store.similarity_search(
            query, k=k, filter=metadata_filter,
            hybrid_alpha=hybrid_alpha if self.hybrid_search else None
        )

        return results

    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Success status
        """
        if self.vector_store is None:
            return False

        try:
            # AstraDB handles deletions in one operation
            if hasattr(self.vector_store, "delete"):
                self.vector_store.delete(document_ids)
                return True
            elif hasattr(self.vector_store, "delete_by_filter"):
                for doc_id in document_ids:
                    self.vector_store.delete_by_filter({"document_id": doc_id})
                return True
            return False
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def load_index(self, **kwargs) -> bool:
        """
        Load or reconnect to an existing AstraDB collection.

        Args:
            **kwargs: Additional arguments to pass to AstraDBStore

        Returns:
            Success status
        """
        try:
            # For AstraDB, we just need to reconnect to the existing collection
            self._initialize_vector_store()
            return True
        except Exception as e:
            print(f"Error connecting to AstraDB collection: {e}")
            return False
