"""
Document indexer module for Zed-KB.
Handles indexing of documents in a vector store with security awareness.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import uuid
import datetime
import json

from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings


class DocumentIndexer:
    """Handles indexing of documents in vector stores with security filtering."""

    VECTOR_STORES = {"faiss": FAISS, "chroma": Chroma}

    EMBEDDING_MODELS = {
        "openai": lambda **kwargs: OpenAIEmbeddings(
            model=kwargs.get("model", "text-embedding-ada-002")
        ),
        "huggingface": lambda **kwargs: HuggingFaceEmbeddings(
            model_name=kwargs.get("model", "all-MiniLM-L6-v2")
        ),
    }

    def __init__(
        self,
        vector_store_type: str = "faiss",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
        persist_directory: str = None,
    ):
        """
        Initialize the document indexer.

        Args:
            vector_store_type: Type of vector store to use
            embedding_provider: Provider of embedding model
            embedding_model: Name of embedding model to use
            persist_directory: Directory to persist vector store
        """
        if vector_store_type not in self.VECTOR_STORES:
            raise ValueError(
                f"Unsupported vector store type: {vector_store_type}. "
                f"Choose from: {', '.join(self.VECTOR_STORES.keys())}"
            )

        if embedding_provider not in self.EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding provider: {embedding_provider}. "
                f"Choose from: {', '.join(self.EMBEDDING_MODELS.keys())}"
            )

        self.vector_store_type = vector_store_type
        self.persist_directory = persist_directory

        # Initialize embedding model
        self.embedding_model = self.EMBEDDING_MODELS[embedding_provider](
            model=embedding_model
        )

        # Vector store will be initialized when needed
        self.vector_store = None

    def _initialize_vector_store(self, documents: Optional[List[Document]] = None):
        """
        Initialize the vector store.

        Args:
            documents: Optional list of documents to initialize with
        """
        if self.vector_store is not None:
            return

        if self.vector_store_type == "faiss":
            if documents:
                self.vector_store = FAISS.from_documents(
                    documents=documents, embedding=self.embedding_model
                )
            else:
                self.vector_store = FAISS(
                    embedding_function=self.embedding_model, index=None
                )

        elif self.vector_store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents if documents else [],
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
            )

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
        else:
            # Add documents to existing store
            self.vector_store.add_documents(documents)

        # Persist if directory is specified
        if self.persist_directory and hasattr(self.vector_store, "persist"):
            self.vector_store.persist()

        # Return document IDs
        return [doc.metadata["document_id"] for doc in documents]

    def _create_metadata_filter(self, user_info: Dict[str, Any]) -> Callable:
        """
        Create a metadata filter function based on user information.

        Args:
            user_info: Dictionary containing user information

        Returns:
            Filter function that takes document metadata and returns whether the user has access
        """

        def filter_fn(metadata: Dict[str, Any]) -> bool:
            # Always allow if no security info in metadata
            if "security_level" not in metadata:
                return True

            # Get security level and access lists
            security_level = metadata.get("security_level", "public")
            allowed_roles = metadata.get("allowed_roles", [])
            allowed_users = metadata.get("allowed_users", [])

            # Get user info
            user_id = user_info.get("user_id")
            user_roles = user_info.get("roles", [])
            user_clearance = user_info.get("clearance", "public")

            # Check direct user access
            if user_id and allowed_users and user_id in allowed_users:
                return True

            # Check role-based access
            for role in user_roles:
                if role in allowed_roles:
                    return True

            # Check clearance-based access
            clearance_levels = {
                "public": 0,
                "internal": 1,
                "confidential": 2,
                "restricted": 3,
            }

            if clearance_levels.get(user_clearance, -1) >= clearance_levels.get(
                security_level, 0
            ):
                return True

            return False

        return filter_fn

    def search(
        self,
        query: str,
        user_info: Dict[str, Any] = None,
        k: int = 5,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[Document]:
        """
        Search for documents in the vector store.

        Args:
            query: Search query
            user_info: User information for security filtering
            k: Number of documents to return
            filter_metadata: Additional metadata filter

        Returns:
            List of matching documents
        """
        if self.vector_store is None:
            # No documents indexed yet
            return []

        # Create metadata filter
        metadata_filter = filter_metadata

        # Apply security filtering if user info is provided
        if user_info:
            filter_fn = self._create_metadata_filter(user_info)

            # Get all relevant documents and filter manually
            # This is a simple approach for FAISS which doesn't support filter functions
            if self.vector_store_type == "faiss":
                results = self.vector_store.similarity_search(
                    query, k=k * 5
                )  # Get more to filter down
                filtered_results = [doc for doc in results if filter_fn(doc.metadata)]
                return filtered_results[:k]

            # For Chroma, we can use the filter argument
            elif self.vector_store_type == "chroma":
                # First, filter by department if provided in user_info
                user_dept = user_info.get("department")
                if user_dept and not metadata_filter:
                    metadata_filter = {"department": user_dept}
                elif user_dept and metadata_filter:
                    metadata_filter = {**metadata_filter, "department": user_dept}

        # Perform the search
        results = self.vector_store.similarity_search(
            query, k=k, filter=metadata_filter
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
            # Different vector stores have different deletion methods
            if hasattr(self.vector_store, "delete"):
                for doc_id in document_ids:
                    self.vector_store.delete(doc_id)
            elif hasattr(self.vector_store, "delete_by_filter"):
                for doc_id in document_ids:
                    self.vector_store.delete_by_filter({"document_id": doc_id})

            # Persist if directory is specified
            if self.persist_directory and hasattr(self.vector_store, "persist"):
                self.vector_store.persist()

            return True

        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def save_index(self, file_path: str) -> bool:
        """
        Save the vector store index to a file.

        Args:
            file_path: Path to save the index

        Returns:
            Success status
        """
        if self.vector_store is None:
            return False

        try:
            if self.vector_store_type == "faiss":
                self.vector_store.save_local(file_path)
                return True
            elif self.vector_store_type == "chroma" and self.persist_directory:
                self.vector_store.persist()
                return True

            return False

        except Exception as e:
            print(f"Error saving index: {e}")
            return False

    def load_index(self, file_path: str) -> bool:
        """
        Load a vector store index from a file.

        Args:
            file_path: Path to load the index from

        Returns:
            Success status
        """
        try:
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.load_local(file_path, self.embedding_model)
                return True
            elif self.vector_store_type == "chroma" and os.path.exists(
                self.persist_directory
            ):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                )
                return True

            return False

        except Exception as e:
            print(f"Error loading index: {e}")
            return False
