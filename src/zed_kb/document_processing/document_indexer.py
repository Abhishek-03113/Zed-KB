"""
Document indexer module for Zed-KB.
Handles indexing of documents in vector stores.
"""

from typing import List, Dict, Any, Optional, Callable
import uuid
import datetime

from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Import our custom vector store and embedding models
from ..vector_store.gemini_embeddings import GeminiEmbeddings
from ..vector_store.openai_embeddings import OpenAIEmbeddings
from ..vector_store.pinecone_store import PineconeStore


class DocumentIndexer:
    """Handles indexing of documents in vector stores."""

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
        hybrid_search: bool = False,
        pinecone_config: Optional[Dict[str, Any]] = None,
        vector_store_type: str = "pinecone",
    ):
        """
        Initialize the document indexer.

        Args:
            embedding_provider: Provider of embedding model ('openai' or 'gemini')
            embedding_model: Name of embedding model to use
            collection_name: Name of the collection/index to use
            hybrid_search: Whether to enable hybrid search (if supported)
            pinecone_config: Configuration for Pinecone connection
            vector_store_type: Type of vector store to use ('pinecone' or 'memory')
        """
        if embedding_provider not in self.EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding provider: {embedding_provider}. "
                f"Choose from: {', '.join(self.EMBEDDING_MODELS.keys())}"
            )

        self.hybrid_search = hybrid_search
        self.pinecone_config = pinecone_config or {}
        self.vector_store_type = vector_store_type.lower()

        # Get the collection/index name
        self.collection_name = collection_name
        if self.vector_store_type == "pinecone":
            self.collection_name = self.pinecone_config.get("index_name", collection_name)

        # Initialize embedding model
        self.embedding_model = self.EMBEDDING_MODELS[embedding_provider](
            model=embedding_model
        )

        # Vector store will be initialized when needed
        self.vector_store = None
        
        # Track documents and chunks for debugging purposes
        self.document_count = 0
        self.chunk_count = 0

    def _initialize_vector_store(self, documents: Optional[List[Document]] = None):
        """
        Initialize the vector store based on configuration.

        Args:
            documents: Optional list of documents to initialize with
        """
        if self.vector_store is not None:
            return

        # Create the appropriate vector store
        if self.vector_store_type == "pinecone":
            # Create Pinecone store with proper configuration
            print(f"Initializing Pinecone store with index name: {self.collection_name}")
            self.vector_store = PineconeStore(
                embedding_function=self.embedding_model,
                index_name=self.collection_name,
                namespace=self.pinecone_config.get("namespace"),
                api_key=self.pinecone_config.get("api_key"),
                environment=self.pinecone_config.get("environment", "us-west1-gcp"),
                cloud=self.pinecone_config.get("cloud", "aws"),
                region=self.pinecone_config.get("region", "us-east-1"),
            )
        else:
            raise ValueError(
                f"Unsupported vector store type: {self.vector_store_type}. "
                f"Choose from: 'pinecone', 'memory'."
            )

        # Add initial documents if provided
        if documents:
            self.add_documents(documents)

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
            self._initialize_vector_store()
            
        # Track how many documents/chunks we're processing
        self.document_count += len(documents)
        print(f"Adding {len(documents)} documents to {self.vector_store_type} store. Total documents: {self.document_count}")
            
        # Add documents to store in reasonable batches
        if len(documents) > 50:
            # Process in batches of 50 for large document sets
            doc_ids = []
            for i in range(0, len(documents), 50):
                batch = documents[i:i+50]
                print(f"Processing batch {i//50 + 1}/{(len(documents)-1)//50 + 1} ({len(batch)} documents)")
                batch_ids = self.vector_store.add_documents(batch)
                doc_ids.extend(batch_ids)
            return doc_ids
        else:
            # Add documents to existing store directly for small batches
            return self.vector_store.add_documents(documents)
            
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Dict[str, Any] = None,
        hybrid_alpha: float = 0.5,  # Balance between vector and keyword search
    ) -> List[Document]:
        """
        Search for documents in the vector store.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Metadata to filter results
            hybrid_alpha: Control hybrid search between keyword and vector search

        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            self._initialize_vector_store()

        # Set up metadata filter
        metadata_filter = filter_metadata or {}
        
        # Handle hybrid search if enabled
        hybrid_param = None
        if self.hybrid_search:
            hybrid_param = hybrid_alpha
        
        # For Pinecone, hybrid search requires additional configuration and may not work the same way
        if self.vector_store_type == "pinecone" and hybrid_param is not None:
            # Just pass it through - the PineconeStore will handle it appropriately
            pass
            
        results = self.vector_store.similarity_search(
            query, k=k, filter=metadata_filter,
            hybrid_alpha=hybrid_param
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
            self.vector_store.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def load_index(self, **kwargs) -> bool:
        """
        Load or reconnect to an existing vector store collection/index.

        Args:
            **kwargs: Additional arguments to pass to the vector store

        Returns:
            Success status
        """
        try:
            # This just initializes the connection to the existing collection/index
            self._initialize_vector_store()
            return True
        except Exception as e:
            print(f"Error connecting to vector store: {e}")
            return False
