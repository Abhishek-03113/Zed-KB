"""
AstraDB vector store implementation for Zed-KB.
Provides integration with DataStax Astra DB for vector search with security level filtering.
"""

from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Type
import uuid
import os
import json
import logging

from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

try:
    # Updated imports for the current astrapy API (v1.5.2+)
    from astrapy.db import AstraDB
    HAS_ASTRA = True
except ImportError:
    HAS_ASTRA = False

logger = logging.getLogger(__name__)


class AstraDBStore(VectorStore):
    """Vector store implementation using DataStax AstraDB with security level filtering."""

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "documents",
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_id: Optional[str] = None,
        astra_db_region: Optional[str] = None,
        astra_db_keyspace: Optional[str] = "default_keyspace",
        namespace: Optional[str] = None,
        hybrid_search: bool = False,  # Kept for backwards compatibility
        index_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the AstraDB vector store.

        Args:
            embedding_function: Function to convert text to embedding vectors
            collection_name: Name of the collection to use
            token: AstraDB token
            api_endpoint: AstraDB API endpoint 
            astra_db_id: AstraDB database ID
            astra_db_region: AstraDB database region
            astra_db_keyspace: AstraDB keyspace name
            namespace: Collection namespace
            hybrid_search: Whether to enable hybrid search (kept for backwards compatibility)
            index_config: Configuration for vector index creation
        """
        if not HAS_ASTRA:
            raise ImportError(
                "Could not import astrapy package. "
                "Please install it with `pip install astrapy`."
            )

        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.hybrid_search = hybrid_search

        # Initialize connection
        self.token = token or os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
        if not self.token:
            raise ValueError(
                "AstraDB token is required. Please provide it as an argument or "
                "set the ASTRA_DB_APPLICATION_TOKEN environment variable."
            )

        # API endpoint takes precedence over database ID and region
        if api_endpoint:
            self.api_endpoint = api_endpoint
        elif astra_db_id and astra_db_region:
            self.api_endpoint = f"https://{astra_db_id}-{astra_db_region}.apps.astra.datastax.com"
        else:
            raise ValueError(
                "Either api_endpoint or both astra_db_id and astra_db_region must be provided."
            )

        # Initialize the AstraDB client
        self.astra_db = AstraDB(
            api_endpoint=self.api_endpoint,
            token=self.token,
            namespace=namespace,
        )

        # Configure the vector dimension based on the embedding function
        sample_vector = embedding_function.embed_query(
            "Sample text to determine vector dimension")
        vector_dimension = len(sample_vector)

        # Create collection or use existing one
        try:
            # Create the collection if it doesn't exist with indexed security fields
            # Note: AstraDB requires explicit indexing of fields used for filtering
            collection_options = {
                "vector": {
                    "dimension": vector_dimension,
                    "metric": "cosine"  # Updated from 'similarity' to 'metric'
                },
                "indexing": {
                    # Fields to exclude from indexing
                    "deny": []  # Updated from 'deny_list'
                }
            }

            try:
                # Create new collection with proper indexing
                collection_info = self.astra_db.create_collection(
                    collection_name=self.collection_name,
                    options=collection_options
                )
                logger.info(
                    f"Created AstraDB collection: {self.collection_name}")
            except Exception as e:
                # Collection might already exist
                if "already exists" not in str(e):
                    logger.warning(f"Error creating collection: {e}")

            # Get the collection
            self.collection = self.astra_db.collection(self.collection_name)

            logger.info(
                f"Connected to AstraDB collection: {self.collection_name}")
        except Exception as e:
            raise ConnectionError(f"Error connecting to AstraDB: {e}")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store with optional metadata.

        Args:
            texts: List of texts to add
            metadatas: List of metadata dicts to associate with texts
            ids: Optional list of IDs for the documents

        Returns:
            List of document IDs
        """
        # Generate embeddings for the texts
        embeddings = self.embedding_function.embed_documents(texts)

        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Add metadatas if not provided
        if not metadatas:
            metadatas = [{} for _ in texts]

        # Prepare documents for insertion
        documents = []
        for i, (text, embedding, metadata, doc_id) in enumerate(zip(texts, embeddings, metadatas, ids)):
            # Ensure security fields exist for filtering
            security_metadata = {
                "security_level": metadata.get("security_level", "public"),
            }

            # Add allowed_roles as a simple string array that can be indexed and filtered
            allowed_roles = metadata.get("allowed_roles", [])
            if allowed_roles:
                security_metadata["allowed_roles"] = allowed_roles

            # Merge with other metadata
            combined_metadata = {**metadata, **security_metadata}

            # Create document
            doc = {
                "_id": doc_id,
                "text": text,
                "$vector": embedding,
                **combined_metadata
            }
            documents.append(doc)

        # Insert documents in batches to handle potential API limits
        batch_size = 20  # Adjust based on AstraDB's limits
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            # Use insert_many with the updated API
            self.collection.insert_many(batch)

        return ids

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            ids: Optional list of IDs for the documents

        Returns:
            List of document IDs
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
    
    @classmethod
    def from_texts(
        cls: Type["AstraDBStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "documents",
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_id: Optional[str] = None,
        astra_db_region: Optional[str] = None,
        **kwargs: Any,
    ) -> "AstraDBStore":
        """
        Create an AstraDBStore from texts.

        Args:
            texts: List of texts to add
            embedding: Embedding function
            metadatas: Optional list of metadatas
            ids: Optional list of document IDs
            collection_name: Name of the collection to use
            token: AstraDB token
            api_endpoint: AstraDB API endpoint
            astra_db_id: AstraDB database ID
            astra_db_region: AstraDB database region
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            AstraDBStore instance
        """
        store = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_id=astra_db_id,
            astra_db_region=astra_db_region,
            **kwargs,
        )
        
        if texts:
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            
        return store

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        hybrid_alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents similar to the query.

        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional metadata filter
            hybrid_alpha: Optional hybrid search parameter (0-1)
                          0 = full text search, 1 = vector search (not currently used)
            **kwargs: Additional arguments

        Returns:
            List of similar Documents
        """
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Prepare search parameters - pass as direct parameters instead of 'options'
        search_params = {
            "vector": query_embedding,
            "limit": k,
        }
        
        # Add filter if provided
        if filter:
            search_params["filter"] = filter
            
        # Note: Hybrid search is not currently supported in the API version
        # We'll log a warning if it was requested but not available
        if self.hybrid_search and hybrid_alpha is not None:
            logger.warning("Hybrid search requested but not available in current AstraDB API version")
                
        try:
            # Execute the search using the updated parameter format
            results = self.collection.vector_find(**search_params)
            
            # Convert results to Document objects
            documents = []
            for result in results:
                # Extract metadata (exclude internal fields)
                metadata = {k: v for k, v in result.items() 
                          if not k.startswith("_") and not k.startswith("$") and k != "text"}
                
                documents.append(
                    Document(
                        page_content=result.get("text", ""),
                        metadata=metadata
                    )
                )
                
            return documents
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            # Return empty list on error
            return []
