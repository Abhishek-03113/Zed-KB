"""
AstraDB vector store implementation for Zed-KB.
Provides integration with DataStax Astra DB for vector search with hybrid capabilities.
"""

from typing import List, Dict, Any, Optional, Callable, Union
import uuid
import os
import json
import logging

from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

try:
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.query import dict_factory
    from astrapy.db import AstraDB, AstraDBCollection
    HAS_ASTRA = True
except ImportError:
    HAS_ASTRA = False

logger = logging.getLogger(__name__)


class AstraDBStore(VectorStore):
    """Vector store implementation using DataStax AstraDB."""

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "documents",
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_id: Optional[str] = None,
        astra_db_region: Optional[str] = None,
        astra_db_keyspace: Optional[str] = "vector_keyspace",
        namespace: Optional[str] = None,
        hybrid_search: bool = True,
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
            hybrid_search: Whether to enable hybrid search (vector + BM25 text search)
            index_config: Configuration for vector index creation
        """
        if not HAS_ASTRA:
            raise ImportError(
                "Could not import astrapy package. "
                "Please install it with `pip install astrapy cassandra-driver`."
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
            self.astra_db = AstraDB(token=self.token, api_endpoint=api_endpoint)
        elif astra_db_id and astra_db_region:
            self.api_endpoint = f"https://{astra_db_id}-{astra_db_region}.apps.astra.datastax.com"
            self.astra_db = AstraDB(
                token=self.token,
                api_endpoint=self.api_endpoint,
            )
        else:
            raise ValueError(
                "Either api_endpoint or both astra_db_id and astra_db_region must be provided."
            )

        # Get or create collection
        self.namespace = namespace
        
        # Configure the vector dimension based on the embedding function
        # Use a simple text to get the vector dimension
        sample_vector = embedding_function.embed_query("Sample text to determine vector dimension")
        vector_dimension = len(sample_vector)
        
        # Default index configuration if not provided
        if not index_config:
            index_config = {
                "vector": {
                    "dimensions": vector_dimension,
                    "metric": "cosine",
                }
            }
            
            # Add text index configuration if hybrid search is enabled
            if self.hybrid_search:
                index_config["text"] = {
                    "analyzer": "en.english",
                }
        
        try:
            # Try to get the collection, create it if it doesn't exist
            if namespace:
                self.collection = self.astra_db.create_collection(
                    collection_name=self.collection_name,
                    dimension=vector_dimension,
                    namespace=namespace,
                    options={"indexConfig": index_config}
                )
            else:
                self.collection = self.astra_db.create_collection(
                    collection_name=self.collection_name,
                    dimension=vector_dimension,
                    options={"indexConfig": index_config}
                )
                
            logger.info(f"Connected to AstraDB collection: {self.collection_name}")
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
            # Create document with text for hybrid search
            doc = {
                "_id": doc_id,
                "text": text,
                "$vector": embedding,
                **metadata
            }
            documents.append(doc)

        # Insert documents in batches to handle potential API limits
        batch_size = 100  # Adjust based on AstraDB's limits
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
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

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents similar to the query text.

        Args:
            query: Text to search for
            k: Number of results to return
            filter: Optional metadata filter dict

        Returns:
            List of documents most similar to the query text
        """
        # Get embedding for the query
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, filter, query, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents similar to the embedding vector.

        Args:
            embedding: Embedding vector to search for
            k: Number of results to return
            filter: Optional metadata filter dict
            query_text: Optional raw text of the query for hybrid search

        Returns:
            List of documents most similar to the embedding
        """
        search_query = {"$vector": embedding}
        
        # Add hybrid search if enabled and query text is provided
        if self.hybrid_search and query_text:
            search_query["text"] = {"$contains": query_text}
            if "hybrid_alpha" in kwargs:
                # The ratio between vector and text search in hybrid searches (0.0 to 1.0)
                # 0 means only text search, 1 means only vector search
                search_query["$options"] = {"hybridSearch": {"alpha": kwargs["hybrid_alpha"]}}
                
        # Add filter if provided
        if filter:
            for key, value in filter.items():
                search_query[key] = value

        # Execute the search
        results = self.collection.vector_find(
            search_query,
            limit=k,
            include_similarity=True,
            fields=["text", "*"],  # Include all fields
        )

        # Convert results to documents
        documents = []
        for item in results:
            # Extract metadata (all keys except _id, text, $vector, and $similarity)
            metadata = {
                k: v for k, v in item.items() 
                if k not in ["_id", "text", "$vector", "$similarity"]
            }
            
            # Add document ID and similarity score to metadata
            metadata["document_id"] = item["_id"]
            if "$similarity" in item:
                metadata["similarity"] = item["$similarity"]
                
            # Create document
            doc = Document(page_content=item["text"], metadata=metadata)
            documents.append(doc)

        return documents

    def delete(self, ids: List[str], **kwargs: Any) -> Optional[bool]:
        """
        Delete documents from the vector store.

        Args:
            ids: List of IDs to delete

        Returns:
            Success flag
        """
        for doc_id in ids:
            self.collection.delete_one({"_id": doc_id})
        return True

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str = "documents",
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "AstraDBStore":
        """
        Create an AstraDB vector store from documents.

        Args:
            documents: List of documents to add
            embedding: Embedding function
            collection_name: Name of the collection to use
            ids: Optional list of document IDs

        Returns:
            AstraDB vector store containing the documents
        """
        astra_store = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            **kwargs,
        )
        astra_store.add_documents(documents=documents, ids=ids)
        return astra_store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = "documents",
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "AstraDBStore":
        """
        Create an AstraDB vector store from texts.

        Args:
            texts: List of texts to add
            embedding: Embedding function
            metadatas: Optional list of metadatas
            collection_name: Name of the collection to use
            ids: Optional list of document IDs

        Returns:
            AstraDB vector store containing the texts
        """
        astra_store = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            **kwargs,
        )
        astra_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return astra_store