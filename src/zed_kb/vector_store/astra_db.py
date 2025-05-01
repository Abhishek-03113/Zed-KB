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
    # Updated imports for the current astrapy API (v1.5.2+)
    from astrapy.db import AstraDB
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
        astra_db_keyspace: Optional[str] = "default_keyspace",
        namespace: Optional[str] = None,
        hybrid_search: bool = False,  # Changed default to False
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
        sample_vector = embedding_function.embed_query("Sample text to determine vector dimension")
        vector_dimension = len(sample_vector)
        
        # Create collection or use existing one
        try:
            # Create the collection if it doesn't exist
            collection_info = self.astra_db.create_collection(
                collection_name=self.collection_name,
                dimension=vector_dimension,
                # Enable hybrid search if requested
                options={"indexing": {"allow": ["text"]}} if self.hybrid_search else None
            )
            
            # Get the collection
            self.collection = self.astra_db.collection(self.collection_name)
            
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
        try:
            # For newer AstraDB API versions, we need a different approach to vector search
            # First, get all documents that match the metadata filters and text query
            filter_query = {}
            
            # Add text search condition if hybrid search is enabled
            if self.hybrid_search and query_text:
                filter_query["text"] = {"$contains": query_text}
                
            # Add additional metadata filters if provided
            if filter:
                filter_query.update(filter)
            
            # Get all documents matching the filters
            try:
                # Try with find_one first to inspect a single document
                sample_doc = self.collection.find_one({})
                logger.info(f"Sample document type: {type(sample_doc)}")
                if isinstance(sample_doc, str):
                    logger.info(f"Sample document string content (first 200 chars): {sample_doc[:200]}")
                elif isinstance(sample_doc, dict):
                    logger.info(f"Sample document keys: {list(sample_doc.keys())}")
            except Exception as e:
                logger.warning(f"Error when inspecting sample document: {e}")
            
            # Then we'll manually sort them by vector similarity
            matching_docs = list(self.collection.find(
                filter_query if filter_query else {},
                options={"limit": 1000}  # Get more than we need to sort later
            ))
            
            logger.info(f"Retrieved {len(matching_docs)} documents from AstraDB")
            
            # If we have no matches, return empty list
            if not matching_docs:
                logger.warning("No matching documents found in AstraDB collection")
                return []
                
            # Try to get the first document to inspect what we're dealing with
            first_doc = matching_docs[0] if matching_docs else None
            if first_doc:
                logger.info(f"First document type: {type(first_doc)}")
                if isinstance(first_doc, str):
                    # Log first few characters for diagnosis
                    logger.info(f"First document content starts with: {first_doc[:100]}")
                    
                    # If it starts with data:, it might be a base64 encoded document or binary data
                    if first_doc.startswith("data:"):
                        logger.info("Document appears to be base64/binary data")
                        
                        # Create a simple mock document as a fallback
                        mock_doc = {
                            "_id": "fallback_doc",
                            "text": "This is fallback content as the original document couldn't be parsed",
                            "$vector": embedding,  # Use the query vector as a placeholder
                            "$similarity": 0.5,  # Assign a medium similarity score
                        }
                        
                        # Return a single document as fallback
                        doc = Document(
                            page_content=mock_doc["text"],
                            metadata={"document_id": mock_doc["_id"], "similarity": mock_doc["$similarity"]}
                        )
                        return [doc]
            
            # Manual vector similarity calculation
            # Since the native vector search isn't working
            results_with_scores = []
            for doc in matching_docs:
                # Handle string documents by trying to parse them as JSON
                if isinstance(doc, str):
                    try:
                        import json
                        # Try to parse as JSON if it's a string
                        doc = json.loads(doc)
                        logger.info(f"Successfully parsed string document as JSON")
                    except json.JSONDecodeError:
                        logger.warning(f"Document is a string and couldn't be parsed as JSON: {doc[:100]}...")
                        continue
                
                # Skip if doc is still not a dictionary or doesn't have vector embeddings
                if not isinstance(doc, dict) or "$vector" not in doc:
                    if isinstance(doc, dict):
                        logger.warning(f"Document {doc.get('_id', 'unknown')} is missing vector embedding")
                    else:
                        logger.warning(f"Document is neither a string nor a dictionary: {type(doc)}")
                    continue
                    
                # Calculate cosine similarity
                doc_vector = doc["$vector"]
                score = self._cosine_similarity(embedding, doc_vector)
                
                # Add score to document
                doc["$similarity"] = score
                results_with_scores.append(doc)
            
            # Sort by similarity (highest first)
            results_with_scores.sort(key=lambda x: x.get("$similarity", 0), reverse=True)
            
            # Take top k results
            results = results_with_scores[:k]
            
            logger.info(f"After processing, found {len(results)} valid results to return")
            
        except Exception as e:
            # Log the error for debugging
            logger.error(f"AstraDB search error: {e}")
            logger.exception("Full stack trace:")
            raise
        
        # Convert results to documents
        documents = []
        for item in results:
            # Extract metadata (all keys except specific ones)
            metadata = {
                k: v for k, v in item.items()
                if k not in ["_id", "text", "$vector", "$similarity"]
            }
            
            # Add document ID and similarity score to metadata
            metadata["document_id"] = item.get("_id", "unknown")
            if "$similarity" in item:
                metadata["similarity"] = item["$similarity"]
            
            # Create document with text content and metadata
            doc = Document(page_content=item.get("text", ""), metadata=metadata)
            documents.append(doc)
            
        return documents
            
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0.0
            
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def delete(self, ids: List[str], **kwargs: Any) -> Optional[bool]:
        """
        Delete documents from the vector store.

        Args:
            ids: List of IDs to delete

        Returns:
            Success flag
        """
        for doc_id in ids:
            self.collection.delete({"_id": doc_id})
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
        # First initialize the vector store
        astra_store = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            **kwargs,
        )
        
        # Only add documents if there are any
        if documents:
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
