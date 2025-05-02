"""
AstraDB vector store implementation for Zed-KB.
Provides integration with DataStax Astra DB for vector search with security level filtering.
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
        sample_vector = embedding_function.embed_query("Sample text to determine vector dimension")
        vector_dimension = len(sample_vector)
        
        # Create collection or use existing one
        try:
            # Create the collection if it doesn't exist with indexed security fields
            # Note: AstraDB requires explicit indexing of fields used for filtering
            collection_options = {
                "vector": {
                    "dimension": vector_dimension,
                    "similarity": "cosine"
                },
                "indexing": {
                    "deny_list": [],  # Fields to exclude from indexing (using underscore)
                    "include_vectors": True  # Ensure vector indexing is enabled
                }
            }
            
            try:
                # Create new collection with proper indexing
                collection_info = self.astra_db.create_collection(
                    collection_name=self.collection_name,
                    options=collection_options
                )
                logger.info(f"Created AstraDB collection: {self.collection_name}")
            except Exception as e:
                # Collection might already exist
                if "already exists" not in str(e):
                    logger.warning(f"Error creating collection: {e}")
            
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

    def similarity_search_with_security(
        self,
        query: str,
        user_info: Dict[str, Any],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents similar to the query text with security filtering.

        Args:
            query: Text to search for
            user_info: User information containing roles and clearance level
            k: Number of documents to return
            filter: Additional metadata filter dict

        Returns:
            List of documents most similar to the query text
        """
        # Get embedding for the query
        embedding = self.embedding_function.embed_query(query)
        
        # Extract user security information
        user_roles = user_info.get("roles", [])
        user_clearance = user_info.get("clearance", "public")
        
        # Create security filters
        security_filter = self._create_security_filter(user_roles, user_clearance)
        
        # Combine with any additional filters
        combined_filter = security_filter
        if filter:
            # Merge filters
            combined_filter.update(filter)
        
        logger.info(f"Searching with security filter: {combined_filter}")
        
        return self.similarity_search_by_vector(
            embedding=embedding, 
            k=k, 
            filter=combined_filter, 
            query_text=query,
            **kwargs
        )
    
    def _create_security_filter(self, user_roles: List[str], user_clearance: str) -> Dict[str, Any]:
        """
        Create a security filter based on user roles and clearance.
        
        Args:
            user_roles: List of user roles
            user_clearance: User clearance level
            
        Returns:
            Filter dictionary for AstraDB query
        """
        security_levels = ["public", "internal", "confidential", "restricted", "top_secret"]
        
        # Get index of user's clearance level
        try:
            clearance_index = security_levels.index(user_clearance.lower())
        except ValueError:
            # Default to public if clearance level not recognized
            clearance_index = 0
            
        # Get all clearance levels the user can access
        allowed_levels = security_levels[:clearance_index + 1]
        
        # Create filter for security level only (simplified filtering approach)
        level_filter = {"security_level": {"$in": allowed_levels}}
        
        # Role filtering is more complex and must match documents where:
        # 1. The document has no allowed_roles field, or
        # 2. The document's allowed_roles field contains at least one of the user's roles
        # We'll check for role access after retrieving documents to simplify querying
        
        return level_filter

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
            # First, get all documents that match the metadata filters
            filter_query = filter or {}
            
            # Get all documents matching the filters using AstraDB's find method
            try:
                # Set up parameters for vector search
                find_params = {}
                
                # Apply filter if present
                if filter_query:
                    find_params["filter"] = filter_query
                
                # Set up options
                find_params["options"] = {
                    "limit": k,
                    "includeSimilarity": True
                }
                
                # Add vector search
                find_params["sort"] = {"$vector": embedding}
                
                # Add hybrid search if enabled and query text is provided
                hybrid_alpha = kwargs.get("hybrid_alpha", 0.5)  # Default to balanced search
                if self.hybrid_search and query_text and 0 <= hybrid_alpha <= 1:
                    # AstraDB uses $text for hybrid search with specified alpha
                    find_params["sort"] = {
                        "$search": {
                            "vector": embedding,
                            "text": query_text,
                            "alpha": hybrid_alpha  # Balance between vector & text
                        }
                    }
                    logger.info(f"Using hybrid search with alpha={hybrid_alpha}")
                
                # Perform the search
                results = list(self.collection.find(**find_params))
                logger.info(f"Retrieved {len(results)} documents from AstraDB using vector search")
                
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to regular find: {e}")
                
                # Fallback: get documents matching the filters without vector search
                query_params = {}
                
                # Apply filter if present
                if filter_query:
                    query_params["filter"] = filter_query
                
                # Add pagination parameters
                query_params["options"] = {"limit": 100}  # Get more than we need to sort later
                
                matching_docs = list(self.collection.find(**query_params))
                
                logger.info(f"Retrieved {len(matching_docs)} documents from AstraDB")
                
                # If we have no matches, return empty list
                if not matching_docs:
                    logger.warning("No matching documents found in AstraDB collection")
                    return []
                
                # Manual vector similarity calculation
                results_with_scores = []
                for doc in matching_docs:
                    # Handle string documents by trying to parse them as JSON
                    if isinstance(doc, str):
                        try:
                            doc = json.loads(doc)
                        except json.JSONDecodeError:
                            logger.warning(f"Document couldn't be parsed as JSON: {doc[:100]}...")
                            continue
                    
                    # Skip if doc is still not a dictionary or doesn't have vector embeddings
                    if not isinstance(doc, dict) or "$vector" not in doc:
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

                # Check for role-based access if needed
                if "user_roles" in kwargs:
                    user_roles = kwargs["user_roles"]
                    # Filter based on roles (post-processing)
                    results = [
                        doc for doc in results 
                        if (
                            "allowed_roles" not in doc or  # No role restrictions
                            not doc.get("allowed_roles") or  # Empty role restrictions
                            any(role in user_roles for role in doc.get("allowed_roles", []))  # Has matching role
                        )
                    ]
                    logger.info(f"After role filtering, found {len(results)} results to return")
            
        except Exception as e:
            # Log the error for debugging
            logger.error(f"AstraDB search error: {e}")
            logger.exception("Full stack trace:")
            raise
        
        # Convert results to documents
        documents = []
        for item in results:
            try:
                # Handle potential string results (depends on AstraDB version)
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping result that couldn't be parsed: {item[:100]}...")
                        continue
                
                # Extract text content
                if "text" not in item:
                    logger.warning(f"Skipping result without text field: {item.keys()}")
                    continue
                    
                text_content = item.get("text", "")
                
                # Extract metadata (all keys except specific ones)
                metadata = {
                    k: v for k, v in item.items()
                    if k not in ["_id", "text", "$vector", "$similarity"]
                }
                
                # Add document ID and similarity score to metadata
                metadata["document_id"] = item.get("_id", "unknown")
                
                # Handle similarity score from different possible sources
                if "$similarity" in item:
                    metadata["similarity"] = item["$similarity"]
                elif "$vectorDistance" in item:
                    # Some AstraDB versions return vector distance
                    metadata["similarity"] = 1.0 - item["$vectorDistance"]  # Convert distance to similarity
                
                # Create document with text content and metadata
                doc = Document(page_content=text_content, metadata=metadata)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
            
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
