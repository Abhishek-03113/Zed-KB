"""
Pinecone vector store implementation for Zed-KB.
Provides integration with Pinecone for vector search with security level filtering.
"""

from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Type
import uuid
import os
import logging
import json

from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):
    """Vector store implementation using Pinecone with security level filtering."""

    def __init__(
        self,
        embedding_function: Embeddings,
        index_name: str = "documents",
        namespace: Optional[str] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = "us-east-1",
        text_field: str = "text",
        metadata_field: str = "metadata",
        dimension: Optional[int] = None,
        cloud: str = "aws",
        region: Optional[str] = None,
    ):
        """
        Initialize the Pinecone vector store.

        Args:
            embedding_function: Function to convert text to embedding vectors
            index_name: Name of the Pinecone index to use
            namespace: Optional Pinecone namespace
            api_key: Pinecone API key
            environment: Pinecone environment (legacy parameter)
            text_field: Field name for document content
            metadata_field: Field name for metadata
            dimension: Vector dimension (calculated from embedding if not provided)
            cloud: Cloud provider for serverless ('aws' or 'gcp')
            region: Region for the serverless index (if not provided, uses environment)
        """
        if not HAS_PINECONE:
            raise ImportError(
                "Could not import pinecone package. "
                "Please install it with `pip install pinecone-client`."
            )

        self.embedding_function = embedding_function
        self.index_name = index_name
        self.namespace = namespace
        self.text_field = text_field
        self.metadata_field = metadata_field

        # Initialize connection
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key is required. Please provide it as an argument or "
                "set the PINECONE_API_KEY environment variable."
            )
        
        # Set up the Pinecone client - using new class-based API
        self.pc = Pinecone(api_key=self.api_key)
        
        # Get or create the Pinecone index
        if dimension is None:
            # Get vector dimension from the embedding function
            sample_vector = embedding_function.embed_query(
                "Sample text to determine vector dimension")
            self.dimension = len(sample_vector)
        else:
            self.dimension = dimension
        
        # Determine region if not provided
        if not region:
            region = "us-east-1"  # Default to us-east-1 as the current server region
            
        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            logger.info(f"Created Pinecone index: {self.index_name}")
        
        # Get index interface
        self.index = self.pc.Index(self.index_name)
        
        logger.info(f"Connected to Pinecone index: {self.index_name}")

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
        # Validate inputs
        if not texts:
            logger.warning("No texts provided to add_texts method. Skipping.")
            return []
            
        # Generate embeddings for the texts
        embeddings = self.embedding_function.embed_documents(texts)
        
        # Log embedding generation for debugging
        logger.info(f"Generated {len(embeddings)} embeddings for texts")

        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Add metadatas if not provided
        if not metadatas:
            metadatas = [{} for _ in texts]

        # Prepare vectors for insertion
        vectors = []
        for i, (text, embedding, metadata, doc_id) in enumerate(zip(texts, embeddings, metadatas, ids)):
            # Skip if embedding is malformed
            if not embedding or len(embedding) != self.dimension:
                logger.warning(f"Skipping document {doc_id} due to malformed embedding (expected {self.dimension} dimensions, got {len(embedding) if embedding else 0})")
                continue
                
            # Ensure security fields exist for filtering
            security_metadata = {
                "security_level": metadata.get("security_level", "public"),
            }

            # Add allowed_roles as a simple string array that can be indexed and filtered
            allowed_roles = metadata.get("allowed_roles", [])
            if allowed_roles:
                security_metadata["allowed_roles"] = allowed_roles
                
            # Add chunk information if available
            chunk_metadata = {}
            if "chunk_index" in metadata and "chunk_total" in metadata:
                chunk_metadata = {
                    "chunk_index": metadata.get("chunk_index"),
                    "chunk_total": metadata.get("chunk_total")
                }

            # Merge with other metadata
            combined_metadata = {**metadata, **security_metadata, **chunk_metadata}
            
            # Ensure metadata values are JSON serializable
            # Convert any non-serializable types to strings
            for key, value in combined_metadata.items():
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    continue
                combined_metadata[key] = str(value)

            # Create vector object for Pinecone
            vector = {
                'id': doc_id,
                'values': embedding,
                'metadata': {
                    self.text_field: text,
                    **combined_metadata
                }
            }
            vectors.append(vector)

        # Check if we have any valid vectors to insert
        if not vectors:
            logger.warning("No valid vectors to insert")
            return []

        # Insert vectors in batches to handle potential API limits
        batch_size = 50  # Adjusted batch size for better reliability
        successful_ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                logger.info(f"Inserting batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} with {len(batch)} vectors")
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace
                )
                successful_ids.extend([v['id'] for v in batch])
            except Exception as e:
                logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                # Continue with next batch instead of failing completely

        return successful_ids

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
        cls: Type["PineconeStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        index_name: str = "documents",
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> "PineconeStore":
        """
        Create a PineconeStore from texts.

        Args:
            texts: List of texts to add
            embedding: Embedding function
            metadatas: Optional list of metadatas
            ids: Optional list of document IDs
            index_name: Name of the Pinecone index to use
            namespace: Optional Pinecone namespace
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            PineconeStore instance
        """
        store = cls(
            embedding_function=embedding,
            index_name=index_name,
            namespace=namespace,
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
            0 = full text search, 1 = vector search
            **kwargs: Additional arguments

        Returns:
            List of similar Documents
        """
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)

        # Prepare search parameters
        search_params = {
            "vector": query_embedding,
            "top_k": k,
            "namespace": self.namespace,
            "include_metadata": True
        }

        # Add filter if provided
        if filter:
            # Convert filter to Pinecone format
            pinecone_filter = {}
            for key, value in filter.items():
                if isinstance(value, list):
                    # For list values (like allowed_roles), use $in operator
                    pinecone_filter[key] = {"$in": value}
                else:
                    # For scalar values, use direct equality
                    pinecone_filter[key] = value
                    
            search_params["filter"] = pinecone_filter

        # Handle hybrid search if alpha is provided
        if hybrid_alpha is not None:
            # Pinecone has built-in hybrid search via the 'sparse_vector' parameter,
            # but this would require additional sparse vector generation.
            # This is a basic implementation that would need additional work
            # for full hybrid search support.
            logger.warning(
                "Full hybrid search requires additional sparse vector configuration.")

        try:
            # Execute the search
            results = self.index.query(**search_params)

            # Convert results to Document objects
            documents = []
            for match in results.matches:
                metadata = match.metadata
                text = metadata.pop(self.text_field, "")
                
                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )

            return documents

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            # Return empty list on error
            return []
    
    def delete(self, ids: List[str], namespace: Optional[str] = None) -> None:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace override
        """
        if not ids:
            return
            
        # Use the instance namespace if not overridden
        ns = namespace if namespace is not None else self.namespace
        
        try:
            self.index.delete(ids=ids, namespace=ns)
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            
    def delete_by_filter(self, filter: Dict[str, Any], namespace: Optional[str] = None) -> None:
        """
        Delete vectors by metadata filter.
        
        Args:
            filter: Metadata filter to select vectors to delete
            namespace: Optional namespace override
        """
        # Use the instance namespace if not overridden
        ns = namespace if namespace is not None else self.namespace
        
        # Convert filter to Pinecone format
        pinecone_filter = {}
        for key, value in filter.items():
            if isinstance(value, list):
                pinecone_filter[key] = {"$in": value}
            else:
                pinecone_filter[key] = value
                
        try:
            self.index.delete(filter=pinecone_filter, namespace=ns)
        except Exception as e:
            logger.error(f"Error deleting vectors by filter: {e}")