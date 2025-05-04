"""
Pinecone vector store for Zed-KB.
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
    """Pinecone vector store with security filtering."""

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

        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key is required. Please provide it as an argument or "
                "set the PINECONE_API_KEY environment variable."
            )
        
        self.pc = Pinecone(api_key=self.api_key)
        
        if dimension is None:
            sample_vector = embedding_function.embed_query(
                "Sample text to determine vector dimension")
            self.dimension = len(sample_vector)
        else:
            self.dimension = dimension
        
        if not region:
            region = "us-east-1"
            
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
        if not texts:
            logger.warning("No texts provided to add_texts method. Skipping.")
            return []
            
        embeddings = self.embedding_function.embed_documents(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings for texts")

        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        vectors = []
        for i, (text, embedding, metadata, doc_id) in enumerate(zip(texts, embeddings, metadatas, ids)):
            if not embedding or len(embedding) != self.dimension:
                logger.warning(f"Skipping document {doc_id} due to malformed embedding (expected {self.dimension} dimensions, got {len(embedding) if embedding else 0})")
                continue
                
            security_metadata = {
                "security_level": metadata.get("security_level", "public"),
            }

            allowed_roles = metadata.get("allowed_roles", [])
            if allowed_roles:
                security_metadata["allowed_roles"] = allowed_roles
                
            chunk_metadata = {}
            if "chunk_index" in metadata and "chunk_total" in metadata:
                chunk_metadata = {
                    "chunk_index": metadata.get("chunk_index"),
                    "chunk_total": metadata.get("chunk_total")
                }

            combined_metadata = {**metadata, **security_metadata, **chunk_metadata}
            
            for key, value in combined_metadata.items():
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    continue
                combined_metadata[key] = str(value)

            vector = {
                'id': doc_id,
                'values': embedding,
                'metadata': {
                    self.text_field: text,
                    **combined_metadata
                }
            }
            vectors.append(vector)

        if not vectors:
            logger.warning("No valid vectors to insert")
            return []

        batch_size = 50
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
        query_embedding = self.embedding_function.embed_query(query)

        search_params = {
            "vector": query_embedding,
            "top_k": k,
            "namespace": self.namespace,
            "include_metadata": True
        }

        if filter:
            pinecone_filter = {}
            for key, value in filter.items():
                if isinstance(value, list):
                    pinecone_filter[key] = {"$in": value}
                else:
                    pinecone_filter[key] = value
                    
            search_params["filter"] = pinecone_filter

        if hybrid_alpha is not None:
            logger.warning(
                "Full hybrid search requires additional sparse vector configuration.")

        try:
            results = self.index.query(**search_params)

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
        ns = namespace if namespace is not None else self.namespace
        
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