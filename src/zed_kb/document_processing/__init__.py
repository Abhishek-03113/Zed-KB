"""
Document processing package for Zed-KB.
Contains components for loading, chunking, extracting metadata, and indexing documents.
"""

from .document_loader import DocumentLoader
from .document_chunker import DocumentChunker
from .metadata_extractor import MetadataExtractor
from .document_indexer import DocumentIndexer


class DocumentProcessor:
    """Main class that coordinates the document processing pipeline."""

    def __init__(
        self,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-ada-002",
        collection_name: str = "zed_kb_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        hybrid_search: bool = True,
        astra_config: dict = None,
    ):
        """
        Initialize the document processor.

        Args:
            embedding_provider: Provider of embedding model ('openai' or 'gemini')
            embedding_model: Name of embedding model to use
            collection_name: Name of the AstraDB collection to use
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            hybrid_search: Enable hybrid search (vector + keyword) for AstraDB
            astra_config: Configuration for AstraDB connection
        """
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.metadata_extractor = MetadataExtractor()
        self.indexer = DocumentIndexer(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            collection_name=collection_name,
            hybrid_search=hybrid_search,
            astra_config=astra_config,
        )

    def process_file(
        self,
        file_path: str,
        metadata: dict = None,
        security_level: str = "internal",
        allowed_roles: list = None,
        allowed_users: list = None,
    ) -> list:
        """
        Process a single file through the full pipeline.

        Args:
            file_path: Path to the file to process
            metadata: Additional metadata to add
            security_level: Security level for access control
            allowed_roles: List of roles that can access
            allowed_users: List of users that can access

        Returns:
            List of document IDs
        """
        # 1. Load the document
        documents = self.loader.load_file(file_path, metadata)

        # 2. Extract metadata
        documents = self.metadata_extractor.extract_batch_metadata(documents)

        # 3. Add security metadata
        documents = self.metadata_extractor.add_batch_security_metadata(
            documents,
            security_level=security_level,
            allowed_roles=allowed_roles,
            allowed_users=allowed_users,
        )

        # 4. Chunk the documents
        chunked_docs = self.chunker.chunk_documents(documents)

        # 5. Index the chunks
        doc_ids = self.indexer.add_documents(chunked_docs)

        return doc_ids

    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: dict = None,
        security_level: str = "internal",
        allowed_roles: list = None,
        allowed_users: list = None,
    ) -> list:
        """
        Process all documents in a directory.

        Args:
            directory_path: Path to the directory to process
            recursive: Whether to search recursively
            metadata: Additional metadata to add
            security_level: Security level for access control
            allowed_roles: List of roles that can access
            allowed_users: List of users that can access

        Returns:
            List of document IDs
        """
        # 1. Load documents from directory
        documents = self.loader.load_directory(directory_path, recursive, metadata)

        # 2. Extract metadata
        documents = self.metadata_extractor.extract_batch_metadata(documents)

        # 3. Add security metadata
        documents = self.metadata_extractor.add_batch_security_metadata(
            documents,
            security_level=security_level,
            allowed_roles=allowed_roles,
            allowed_users=allowed_users,
        )

        # 4. Chunk the documents
        chunked_docs = self.chunker.chunk_documents(documents)

        # 5. Index the chunks
        doc_ids = self.indexer.add_documents(chunked_docs)

        return doc_ids

    def search(
        self,
        query: str,
        user_info: dict = None,
        k: int = 5,
        filter_metadata: dict = None,
        hybrid_alpha: float = 0.5,
    ) -> list:
        """
        Search for documents with security filtering.

        Args:
            query: Search query
            user_info: User information for security filtering
            k: Number of documents to return
            filter_metadata: Additional metadata filter
            hybrid_alpha: Ratio between vector and keyword search (0 = only keywords, 1 = only vector)

        Returns:
            List of matching documents
        """
        return self.indexer.search(
            query, user_info, k, filter_metadata, hybrid_alpha=hybrid_alpha
        )
