"""
Handles document chunking for Zed-KB.
"""

from typing import List, Dict, Any, Optional, Union
import re

from langchain.schema import Document
from langchain_text_splitters.character import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.text_splitter import TokenTextSplitter  # Updated import path
from langchain_text_splitters.markdown import MarkdownTextSplitter
from langchain_text_splitters.html import HTMLHeaderTextSplitter


class DocumentChunker:
    """Chunks documents into pieces."""

    CHUNK_STRATEGIES = {
        "recursive": RecursiveCharacterTextSplitter,
        "token": TokenTextSplitter,
        "character": CharacterTextSplitter,
        "markdown": MarkdownTextSplitter,
        # Updated to use HTMLHeaderTextSplitter instead of HTMLTextSplitter
        "html": HTMLHeaderTextSplitter,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy to use
        """
        if strategy not in self.CHUNK_STRATEGIES:
            raise ValueError(
                f"Unsupported chunking strategy: {strategy}. "
                f"Choose from: {', '.join(self.CHUNK_STRATEGIES.keys())}"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        # Use appropriate separator based on strategy
        separators = ["\n\n", "\n", " ", ""]
        if strategy == "recursive":
            self.text_splitter = self.CHUNK_STRATEGIES[strategy](
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        elif strategy == "markdown":
            self.text_splitter = self.CHUNK_STRATEGIES[strategy](
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif strategy == "html":
            self.text_splitter = self.CHUNK_STRATEGIES[strategy](
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            self.text_splitter = self.CHUNK_STRATEGIES[strategy](
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

    def _detect_document_format(self, document: Document) -> str:
        """
        Detect the format of a document to select appropriate chunking strategy.

        Args:
            document: The document to analyze

        Returns:
            The detected format
        """
        source = (
            document.metadata.get(
                "source", "").lower() if document.metadata else ""
        )

        if source.endswith(".md") or source.endswith(".markdown"):
            return "markdown"
        elif source.endswith(".html") or source.endswith(".htm"):
            return "html"
        else:
            # Default to recursive for most documents
            return "recursive"

    def _enrich_metadata_with_chunk_info(
        self, chunks: List[Document], parent_document: Optional[Document] = None
    ) -> List[Document]:
        """
        Add chunking metadata to document chunks.

        Args:
            chunks: List of document chunks
            parent_document: Original document

        Returns:
            List of chunks with enhanced metadata
        """
        doc_len = len(chunks)

        # Create and copy metadata
        for i, chunk in enumerate(chunks):
            # Add chunk-specific metadata
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = doc_len

            # Add parent document ID if available
            if parent_document and parent_document.metadata.get("doc_id"):
                chunk.metadata["parent_doc_id"] = parent_document.metadata["doc_id"]
            
            # Inherit security metadata from parent document if available
            if parent_document and parent_document.metadata:
                # Copy security metadata
                for security_field in ["security_level", "allowed_roles", "allowed_users"]:
                    if security_field in parent_document.metadata:
                        chunk.metadata[security_field] = parent_document.metadata[security_field]

        return chunks

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks.

        Args:
            document: Document to split

        Returns:
            List of document chunks
        """
        # Select appropriate chunking strategy based on document format
        detected_format = self._detect_document_format(document)

        # If the format is different from the initialized one, create a new splitter
        if (
            detected_format != self.strategy
            and detected_format in self.CHUNK_STRATEGIES
        ):
            if detected_format == "markdown":
                splitter = MarkdownTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
            elif detected_format == "html":
                splitter = HTMLHeaderTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
            else:
                splitter = self.text_splitter
        else:
            splitter = self.text_splitter

        # Split the document
        # Check if the document has content to avoid errors
        if not document.page_content or len(document.page_content.strip()) == 0:
            print(f"Warning: Document has empty content. Source: {document.metadata.get('source', 'Unknown')}")
            return []
            
        chunks = splitter.split_documents([document])
        
        # Log chunking results for debugging
        print(f"Document chunked into {len(chunks)} parts (source: {document.metadata.get('source', 'Unknown')})")
        
        # If we got no chunks, there might be an issue with the content
        if not chunks:
            print(f"Warning: Document produced 0 chunks. Content length: {len(document.page_content)}")
            # Create a single chunk with the original content as a fallback
            # This prevents documents from disappearing from search results
            chunks = [document]

        # Add metadata to chunks
        enriched_chunks = self._enrich_metadata_with_chunk_info(
            chunks, document)

        return enriched_chunks

    def chunk_documents(
        self, documents: List[Document], by_source: bool = True
    ) -> List[Document]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of documents to split
            by_source: Whether to process documents with the same source together

        Returns:
            List of document chunks
        """
        if not by_source:
            all_chunks = []
            for document in documents:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            return all_chunks

        # Group documents by source
        source_docs = {}
        for document in documents:
            source = (
                document.metadata.get("source", "unknown")
                if document.metadata
                else "unknown"
            )
            if source not in source_docs:
                source_docs[source] = []
            source_docs[source].append(document)

        # Chunk documents by source
        all_chunks = []
        for source, docs in source_docs.items():
            for document in docs:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)

        return all_chunks
