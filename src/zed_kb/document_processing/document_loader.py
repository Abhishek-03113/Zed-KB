"""
Document loader module for Zed-KB.
Handles loading of various document formats including PDFs, Word documents,
HTML, and text files.
"""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    NotionDirectoryLoader,
    ConfluenceLoader,
)
from langchain.schema import Document


class DocumentLoader:
    """Handles the loading of various document formats into a unified format."""

    SUPPORTED_EXTENSIONS = {
        # Document formats
        ".pdf": PyMuPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".txt": TextLoader,
        # Add more formats as needed
    }

    def __init__(self):
        """Initialize the document loader."""
        pass

    def load_file(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load a single file from the given path.

        Args:
            file_path: Path to the file to load
            metadata: Optional metadata to add to the document

        Returns:
            List of Document objects
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {file_extension}")

        loader_class = self.SUPPORTED_EXTENSIONS[file_extension]
        loader = loader_class(file_path)

        # Load documents
        documents = loader.load()

        # Add additional metadata if provided
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        return documents

    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to the directory containing documents
            recursive: Whether to search recursively
            metadata: Optional metadata to add to all documents

        Returns:
            List of Document objects
        """
        documents = []
        base_path = Path(directory_path)

        # Get all files with supported extensions
        glob_patterns = [
            f"**/*{ext}" if recursive else f"*{ext}"
            for ext in self.SUPPORTED_EXTENSIONS.keys()
        ]

        for pattern in glob_patterns:
            for file_path in base_path.glob(pattern):
                try:
                    # Extract department info from the file path
                    relative_path = file_path.relative_to(base_path)
                    file_metadata = {"source": str(file_path)}

                    # Add the directory structure to metadata to assist with access control
                    parts = relative_path.parts
                    if len(parts) > 1:  # If file is in a subdirectory
                        file_metadata["department"] = parts[0]

                    # Combine with user-provided metadata
                    if metadata:
                        file_metadata.update(metadata)

                    # Load the file and add it to our documents
                    docs = self.load_file(str(file_path), file_metadata)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents
    
    # following are future additions for loading Notion and Confluence content

    def load_notion(
        self, directory_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load Notion content exported as Markdown.

        Args:
            directory_path: Path to the directory containing Notion export
            metadata: Optional metadata to add to all documents

        Returns:
            List of Document objects
        """
        loader = NotionDirectoryLoader(directory_path)
        documents = loader.load()

        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        return documents

    def load_confluence(
        self,
        url: str,
        username: str,
        api_key: str,
        space_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load content from Confluence.

        Args:
            url: Confluence URL
            username: Confluence username
            api_key: Confluence API key
            space_key: Optional space key to limit loading
            metadata: Optional metadata to add to all documents

        Returns:
            List of Document objects
        """
        loader = ConfluenceLoader(
            url=url, username=username, api_key=api_key, space_key=space_key
        )

        documents = loader.load()

        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        return documents
