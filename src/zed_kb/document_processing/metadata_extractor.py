"""
Extracts metadata for Zed-KB documents.
"""
from typing import List, Dict, Any, Optional
from langchain.schema import Document


class MetadataExtractor:
    """Extracts and assigns metadata."""

    def __init__(self):
        """Initialize the metadata extractor."""
        pass

    def extract_metadata(self, document: Document) -> Document:
        """
        Apply uniform metadata schema to a document.
        Does not extract information from content or filesystem.

        Args:
            document: Document to apply metadata schema to

        Returns:
            Document with uniform metadata schema
        """
        # Create a copy of the document to avoid modifying the original
        enhanced_doc = Document(
            page_content=document.page_content,
            metadata=document.metadata.copy() if document.metadata else {},
        )

        # Apply uniform metadata schema with default values
        # Security metadata
        if "security_level" not in enhanced_doc.metadata:
            enhanced_doc.metadata["security_level"] = "internal"

        if "allowed_roles" not in enhanced_doc.metadata:
            enhanced_doc.metadata["allowed_roles"] = []

        if "allowed_users" not in enhanced_doc.metadata:
            enhanced_doc.metadata["allowed_users"] = []

        # Document identifier metadata (preserve existing or initialize)
        if "doc_id" not in enhanced_doc.metadata and "source" in enhanced_doc.metadata:
            enhanced_doc.metadata["doc_id"] = enhanced_doc.metadata["source"]

        return enhanced_doc

    def extract_batch_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Apply uniform metadata schema to a batch of documents.

        Args:
            documents: List of documents to apply metadata schema to

        Returns:
            List of documents with uniform metadata schema
        """
        return [self.extract_metadata(doc) for doc in documents]

    def add_security_metadata(
        self,
        document: Document,
        security_level: str,
        allowed_roles: List[str] = None,
        allowed_users: List[str] = None,
    ) -> Document:
        """
        Add security and access control metadata to a document.

        Args:
            document: Document to add metadata to
            security_level: Security level (public, internal, confidential, restricted)
            allowed_roles: List of roles that can access the document
            allowed_users: List of specific users that can access the document

        Returns:
            Document with security metadata
        """
        # Create a copy of the document to avoid modifying the original
        secure_doc = Document(
            page_content=document.page_content,
            metadata=document.metadata.copy() if document.metadata else {},
        )

        # Add security metadata
        secure_doc.metadata["security_level"] = security_level

        if allowed_roles:
            secure_doc.metadata["allowed_roles"] = allowed_roles

        if allowed_users:
            secure_doc.metadata["allowed_users"] = allowed_users

        return secure_doc

    def add_batch_security_metadata(
        self,
        documents: List[Document],
        security_level: str,
        allowed_roles: List[str] = None,
        allowed_users: List[str] = None,
    ) -> List[Document]:
        """
        Add security metadata to a batch of documents.

        Args:
            documents: List of documents to add metadata to
            security_level: Security level
            allowed_roles: List of roles that can access the documents
            allowed_users: List of specific users that can access the documents

        Returns:
            List of documents with security metadata
        """
        return [
            self.add_security_metadata(
                doc, security_level, allowed_roles, allowed_users
            )
            for doc in documents
        ]

    def add_document_metadata(
        self,
        document: Document,
        metadata: Dict[str, Any]
    ) -> Document:
        """
        Add plain document metadata to a document.

        Args:
            document: Document to add metadata to
            metadata: Dictionary of metadata to add

        Returns:
            Document with added metadata
        """
        # Create a copy of the document to avoid modifying the original
        enhanced_doc = Document(
            page_content=document.page_content,
            metadata=document.metadata.copy() if document.metadata else {},
        )

        # Add the metadata
        for key, value in metadata.items():
            enhanced_doc.metadata[key] = value

        return enhanced_doc

    def add_batch_document_metadata(
        self,
        documents: List[Document],
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Add plain document metadata to a batch of documents.

        Args:
            documents: List of documents to add metadata to
            metadata: Dictionary of metadata to add

        Returns:
            List of documents with added metadata
        """
        return [
            self.add_document_metadata(doc, metadata)
            for doc in documents
        ]
