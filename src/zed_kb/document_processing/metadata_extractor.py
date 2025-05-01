"""
Metadata extractor module for Zed-KB.
Enhances document metadata by extracting useful information.
"""

from typing import List, Dict, Any, Optional
import re
import datetime
from pathlib import Path
import os

from langchain.schema import Document


class MetadataExtractor:
    """Extracts and enhances metadata from documents."""

    def __init__(self):
        """Initialize the metadata extractor."""
        pass

    def extract_metadata(self, document: Document) -> Document:
        """
        Extract metadata from a document and attach it to the document.

        Args:
            document: Document to extract metadata from

        Returns:
            Document with enhanced metadata
        """
        # Create a copy of the document to avoid modifying the original
        enhanced_doc = Document(
            page_content=document.page_content,
            metadata=document.metadata.copy() if document.metadata else {},
        )

        # Add timestamp metadata
        enhanced_doc.metadata["processed_at"] = datetime.datetime.now().isoformat()

        # Extract content-based metadata
        content_metadata = self._extract_content_metadata(enhanced_doc.page_content)
        enhanced_doc.metadata.update(content_metadata)

        # Extract path-based metadata
        if "source" in enhanced_doc.metadata:
            path_metadata = self._extract_path_metadata(enhanced_doc.metadata["source"])
            enhanced_doc.metadata.update(path_metadata)

        return enhanced_doc

    def extract_batch_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Extract metadata from a batch of documents.

        Args:
            documents: List of documents to extract metadata from

        Returns:
            List of documents with enhanced metadata
        """
        return [self.extract_metadata(doc) for doc in documents]

    def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from document content.

        Args:
            content: Document content

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        # Extract content type/category through simple heuristics
        if re.search(r"(contract|agreement|terms|conditions)", content, re.IGNORECASE):
            metadata["content_type"] = "legal"
            metadata["sensitivity"] = "high"

        elif re.search(r"(confidential|proprietary|sensitive)", content, re.IGNORECASE):
            metadata["sensitivity"] = "high"

        elif re.search(r"(financial|revenue|profit|loss)", content, re.IGNORECASE):
            metadata["content_type"] = "financial"
            metadata["sensitivity"] = "high"

        elif re.search(r"(report|analysis|study|research)", content, re.IGNORECASE):
            metadata["content_type"] = "report"

        elif re.search(r"(manual|guide|instruction|how\sto)", content, re.IGNORECASE):
            metadata["content_type"] = "manual"
            metadata["sensitivity"] = "low"

        # Extract dates - look for common date formats
        date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        dates = re.findall(date_pattern, content)
        if dates:
            metadata["extracted_dates"] = dates[:5]  # Take only the first 5 dates

        return metadata

    def _extract_path_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from file path.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        path = Path(file_path)

        # File metadata
        metadata["file_name"] = path.name
        metadata["file_extension"] = path.suffix.lower()
        metadata["file_size"] = (
            os.path.getsize(file_path) if os.path.exists(file_path) else None
        )

        # Last modified time
        if os.path.exists(file_path):
            metadata["last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat()

        # Directory structure metadata
        parts = path.parts
        if len(parts) >= 2:
            # Extract department or category from directory structure
            metadata["directory"] = parts[-2]

            # Try to infer department from path
            lower_path = file_path.lower()
            if "hr" in lower_path or "human resources" in lower_path:
                metadata["department"] = "HR"
            elif "finance" in lower_path:
                metadata["department"] = "Finance"
            elif "legal" in lower_path:
                metadata["department"] = "Legal"
            elif "sales" in lower_path:
                metadata["department"] = "Sales"
            elif "marketing" in lower_path:
                metadata["department"] = "Marketing"
            elif "engineering" in lower_path or "dev" in lower_path:
                metadata["department"] = "Engineering"
            elif "product" in lower_path:
                metadata["department"] = "Product"
            elif "support" in lower_path or "customer" in lower_path:
                metadata["department"] = "Customer Support"

        return metadata

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
