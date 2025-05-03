"""
User management module for Zed-KB.
Simplified implementation that handles user profiles and integrates with Permit.io.
"""

from typing import Dict, Any, Optional, List, Union
import os
import logging
import json
from functools import lru_cache

from .permit_client import get_permit_client, PermitClient

logger = logging.getLogger(__name__)


class UserManager:
    """
    Manages user profiles and document access permissions using Permit.io.
    Simplified implementation based on direct Permit.io API usage.
    """

    def __init__(
        self,
        permit_client: Optional[PermitClient] = None,
        default_tenant: Optional[str] = None,
    ):
        """
        Initialize the user manager.

        Args:
            permit_client: PermitClient instance (created if not provided)
            default_tenant: Default tenant ID for multi-tenant setups
        """
        self.permit = permit_client or get_permit_client()
        self.default_tenant = default_tenant

    async def check_document_access(
        self,
        user_id: str,
        document_id: str,
        action: str = "read",
        tenant: Optional[str] = None,
    ) -> bool:
        """
        Check if a user has access to a specific document.

        Args:
            user_id: User identifier
            document_id: Document identifier
            action: The action to check (read, write, delete, etc.)
            tenant: Optional tenant identifier

        Returns:
            True if access is granted, False otherwise
        """
        tenant = tenant or self.default_tenant
        
        # Call permit check directly
        return await self.permit.check_permission(
            user_id=user_id,
            action=action,
            resource=document_id,
            tenant=tenant,
        )

    async def check_collection_access(
        self,
        user_id: str,
        collection_id: str,
        action: str = "read",
        tenant: Optional[str] = None,
    ) -> bool:
        """
        Check if a user has access to a specific collection.

        Args:
            user_id: User identifier
            collection_id: Collection identifier
            action: The action to check (read, write, delete, etc.)
            tenant: Optional tenant identifier

        Returns:
            True if access is granted, False otherwise
        """
        tenant = tenant or self.default_tenant
        
        # Call permit check directly
        return await self.permit.check_permission(
            user_id=user_id,
            action=action,
            resource=collection_id,
            tenant=tenant,
        )

    async def filter_documents(
        self,
        user_id: str,
        documents: List[Dict[str, Any]],
        action: str = "read",
        tenant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of documents based on user permissions.

        Args:
            user_id: User identifier
            documents: List of document dictionaries
            action: The action to check
            tenant: Optional tenant identifier

        Returns:
            Filtered list of documents
        """
        tenant = tenant or self.default_tenant
        
        # Simple approach: Check each document individually
        filtered_docs = []
        
        for doc in documents:
            doc_id = doc.get("doc_id") or doc.get("document_id", str(id(doc)))
            
            # Check permission for this document
            permitted = await self.permit.check_permission(
                user_id=user_id,
                action=action,
                resource=doc_id,
                tenant=tenant,
            )
            
            if permitted:
                filtered_docs.append(doc)
                
        return filtered_docs

    def get_vector_store_filter(
        self,
        user_id: str,
        action: str = "read",
        tenant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a simplified filter dictionary for vector store queries.

        Args:
            user_id: User identifier
            action: The action to check (default: read)
            tenant: Optional tenant identifier

        Returns:
            Filter dictionary for vector store
        """
        return self.permit.get_resource_filter(
            user_id=user_id,
            action=action,
        )

    async def sync_user(
        self,
        user: Dict[str, Any],
        tenant: Optional[str] = None,
    ) -> bool:
        """
        Sync user data with Permit.io.

        Args:
            user: User dictionary containing user information
            tenant: Optional tenant identifier

        Returns:
            True if successful, False otherwise
        """
        return await self.permit.sync_user(
            user=user,
            tenant=tenant,
        )


# Singleton instance
@lru_cache(maxsize=1)
def get_user_manager(
    permit_client: Optional[PermitClient] = None,
    default_tenant: Optional[str] = None,
) -> UserManager:
    """
    Get or create a singleton instance of UserManager.

    Args:
        permit_client: Optional PermitClient instance
        default_tenant: Optional default tenant ID

    Returns:
        UserManager instance
    """
    return UserManager(permit_client=permit_client, default_tenant=default_tenant)
