"""
User management module for Zed-KB.
Handles user profiles, authentication state, and integrates with Permit.io.
"""

from typing import Dict, Any, Optional, List, Union
import os
import logging
import json
from functools import lru_cache

from .permit_client import get_permit_client, PermitClient

logger = logging.getLogger(__name__)


class UserManager:
    """Manages user profiles and permissions using Permit.io."""

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

    def get_user_info(
        self,
        user_id: str,
        tenant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive user information including roles and permissions.

        Args:
            user_id: User identifier
            tenant: Optional tenant identifier

        Returns:
            Dictionary containing user information
        """
        tenant = tenant or self.default_tenant

        # Get user roles
        roles = self.permit.get_user_roles(user_id, tenant)

        # Get user permissions
        permissions = self.permit.get_user_permissions(user_id, tenant)

        # Determine clearance level based on roles
        clearance_level = "public"  # Default lowest level
        if "admin" in roles:
            clearance_level = "confidential"  # Highest level
        elif "user" in roles:
            clearance_level = "internal"  # Mid level

        # Build user info object
        user_info = {
            "user_id": user_id,
            "roles": roles,
            "permissions": permissions,
            "clearance": clearance_level,
        }

        if tenant:
            user_info["tenant"] = tenant

        return user_info

    def check_document_access(
        self,
        user_id: str,
        document: Dict[str, Any],
        action: str = "read",
        tenant: Optional[str] = None,
    ) -> bool:
        """
        Check if a user has access to a specific document.

        Args:
            user_id: User identifier
            document: Document dictionary with metadata
            action: The action to check (read, write, delete, etc.)
            tenant: Optional tenant identifier

        Returns:
            True if access is granted, False otherwise
        """
        tenant = tenant or self.default_tenant

        # Convert document to a resource representation
        doc_id = document.get("doc_id") or document.get(
            "document_id", str(id(document)))

        # Create resource attributes from document metadata
        resource = {
            "id": doc_id,
            "security_level": document.get("security_level", "public"),
            "type": "document",
        }

        # Add allowed roles if available
        if "allowed_roles" in document:
            resource["allowed_roles"] = document["allowed_roles"]

        # Add allowed users if available
        if "allowed_users" in document:
            resource["allowed_users"] = document["allowed_users"]

        # Call permit check
        return self.permit.check(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_type="document",
            tenant=tenant,
        )

    def filter_documents(
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
        # Get user roles for this tenant
        roles = self.permit.get_user_roles(user_id, tenant)

        # Get user's clearance level
        clearance_level = "public"  # Default
        if "admin" in roles:
            clearance_level = "confidential"
        elif "user" in roles:
            clearance_level = "internal"

        # Map clearance levels to numeric values for comparison
        clearance_levels = {
            "public": 0,
            "internal": 1,
            "confidential": 2
        }
        user_level = clearance_levels.get(clearance_level, 0)

        # Filter documents
        filtered_docs = []
        for doc in documents:
            # Check security level
            doc_level = doc.get("security_level", "public")
            doc_level_value = clearance_levels.get(doc_level, 0)

            # Basic security level check
            if doc_level_value > user_level:
                continue

            # Check if document specifies allowed roles
            allowed_roles = doc.get("allowed_roles", [])
            if allowed_roles and not set(roles).intersection(set(allowed_roles)):
                continue

            # Check if document specifies allowed users
            allowed_users = doc.get("allowed_users", [])
            if allowed_users and user_id not in allowed_users and "*" not in allowed_users:
                continue

            # Document passed all checks - add to filtered list
            filtered_docs.append(doc)

        return filtered_docs

    def get_vector_store_filter(
        self,
        user_id: str,
        action: str = "read",
        tenant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a filter dictionary for AstraDB vector store queries.

        Args:
            user_id: User identifier
            action: The action to check (default: read)
            tenant: Optional tenant identifier

        Returns:
            Filter dictionary for AstraDB vector store
        """
        return self.permit.get_secured_filter(
            user_id=user_id,
            tenant=tenant,
            action=action,
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
