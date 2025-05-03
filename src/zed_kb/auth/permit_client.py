"""
Permit.io client for authorization in Zed-KB.
Handles integration with Permit.io's RBAC and ABAC systems.
"""

import os
from typing import Dict, Any, Optional, List, Union
import logging
from functools import lru_cache
from datetime import datetime, timedelta

# Updated imports to match the current Permit API
from permit import Permit, User, Resource
from permit.enforcement import EnforcementContext
# Note: PdpOptions doesn't exist in the current permit library version,
# we'll use RemoteConfig instead for configuration

logger = logging.getLogger(__name__)

# Default Permit.io context key for user info
USER_CONTEXT_KEY = "user"

# Default resource types
DOCUMENT_RESOURCE = "document"
COLLECTION_RESOURCE = "collection"


class PermitClient:
    """Client for Permit.io authorization service."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        pdp_address: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the Permit client with API key.

        Args:
            api_key: Permit.io API key (defaults to PERMIT_IO_API_KEY env var)
            pdp_address: Custom PDP address (if using a self-hosted PDP)
            debug: Enable debug logging
        """
        self.api_key = api_key or os.getenv("PERMIT_IO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Permit.io API key is required. Either pass it to the constructor "
                "or set the PERMIT_IO_API_KEY environment variable."
            )

        # Initialize Permit.io client with updated configuration
        config_options = {}
        if pdp_address:
            config_options["pdp"] = {"address": pdp_address}
        
        self.permit = Permit(
            token=self.api_key,
            config=config_options,
        )

    def initialize(self) -> bool:
        """
        Initialize the Permit.io client connection.

        Returns:
            True if successful, False otherwise
        """
        try:
            # In newer versions, sync is automatically called on init
            # but we'll keep this for explicitness
            self.permit.api.sync()
            logger.info("Permit.io client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Permit.io client: {e}")
            return False

    def check(
        self,
        user_id: str,
        action: str,
        resource: Union[str, Dict[str, Any]],
        resource_type: str = DOCUMENT_RESOURCE,
        tenant: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if a user has permission to perform an action on a resource.

        Args:
            user_id: Unique identifier of the user
            action: Action the user wants to perform (read, write, delete, etc.)
            resource: Resource identifier or resource object
            resource_type: Type of resource (document, collection, etc.)
            tenant: Optional tenant identifier for multi-tenant applications
            context: Optional context data for attribute-based access control

        Returns:
            True if permission is granted, False otherwise
        """
        try:
            # Create user object
            user = User(key=user_id)

            # Handle resource as string ID or object
            if isinstance(resource, dict):
                resource_key = resource.get(
                    "id", resource.get("key", str(resource)))
                resource_obj = Resource(
                    type=resource_type,
                    key=resource_key,
                    attributes=resource,
                )
            else:
                resource_obj = Resource(
                    type=resource_type,
                    key=str(resource),
                )

            # Set up enforcement context
            enforcement_context = None
            if context:
                enforcement_context = EnforcementContext(context=context)

            # Run permission check
            permitted = self.permit.check(
                user=user,
                action=action,
                resource=resource_obj,
                tenant=tenant,
                context=enforcement_context
            )

            logger.debug(f"Permission check for user {user_id}, action {action}, "
                         f"resource {resource}: {'GRANTED' if permitted else 'DENIED'}")

            return permitted

        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            # Default deny on error
            return False

    def bulk_check(
        self,
        user_id: str,
        action: str,
        resources: List[Dict[str, Any]],
        resource_type: str = DOCUMENT_RESOURCE,
        tenant: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """
        Check permissions for multiple resources at once.

        Args:
            user_id: Unique identifier of the user
            action: Action the user wants to perform
            resources: List of resource objects
            resource_type: Type of resources
            tenant: Optional tenant identifier 
            context: Optional context data

        Returns:
            Dictionary mapping resource IDs to permission results
        """
        results = {}

        for resource in resources:
            resource_id = resource.get(
                "id", resource.get("key", str(resource)))
            permitted = self.check(
                user_id=user_id,
                action=action,
                resource=resource,
                resource_type=resource_type,
                tenant=tenant,
                context=context
            )
            results[str(resource_id)] = permitted

        return results

    def get_user_roles(
        self,
        user_id: str,
        tenant: Optional[str] = None
    ) -> List[str]:
        """
        Get roles assigned to a user.

        Args:
            user_id: Unique identifier of the user
            tenant: Optional tenant identifier

        Returns:
            List of role keys assigned to the user
        """
        try:
            # Get user roles from Permit.io
            user = User(key=user_id)
            roles = self.permit.get_user_roles(user, tenant)

            # Extract role keys
            role_keys = [role.key for role in roles] if roles else []
            logger.debug(f"User {user_id} has roles: {', '.join(role_keys)}")

            return role_keys
        except Exception as e:
            logger.error(f"Error getting user roles: {e}")
            return []

    def get_user_permissions(
        self,
        user_id: str,
        tenant: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get all permissions assigned to a user.

        Args:
            user_id: Unique identifier of the user
            tenant: Optional tenant identifier

        Returns:
            List of permission objects with action and resource type
        """
        try:
            # Get user permissions from Permit.io
            user = User(key=user_id)
            permissions = self.permit.get_user_permissions(user, tenant)

            # Format permissions as dictionaries
            formatted_permissions = [
                {"action": perm.action, "resource": perm.resource}
                for perm in permissions
            ]

            return formatted_permissions
        except Exception as e:
            logger.error(f"Error getting user permissions: {e}")
            return []

    def get_secured_filter(
        self,
        user_id: str,
        tenant: Optional[str] = None,
        action: str = "read",
        resource_type: str = DOCUMENT_RESOURCE,
    ) -> Dict[str, Any]:
        """
        Generate a metadata filter for vector store queries based on user permissions.

        Args:
            user_id: Unique identifier of the user
            tenant: Optional tenant identifier
            action: Action to filter by (default: read)
            resource_type: Resource type (default: document)

        Returns:
            Dictionary containing metadata filter for security levels and allowed_roles
        """
        # Get user roles and convert security clearance to level
        roles = self.get_user_roles(user_id, tenant)

        # Create filter dictionary based on user roles
        filter_dict = {}

        # Always include allowed_roles filter if roles exist
        if roles:
            filter_dict["allowed_roles"] = {
                "$in": roles
            }

        # Add user-specific conditions
        filter_dict["allowed_users"] = {
            # Match documents explicitly allowing this user or all users
            "$in": [user_id, "*"]
        }

        # Determine security level clearance based on roles
        # This is a simplified approach; in a real system, you might want to query
        # Permit.io for more sophisticated attribute-based decisions
        clearance_level = "public"  # Default lowest level
        if "admin" in roles:
            clearance_level = "confidential"  # Highest level
        elif "user" in roles:
            clearance_level = "internal"  # Mid level

        # Map clearance level to numeric values
        clearance_levels = {
            "public": 0,
            "internal": 1,
            "confidential": 2
        }

        user_clearance_level = clearance_levels.get(clearance_level, 0)

        # Get all security levels the user has access to
        eligible_levels = [
            level for level, value in clearance_levels.items()
            if value <= user_clearance_level
        ]

        # Add security level filter
        if eligible_levels:
            filter_dict["security_level"] = {
                "$in": eligible_levels
            }

        return filter_dict


# Create singleton instance
@lru_cache(maxsize=1)
def get_permit_client(
    api_key: Optional[str] = None,
    pdp_address: Optional[str] = None,
    debug: bool = False
) -> PermitClient:
    """
    Get or create a singleton instance of PermitClient.

    Args:
        api_key: Permit.io API key (optional)
        pdp_address: Custom PDP address (optional)
        debug: Enable debug logging (default: False)

    Returns:
        PermitClient instance
    """
    client = PermitClient(
        api_key=api_key, pdp_address=pdp_address, debug=debug)
    client.initialize()
    return client
