"""
Permit.io client for authorization in Zed-KB.
Simplified implementation based on direct Permit.io API usage.
"""

import os
from typing import Dict, Any, Optional, List, Union
import logging
from functools import lru_cache

from permit import Permit, User, Resource

logger = logging.getLogger(__name__)

# Default resource types
DOCUMENT_RESOURCE = "document"
COLLECTION_RESOURCE = "collection"


class PermitClient:
    """Simplified client for Permit.io authorization service."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        pdp_url: str = "https://cloudpdp.api.permit.io",
        debug: bool = False,
    ):
        """
        Initialize the Permit client with API key.

        Args:
            api_key: Permit.io API key (defaults to PERMIT_IO_API_KEY env var)
            pdp_url: URL for the Permit.io PDP service
            debug: Enable debug logging
        """
        self.api_key = api_key or os.getenv("PERMIT_IO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Permit.io API key is required. Either pass it to the constructor "
                "or set the PERMIT_IO_API_KEY environment variable."
            )

        # Initialize Permit.io client
        self.permit = Permit(
            pdp=pdp_url,
            token=self.api_key,
        )

        if debug:
            logging.getLogger("permit").setLevel(logging.DEBUG)

    async def check_permission(
        self,
        user_id: str,
        action: str,
        resource: Union[str, Dict[str, Any]],
        tenant: Optional[str] = None,
    ) -> bool:
        """
        Check if a user has permission to perform an action on a resource.

        Args:
            user_id: Unique identifier of the user
            action: Action the user wants to perform (read, write, delete, etc.)
            resource: Resource identifier or resource object
            tenant: Optional tenant identifier for multi-tenant applications

        Returns:
            True if permission is granted, False otherwise
        """
        try:
            # Handle resource as string ID or object
            resource_key = resource
            if isinstance(resource, dict):
                resource_key = resource.get(
                    "id", resource.get("key", str(resource)))

            # Run permission check using direct permit.check() method
            permitted = await self.permit.check(
                user_id,
                action,
                resource_key,
                tenant=tenant
            )

            logger.debug(f"Permission check for user {user_id}, action {action}, "
                         f"resource {resource_key}: {'GRANTED' if permitted else 'DENIED'}")

            return permitted

        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            # Default deny on error
            return False

    async def bulk_check_permissions(
        self,
        user_id: str,
        action: str,
        resources: List[Union[str, Dict[str, Any]]],
        tenant: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Check permissions for multiple resources at once.

        Args:
            user_id: Unique identifier of the user
            action: Action the user wants to perform
            resources: List of resource IDs or objects
            tenant: Optional tenant identifier 

        Returns:
            Dictionary mapping resource IDs to permission results
        """
        results = {}

        for resource in resources:
            resource_id = resource
            if isinstance(resource, dict):
                resource_id = resource.get(
                    "id", resource.get("key", str(resource)))

            permitted = await self.check_permission(
                user_id=user_id,
                action=action,
                resource=resource_id,
                tenant=tenant
            )
            results[str(resource_id)] = permitted

        return results

    async def sync_user(
        self,
        user: Dict[str, Any],
        tenant: Optional[str] = None
    ) -> bool:
        """
        Sync user data with Permit.io.

        Args:
            user: User dictionary containing at least 'id' key
            tenant: Optional tenant identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure user has required fields
            if 'id' not in user:
                logger.error("User dictionary must contain 'id' key")
                return False

            # Sync user
            await self.permit.api.sync_user(user)

            logger.info(f"User {user['id']} synced with Permit.io")
            return True

        except Exception as e:
            logger.error(f"Error syncing user: {e}")
            return False

    def get_resource_filter(
        self,
        user_id: str,
        action: str = "read",
    ) -> Dict[str, Any]:
        """
        Generate a simplified metadata filter for vector store queries based on user permissions.

        Args:
            user_id: Unique identifier of the user
            action: Action to filter by (default: read)

        Returns:
            Dictionary containing basic metadata filter
        """
        # Create a simple filter dictionary
        filter_dict = {
            "allowed_users": {
                # Match documents allowing this user or all users
                "$in": [user_id, "*"]
            }
        }

        return filter_dict


# Create singleton instance
@lru_cache(maxsize=1)
def get_permit_client(
    api_key: Optional[str] = None,
    pdp_url: str = "https://cloudpdp.api.permit.io",
    debug: bool = False
) -> PermitClient:
    """
    Get or create a singleton instance of PermitClient.

    Args:
        api_key: Permit.io API key (optional)
        pdp_url: URL for the Permit.io PDP service
        debug: Enable debug logging (default: False)

    Returns:
        PermitClient instance
    """
    client = PermitClient(api_key=api_key, pdp_url=pdp_url, debug=debug)
    return client
