"""
Simple authorization utilities for Zed-KB.
Provides straightforward functions for checking permissions using Permit.io.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import logging
import functools
import inspect

from fastapi import HTTPException, Request, status, Depends
from pydantic import BaseModel

from .permit_client import get_permit_client, PermitClient
from .user_manager import get_user_manager, UserManager

logger = logging.getLogger(__name__)


class UserInfo(BaseModel):
    """User information model."""
    id: str
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    email: Optional[str] = None
    tenant: Optional[str] = None


async def check_permission(
    user_id: str,
    action: str,
    resource: str,
    resource_type: Optional[str] = None,
    tenant: Optional[str] = None,
    permit_client: Optional[PermitClient] = None,
) -> bool:
    """
    Simple function to check if a user has permission for an action.
    
    Args:
        user_id: User identifier
        action: Action to check (e.g., "read", "write", "delete")
        resource: Resource identifier
        resource_type: Optional resource type
        tenant: Optional tenant identifier
        permit_client: Optional PermitClient instance
        
    Returns:
        True if permission granted, False otherwise
    """
    client = permit_client or get_permit_client()
    return await client.check_permission(
        user_id=user_id, 
        action=action,
        resource=resource,
        tenant=tenant
    )


async def check_document_access(
    user_id: str,
    document_id: str,
    action: str = "read",
    tenant: Optional[str] = None,
) -> bool:
    """
    Check if a user has permission to access a document.
    
    Args:
        user_id: User identifier
        document_id: Document identifier
        action: Action to check (default: "read")
        tenant: Optional tenant identifier
        
    Returns:
        True if permission granted, False otherwise
    """
    user_manager = get_user_manager()
    return await user_manager.check_document_access(
        user_id=user_id,
        document_id=document_id,
        action=action,
        tenant=tenant
    )


async def sync_user(user: Dict[str, Any], tenant: Optional[str] = None) -> bool:
    """
    Sync a user with Permit.io.
    
    Args:
        user: User information dictionary (must include 'id')
        tenant: Optional tenant identifier
        
    Returns:
        True if successful, False otherwise
    """
    client = get_permit_client()
    return await client.sync_user(user, tenant)


# FastAPI specific utilities for easy integration

def get_user_from_request(request: Request) -> UserInfo:
    """
    Extract user info from a FastAPI request.
    
    This is a placeholder implementation. In a real application, you would
    extract the user information from authentication headers, tokens, etc.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        UserInfo object
    """
    # This is a placeholder - in a real app you would extract from JWT, headers, etc.
    # For demo purposes only - DO NOT USE IN PRODUCTION
    
    # Extract from header if available
    user_id = request.headers.get("X-User-ID", "default_user")
    first_name = request.headers.get("X-User-FirstName", "Default")
    last_name = request.headers.get("X-User-LastName", "User")
    email = request.headers.get("X-User-Email", "default@example.com")
    tenant = request.headers.get("X-Tenant-ID")
    
    return UserInfo(
        id=user_id,
        firstName=first_name,
        lastName=last_name,
        email=email,
        tenant=tenant
    )


def require_permission(
    action: str,
    resource: Union[str, Callable],
    resource_type: Optional[str] = None,
):
    """
    FastAPI dependency for requiring permission.
    
    Usage example:
    ```
    @app.get("/docs/{doc_id}")
    async def get_document(
        doc_id: str,
        _: None = Depends(require_permission("read", lambda req, doc_id: doc_id))
    ):
        # Function will only execute if permission is granted
        return {"doc_id": doc_id, "content": "Document content"}
    ```
    
    Args:
        action: Action to check (e.g., "read", "write")
        resource: Resource ID or callable that returns resource ID
        resource_type: Optional resource type
        
    Returns:
        FastAPI dependency that checks permission
    """
    async def dependency(request: Request, **path_params):
        # Get user info from request
        user_info = get_user_from_request(request)
        
        # Determine resource ID
        resource_id = resource
        if callable(resource):
            resource_id = resource(request, **path_params)
            
        # Check permission
        client = get_permit_client()
        permitted = await client.check_permission(
            user_id=user_info.id,
            action=action,
            resource=resource_id,
            tenant=user_info.tenant,
        )
        
        if not permitted:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "result": f"User {user_info.id} is NOT PERMITTED to {action} {resource_id}"
                }
            )
            
        # Return user info if permission is granted
        return user_info
        
    return dependency


def with_permission_check(
    action: str,
    get_resource_id: Callable,
):
    """
    Decorator for checking permission in any async function.
    
    Usage example:
    ```
    @with_permission_check("read", lambda user_id, doc_id, **kwargs: doc_id)
    async def read_document(user_id: str, doc_id: str):
        # This function only executes if permission is granted
        return load_document(doc_id)
    ```
    
    Args:
        action: Action to check (e.g., "read", "write")
        get_resource_id: Function to extract resource ID from function parameters
        
    Returns:
        Decorated function that checks permission before execution
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract parameters from the function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Find user_id in parameters
            user_id = None
            if 'user_id' in bound_args.arguments:
                user_id = bound_args.arguments['user_id']
            elif 'user' in bound_args.arguments and hasattr(bound_args.arguments['user'], 'id'):
                user_id = bound_args.arguments['user'].id
                
            if not user_id:
                raise ValueError("No user_id found in function parameters")
                
            # Get tenant if available
            tenant = None
            if 'tenant' in bound_args.arguments:
                tenant = bound_args.arguments['tenant']
                
            # Get resource ID
            resource_id = get_resource_id(**bound_args.arguments)
            
            # Check permission
            permit_client = get_permit_client()
            permitted = await permit_client.check_permission(
                user_id=user_id,
                action=action,
                resource=resource_id,
                tenant=tenant
            )
            
            if not permitted:
                logger.warning(f"Permission denied: user {user_id} cannot {action} resource {resource_id}")
                raise PermissionError(f"Permission denied: {action} resource {resource_id}")
                
            # Permission granted, execute function
            return await func(*args, **kwargs)
            
        return wrapper
        
    return decorator