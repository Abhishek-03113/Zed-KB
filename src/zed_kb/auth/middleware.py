"""
Authorization middleware for Zed-KB.
Provides decorators and middleware for enforcing permissions.
"""

from typing import Dict, Any, Optional, List, Union, Callable, Type, TypeVar, cast
import logging
import functools
import inspect
from dataclasses import dataclass, field

from .user_manager import get_user_manager, UserManager
from .permit_client import get_permit_client, PermitClient

logger = logging.getLogger(__name__)

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


@dataclass
class AuthContext:
    """Authorization context for request processing."""
    user_id: str
    tenant: Optional[str] = None
    user_info: Dict[str, Any] = field(default_factory=dict)
    permit_client: Optional[PermitClient] = None
    user_manager: Optional[UserManager] = None
    
    def __post_init__(self):
        # Initialize user info if not provided
        if not self.user_info and self.user_id:
            # Get user manager
            self.user_manager = self.user_manager or get_user_manager()
            
            # Get user information
            self.user_info = self.user_manager.get_user_info(
                self.user_id, self.tenant
            )


def require_permission(
    action: str,
    resource_type: str,
    get_resource_id: Optional[Callable] = None,
    get_user_id: Optional[Callable] = None,
    get_tenant: Optional[Callable] = None,
) -> Callable[[F], F]:
    """
    Decorator to enforce permission checks on functions.
    
    Args:
        action: The action to check
        resource_type: The resource type to check
        get_resource_id: Optional function to extract resource ID from args/kwargs
        get_user_id: Optional function to extract user ID from args/kwargs
        get_tenant: Optional function to extract tenant ID from args/kwargs
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get permit client
            permit_client = get_permit_client()
            
            # Extract user ID
            user_id = None
            if get_user_id:
                user_id = get_user_id(*args, **kwargs)
            elif 'user_id' in kwargs:
                user_id = kwargs['user_id']
            elif 'user_info' in kwargs and isinstance(kwargs['user_info'], dict):
                user_id = kwargs['user_info'].get('user_id')
                
            if not user_id:
                logger.error(f"Unable to determine user ID for permission check in {func.__name__}")
                raise ValueError("User ID is required for permission check")
                
            # Extract resource ID
            resource_id = None
            if get_resource_id:
                resource_id = get_resource_id(*args, **kwargs)
            elif 'resource_id' in kwargs:
                resource_id = kwargs['resource_id']
            elif 'doc_id' in kwargs:
                resource_id = kwargs['doc_id']
                
            # No specific resource ID, use the resource type as a general check
            if not resource_id:
                resource_id = resource_type
                
            # Extract tenant
            tenant = None
            if get_tenant:
                tenant = get_tenant(*args, **kwargs)
            elif 'tenant' in kwargs:
                tenant = kwargs['tenant']
            elif 'user_info' in kwargs and isinstance(kwargs['user_info'], dict):
                tenant = kwargs['user_info'].get('tenant')
                
            # Check permission
            if not permit_client.check(
                user_id=user_id,
                action=action,
                resource=resource_id,
                resource_type=resource_type,
                tenant=tenant
            ):
                logger.warning(
                    f"Permission denied: user {user_id} cannot {action} {resource_type} {resource_id}"
                )
                raise PermissionError(f"Permission denied: {action} {resource_type}")
                
            # Permission granted, proceed with function
            return func(*args, **kwargs)
            
        return cast(F, wrapper)
        
    return decorator


def with_auth_context(
    get_user_id: Optional[Callable] = None,
    get_tenant: Optional[Callable] = None,
) -> Callable[[F], F]:
    """
    Decorator to inject authorization context into function.
    
    Args:
        get_user_id: Optional function to extract user ID from args/kwargs
        get_tenant: Optional function to extract tenant ID from args/kwargs
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract user ID
            user_id = None
            if get_user_id:
                user_id = get_user_id(*args, **kwargs)
            elif 'user_id' in kwargs:
                user_id = kwargs['user_id']
            elif 'user_info' in kwargs and isinstance(kwargs['user_info'], dict):
                user_id = kwargs['user_info'].get('user_id')
                
            if not user_id:
                logger.error(f"Unable to determine user ID for auth context in {func.__name__}")
                raise ValueError("User ID is required for auth context")
                
            # Extract tenant
            tenant = None
            if get_tenant:
                tenant = get_tenant(*args, **kwargs)
            elif 'tenant' in kwargs:
                tenant = kwargs['tenant']
            elif 'user_info' in kwargs and isinstance(kwargs['user_info'], dict):
                tenant = kwargs['user_info'].get('tenant')
                
            # Create auth context
            auth_ctx = AuthContext(
                user_id=user_id,
                tenant=tenant,
                user_info=kwargs.get('user_info', {})
            )
            
            # Check if function expects auth_context parameter
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            if 'auth_context' in param_names:
                # Inject auth_context as named parameter
                kwargs['auth_context'] = auth_ctx
            else:
                # Look for a parameter with AuthContext type annotation
                for param_name, param in sig.parameters.items():
                    if param.annotation == AuthContext:
                        kwargs[param_name] = auth_ctx
                        break
                        
            # Proceed with function
            return func(*args, **kwargs)
            
        return cast(F, wrapper)
        
    return decorator


class AuthorizationMiddleware:
    """Middleware for enforcing authorization in API requests."""
    
    def __init__(
        self,
        permit_client: Optional[PermitClient] = None,
        user_manager: Optional[UserManager] = None,
        get_user_id: Optional[Callable] = None,
        get_tenant: Optional[Callable] = None,
    ):
        """
        Initialize the authorization middleware.
        
        Args:
            permit_client: Optional PermitClient instance
            user_manager: Optional UserManager instance
            get_user_id: Function to extract user ID from request
            get_tenant: Function to extract tenant ID from request
        """
        self.permit = permit_client or get_permit_client()
        self.user_manager = user_manager or get_user_manager(permit_client=self.permit)
        self.get_user_id = get_user_id
        self.get_tenant = get_tenant
        
    def check_permission(
        self,
        request: Any,
        action: str,
        resource: Union[str, Dict[str, Any]],
        resource_type: str = "document",
    ) -> bool:
        """
        Check if the request has permission for an action.
        
        Args:
            request: The request object
            action: The action to check
            resource: The resource to check
            resource_type: The resource type
            
        Returns:
            True if permitted, False otherwise
        """
        # Get user ID from request
        user_id = self._extract_user_id(request)
        if not user_id:
            logger.warning("No user ID in request")
            return False
            
        # Get tenant ID from request if available
        tenant = None
        if self.get_tenant:
            tenant = self.get_tenant(request)
            
        # Check permission
        return self.permit.check(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_type=resource_type,
            tenant=tenant
        )
        
    def get_user_info_from_request(self, request: Any) -> Dict[str, Any]:
        """
        Extract user info from a request.
        
        Args:
            request: The request object
            
        Returns:
            User info dictionary
        """
        # Get user ID
        user_id = self._extract_user_id(request)
        if not user_id:
            return {}
            
        # Get tenant if available
        tenant = None
        if self.get_tenant:
            tenant = self.get_tenant(request)
            
        # Get user info
        return self.user_manager.get_user_info(user_id, tenant)
        
    def _extract_user_id(self, request: Any) -> Optional[str]:
        """
        Extract user ID from request.
        
        Args:
            request: The request object
            
        Returns:
            User ID or None
        """
        if self.get_user_id:
            return self.get_user_id(request)
            
        # Generic extraction - will need to be customized for specific frameworks
        if hasattr(request, 'user') and hasattr(request.user, 'id'):
            return str(request.user.id)
            
        if hasattr(request, 'user') and hasattr(request.user, 'user_id'):
            return str(request.user.user_id)
            
        if hasattr(request, 'headers') and request.headers.get('X-User-ID'):
            return request.headers.get('X-User-ID')
            
        return None