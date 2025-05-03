"""
Authorization and access control module for Zed-KB.
Integrates with Permit.io for tiered security and access control.
"""

from .permit_client import PermitClient, get_permit_client
from .user_manager import UserManager, get_user_manager
from .middleware import (
    AuthContext, 
    require_permission,
    with_auth_context,
    AuthorizationMiddleware
)

__all__ = [
    "PermitClient", 
    "get_permit_client",
    "UserManager", 
    "get_user_manager",
    "AuthContext",
    "require_permission",
    "with_auth_context",
    "AuthorizationMiddleware"
]