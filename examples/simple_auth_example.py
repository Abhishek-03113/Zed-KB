"""
Example demonstrating the simplified authorization system.

This example shows how to use the new simplified authorization approach
with the Permit.io API for checking user permissions.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse

# Import our simplified authorization components
from zed_kb.auth.permit_client import get_permit_client
from zed_kb.auth.simple_auth import (
    check_permission, 
    sync_user, 
    require_permission,
    with_permission_check,
    get_user_from_request,
    UserInfo
)

# Create FastAPI app
app = FastAPI(title="Zed-KB Simple Auth Example")

# Sample user data
SAMPLE_USERS = {
    "user1": {
        "id": "user1",
        "firstName": "Alice",
        "lastName": "Smith",
        "email": "alice@example.com",
    },
    "user2": {
        "id": "user2",
        "firstName": "Bob",
        "lastName": "Jones",
        "email": "bob@example.com",
    }
}

# Sample documents
DOCUMENTS = {
    "doc1": {"id": "doc1", "title": "Project Overview", "content": "This is a confidential document..."},
    "doc2": {"id": "doc2", "title": "Public Report", "content": "This report is available to all users..."},
    "doc3": {"id": "doc3", "title": "Financial Data", "content": "The quarterly financial data shows..."}
}

# ---- FastAPI Routes ----

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Zed-KB Simple Auth Example"}


@app.get("/sync-user/{user_id}")
async def sync_user_endpoint(user_id: str):
    """Sync a user with Permit.io."""
    if user_id not in SAMPLE_USERS:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = SAMPLE_USERS[user_id]
    success = await sync_user(user_data)
    
    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Failed to sync user {user_id}"}
        )
        
    return {"message": f"User {user_id} synced successfully"}


@app.get("/check/{user_id}/document/{doc_id}")
async def check_document_permission(user_id: str, doc_id: str, action: str = "read"):
    """Check if a user has permission to access a document."""
    if user_id not in SAMPLE_USERS:
        raise HTTPException(status_code=404, detail="User not found")
        
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Use the simple permission check function
    permitted = await check_permission(
        user_id=user_id,
        action=action,
        resource=doc_id,
        resource_type="document"
    )
    
    if not permitted:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "permitted": False,
                "message": f"{SAMPLE_USERS[user_id]['firstName']} is NOT PERMITTED to {action} document {doc_id}"
            }
        )
        
    return {
        "permitted": True,
        "message": f"{SAMPLE_USERS[user_id]['firstName']} is PERMITTED to {action} document {doc_id}",
        "user": SAMPLE_USERS[user_id]
    }


# Example using the FastAPI dependency for permission
@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: str, 
    user_info: UserInfo = Depends(require_permission(
        action="read", 
        resource=lambda req, doc_id: doc_id,
        resource_type="document"
    ))
):
    """Get document if user has permission."""
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # If we get here, permission has been granted
    return {
        "document": DOCUMENTS[doc_id],
        "user": user_info,
    }


# Example using the decorator approach
@app.get("/api/documents/{doc_id}")
@with_permission_check(action="read", get_resource_id=lambda doc_id, **kwargs: doc_id)
async def get_document_api(doc_id: str, user_id: str):
    """Get document using the decorator approach."""
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # If we get here, permission has been granted
    return {
        "document": DOCUMENTS[doc_id],
        "user_id": user_id
    }


# ---- Standalone example functions ----

async def example_direct_check():
    """Example of direct permission checks."""
    permit_client = get_permit_client()
    
    print("Checking permissions directly:")
    for user_id, user in SAMPLE_USERS.items():
        for doc_id in DOCUMENTS:
            permitted = await permit_client.check_permission(
                user_id=user_id,
                action="read",
                resource=doc_id
            )
            print(f"User {user_id} read {doc_id}: {'✓' if permitted else '✗'}")


# Run example if this file is executed directly
if __name__ == "__main__":
    # Set API key from environment or hardcode for testing
    api_key = os.getenv("PERMIT_IO_API_KEY", 
        "permit_key_mcuHJSmHb3TGCrEYAQSjY8pxbD2O1KrmQAf6Wt8kkqBvsQLHr656NPYFPrHSykUZO2dkHlVb8uDML4qKJImMml")

    # Create client with API key (not needed if PERMIT_IO_API_KEY is set)
    client = get_permit_client(api_key=api_key)
    
    print("Running standalone example...")
    asyncio.run(example_direct_check())
    print("\nTo run the FastAPI server, execute: uvicorn examples.simple_auth_example:app --reload")