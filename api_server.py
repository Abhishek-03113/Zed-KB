#!/usr/bin/env python
"""
RAG Chat Application - FastAPI Backend

This file contains the FastAPI backend for the RAG chat application, which:
- Handles document upload and processing
- Manages the RAG pipeline for answering queries
- Provides API endpoints for the Streamlit frontend
- Implements authentication and authorization
"""

from src.zed_kb.llm import GeminiLLM, RAGPipeline
from src.zed_kb.vector_store.gemini_embeddings import GeminiEmbeddings
from src.zed_kb.vector_store.pinecone_store import PineconeStore
from src.zed_kb.document_processing.document_loader import DocumentLoader
from src.zed_kb.document_processing import DocumentProcessor
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# FastAPI imports
import uvicorn
from fastapi import FastAPI, Request, Response, File, UploadFile, BackgroundTasks, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Environment variables
import dotenv

# Add project root to path for importing zed_kb
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
dotenv.load_dotenv()

# Create FastAPI app
app = FastAPI(title="RAG Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for document processor and RAG pipeline
document_processor = None
rag_pipeline = None

# Directory to store uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Security
security = HTTPBearer()

# User authentication functions


def get_users():
    """Load user data from config file"""
    config_path = os.path.join("src", "zed_kb", "config", "schema.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("metadata", {}).get("allowed_users", [])
    return []


def authenticate(username: str, password: str):
    """Authenticate a user"""
    users = get_users()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    for user in users:
        if user.get("username") == username and user.get("password") == hashed_password:
            return user
    return None


async def validate_token(request: Request):
    """Validate authorization token from request headers"""
    try:
        # Get authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(
                status_code=401, detail="No authorization header")

        # Parse authorization header
        try:
            auth_data = json.loads(auth_header)
            username = auth_data.get("username")
            password = auth_data.get("password")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid authorization format")

        # Authenticate user
        user = authenticate(username, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Authentication error: {str(e)}")


async def validate_admin(request: Request):
    """Validate that the user is an admin"""
    user = await validate_token(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def validate_user(request: Request):
    """Validate that the user is authenticated (any role)"""
    return await validate_token(request)

# Create API endpoints


def create_default_test_users():
    """Create default test users if they don't exist"""
    try:
        config_path = os.path.join("src", "zed_kb", "config", "schema.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            users = config.get("metadata", {}).get("allowed_users", [])

            # Check if users already exist
            has_admin = any(u.get("username") == "admin" for u in users)
            has_test_user = any(u.get("username") ==
                                "test_user" for u in users)

            # Add missing users
            changes_made = False
            if not has_admin:
                # Admin password: "admin"
                admin_hash = hashlib.sha256("admin".encode()).hexdigest()
                users.append({
                    "username": "admin",
                    "password": admin_hash,
                    "role": "admin"
                })
                changes_made = True
                print("Created default admin user (username: admin, password: admin)")

            if not has_test_user:
                # Test user password: "password"
                user_hash = hashlib.sha256("password".encode()).hexdigest()
                users.append({
                    "username": "test_user",
                    "password": user_hash,
                    "role": "user"
                })
                changes_made = True
                print(
                    "Created default test user (username: test_user, password: password)")

            # Save changes if any users were added
            if changes_made:
                config["metadata"]["allowed_users"] = users
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print("Updated user configuration file")

            return True
        else:
            print(f"Warning: Config file not found at {config_path}")
            return False
    except Exception as e:
        print(f"Error creating default test users: {e}")
        return False


@app.on_event("startup")
async def startup():
    """Initialize components on startup"""
    global document_processor, rag_pipeline

    # Create default test users for development
    create_default_test_users()

    # Get Pinecone credentials from environment variables
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_environment = os.environ.get(
        "PINECONE_ENVIRONMENT", "us-west1-gcp")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "zed-kb-documents")

    # Set up Pinecone configuration
    if (pinecone_api_key):
        pinecone_config = {
            "api_key": pinecone_api_key,
            "environment": pinecone_environment,
            "index_name": index_name,
            "namespace": "default"
        }
        vector_store_type = "pinecone"
        print(f"Using Pinecone vector store with index: {index_name}")
    else:
        print("Warning: Pinecone API key not found. Using in-memory vector store.")
        pinecone_config = None
        vector_store_type = "memory"

    # Initialize document processor
    try:
        document_processor = DocumentProcessor(
            embedding_provider="gemini",
            embedding_model="embedding-001",
            collection_name=index_name,
            hybrid_search=False,  # Hybrid search not available in basic Pinecone setup
            vector_store_type=vector_store_type,
            pinecone_config=pinecone_config,
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            document_indexer=document_processor.indexer,
            llm=GeminiLLM(
                model_name="gemini-2.0-flash",  # Updated to use Gemini 2.0 flash
                temperature=0.2,
                max_output_tokens=1024,
            ),
            num_results=5,
            access_level="user",
        )
        print("Successfully initialized document processor and RAG pipeline")
    except Exception as e:
        print(f"Error initializing components: {e}")
        # Initialize with minimal functionality if there's an error
        if document_processor is None:
            # Simple initialization of document loader as fallback
            document_processor = {"loader": DocumentLoader()}


@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/documents")
async def list_documents(user: dict = Depends(validate_admin)):
    """List all processed documents (admin only)"""
    try:
        # List all files in the upload directory
        files = []
        if os.path.exists(UPLOAD_DIR):
            files = os.listdir(UPLOAD_DIR)
        return {"documents": files}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list documents: {str(e)}"}
        )


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    security_level: str = Form("public"),
    user: dict = Depends(validate_admin)
):
    """Upload and process document endpoint (admin only)"""
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the document in background
        background_tasks.add_task(
            process_document,
            file_path,
            security_level
        )

        return {"filename": file.filename, "status": "processing"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to upload document: {str(e)}"}
        )


@app.post("/chat")
async def chat_endpoint(request: Request):
    """Chat endpoint for RAG queries (available to all authenticated users)"""
    try:
        # Authenticate user
        user = await validate_user(request)

        # Parse request body
        data = await request.json()
        query = data.get("query")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "No query provided"}
            )

        # Use RAG pipeline to answer the query
        if rag_pipeline:
            # Set user info for querying based on user role
            user_info = {
                "roles": [user.get("role", "user")],
                "clearance": "public" if user.get("role") == "user" else "admin"
            }

            # Process the query through RAG pipeline
            result = rag_pipeline.run(
                query=query,
                user_info=user_info
            )

            return {
                "answer": result["answer"],
                "sources": [
                    {
                        "title": doc.metadata.get("source", "Unknown"),
                        "snippet": doc.page_content[:200] + "..."
                    }
                    for doc in result.get("documents", [])
                ],
                "source_count": result.get("source_count", 0)
            }
        else:
            return {
                "answer": "I'm sorry, the RAG pipeline is not available. Please ensure that your environment is properly configured.",
                "sources": [],
                "source_count": 0
            }
    except HTTPException as he:
        # Pass through HTTP exceptions
        return JSONResponse(
            status_code=he.status_code,
            content={"error": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing chat request: {str(e)}"}
        )


def process_document(file_path: str, security_level: str):
    """Process a document and add to vector store (runs in background)"""
    try:
        if document_processor:
            # Process the document with the appropriate security level
            doc_ids = document_processor.process_file(
                file_path=file_path,
                metadata={"source": os.path.basename(file_path)},
                security_level=security_level,
                allowed_roles=["user", "admin"]
            )
            print(f"Processed document: {file_path}, IDs: {doc_ids}")
        else:
            print(
                f"Document processor not available, cannot process: {file_path}")
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")


if __name__ == "__main__":
    print("Starting RAG Chat API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
