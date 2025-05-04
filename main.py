#!/usr/bin/env python
# RAG Chat Application - FastAPI Backend
# Handles API endpoints, authentication, document upload, and chat

# Imports
from pymongo import MongoClient
import pymongo
from src.zed_kb.llm import GeminiLLM, RAGPipeline
from src.zed_kb.vector_store.gemini_embeddings import GeminiEmbeddings
from src.zed_kb.vector_store.pinecone_store import PineconeStore
from src.zed_kb.document_processing.document_loader import DocumentLoader
from src.zed_kb.document_processing import DocumentProcessor
from permit import Permit
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Request, Response, File, UploadFile, BackgroundTasks, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import dotenv
from pydantic import BaseModel

# Classes


class Usermodel(BaseModel):
    username: str
    password: str
    role: str
    security_level: str


class User(BaseModel):
    username: str
    password: str
    role: str
    security_level: str


# Globals and Config
users_ids = {
    "admin": "0001",
    "user": "user",
    "superuser": "superuser"
}
resources = ("KnowledgeBase", "ChatBot")
actions = ("read", "write", "delete")
sys.path.insert(0, str(Path(__file__).parent))
dotenv.load_dotenv()
app = FastAPI(title="RAG Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
document_processor = None
rag_pipeline = None
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
security = HTTPBearer()
username = os.environ.get("MONGO_DATABASE_USERNAME")
mongopass = os.environ.get("MONGO_DATABASE_PASSWORD")
uri = f"mongodb+srv://{username}:{mongopass}@userscluster.eowq0cp.mongodb.net/?retryWrites=true&w=majority&appName=UsersCluster"
mongo_client = MongoClient(uri)
try:
    mongo_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
db = mongo_client["users"]
collection = db["users"]
if "users" not in db.list_collection_names():
    db.create_collection("users")
    print("Created collection: users")
else:
    print("Collection already exists: users")
permit_client = None  # Will be initialized in startup

# Functions


def create_user(user: User):
    """Create a user in MongoDB"""
    try:
        existing_user = collection.find_one({"username": user.username})
        if (existing_user):
            print(f"User {user.username} already exists.")
            return False
        collection.insert_one(user.dict())
        print(f"User {user.username} created successfully.")
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False


def get_user(username: str):
    """Get a user from MongoDB"""
    try:
        user = collection.find_one({"username": username})
        if user:
            return User(**user)
        else:
            print(f"User {username} not found.")
            return None
    except Exception as e:
        print(f"Error retrieving user: {e}")
        return None


def user_data(User):
    return {
        "username": User.username,
        "role": User.role,
        "security_level": User.security_level
    }


def authenticate(username: str, password: str):
    """Authenticate a user by username and password"""
    try:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        user = collection.find_one(
            {"username": username, "password": hashed_password})
        if user:
            return User(**user)
        else:
            print("Invalid credentials")
            return None
    except Exception as e:
        print(f"Error during authentication: {e}")
        return None


async def validate_token(request: Request):
    """Validate authorization token from request headers"""
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(
                status_code=401, detail="No authorization header")
        try:
            auth_data = json.loads(auth_header)
            username = auth_data.get("username")
            password = auth_data.get("password")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid authorization format")
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
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def validate_user(request: Request):
    """Validate that the user is authenticated (any role)"""
    return await validate_token(request)


def process_document(file_path: str, security_level: str):
    """Process a document and add to vector store (runs in background)"""
    try:
        if document_processor:
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

# Routes


@app.post("/signup")
async def signup(user: Usermodel):
    """Sign up a new user"""
    try:
        hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
        new_user = User(
            username=user.username,
            password=hashed_password,
            role="user",
            security_level="public"
        )
        if create_user(new_user):
            return {"message": "User created successfully"}
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "User already exists"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error creating user: {str(e)}"}
        )


@app.post("/login")
async def login(user: Usermodel):
    """Log in a user"""
    try:
        authenticated_user = authenticate(user.username, user.password)
        if authenticated_user:
            return {"message": "Login successful", "user": user_data(authenticated_user)}
        else:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid credentials"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error logging in: {str(e)}"}
        )


@app.get("/users")
async def get_users():
    """Get all users (admin only)"""
    try:
        users = list(collection.find())
        return {"users": [user_data(User(**user)) for user in users]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving users: {str(e)}"}
        )


@app.get("/give_role/{username}/{role}")
async def give_role(username: str, role: str):
    """Give a role to a user (admin only)"""
    try:
        result = collection.update_one(
            {"username": username},
            {"$set": {"role": role}}
        )
    except pymongo.errors.PyMongoError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error updating user role: {str(e)}"}
        )
    if result.modified_count > 0:
        return {"message": f"Role '{role}' assigned to user '{username}'"}
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"User '{username}' not found"}
        )


@app.get("/remove_role/{username}")
async def remove_role(username: str):
    """Remove a role from a user (admin only)"""
    try:
        result = collection.update_one(
            {"username": username},
            {"$set": {"role": "user"}}
        )
    except pymongo.errors.PyMongoError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error updating user role: {str(e)}"}
        )
    if result.modified_count > 0:
        return {"message": f"Role removed from user '{username}'"}
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"User '{username}' not found"}
        )


@app.on_event("startup")
async def startup():
    """Initialize components on startup"""
    global document_processor, rag_pipeline, permit_client
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_environment = os.environ.get(
        "PINECONE_ENVIRONMENT", "us-west1-gcp")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "zed-kb-documents")
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
    try:
        document_processor = DocumentProcessor(
            embedding_provider="gemini",
            embedding_model="embedding-001",
            collection_name=index_name,
            hybrid_search=False,
            vector_store_type=vector_store_type,
            pinecone_config=pinecone_config,
            chunk_size=1000,
            chunk_overlap=200,
        )
        rag_pipeline = RAGPipeline(
            document_indexer=document_processor.indexer,
            llm=GeminiLLM(
                model_name="gemini-2.0-flash",
                temperature=0.2,
                max_output_tokens=1024,
            ),
            num_results=5,
            access_level="user",
        )
        # Initialize async Permit client
        permit_client = Permit(
            pdp="https://cloudpdp.api.permit.io",
            token="permit_key_mcuHJSmHb3TGCrEYAQSjY8pxbD2O1KrmQAf6Wt8kkqBvsQLHr656NPYFPrHSykUZO2dkHlVb8uDML4qKJImMml"
        )
        # Optionally test permission on startup
        permission = await permit_client.check(
            users_ids['admin'], action='read', resource='KnowledgeBase')
        print(permission)
        print(
            "Successfully initialized document processor, RAG pipeline, and Permit client")
    except Exception as e:
        print(f"Error initializing components: {e}")
        if document_processor is None:
            document_processor = {"loader": DocumentLoader()}


@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/documents")
async def list_documents(user: dict = Depends(validate_admin)):
    """List all processed documents (admin only)"""

    # permission check
    permission = await permit_client.check(
        users_ids[user.role], action='read', resource='KnowledgeBase')
    if not permission:
        return JSONResponse(
            status_code=403,
            content={"error": "You do not have permission to view documents."}
        )
    else:

        try:
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

    # permiision check

    permission = await permit_client.check(
        users_ids[user.role], action='update', resource='KnowledgeBase')

    if not permission:
        return JSONResponse(
            status_code=403,
            content={"error": "You do not have permission to upload documents."}
        )
    else:
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
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
        user = await validate_user(request)
        data = await request.json()
        query = data.get("query")
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "No query provided"}
            )
        if rag_pipeline:
            user_info = {
                "roles": [user.role],
                "clearance": "public" if user.role == "user" else "admin"
            }

            if await permit_client.check(
                    users_ids[user.role], action='read', resource='ChatBot'):
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
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "You do not have permission to access the RAG pipeline."}
                )
        else:
            return {
                "answer": "I'm sorry, the RAG pipeline is not available. Please ensure that your environment is properly configured.",
                "sources": [],
                "source_count": 0
            }
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"error": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing chat request: {str(e)}"}
        )

if __name__ == "__main__":
    print("Starting RAG Chat API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
