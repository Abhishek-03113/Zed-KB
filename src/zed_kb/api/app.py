from fastapi import FastAPI, Request, Response 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import File, UploadFile
import os 
import json 

import os
import sys
import dotenv
from pathlib import Path

# Add the parent directory to the path so we can import zed_kb
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
dotenv.load_dotenv()

from src.zed_kb.document_processing import DocumentProcessor
from src.zed_kb.llm import GeminiLLM, RAGPipeline

app = FastAPI() 

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def HealthCheck():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.post("/upload") 
async def upload_file(request: Request, file: UploadFile = File(...)): 
    # Save the uploaded file to a specific directory
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return JSONResponse(content={"filename": file.filename}, status_code=200)

@app.get("/files") 
async def list_files():
    # List all files in the upload directory
    upload_dir = "uploads"
    files = os.listdir(upload_dir)
    return JSONResponse(content={"files": files}, status_code=200)

