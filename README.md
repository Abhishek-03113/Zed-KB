# Zed-KB

Zed-KB is a secure AI-powered internal knowledge base with tiered access and authorization controls. The system integrates document processing, vector storage, and retrieval-augmented generation (RAG) capabilities with comprehensive security controls.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)

## About

Zed-KB is designed to ingest various document formats, process them into semantically meaningful chunks, store them in vector databases, and provide a secure retrieval mechanism using modern language models. The system implements tiered access controls ensuring users only retrieve information they are authorized to access.

## Features

- Document ingestion for multiple formats (PDF, DOCX, HTML, TXT, etc.)
- Intelligent document chunking with configurable parameters
- Metadata extraction and security classification
- Vector database integration (Pinecone)
- RAG pipeline with support for Google Gemini LLMs
- Security with document-level access controls
- FastAPI backend and Streamlit frontend

## Project Structure

```
.
├── main.py                  # FastAPI backend entry point
├── streamlit_app.py         # Streamlit frontend
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── src/
│   └── zed_kb/
│       ├── document_processing/  # Document processing pipeline
│       ├── llm/                 # LLM integration components
│       └── vector_store/        # Vector database integration
└── uploads/                 # Document upload directory
```

## Dependencies

### Core Requirements
These dependencies are already listed in requirements.txt:

```
click==8.1.7
cryptography==41.0.3
fastapi==0.115.12
langchain==0.3.25
langchain_community==0.3.23
langchain_core==0.3.58
langchain_text_splitters==0.3.8
permit==2.7.5
pinecone==6.0.2
protobuf==6.30.2
pydantic==2.11.4
pymongo==3.11.0
python-dotenv==1.1.0
requests==2.32.3
simplejson==3.19.2
streamlit==1.45.0
uvicorn==0.34.2
watchdog==3.0.0
```

### Additional Dependencies
These dependencies are being used in the code but aren't explicitly listed in requirements.txt:

```
# For document processing
PyMuPDF  # For PDF document processing
unstructured  # For docx, html document processing
markdown  # For markdown document processing

# For Google Gemini integration
google-generativeai  # For Gemini API access
```

To install all dependencies:

```bash
pip install -r requirements.txt
pip install PyMuPDF unstructured markdown google-generativeai
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Zed-KB
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install PyMuPDF unstructured markdown google-generativeai
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Fill in the required API keys and configuration values

4. **Run the backend:**
   ```bash
   python main.py
   ```

5. **Run the Streamlit frontend (in a separate terminal):**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

### Backend API

The system provides several API endpoints:
- User management: `/signup`, `/login`
- Document management: `/upload`, `/documents`
- RAG functionality: `/chat`

### Streamlit Interface

The Streamlit interface provides:
- User authentication
- Document uploading and management
- Chat interface for querying the knowledge base

## Environment Variables

Required environment variables:

```
GOOGLE_API_KEY = "Your Gemini API key"
OPENAI_API_KEY = "Your OpenAI API key" 
HUGGINGFACE_API_KEY = "Your HuggingFace API key"
PERMIT_IO_API_KEY = "Your Permit.io API key"
PINECONE_API_KEY = "Your Pinecone API key"
MONGO_DATABASE_USERNAME = "Your MongoDB username"
MONGO_DATABASE_PASSWORD = "Your MongoDB password"
```

Optional variables:
```
PINECONE_ENVIRONMENT = "us-west1-gcp"
PINECONE_INDEX_NAME = "zed-kb-documents"
```

## Deployment

The project includes a Dockerfile for containerized deployment:

```bash
docker build -t zed-kb .
docker run -p 8000:80 --env-file .env zed-kb
```

For production deployment, consider:
- Using a reverse proxy like Nginx
- Implementing proper authentication mechanisms
- Securing environment variables
- Setting up database backups
