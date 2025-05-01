"""
Example script demonstrating AstraDB Vector Store with hybrid search capabilities.
This example shows how to use OpenAI or Gemini embeddings with AstraDB vector database.
"""

import os
import sys
import dotenv
from pathlib import Path

# Add the parent directory to the path so we can import zed_kb
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
dotenv.load_dotenv()

from src.zed_kb.document_processing import DocumentProcessor


def astra_openai_example():
    """Example using AstraDB with OpenAI embeddings"""
    
    print("\n*** AstraDB with OpenAI Embeddings Example ***\n")
    
    # Configure AstraDB connection (replace with your own values)
    astra_config = {
        "token": os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
        "api_endpoint": os.environ.get("ASTRA_DB_API_ENDPOINT"),
        # Or use database ID and region:
        # "astra_db_id": "your-db-id",
        # "astra_db_region": "your-db-region",
        "collection_name": "zed_kb_openai_documents"
    }

    # Set up document processor with AstraDB and OpenAI embeddings
    processor = DocumentProcessor(
        vector_store_type="astradb",
        embedding_provider="openai",  # Using OpenAI embeddings
        embedding_model="text-embedding-ada-002",  # Or "text-embedding-3-small" for newer model
        hybrid_search=True,  # Enable hybrid search
        astra_config=astra_config,
    )
    
    # Process a document with access control metadata
    print("Processing document with OpenAI embeddings...")
    doc_ids = processor.process_file(
        file_path="./data/test.pdf",
        metadata={
            "department": "legal",
            "project": "compliance",
            "importance": "high"
        },
        security_level="confidential",
        allowed_roles=["legal", "executive", "compliance_officer"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Search for documents with role-based access
    print("\nSearching as Legal Team Member:")
    legal_user = {
        "user_id": "jane.legal@company.com",
        "roles": ["legal"],
        "clearance": "confidential",
        "department": "legal",
    }

    # Search with hybrid search (mix of vector and keyword)
    results = processor.search(
        query="compliance requirements for financial institutions",
        user_info=legal_user,
        k=3,
        hybrid_alpha=0.5,  # Balance between vector and keyword search
    )

    print(f"  Found {len(results)} results for legal team member")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Department: {doc.metadata.get('department', 'Unknown')}"
        )
        print(f"     Similarity: {doc.metadata.get('similarity', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")


def astra_gemini_example():
    """Example using AstraDB with Gemini embeddings"""
    
    print("\n*** AstraDB with Gemini Embeddings Example ***\n")
    
    # Configure AstraDB connection (replace with your own values)
    astra_config = {
        "token": os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
        "api_endpoint": os.environ.get("ASTRA_DB_API_ENDPOINT"),
        "collection_name": "zed_kb_gemini_documents"
    }

    # Set up document processor with AstraDB and Google Gemini embeddings
    processor = DocumentProcessor(
        vector_store_type="astradb",
        embedding_provider="gemini",  # Using Google Gemini embeddings
        embedding_model="models/embedding-001",
        hybrid_search=True,  # Enable hybrid search
        astra_config=astra_config,
    )
    
    # Process the document with role-based access control
    print("Processing document with Gemini embeddings...")
    doc_ids = processor.process_file(
        file_path="./data/test.pdf",
        metadata={
            "department": "engineering",
            "project": "api_development",
            "importance": "medium"
        },
        security_level="internal",
        allowed_roles=["developer", "tech_lead", "product_manager"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Search for documents with role-based access
    print("\nSearching as Developer:")
    developer = {
        "user_id": "alex.dev@company.com",
        "roles": ["developer"],
        "clearance": "internal",
        "department": "engineering",
    }

    # Vector-focused search (higher alpha means more emphasis on vector similarity)
    results = processor.search(
        query="api authentication methods",
        user_info=developer,
        k=3,
        hybrid_alpha=0.8,  # Emphasize vector search over keyword search
    )

    print(f"  Found {len(results)} results for developer (vector-focused search)")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Project: {doc.metadata.get('project', 'Unknown')}"
        )
        print(f"     Similarity: {doc.metadata.get('similarity', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")

    # Keyword-focused search (lower alpha means more emphasis on keyword matching)
    print("\nPerforming keyword-focused search:")
    results = processor.search(
        query="api authentication methods",
        user_info=developer,
        k=3,
        hybrid_alpha=0.2,  # Emphasize keyword search over vector search
    )

    print(f"  Found {len(results)} results for developer (keyword-focused search)")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Project: {doc.metadata.get('project', 'Unknown')}"
        )
        print(f"     Similarity: {doc.metadata.get('similarity', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")


if __name__ == "__main__":
    # Check for necessary environment variables
    if not os.environ.get("ASTRA_DB_APPLICATION_TOKEN") or not os.environ.get("ASTRA_DB_API_ENDPOINT"):
        print("WARNING: ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT are required for AstraDB.")
        print("Please set these environment variables in your .env file.")
        print("Skipping AstraDB examples...")
    else:
        # Run OpenAI example if OpenAI API key is available
        if os.environ.get("OPENAI_API_KEY"):
            astra_openai_example()
        else:
            print("WARNING: OPENAI_API_KEY not found. Skipping OpenAI embeddings example.")
        
        # Run Gemini example if Google API key is available
        if os.environ.get("GOOGLE_API_KEY"):
            astra_gemini_example() 
        else:
            print("WARNING: GOOGLE_API_KEY not found. Skipping Gemini embeddings example.")