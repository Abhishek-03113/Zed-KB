#!/usr/bin/env python
"""
Example script demonstrating how to use AstraDB for security-aware
Retrieval Augmented Generation (RAG) with Zed-KB.

This example:
1. Connects to AstraDB vector store
2. Creates sample documents with different security levels
3. Demonstrates security filtering during retrieval
4. Shows how to use the RAG pipeline with different user permissions

Requirements:
- AstraDB account and credentials (API key, database ID, region)
- Google API key for Gemini model
- Sample documents (this example uses a PDF from the data folder)
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.zed_kb.document_processing.document_loader import DocumentLoader
from src.zed_kb.document_processing.document_chunker import DocumentChunker
from src.zed_kb.document_processing.document_indexer import DocumentIndexer
from src.zed_kb.vector_store.gemini_embeddings import GeminiEmbeddings
from src.zed_kb.llm.gemini_model import GeminiLLM
from src.zed_kb.llm.rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_security_schema():
    """Load security schema from config file."""
    try:
        schema_path = os.path.join(os.path.dirname(__file__), 
                                  '../src/zed_kb/config/schema.json')
        with open(schema_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading security schema: {e}")
        return {
            "metadata": {
                "security_level": ["public", "internal", "confidential"],
                "allowed_roles": ["admin", "user", "guest", "superuser"],
                "allowed_users": []
            }
        }


def create_sample_documents():
    """Create sample documents with different security levels."""
    # Create list to hold our document examples with different security levels
    documents = []
    
    # Public document
    documents.append({
        "page_content": "This is a public document that anyone can access. " 
                       "It contains general information about the company's " 
                       "public products and services.",
        "metadata": {
            "security_level": "public",
            "source": "company_website",
            "allowed_roles": ["admin", "user", "guest"],
            "doc_id": "doc_public_001"
        }
    })
    
    # Internal document
    documents.append({
        "page_content": "This is an internal document with limited distribution. " 
                       "It contains information about upcoming product releases " 
                       "and internal processes.",
        "metadata": {
            "security_level": "internal",
            "source": "internal_wiki",
            "allowed_roles": ["admin", "user"],
            "doc_id": "doc_internal_001" 
        }
    })
    
    # Confidential document
    documents.append({
        "page_content": "This is a confidential document with restricted access. " 
                       "It contains sensitive financial projections and strategic " 
                       "planning information that should not be widely shared.",
        "metadata": {
            "security_level": "confidential",
            "source": "executive_reports",
            "allowed_roles": ["admin"],
            "doc_id": "doc_confidential_001"
        }
    })
    
    return documents


def main():
    """Main function demonstrating AstraDB for security-aware RAG."""
    # Load security schema
    schema = load_security_schema()
    print(f"Loaded security schema with levels: {schema['metadata']['security_level']}")
    
    # Get AstraDB credentials from environment variables
    astra_token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    astra_db_id = os.environ.get("ASTRA_DB_ID")
    astra_db_region = os.environ.get("ASTRA_DB_REGION")
    
    if not (astra_token and astra_db_id and astra_db_region):
        print("Error: Missing AstraDB credentials. Please set the following environment variables:")
        print("  ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID, ASTRA_DB_REGION")
        print("\nExample:")
        print("  export ASTRA_DB_APPLICATION_TOKEN='your_token_here'")
        print("  export ASTRA_DB_ID='your_db_id_here'")
        print("  export ASTRA_DB_REGION='your_region_here'")
        return
    
    # Check for Google API key for Gemini model
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("  Set it to use Gemini models: export GOOGLE_API_KEY='your_key_here'")
    
    # Create AstraDB configuration
    astra_config = {
        "token": astra_token,
        "astra_db_id": astra_db_id,
        "astra_db_region": astra_db_region,
        "collection_name": "zed_kb_secure_documents"
    }
    
    # Initialize document indexer with Gemini embeddings
    print("Initializing document indexer with Gemini embeddings...")
    try:
        indexer = DocumentIndexer(
            embedding_provider="gemini",  # Use Gemini embeddings (or "openai")
            embedding_model="embedding-001",
            collection_name=astra_config["collection_name"],
            hybrid_search=True,
            astra_config=astra_config
        )
    except Exception as e:
        print(f"Error initializing document indexer: {e}")
        return
    
    # Create and add sample documents
    print("Creating sample documents with different security levels...")
    sample_docs = create_sample_documents()
    from langchain.schema import Document
    langchain_docs = [Document(**doc) for doc in sample_docs]
    
    # Index documents
    print("Indexing documents...")
    try:
        doc_ids = indexer.add_documents(langchain_docs)
        print(f"Indexed {len(doc_ids)} documents")
    except Exception as e:
        print(f"Error indexing documents: {e}")
        return
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    try:
        rag = RAGPipeline(
            document_indexer=indexer,
            llm=GeminiLLM(
                model_name="gemini-1.5-pro-latest",
                temperature=0.2
            ),
            num_results=5
        )
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        return
    
    # Define user profiles with different security clearances
    users = {
        "guest_user": {
            "roles": ["guest"],
            "clearance": "public"
        },
        "standard_user": {
            "roles": ["user"],
            "clearance": "internal"
        },
        "admin_user": {
            "roles": ["admin"],
            "clearance": "confidential"
        }
    }
    
    # Test queries
    query = "What information do we have about the company?"
    
    print("\n" + "="*50)
    print("Testing RAG with different user permissions")
    print("="*50)
    
    # Test with different user permissions
    for user_name, user_info in users.items():
        print(f"\nQuerying as {user_name} with clearance: {user_info['clearance']}")
        result = rag.run(query, user_info=user_info)
        
        print(f"Documents found: {len(result['documents'])}")
        print(f"Security levels retrieved: {[doc.metadata.get('security_level') for doc in result['documents']]}")
        print(f"Answer: {result['answer']}")
    
    print("\n" + "="*50)
    print("Security filtering demonstration complete!")


if __name__ == "__main__":
    main()