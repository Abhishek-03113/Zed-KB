#!/usr/bin/env python
"""
Example script demonstrating Permit.io authorization integration with Zed-KB.

This example:
1. Sets up Permit.io client with a test API key
2. Creates test users with different permission levels
3. Adds documents with different security classifications
4. Demonstrates how the RAG pipeline filters results based on user permissions

Requirements:
- Permit.io API key (set as PERMIT_IO_API_KEY environment variable)
- AstraDB credentials for vector storage
- Google or OpenAI API key for embeddings and LLM
"""

import os
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import zed_kb
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import the modules after the path is set
from src.zed_kb.llm import GeminiLLM, RAGPipeline
from src.zed_kb.auth import get_permit_client, get_user_manager
from src.zed_kb.document_processing import DocumentIndexer
from src.zed_kb.document_processing.metadata_extractor import MetadataExtractor
from src.zed_kb.document_processing.document_chunker import DocumentChunker
from src.zed_kb.document_processing.document_loader import DocumentLoader

# Load environment variables from .env
load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_documents() -> List[Dict[str, Any]]:
    """Create test documents with different security levels."""

    # Sample documents with different security classifications
    documents = [
        {
            "title": "Public Company Overview",
            "content": """
            Zed-KB is a secure AI-powered knowledge base system that provides 
            tiered access to information based on user permissions. Our system
            allows organizations to safely store and retrieve information with
            appropriate access controls.
            """,
            "metadata": {
                "security_level": "public",
                "allowed_roles": ["guest", "user", "admin"],
                "department": "Marketing",
                "doc_type": "overview"
            }
        },
        {
            "title": "Internal Engineering Documentation",
            "content": """
            Technical specifications for the Zed-KB system architecture:
            
            1. Vector store integration with AstraDB
            2. Embedding models from OpenAI and Google
            3. Security layers with Permit.io authorization
            4. RAG pipeline optimization with chunking strategies
            
            The system uses a multi-tier caching approach with token-based
            authentication and fine-grained access control.
            """,
            "metadata": {
                "security_level": "internal",
                "allowed_roles": ["user", "admin"],
                "department": "Engineering",
                "doc_type": "technical"
            }
        },
        {
            "title": "Confidential Strategy Document",
            "content": """
            CONFIDENTIAL - LIMITED DISTRIBUTION
            
            Our 2025 strategic roadmap includes:
            
            1. Expanding the security features with behavioral analysis
            2. Implementing advanced redaction for mixed-security documents
            3. Adding multi-modal support for images and audio content
            4. Potential acquisition of SecurityAI Inc. in Q3
            
            Financial projections indicate 40% growth with estimated
            valuation of $75M by end of year.
            """,
            "metadata": {
                "security_level": "confidential",
                "allowed_roles": ["admin"],
                "department": "Executive",
                "doc_type": "strategy"
            }
        }
    ]

    return documents


def index_test_documents(indexer: DocumentIndexer) -> List[str]:
    """
    Process and index test documents.

    Args:
        indexer: DocumentIndexer instance

    Returns:
        List of document IDs
    """
    from langchain.schema import Document

    # Create test documents
    test_docs = create_test_documents()

    # Convert to LangChain Document objects
    documents = []
    for doc in test_docs:
        documents.append(
            Document(
                page_content=f"{doc['title']}\n\n{doc['content']}",
                metadata=doc['metadata']
            )
        )

    # Add document IDs if not present
    for doc in documents:
        if "doc_id" not in doc.metadata:
            doc.metadata["doc_id"] = f"doc_{doc.metadata.get('security_level')}_{len(documents)}"

    # Index documents
    doc_ids = indexer.add_documents(documents)
    logger.info(f"Indexed {len(doc_ids)} documents")

    return doc_ids


def setup_permit():
    """Set up Permit.io client and test connection."""
    # Check if API key is available
    if not os.environ.get("PERMIT_IO_API_KEY"):
        logger.error("PERMIT_IO_API_KEY environment variable not set")
        print("\nIMPORTANT: You need a Permit.io API key to run this example.")
        print("Set it in your .env file or as an environment variable:")
        print("export PERMIT_IO_API_KEY='your_api_key'")
        return False

    # Initialize permit client
    try:
        permit_client = get_permit_client()
        success = permit_client.initialize()

        if success:
            logger.info("✅ Permit.io client initialized successfully")
            return permit_client
        else:
            logger.error("Failed to initialize Permit.io client")
            return None
    except Exception as e:
        logger.error(f"Error initializing Permit.io client: {e}")
        return None


def main():
    """Main function to run the authorization example."""
    print("\n===== Zed-KB Authorization Example with Permit.io =====\n")

    # Step 1: Set up Permit.io client
    print("Setting up Permit.io client...")
    permit_client = setup_permit()
    if not permit_client:
        print("❌ Failed to set up Permit.io. Exiting.")
        return

    # Step 2: Set up user manager
    user_manager = get_user_manager(permit_client=permit_client)

    # Step 3: Initialize document indexer
    print("\nInitializing document indexer...")
    indexer = DocumentIndexer(
        embedding_provider="openai",  # Use OpenAI for simplicity in this example
        embedding_model="text-embedding-ada-002",
        collection_name="zed_kb_auth_example",
        user_manager=user_manager
    )

    # Step 4: Index test documents
    print("Indexing test documents with different security levels...")
    doc_ids = index_test_documents(indexer)

    # Step 5: Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")

    # Try to use Gemini model, fall back to a mock if no API key
    llm = None
    try:
        llm = GeminiLLM(
            model_name="gemini-1.5-pro-latest",
            temperature=0.2
        )
    except:
        # If Gemini API key is not available, we'll see error messages but continue
        from langchain.llms.fake import FakeListLLM
        llm = FakeListLLM(
            responses=["This is a mock response since no valid API key was provided."])
        print("⚠️ Using mock LLM because Gemini API key is not available.")

    # Create RAG pipeline
    rag = RAGPipeline(
        document_indexer=indexer,
        llm=llm,
        num_results=5,
        user_manager=user_manager
    )

    # Step 6: Define test users with different permission levels
    print("\nSetting up test users with different permission levels...")
    users = {
        "guest_user": {
            "user_id": "guest123",
            "roles": ["guest"],
            "clearance": "public"
        },
        "regular_user": {
            "user_id": "user456",
            "roles": ["user"],
            "clearance": "internal"
        },
        "admin_user": {
            "user_id": "admin789",  # This could be the admin user with ID 0001
            "roles": ["admin"],
            "clearance": "confidential"
        }
    }

    # Step 7: Run queries with different users
    print("\n===== Testing Authorization with Different Users =====")

    # The query that we'll test with all users
    query = "Tell me about the technical architecture and strategic plans for Zed-KB?"

    for user_type, user_info in users.items():
        print(f"\n----- As {user_type.upper()} -----")
        print(f"User ID: {user_info['user_id']}")
        print(f"Roles: {user_info['roles']}")
        print(f"Clearance: {user_info['clearance']}")
        print(f"\nQuery: {query}")

        # Run the query with this user's context
        result = rag.run(query=query, user_info=user_info)

        # Print results
        print("\nRetrieved document security levels:")
        security_levels = [doc.metadata.get("security_level", "unknown")
                           for doc in result.get("documents", [])]
        print(f"  {security_levels}")

        print(f"\nAnswer:")
        print(f"{result['answer']}")
        print("\n" + "-" * 50)

    print("\n✅ Authorization example completed.")
    print("Note: In a real application, Permit.io would manage actual user permissions")
    print("and enforce complex policies including attribute-based access control.")


if __name__ == "__main__":
    main()
