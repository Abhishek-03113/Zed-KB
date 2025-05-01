"""
Example script demonstrating the document processing pipeline with Google's Gemini embeddings.
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


def main():
    # Set up the document processor with Gemini embeddings
    processor = DocumentProcessor(
        vector_store_type="faiss",  # Can also use "chroma"
        embedding_provider="gemini",  # Use the Gemini provider
        embedding_model="models/embedding-001",  # Currently the main Gemini embedding model
        persist_directory="./data/gemini_vector_store",
        chunk_size=1000,
        chunk_overlap=200,
    )

    print("\n*** Processing Documents with Gemini Embeddings ***\n")
    
    # Process a technical document
    print("Processing technical documentation...")
    doc_ids = processor.process_file(
        file_path="./data/test.pdf",
        metadata={"category": "technical", "topic": "api"},
        security_level="internal",
        allowed_roles=["developer", "product", "technical_writer"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Process a research document
    print("\nProcessing research document...")
    doc_ids = processor.process_file(
        file_path="./data/test.pdf",
        metadata={"category": "research", "topic": "market_analysis"},
        security_level="confidential",
        allowed_roles=["research", "executive", "marketing_lead"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Searching with different user permissions
    print("\n*** Searching Documents with Gemini Embeddings ***\n")

    # Developer search
    print("Developer search:")
    developer = {
        "user_id": "alex.dev@company.com",
        "roles": ["developer"],
        "clearance": "internal",
        "department": "engineering",
    }

    results = processor.search(query="api authentication methods", user_info=developer, k=3)

    print(f"  Found {len(results)} results for developer")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Security: {doc.metadata.get('security_level', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")

    # Marketing lead search
    print("\nMarketing lead search:")
    marketing_lead = {
        "user_id": "sara.marketing@company.com",
        "roles": ["marketing_lead"],
        "clearance": "confidential",
        "department": "marketing",
    }

    results = processor.search(query="market trends analysis", user_info=marketing_lead, k=3)

    print(f"  Found {len(results)} results for marketing lead")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Security: {doc.metadata.get('security_level', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()