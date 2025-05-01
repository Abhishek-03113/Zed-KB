"""
Example script demonstrating the document processing pipeline.
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
    # Set up the document processor
    processor = DocumentProcessor(
        vector_store_type="faiss",
        embedding_provider="gemini",  # Using Google Gemini embeddings
        embedding_model="models/embedding-001",
        persist_directory="./data/vector_store",
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Example document processing with different security levels and access controls
    print("\n*** Processing Documents with Security Controls ***\n")

    # Process a public document
    print("Processing public document...")
    doc_ids = processor.process_file(
        file_path="./data/test.pdf",
        metadata={"category": "company", "topic": "overview"},
        security_level="public",
        allowed_roles=["employee", "contractor", "customer"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Process a confidential HR document
    print("\nProcessing confidential HR document...")
    doc_ids = processor.process_file(
        file_path="./data/documents/hr/employee_handbook.pdf",
        metadata={"category": "hr", "topic": "policies"},
        security_level="confidential",
        allowed_roles=["hr", "manager", "executive"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Process a restricted financial document
    print("\nProcessing restricted financial document...")
    doc_ids = processor.process_file(
        file_path="./data/documents/finance/q2_forecast.pdf",
        metadata={"category": "finance", "topic": "forecast", "quarter": "Q2"},
        security_level="restricted",
        allowed_roles=["finance", "executive"],
        allowed_users=["john.cfo@company.com"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Example searching with different user permissions
    print("\n*** Searching Documents with Different User Permissions ***\n")

    # Regular employee search
    print("Regular employee search:")
    regular_user = {
        "user_id": "jane.employee@company.com",
        "roles": ["employee"],
        "clearance": "internal",
        "department": "marketing",
    }

    results = processor.search(query="company policies", user_info=regular_user, k=3)

    print(f"  Found {len(results)} results for regular employee")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Security: {doc.metadata.get('security_level', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")

    # HR manager search
    print("\nHR manager search:")
    hr_manager = {
        "user_id": "hr.manager@company.com",
        "roles": ["hr", "manager"],
        "clearance": "confidential",
        "department": "hr",
    }

    results = processor.search(query="company policies", user_info=hr_manager, k=3)

    print(f"  Found {len(results)} results for HR manager")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Security: {doc.metadata.get('security_level', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")

    # Executive search
    print("\nExecutive search:")
    executive = {
        "user_id": "john.cfo@company.com",
        "roles": ["finance", "executive"],
        "clearance": "restricted",
        "department": "finance",
    }

    results = processor.search(query="financial forecast", user_info=executive, k=3)

    print(f"  Found {len(results)} results for executive")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Security: {doc.metadata.get('security_level', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
