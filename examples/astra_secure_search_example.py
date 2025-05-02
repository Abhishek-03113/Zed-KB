"""
Example script demonstrating AstraDB Vector Store with secure multi-level search capabilities.

This example illustrates:
1. Multi-level search based on security clearance and user roles
2. Vector search on the filtered results for accurate retrieval
3. Handling security permissions for different user types
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


def astra_secure_multi_level_search():
    """Example demonstrating AstraDB with security-level filtered search before vector search"""
    
    print("\n*** AstraDB Secure Multi-Level Search Example ***\n")
    
    # Configure AstraDB connection
    astra_config = {
        "token": os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
        "api_endpoint": os.environ.get("ASTRA_DB_API_ENDPOINT"),
        "collection_name": "secure_documents_collection"
    }

    # Set up document processor with AstraDB and Google Gemini embeddings
    processor = DocumentProcessor(
        vector_store_type="astradb",
        embedding_provider="gemini",  # Using Google Gemini embeddings
        embedding_model="models/embedding-001",
        hybrid_search=False,  # Using pure vector search since hybrid isn't available in all regions
        astra_config=astra_config,
    )
    
    # Get absolute path to the test.pdf file
    base_dir = Path(__file__).parent.parent
    pdf_path = os.path.join(base_dir, "data", "test.pdf")
    
    # Process the document with different security levels
    print(f"Processing documents with multiple security levels from: {pdf_path}")
    
    # Document with public access level
    public_doc_ids = processor.process_file(
        file_path=pdf_path,
        metadata={
            "category": "climate_science", 
            "topic": "temperature_trends", 
            "source": "public_data",
            "analysis_type": "basic_statistics",
            "importance": "medium"
        },
        security_level="public",  # Accessible to everyone
        allowed_roles=[]  # No role restrictions
    )
    print(f"  Processed public document IDs: {public_doc_ids}")

    # Document with internal access level
    internal_doc_ids = processor.process_file(
        file_path=pdf_path,
        metadata={
            "category": "climate_science", 
            "topic": "warming_factors", 
            "source": "internal_research",
            "analysis_type": "correlation_analysis",
            "importance": "high"
        },
        security_level="internal",  # Accessible to employees
        allowed_roles=["researcher", "analyst"]  # Role restrictions
    )
    print(f"  Processed internal document IDs: {internal_doc_ids}")

    # Document with confidential access level
    confidential_doc_ids = processor.process_file(
        file_path=pdf_path,
        metadata={
            "category": "climate_science", 
            "topic": "policy_recommendations", 
            "source": "confidential_analysis",
            "analysis_type": "predictive_modeling",
            "importance": "critical"
        },
        security_level="confidential",  # Limited access
        allowed_roles=["senior_researcher", "policy_advisor"]  # Stricter role restrictions
    )
    print(f"  Processed confidential document IDs: {confidential_doc_ids}")
    
    # Document with restricted access level
    restricted_doc_ids = processor.process_file(
        file_path=pdf_path,
        metadata={
            "category": "climate_science", 
            "topic": "strategic_response", 
            "source": "government_briefing",
            "analysis_type": "scenario_planning",
            "importance": "highest"
        },
        security_level="restricted",  # Highly limited access
        allowed_roles=["director", "executive"]  # Very limited role access
    )
    print(f"  Processed restricted document IDs: {restricted_doc_ids}")

    # Now demonstrate searching with different user types
    search_query = "climate change temperature analysis"

    # 1. Public user search (lowest clearance)
    print("\n[1] Searching as Public User (no special clearance):")
    public_user = {
        "user_id": "public.user@example.com",
        "roles": ["viewer"],
        "clearance": "public",
        "department": "external",
    }

    results = processor.search(
        query=search_query,
        user_info=public_user,
        k=3,
    )

    print(f"  Found {len(results)} results for public user")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security Level: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Topic: {doc.metadata.get('topic', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")

    # 2. Researcher search (internal clearance)
    print("\n[2] Searching as Researcher (internal clearance):")
    researcher = {
        "user_id": "researcher@organization.com",
        "roles": ["researcher", "analyst"],
        "clearance": "internal",
        "department": "research",
    }

    results = processor.search(
        query=search_query,
        user_info=researcher,
        k=3,
    )

    print(f"  Found {len(results)} results for researcher")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security Level: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Topic: {doc.metadata.get('topic', 'Unknown')}"
        )
        print(f"     Similarity: {doc.metadata.get('similarity', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")

    # 3. Senior researcher search (confidential clearance)
    print("\n[3] Searching as Senior Researcher (confidential clearance):")
    senior_researcher = {
        "user_id": "senior.researcher@organization.com",
        "roles": ["senior_researcher"],
        "clearance": "confidential",
        "department": "research",
    }

    results = processor.search(
        query=search_query,
        user_info=senior_researcher,
        k=3,
    )

    print(f"  Found {len(results)} results for senior researcher")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security Level: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Topic: {doc.metadata.get('topic', 'Unknown')}"
        )
        print(f"     Importance: {doc.metadata.get('importance', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")

    # 4. Executive search (restricted clearance + role)
    print("\n[4] Searching as Executive (restricted clearance):")
    executive = {
        "user_id": "executive@organization.com",
        "roles": ["executive", "director"],
        "clearance": "restricted",
        "department": "leadership",
    }

    results = processor.search(
        query=search_query,
        user_info=executive,
        k=3,
    )

    print(f"  Found {len(results)} results for executive")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security Level: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Topic: {doc.metadata.get('topic', 'Unknown')}"
        )
        print(f"     Importance: {doc.metadata.get('importance', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")
    
    # 5. Director with role but insufficient clearance
    print("\n[5] Searching as Junior Director (with role but insufficient clearance):")
    junior_director = {
        "user_id": "junior.director@organization.com",
        "roles": ["director"],  # Has the right role
        "clearance": "internal", # But insufficient clearance
        "department": "leadership",
    }

    results = processor.search(
        query=search_query,
        user_info=junior_director,
        k=3,
    )

    print(f"  Found {len(results)} results for junior director")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Security Level: {doc.metadata.get('security_level', 'Unknown')} - "
            f"Topic: {doc.metadata.get('topic', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")


if __name__ == "__main__":
    # Check for necessary environment variables
    if not os.environ.get("ASTRA_DB_APPLICATION_TOKEN") or not os.environ.get("ASTRA_DB_API_ENDPOINT"):
        print("WARNING: ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT are required.")
        print("Please set these environment variables in your .env file.")
        print("Skipping AstraDB example...")
    else:
        # Run Gemini example if Google API key is available
        if os.environ.get("GOOGLE_API_KEY"):
            astra_secure_multi_level_search()
        else:
            print("WARNING: GOOGLE_API_KEY not found. Skipping Gemini embeddings example.")
            print("Please set the GOOGLE_API_KEY environment variable in your .env file.")