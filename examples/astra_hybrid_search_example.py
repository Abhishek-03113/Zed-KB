"""
Example script demonstrating AstraDB Vector Store with hybrid search capabilities.
This example shows how to use Gemini embeddings with AstraDB vector database for
analyzing Twitter data related to climate change.
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


def astra_gemini_climate_analytics():
    """Example using AstraDB with Gemini embeddings for Twitter climate change analytics"""
    
    print("\n*** AstraDB with Gemini Embeddings for Twitter Climate Analytics Example ***\n")
    
    # Configure AstraDB connection
    astra_config = {
        "token": os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
        "api_endpoint": os.environ.get("ASTRA_DB_API_ENDPOINT"),
        "collection_name": "twitter_climate_analytics"
    }

    # Set up document processor with AstraDB and Google Gemini embeddings
    processor = DocumentProcessor(
        vector_store_type="astradb",
        embedding_provider="gemini",  # Using Google Gemini embeddings
        embedding_model="models/embedding-001",
        hybrid_search=True,  # Enable hybrid search
        astra_config=astra_config,
    )
    
    # Process the Twitter climate change report
    print("Processing Twitter climate change analytics report...")
    doc_ids = processor.process_file(
        file_path="./data/test.pdf",
        metadata={
            "category": "social_media_analytics", 
            "topic": "climate_change", 
            "source": "twitter",
            "analysis_type": "sentiment_analysis",
            "project": "climate_research",
            "importance": "high"
        },
        security_level="internal",
        allowed_roles=["researcher", "analyst", "data_scientist", "policy_advisor"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Different user roles searching for climate data
    
    # Climate researcher search
    print("\nSearching as Climate Researcher:")
    researcher = {
        "user_id": "climate.researcher@org.com",
        "roles": ["researcher"],
        "clearance": "internal",
        "department": "climate_science",
    }

    # Vector-focused search (higher alpha means more emphasis on vector similarity)
    results = processor.search(
        query="sentiment analysis of climate change denial tweets",
        user_info=researcher,
        k=3,
        hybrid_alpha=0.8,  # Emphasize vector search over keyword search
    )

    print(f"  Found {len(results)} results for climate researcher (vector-focused search)")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Topic: {doc.metadata.get('topic', 'Unknown')} - "
            f"Source: {doc.metadata.get('source', 'Unknown')}"
        )
        print(f"     Similarity: {doc.metadata.get('similarity', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")

    # Data analyst search with keyword focus
    print("\nSearching as Data Analyst:")
    analyst = {
        "user_id": "data.analyst@org.com",
        "roles": ["analyst"],
        "clearance": "internal",
        "department": "data_analytics",
    }

    # Keyword-focused search (lower alpha means more emphasis on keyword matching)
    results = processor.search(
        query="hashtag frequency climate activism twitter",
        user_info=analyst,
        k=3,
        hybrid_alpha=0.2,  # Emphasize keyword search over vector search
    )

    print(f"  Found {len(results)} results for data analyst (keyword-focused search)")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Analysis Type: {doc.metadata.get('analysis_type', 'Unknown')} - "
            f"Category: {doc.metadata.get('category', 'Unknown')}"
        )
        print(f"     Similarity: {doc.metadata.get('similarity', 'Unknown')}")
        print(f"     {doc.page_content[:100]}...")
    
    # Policy advisor search with balanced hybrid search
    print("\nSearching as Policy Advisor:")
    advisor = {
        "user_id": "policy.advisor@gov.org",
        "roles": ["policy_advisor"],
        "clearance": "internal",
        "department": "policy",
    }

    # Balanced hybrid search
    results = processor.search(
        query="public opinion trends on climate policy based on twitter data",
        user_info=advisor,
        k=3,
        hybrid_alpha=0.5,  # Balance between vector and keyword search
    )

    print(f"  Found {len(results)} results for policy advisor (balanced hybrid search)")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. Topic: {doc.metadata.get('topic', 'Unknown')} - "
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
        # Run Gemini example if Google API key is available
        if os.environ.get("GOOGLE_API_KEY"):
            astra_gemini_climate_analytics()
        else:
            print("WARNING: GOOGLE_API_KEY not found. Skipping Gemini embeddings example.")
            print("Please set the GOOGLE_API_KEY environment variable in your .env file.")