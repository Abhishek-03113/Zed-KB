"""
Example script demonstrating document processing pipeline for climate change Twitter data analytics.
"""

import os
import sys
import dotenv
from pathlib import Path

# Add the parent directory to the path so we can import zed_kb
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.zed_kb.document_processing import DocumentProcessor

# Load environment variables from .env
dotenv.load_dotenv()


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

    # Process the Twitter data analytics document for climate change
    print("\n*** Processing Twitter Data Analytics for Climate Change ***\n")

    # Process the climate change Twitter data report
    print("Processing Twitter data analytics report...")
    doc_ids = processor.process_file(
        file_path="./data/test.pdf",
        metadata={
            "category": "social_media_analytics", 
            "topic": "climate_change", 
            "source": "twitter",
            "analysis_type": "sentiment_analysis"
        },
        security_level="internal",
        allowed_roles=["researcher", "analyst", "data_scientist"],
    )
    print(f"  Processed document IDs: {doc_ids}")

    # Example searching the Twitter climate change data with different queries
    print("\n*** Searching Twitter Climate Change Data ***\n")

    # Climate researcher search
    print("Climate researcher search:")
    researcher = {
        "user_id": "climate.researcher@org.com",
        "roles": ["researcher"],
        "clearance": "internal",
        "department": "climate_science",
    }

    results = processor.search(
        query="climate change sentiment trends", user_info=researcher, k=3)

    print(f"  Found {len(results)} results for climate researcher")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Topic: {doc.metadata.get('topic', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")

    # Data analyst search
    print("\nData analyst search:")
    analyst = {
        "user_id": "data.analyst@org.com",
        "roles": ["analyst"],
        "clearance": "internal",
        "department": "data_analytics",
    }

    results = processor.search(
        query="twitter hashtag frequency climate", user_info=analyst, k=3)

    print(f"  Found {len(results)} results for data analyst")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Analysis: {doc.metadata.get('analysis_type', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")

    # Policy advisor search
    print("\nPolicy advisor search:")
    advisor = {
        "user_id": "policy.advisor@gov.org",
        "roles": ["researcher", "analyst"],
        "clearance": "internal",
        "department": "policy",
    }

    results = processor.search(
        query="public opinion climate policy", user_info=advisor, k=3)

    print(f"  Found {len(results)} results for policy advisor")
    for i, doc in enumerate(results):
        print(
            f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Topic: {doc.metadata.get('topic', 'Unknown')}"
        )
        print(f"     {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
