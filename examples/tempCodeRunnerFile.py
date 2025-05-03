"""
Example script demonstrating the RAG pipeline with Gemini 2.5 for Zed-KB.
Shows how to use the LLM orchestration layer to answer questions against a document store.
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
from src.zed_kb.llm import GeminiLLM, RAGPipeline


def main():
    # Get AstraDB credentials from environment variables
    astra_token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    astra_api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
    
    # Fallback to id and region if api_endpoint is not available
    astra_db_id = os.environ.get("ASTRA_DB_ID")
    astra_db_region = os.environ.get("ASTRA_DB_REGION")
    
    if not astra_token or (not astra_api_endpoint and not (astra_db_id and astra_db_region)):
        print("Warning: Missing AstraDB credentials. Using in-memory vector store.")
        astra_config = None
    else:
        # Create AstraDB configuration
        astra_config = {
            "token": astra_token,
            "collection_name": "zed_kb_gemini_example"
        }
        
        # Add either the full API endpoint or the db_id and region
        if astra_api_endpoint:
            astra_config["api_endpoint"] = astra_api_endpoint
        else:
            astra_config["astra_db_id"] = astra_db_id
            astra_config["astra_db_region"] = astra_db_region
    
    # Set up the document processor with Gemini embeddings for vector search
    processor = DocumentProcessor(
        embedding_provider="gemini",
        embedding_model="embedding-001",
        collection_name="zed_kb_gemini_example",
        hybrid_search=True,
        astra_config=astra_config,  # Pass AstraDB config if available
        chunk_size=1000,
        chunk_overlap=200,
    )

    print("\n*** Processing Documents with Zed-KB RAG Pipeline ***\n")
    
    # Process test documents with security levels
    print("Processing documents with different security levels...")
    
    # Process an internal document
    internal_id = processor.process_file(
        file_path="./data/test.pdf",
        metadata={"category": "technical", "topic": "api", "allow_quotes": True},
        security_level="public",
        allowed_roles=["developer", "product", "technical_writer"],
    )
    print(f"  Processed internal document ID: {internal_id}")
    
    # Process a confidential document
    confidential_id = processor.process_file(
        file_path="./data/test.pdf",
        metadata={"category": "research", "topic": "strategy", "allow_quotes": False},
        security_level="public",
        allowed_roles=["executive", "research"],
    )
    print(f"  Processed confidential document ID: {confidential_id}")

    # Initialize the Gemini LLM
    llm = GeminiLLM(
        model_name="gemini-2.0-flash",
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=1024,
    )
    
    # Initialize the RAG Pipeline with our document processor's indexer
    rag_pipeline = RAGPipeline(
        document_indexer=processor.indexer,
        llm=llm,
        num_results=3,
        security_level="general",  # Default security level
    )
    
    print("\n*** Querying with Different User Access Levels ***\n")
    
    # Define example queries
    queries = [
        "Is twitter data actually usefull for twitter data analysis?",
        "Explain the main strategy recommendations for our company.",
    ]
    
    # Define different user profiles with varying access levels
    users = {
        "Developer": {
            "user_id": "alex.developer@company.com",
            "roles": ["developer"],
            "clearance": "internal",
            "department": "engineering",
        },
        "Executive": {
            "user_id": "sarah.executive@company.com",
            "roles": ["executive"],
            "clearance": "confidential",
            "department": "management",
        }
    }
    
    # Run queries with different user contexts
    for user_name, user_info in users.items():
        print(f"\n=== Queries as {user_name} (Clearance: {user_info['clearance']}) ===\n")
        
        for query in queries:
            print(f"Query: {query}")
            
            # Run the RAG pipeline with this user's context
            result = rag_pipeline.run(
                query=query,
                user_info=user_info,
            )
            
            # Show the results
            print(f"\nAnswer:")
            print(f"{result['answer']}")
            print(f"\nRetrieved {result['source_count']} documents")
            
            # Show document security levels that were accessed
            print("\nAccessed document security levels:")
            for doc in result.get('documents', []):
                security = doc.metadata.get('security_level', 'unknown')
                doc_id = doc.metadata.get('doc_id', 'unknown')
                print(f"  - {security} (ID: {doc_id})")
            
            print("\n" + "-" * 80)
    
    print("\n*** Querying with Citations ***\n")
    
    # Try a query with citations (as Executive who has higher clearance)
    executive_info = users["Executive"]
    query_with_citations = "What are the main technical components described in the documentation?"
    
    result = rag_pipeline.run_with_citations(
        query=query_with_citations,
        user_info=executive_info,
    )
    
    print(f"Query: {query_with_citations}")
    print(f"\nAnswer with citations:")
    print(f"{result['answer']}")
    print(f"\nSource document IDs: {', '.join(result.get('source_ids', []))}")
    

if __name__ == "__main__":
    main()