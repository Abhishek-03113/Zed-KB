"""
RAG (Retrieval Augmented Generation) Pipeline for Zed-KB.
Orchestrates the query → retrieve → generate flow.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from langchain.schema import Document
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser

from .gemini_model import GeminiLLM
from .prompt_templates import create_rag_prompt, format_documents

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the RAG (Retrieval Augmented Generation) pipeline for Zed-KB."""

    def __init__(
        self,
        document_indexer=None,
        llm: Optional[BaseLLM] = None,
        num_results: int = 10,
        access_level: str = "user",
    ):
        """
        Initialize the RAG Pipeline.

        Args:
            document_indexer: Document indexer instance with search capability
            llm: Language model to use for generation (if None, uses GeminiLLM)
            num_results: Number of documents to retrieve for each query
            access_level: Default access level for prompts ('admin' or 'user')
        """
        # Initialize components
        self.document_indexer = document_indexer
        self.llm = llm or GeminiLLM(
            model_name="gemini-2.0-flash", temperature=0.2)
        self.num_results = num_results
        self.access_level = access_level
        self.output_parser = StrOutputParser()

        # Create prompt template
        self.prompt_template = create_rag_prompt(
            access_level=self.access_level)

    def _build_retriever_chain(self, user_info: Optional[Dict[str, Any]] = None):
        """
        Build a retrieval chain for document search.

        Args:
            user_info: Optional user information for metadata filtering

        Returns:
            A retrieval function
        """
        def retrieve(query: str):
            try:
                # Use the document indexer to search
                if self.document_indexer:
                    results = self.document_indexer.search(
                        query=query,
                        k=self.num_results,
                    )
                    return results
                else:
                    logger.error("Document indexer not initialized")
                    return []
            except Exception as e:
                logger.error(f"Error during document retrieval: {e}")
                return []

        return retrieve

    def _get_effective_access_level(self, user_info: Optional[Dict[str, Any]]) -> str:
        """
        Determine the effective access level based on user role and security level.

        Args:
            user_info: User information dict with roles and clearance

        Returns:
            Effective access level for prompt selection ('admin' or 'user')
        """
        if not user_info:
            return self.access_level

        # Check if user has admin role or admin clearance
        roles = user_info.get("roles", [])
        clearance = user_info.get("clearance", "public")
        
        # Admin access is granted if:
        # 1. User has "admin" role in their roles list, OR
        # 2. User's clearance is explicitly set to "admin"
        if "admin" in roles or clearance == "admin":
            return "admin"
        else:
            return "user"

    def run(
        self,
        query: str,
        user_info: Optional[Dict[str, Any]] = None,
        filter_metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline to answer a query.

        Args:
            query: User query to answer
            user_info: User information for metadata filtering (optional)
            filter_metadata: Additional metadata filters for retrieval

        Returns:
            Dictionary with answer and retrieved documents
        """
        # Step 1: Retrieve relevant documents
        retriever = self._build_retriever_chain(user_info)
        retrieved_docs = retriever(query)

        # If no documents found, return early
        if not retrieved_docs:
            return {
                "answer": "I don't have enough information to answer this question.",
                "documents": [],
                "source_count": 0
            }

        # Step 2: Format documents for the prompt
        formatted_docs = format_documents(retrieved_docs)

        # Step 3: Set the appropriate access level based on user's clearance
        effective_access = self._get_effective_access_level(user_info)
        prompt = create_rag_prompt(access_level=effective_access)

        # Step 4: Generate answer
        prompt_inputs = {
            "context": formatted_docs,
            "question": query
        }

        # Run the LLM to generate a response
        try:
            answer = self.llm(prompt.format(**prompt_inputs))
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            answer = "I encountered an error while processing your request."

        # Step 5: Return results
        return {
            "answer": answer,
            "documents": retrieved_docs,
            "source_count": len(retrieved_docs)
        }

    def run_with_citations(
        self,
        query: str,
        user_info: Optional[Dict[str, Any]] = None,
        filter_metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline with explicit citations in the response.

        Args:
            query: User query to answer
            user_info: User information for metadata filtering (optional)
            filter_metadata: Additional metadata filters for retrieval

        Returns:
            Dictionary with answer, citations, and retrieved documents
        """
        # Modify the query to request citations
        query_with_citation_request = f"{query} (Please include citations to specific documents in your answer)"

        # Run the standard pipeline
        result = self.run(query_with_citation_request,
                          user_info, filter_metadata)

        # Get source document IDs for reference
        source_ids = []
        for doc in result.get("documents", []):
            doc_id = doc.metadata.get(
                "doc_id", None) or doc.metadata.get("document_id", None)
            if doc_id:
                source_ids.append(doc_id)

        # Add citation data to result
        result["source_ids"] = source_ids

        return result
