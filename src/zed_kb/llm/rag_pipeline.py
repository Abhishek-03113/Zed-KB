"""
RAG (Retrieval Augmented Generation) Pipeline for Zed-KB.
Orchestrates the query → retrieve → generate flow with security awareness.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from langchain.schema import Document
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser

from .gemini_model import GeminiLLM
from .prompt_templates import create_rag_prompt, format_documents
from ..auth.middleware import require_permission, with_auth_context
from ..auth.user_manager import get_user_manager, UserManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the RAG (Retrieval Augmented Generation) pipeline for Zed-KB."""

    def __init__(
        self,
        document_indexer=None,
        llm: Optional[BaseLLM] = None,
        num_results: int = 5,
        security_level: str = "general",
        user_manager: Optional[UserManager] = None,
    ):
        """
        Initialize the RAG Pipeline.

        Args:
            document_indexer: Document indexer instance with search capability
            llm: Language model to use for generation (if None, uses GeminiLLM)
            num_results: Number of documents to retrieve for each query
            security_level: Default security level for prompts ('general' or 'confidential')
            user_manager: Optional UserManager for authorization checks
        """
        # Initialize components
        self.document_indexer = document_indexer
        self.llm = llm or GeminiLLM(model_name="gemini-1.5-pro-latest", temperature=0.2)
        self.num_results = num_results
        self.security_level = security_level
        self.output_parser = StrOutputParser()
        self.user_manager = user_manager or get_user_manager()
        
        # Create prompt template
        self.prompt_template = create_rag_prompt(security_level=self.security_level)

    def _build_retriever_chain(self, user_info: Optional[Dict[str, Any]] = None):
        """
        Build a security-aware retrieval chain that respects user permissions.

        Args:
            user_info: Optional user information for security filtering

        Returns:
            A retrieval function that incorporates security filters
        """
        
        def retrieve(query: str):
            try:
                # Use the document indexer to search with security awareness
                if self.document_indexer:
                    results = self.document_indexer.search(
                        query=query,
                        user_info=user_info,
                        k=self.num_results,
                    )
                    
                    # If the user has an ID, apply post-retrieval permission filtering
                    if user_info and user_info.get("user_id") and results:
                        user_id = user_info["user_id"]
                        
                        # Convert Document objects to dicts for permission checking 
                        doc_dicts = [
                            {
                                "doc_id": doc.metadata.get("doc_id") or doc.metadata.get("document_id", ""),
                                "security_level": doc.metadata.get("security_level", "public"),
                                "allowed_roles": doc.metadata.get("allowed_roles", []),
                                "allowed_users": doc.metadata.get("allowed_users", []),
                            }
                            for doc in results
                        ]
                        
                        # Use UserManager to filter based on permissions
                        allowed_docs = self.user_manager.filter_documents(
                            user_id=user_id,
                            documents=doc_dicts,
                            action="read"
                        )
                        
                        # Extract IDs of allowed documents
                        allowed_ids = [doc.get("doc_id", "") for doc in allowed_docs]
                        
                        # Filter results to only include documents user has permission to access
                        results = [
                            doc for doc in results
                            if (doc.metadata.get("doc_id", "") in allowed_ids or
                                doc.metadata.get("document_id", "") in allowed_ids)
                        ]
                        
                    return results
                else:
                    logger.error("Document indexer not initialized")
                    return []
            except Exception as e:
                logger.error(f"Error during document retrieval: {e}")
                return []
        
        return retrieve

    def _get_effective_security_level(self, user_info: Optional[Dict[str, Any]]) -> str:
        """
        Determine the effective security level based on user clearance.
        
        Args:
            user_info: User information dict with clearance
            
        Returns:
            Effective security level for prompt selection
        """
        if not user_info:
            return self.security_level
            
        # Map clearance levels to numeric values based on schema.json
        clearance_levels = {
            "public": 0,
            "internal": 1,
            "confidential": 2
        }
        
        # Get user's clearance level
        user_clearance = user_info.get("clearance", "public")
        user_level = clearance_levels.get(user_clearance, 0)
        
        # Use confidential prompt for higher security levels
        if user_level >= 2:  # confidential
            return "confidential"
        else:
            return "general"

    @with_auth_context(get_user_id=lambda *args, **kwargs: kwargs.get("user_info", {}).get("user_id"))
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
            user_info: User information for security filtering
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
        
        # Step 3: Set the appropriate security level based on user's clearance
        effective_security = self._get_effective_security_level(user_info)
        prompt = create_rag_prompt(security_level=effective_security)
        
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
    
    @with_auth_context(get_user_id=lambda *args, **kwargs: kwargs.get("user_info", {}).get("user_id"))
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
            user_info: User information for security filtering
            filter_metadata: Additional metadata filters for retrieval

        Returns:
            Dictionary with answer, citations, and retrieved documents
        """
        # Modify the query to request citations
        query_with_citation_request = f"{query} (Please include citations to specific documents in your answer)"
        
        # Run the standard pipeline
        result = self.run(query_with_citation_request, user_info, filter_metadata)
        
        # Get source document IDs for reference
        source_ids = []
        for doc in result.get("documents", []):
            doc_id = doc.metadata.get("doc_id", None) or doc.metadata.get("document_id", None)
            if doc_id:
                source_ids.append(doc_id)
        
        # Add citation data to result
        result["source_ids"] = source_ids
        
        return result