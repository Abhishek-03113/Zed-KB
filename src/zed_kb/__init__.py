"""
Zed-KB: Secure AI powered internal knowledge base with tiered access and authorization controls.

This package provides secure document processing, embedding, and retrieval capabilities with
built-in access controls and authorization mechanisms.
"""

# Import key modules for easy access
from .document_processing import DocumentProcessor
from .vector_store import GeminiEmbeddings, OpenAIEmbeddings, AstraDBStore

__version__ = "0.1.0"