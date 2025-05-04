"""
Zed-KB: Secure AI powered internal knowledge base with tiered access and authorization controls.
"""
# Main imports
from .document_processing import DocumentProcessor
from .vector_store import GeminiEmbeddings, OpenAIEmbeddings
from .llm import GeminiLLM, RAGPipeline

__version__ = "0.1.0"