"""
Gemini embeddings for Zed-KB.
"""

from typing import List, Dict, Any, Optional
import os

import google.generativeai as genai
from langchain.embeddings.base import Embeddings


class GeminiEmbeddings(Embeddings):
    """Google Gemini embedding model."""

    def __init__(
        self,
        model: str = "embedding-001",
        api_key: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
    ):
        """
        Initialize the Gemini embedding model.

        Args:
            model: The embedding model identifier
            api_key: Google AI API key (if not provided, will look for GOOGLE_API_KEY env var)
            task_type: The embedding task type, one of:
                       "RETRIEVAL_QUERY" (for embedding queries)
                       "RETRIEVAL_DOCUMENT" (for embedding documents)
                       "SEMANTIC_SIMILARITY" (for general semantic similarity)
                       "CLASSIFICATION" (for text classification)
                       "CLUSTERING" (for text clustering)
            title: Optional title to include with the document (for RETRIEVAL_DOCUMENT task type)
        """
        # Set up API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Please provide it as an argument or set the GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=self.api_key)

        # Ensure model name has the correct prefix
        if not model.startswith("models/") and not model.startswith("tunedModels/"):
            self.model = f"models/{model}"
        else:
            self.model = model
            
        self.task_type = task_type
        self.title = title

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using Gemini embedding model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for text in texts:
            if self.task_type == "RETRIEVAL_DOCUMENT" and self.title:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=self.task_type,
                    title=self.title,
                )
            else:
                result = genai.embed_content(
                    model=self.model, content=text, task_type=self.task_type
                )

            embeddings.append(result["embedding"])

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using Gemini embedding model.

        Args:
            texts: List of documents to embed

        Returns:
            List of document embedding vectors
        """
        # For document embeddings, use RETRIEVAL_DOCUMENT task type
        original_task_type = self.task_type
        if self.task_type == "RETRIEVAL_QUERY":
            self.task_type = "RETRIEVAL_DOCUMENT"

        try:
            embeddings = self._embed_texts(texts)
        finally:
            # Restore the original task type
            self.task_type = original_task_type

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text using Gemini embedding model.

        Args:
            text: Query text to embed

        Returns:
            Query embedding vector
        """
        # For query embeddings, use RETRIEVAL_QUERY task type
        original_task_type = self.task_type
        if self.task_type == "RETRIEVAL_DOCUMENT":
            self.task_type = "RETRIEVAL_QUERY"

        try:
            embeddings = self._embed_texts([text])
        finally:
            # Restore the original task type
            self.task_type = original_task_type

        return embeddings[0]
