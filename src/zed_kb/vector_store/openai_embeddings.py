"""
OpenAI embeddings implementation for Zed-KB.
Provides integration with OpenAI's embedding models for high-quality text embeddings.
"""

from typing import List, Dict, Any, Optional
import os

from langchain.embeddings.base import Embeddings
import openai


class OpenAIEmbeddings(Embeddings):
    """Implementation of OpenAI's embedding models."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
    ):
        """
        Initialize the OpenAI embedding model.

        Args:
            model: The embedding model identifier, options include:
                  - text-embedding-ada-002 (default)
                  - text-embedding-3-small
                  - text-embedding-3-large
            api_key: OpenAI API key (if not provided, will look for OPENAI_API_KEY env var)
            dimensions: Output dimensions for the embeddings (model-specific)
            encoding_format: The format of the embeddings, "float" or "base64"
        """
        # Set up API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Please provide it as an argument or set the OPENAI_API_KEY environment variable."
            )

        # Configure the OpenAI client
        openai.api_key = self.api_key

        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using OpenAI embedding model.

        Args:
            texts: List of documents to embed

        Returns:
            List of document embedding vectors
        """
        if not texts:
            return []

        # Process in batches to handle potential API limits
        # OpenAI can handle multiple texts in a single API call
        response = openai.Embedding.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format=self.encoding_format,
        )

        # Extract embeddings from the response
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text using OpenAI embedding model.

        Args:
            text: Query text to embed

        Returns:
            Query embedding vector
        """
        if not text:
            return []

        # Process the query
        response = openai.Embedding.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions,
            encoding_format=self.encoding_format,
        )

        # Extract embedding from the response
        embedding = response["data"][0]["embedding"]
        return embedding