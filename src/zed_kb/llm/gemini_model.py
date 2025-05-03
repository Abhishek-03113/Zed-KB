"""
Gemini 2.5 model integration for Zed-KB.
Provides a LangChain-compatible interface to Google's Gemini 2.5 model.
"""

from typing import List, Dict, Any, Optional, Mapping, ClassVar
import os
import logging

import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """Implementation of Google's Gemini models as a LangChain LLM."""

    # Class attributes for Pydantic V2 compatibility
    model_name: ClassVar[str] = "gemini-1.5-pro-latest"
    temperature: ClassVar[float] = 0.0
    top_p: ClassVar[float] = 0.95
    top_k: ClassVar[int] = 40
    max_output_tokens: ClassVar[int] = 2048
    api_key: ClassVar[Optional[str]] = None
    safety_settings: ClassVar[Optional[List[Dict[str, Any]]]] = None

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro-latest",
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40, 
        max_output_tokens: int = 1024,
        api_key: Optional[str] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Initialize the Gemini LLM.

        Args:
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature between 0.0 and 1.0
            top_p: Nucleus sampling parameter
            top_k: Number of highest probability tokens to keep
            max_output_tokens: Maximum number of tokens to generate
            api_key: Google AI API key (if not provided, will look for GOOGLE_API_KEY env var)
            safety_settings: Optional safety settings for content filtering
        """
        # Initialize the parent class first to ensure Pydantic fields are properly set
        super().__init__(**kwargs)
        
        # Set up API key
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Please provide it as an argument or set the GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=api_key)

        # Store configuration as instance attributes instead of class attributes
        self._model_name = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_output_tokens = max_output_tokens
        self._safety_settings = safety_settings

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_p(self) -> float:
        return self._top_p
        
    @property
    def top_k(self) -> int:
        return self._top_k
        
    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens
        
    @property
    def safety_settings(self) -> Optional[List[Dict[str, Any]]]:
        return self._safety_settings

    def _llm_type(self) -> str:
        return "gemini"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> LLMResult:
        """
        Generate text for a list of prompts.

        Args:
            prompts: List of prompts to generate text for
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional keyword arguments

        Returns:
            LLMResult object containing the generated texts
        """
        generations = []
        for prompt in prompts:
            generation_text = self._call(
                prompt=prompt, 
                stop=stop, 
                run_manager=run_manager, 
                **kwargs
            )
            generations.append([Generation(text=generation_text)])
        
        return LLMResult(generations=generations)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """
        Call the Gemini model with the given prompt.

        Args:
            prompt: The prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional keyword arguments

        Returns:
            Generated text from the model
        """
        try:
            # Create Gemini model
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }

            # Add stop sequences if provided
            if stop:
                generation_config["stop_sequences"] = stop

            # Get the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
            )

            # Generate content
            response = model.generate_content(prompt)

            # Extract and return the text
            if response.text:
                return response.text
            else:
                # If response was blocked by safety filters
                logger.warning("Response was blocked or empty")
                return "I apologize, but I cannot provide a response to that query."

        except Exception as e:
            logger.error(f"Error calling Gemini model: {e}")
            raise

    def with_structured_output(self, schema):
        """Add support for structured output based on a schema (placeholder)."""
        raise NotImplementedError("Structured output is not yet implemented for GeminiLLM")