"""
Google Gemini model adapter for LangGraph.
"""

import os
from typing import Any, Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage

from models.base import BaseLangGraphModel


class GeminiLangGraphModel(BaseLangGraphModel):
    """
    LangGraph adapter for Google Gemini models with tool calling.

    Uses langchain-google-genai v4+ with ChatGoogleGenerativeAI.
    Gemini 3 handles thought signatures automatically.
    """

    def __init__(
        self,
        model_id: str = "gemini-3-pro-preview",
        api_key: str | None = None,
        temperature: float = 1.0,  # Required for Gemini 3
    ):
        super().__init__(model_id=model_id, temperature=temperature)
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._model: ChatGoogleGenerativeAI | None = None

    @property
    def model(self) -> ChatGoogleGenerativeAI:
        """Lazy initialization of the ChatGoogleGenerativeAI model."""
        if self._model is None:
            self._model = ChatGoogleGenerativeAI(
                model=self.model_id,
                google_api_key=self._api_key,
                temperature=self.temperature,
            )
        return self._model

    def bind_tools(self, tools: Sequence[Any]) -> "GeminiLangGraphModel":
        """
        Bind tools for function calling.

        Gemini 3 handles thought signatures automatically via langchain-google-genai.

        Args:
            tools: List of LangChain tool objects

        Returns:
            A new model instance with tools bound
        """
        new_model = GeminiLangGraphModel(
            model_id=self.model_id,
            api_key=self._api_key,
            temperature=self.temperature,
        )
        new_model._model = self.model.bind_tools(tools)
        new_model._tools = list(tools)
        return new_model

    def invoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        """
        Invoke the model with messages.

        Args:
            messages: List of LangChain message objects

        Returns:
            The model's response message
        """
        return self.model.invoke(messages)

    @property
    def provider(self) -> str:
        return "gemini"
