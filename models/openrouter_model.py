"""
OpenRouter model adapter for LangGraph.
"""

import os
from typing import Any, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from models.base import BaseLangGraphModel


class OpenRouterLangGraphModel(BaseLangGraphModel):
    """
    LangGraph adapter for OpenRouter models with tool calling.

    Uses OpenAI-compatible API with custom base URL.
    Supports INTELLECT-3, DeepSeek, Qwen, Kimi, and other OpenRouter models.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        temperature: float = 0.7,
    ):
        super().__init__(model_id=model_id, temperature=temperature)
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._model: ChatOpenAI | None = None

    @property
    def model(self) -> ChatOpenAI:
        """Lazy initialization of the ChatOpenAI model with OpenRouter base URL."""
        if self._model is None:
            self._model = ChatOpenAI(
                model=self.model_id,
                openai_api_key=self._api_key,
                openai_api_base=self.OPENROUTER_BASE_URL,
                temperature=self.temperature,
                default_headers={
                    "HTTP-Referer": "https://github.com/kalshi-agentic",
                    "X-Title": "KalshiAgentic",
                },
            )
        return self._model

    def bind_tools(self, tools: Sequence[Any]) -> "OpenRouterLangGraphModel":
        """
        Bind tools for function calling.

        Args:
            tools: List of LangChain tool objects

        Returns:
            A new model instance with tools bound
        """
        new_model = OpenRouterLangGraphModel(
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
        return "openrouter"
