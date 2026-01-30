"""
Anthropic Claude model adapter for LangGraph.
"""

import os
from typing import Any, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from models.base import BaseLangGraphModel


class AnthropicLangGraphModel(BaseLangGraphModel):
    """
    LangGraph adapter for Anthropic Claude models with tool calling.
    """

    def __init__(
        self,
        model_id: str = "claude-opus-4-5-20251101",
        api_key: str | None = None,
        temperature: float = 0.7,
    ):
        super().__init__(model_id=model_id, temperature=temperature)
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model: ChatAnthropic | None = None

    @property
    def model(self) -> ChatAnthropic:
        """Lazy initialization of the ChatAnthropic model."""
        if self._model is None:
            self._model = ChatAnthropic(
                model=self.model_id,
                anthropic_api_key=self._api_key,
                temperature=self.temperature,
                # No max_tokens - LangChain auto-sets from model profile (64K for Opus 4.5)
            )
        return self._model

    def bind_tools(self, tools: Sequence[Any]) -> "AnthropicLangGraphModel":
        """
        Bind tools for function calling.

        Args:
            tools: List of LangChain tool objects

        Returns:
            A new model instance with tools bound
        """
        new_model = AnthropicLangGraphModel(
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
        return "anthropic"
