"""
XAI Grok model adapter for LangGraph.
"""

import os
from typing import Any, Sequence

from langchain_xai import ChatXAI
from langchain_core.messages import BaseMessage

from models.base import BaseLangGraphModel


class GrokLangGraphModel(BaseLangGraphModel):
    """
    LangGraph adapter for xAI Grok models with tool calling.

    Uses langchain-xai with ChatXAI. Grok 4.1 Fast is xAI's best agentic model.
    """

    def __init__(
        self,
        model_id: str = "grok-4-1-fast",
        api_key: str | None = None,
        temperature: float = 0.7,
    ):
        super().__init__(model_id=model_id, temperature=temperature)
        self._api_key = api_key or os.getenv("XAI_API_KEY")
        self._model: ChatXAI | None = None

    @property
    def model(self) -> ChatXAI:
        """Lazy initialization of the ChatXAI model."""
        if self._model is None:
            self._model = ChatXAI(
                model=self.model_id,
                xai_api_key=self._api_key,
                temperature=self.temperature,
            )
        return self._model

    def bind_tools(self, tools: Sequence[Any]) -> "GrokLangGraphModel":
        """
        Bind tools for function calling.

        Grok 4.1 Fast supports tool calling natively.

        Args:
            tools: List of LangChain tool objects

        Returns:
            A new model instance with tools bound
        """
        new_model = GrokLangGraphModel(
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
        return "grok"
