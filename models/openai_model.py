"""
OpenAI GPT model adapter for LangGraph.
"""

import os
from typing import Any, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from models.base import BaseLangGraphModel


class OpenAILangGraphModel(BaseLangGraphModel):
    """
    LangGraph adapter for OpenAI GPT models with tool calling.

    Supports GPT-5.2 and other OpenAI models with function calling.
    """

    def __init__(
        self,
        model_id: str = "gpt-5.2",
        api_key: str | None = None,
        temperature: float = 1.0,
    ):
        super().__init__(model_id=model_id, temperature=temperature)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model: ChatOpenAI | None = None

    @property
    def model(self) -> ChatOpenAI:
        """Lazy initialization of the ChatOpenAI model."""
        if self._model is None:
            self._model = ChatOpenAI(
                model=self.model_id,
                openai_api_key=self._api_key,
                temperature=self.temperature,
            )
        return self._model

    def bind_tools(self, tools: Sequence[Any]) -> "OpenAILangGraphModel":
        """
        Bind tools for function calling.

        Args:
            tools: List of LangChain tool objects

        Returns:
            A new model instance with tools bound
        """
        new_model = OpenAILangGraphModel(
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
        return "openai"
