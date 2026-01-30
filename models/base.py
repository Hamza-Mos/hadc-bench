"""
Base class for LangGraph model adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

from langchain_core.messages import BaseMessage


class BaseLangGraphModel(ABC):
    """
    Abstract base class for LangGraph-compatible model adapters.

    All model adapters must implement:
    - bind_tools(): Bind tools for function calling
    - invoke(): Execute the model with messages
    - provider property: Return the provider name
    """

    def __init__(self, model_id: str, temperature: float = 0.7):
        self.model_id = model_id
        self.temperature = temperature
        self._tools: list[Any] | None = None

    @abstractmethod
    def bind_tools(self, tools: Sequence[Any]) -> "BaseLangGraphModel":
        """
        Bind tools to the model for function calling.

        Args:
            tools: List of LangChain tool objects

        Returns:
            A new model instance with tools bound
        """
        pass

    @abstractmethod
    def invoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        """
        Invoke the model with a list of messages.

        Args:
            messages: List of LangChain message objects

        Returns:
            The model's response message
        """
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id!r}, temperature={self.temperature})"
