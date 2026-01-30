"""
Model registry for agentic forecasting benchmark.
"""

from typing import Optional

from config import (
    AGENTIC_MODEL_ROUTES,
    MODEL_TEMPERATURES,
    AgenticAPIConfig,
)
from models.base import BaseLangGraphModel
from models.anthropic_model import AnthropicLangGraphModel
from models.openai_model import OpenAILangGraphModel
from models.gemini_model import GeminiLangGraphModel
from models.grok_model import GrokLangGraphModel
from models.openrouter_model import OpenRouterLangGraphModel


def get_agentic_model(
    model_name: str,
    temperature: Optional[float] = None,
    api_config: Optional[AgenticAPIConfig] = None,
) -> BaseLangGraphModel:
    """
    Factory function to get a LangGraph model adapter by name.

    Args:
        model_name: Friendly model name (e.g., 'claude-opus-4.5', 'gpt-5.2-xhigh')
        temperature: Optional temperature override
        api_config: Optional API configuration with keys

    Returns:
        A BaseLangGraphModel instance for the requested model

    Raises:
        ValueError: If model_name is not found in AGENTIC_MODEL_ROUTES
        ImportError: If required dependencies are not installed
    """
    if model_name not in AGENTIC_MODEL_ROUTES:
        available = ", ".join(sorted(AGENTIC_MODEL_ROUTES.keys()))
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )

    provider, model_id = AGENTIC_MODEL_ROUTES[model_name]
    api_config = api_config or AgenticAPIConfig()

    # Use model-specific temperature if not overridden
    if temperature is None:
        temperature = MODEL_TEMPERATURES.get(model_name, 0.7)

    # Get API key for provider
    api_key = api_config.get_key(provider)

    if provider == "anthropic":

        return AnthropicLangGraphModel(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
        )

    elif provider == "openai":

        return OpenAILangGraphModel(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
        )

    elif provider == "gemini":

        return GeminiLangGraphModel(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
        )

    elif provider == "grok":

        return GrokLangGraphModel(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
        )

    elif provider == "openrouter":

        return OpenRouterLangGraphModel(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def list_available_models() -> list[str]:
    """Return list of available model names."""
    return sorted(AGENTIC_MODEL_ROUTES.keys())


def get_model_info(model_name: str) -> dict:
    """Get information about a model."""
    if model_name not in AGENTIC_MODEL_ROUTES:
        raise ValueError(f"Unknown model: {model_name}")

    provider, model_id = AGENTIC_MODEL_ROUTES[model_name]
    return {
        "name": model_name,
        "provider": provider,
        "model_id": model_id,
        "temperature": MODEL_TEMPERATURES.get(model_name, 0.7),
    }
