"""
Configuration for the agentic forecasting benchmark.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgenticAPIConfig:
    """API configuration for agentic benchmark providers."""

    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    xai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("XAI_API_KEY"))
    openrouter_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    serpapi_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("SERPAPI_API_KEY")
    )

    def validate(self, provider: str) -> bool:
        """Check if API key exists for provider."""
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "grok": self.xai_api_key,
            "openrouter": self.openrouter_api_key,
        }
        return bool(key_map.get(provider))

    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for provider."""
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "grok": self.xai_api_key,
            "openrouter": self.openrouter_api_key,
            "serpapi": self.serpapi_api_key,
        }
        return key_map.get(provider)


# Model routing table: friendly name -> (provider, model_id)
AGENTIC_MODEL_ROUTES: dict[str, tuple[str, str]] = {
    # OpenAI
    "gpt-5.2": ("openai", "gpt-5.2"),
    # Anthropic
    "claude-opus-4.5": ("anthropic", "claude-opus-4-5-20251101"),
    # Google Gemini
    "gemini-3-pro": ("gemini", "gemini-3-pro-preview"),
    # XAI Grok
    "grok-4.1-fast": ("grok", "grok-4-1-fast"),
    # OpenRouter models
    "intellect-3": ("openrouter", "prime-intellect/intellect-3"),
    "deepseek-v3.2": ("openrouter", "deepseek/deepseek-v3.2"),
    "kimi-k2": ("openrouter", "moonshotai/kimi-k2-thinking"),
    "kimi-k2.5": ("openrouter", "moonshotai/kimi-k2.5"),
    "trinity-large-preview": ("openrouter", "arcee-ai/trinity-large-preview:free"),
    "qwen3-235b": ("openrouter", "qwen/qwen3-235b-a22b-thinking-2507"),
}

# Temperature settings per model type
MODEL_TEMPERATURES: dict[str, float] = {
    "gpt-5.2": 1.0,  # Reasoning model
    "claude-opus-4.5": 0.7,
    "gemini-3-pro": 1.0,  # Required for Gemini 3
    "grok-4.1-fast": 0.7,
    "intellect-3": 0.7,
    "deepseek-v3.2": 0.7,
    "qwen3-235b": 0.7,
    "kimi-k2.5": 0.7,
    "trinity-large-preview": 0.7,
    "kimi-k2": 0.7,
}

# Default agent configuration
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_MAX_SEARCH_RESULTS = 10
DEFAULT_MAX_RETRIES = 5  # Number of retries per sample on failure (matches kalshi_benchmark)
DEFAULT_RETRY_DELAY = 1.0  # Delay between retries in seconds

# Default benchmark dataset path
DEFAULT_BENCHMARK_DATASET = "dataset/benchmark_dataset_v2.json"

# Valid checkpoints (immutable set for validation)
VALID_CHECKPOINTS = frozenset(["open_plus_1", "pct_25", "pct_50", "pct_75", "close_minus_1"])

# Checkpoint order for display/sorting
CHECKPOINT_ORDER = ["open_plus_1", "pct_25", "pct_50", "pct_75", "close_minus_1"]

# Default checkpoints (all 5)
DEFAULT_CHECKPOINTS = list(CHECKPOINT_ORDER)

# All available models for default run
DEFAULT_MODELS = list(AGENTIC_MODEL_ROUTES.keys())
