"""Model registry for the benchmark (all via OpenRouter)."""
from __future__ import annotations

from .base import GroundingModel, RawResponse, encode_image
from .openrouter import (
    ClaudeOpus46,
    ClaudeOpus47,
    ClaudeSonnet45,
    Gemini25Pro,
    GPT4o,
    GPT41,
    OpenRouterModel,
)


ALL_MODELS: dict[str, callable] = {
    "claude-sonnet-4.5": ClaudeSonnet45,
    "claude-opus-4.6":   ClaudeOpus46,
    "claude-opus-4.7":   ClaudeOpus47,
    "gpt-4o":            GPT4o,
    "gpt-4.1":           GPT41,
    "gemini-2.5-pro":    Gemini25Pro,
}


def available_models() -> list[str]:
    """Return all models if OPENROUTER_API_KEY is set."""
    import os
    if os.environ.get("OPENROUTER_API_KEY"):
        return list(ALL_MODELS.keys())
    return []


__all__ = [
    "GroundingModel", "RawResponse", "encode_image",
    "OpenRouterModel",
    "ClaudeSonnet45", "ClaudeOpus46", "ClaudeOpus47",
    "GPT4o", "GPT41", "Gemini25Pro",
    "ALL_MODELS", "available_models",
]
