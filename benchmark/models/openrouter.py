"""Unified OpenRouter client - OpenAI-compatible API that proxies to
Anthropic, OpenAI, Google and others. One API key gets access to all providers.

Model IDs are in the form 'provider/model', e.g. 'anthropic/claude-sonnet-4.5'.
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .base import GroundingModel


class OpenRouterModel(GroundingModel):
    """Unified client for any OpenRouter-hosted model with vision support."""

    def __init__(self, *, name: str, model_id: str, provider: str,
                 cu_trained: bool, max_tokens: int = 2048) -> None:
        # 2048 is generous: leaves headroom for reasoning models (Gemini 2.5
        # Pro, GPT-4o with thinking) that use hidden reasoning tokens before
        # the final JSON. The actual JSON output is ~15 tokens.
        self.name = name
        self.model_id = model_id
        self.provider = provider
        self.cu_trained = cu_trained
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                # Attribution headers recommended by OpenRouter
                "HTTP-Referer": "https://github.com/hassineabd/TDS-A",
                "X-Title": "TDS-A Mobile Grounding Benchmark",
            },
        )

    def _call(self, image_b64: str, prompt: str,
              media_type: str) -> tuple[str, dict[str, Any]]:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}",
                    }},
                ],
            }],
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        meta: dict[str, Any] = {"model_id": self.model_id}
        if usage is not None:
            meta["input_tokens"] = usage.prompt_tokens
            meta["output_tokens"] = usage.completion_tokens
            # OpenRouter exposes cost in response.usage when available
            cost = getattr(usage, "cost", None)
            if cost is not None:
                meta["cost_usd"] = cost
        return text, meta


# Factory builders
def ClaudeSonnet45() -> OpenRouterModel:
    return OpenRouterModel(
        name="claude-sonnet-4.5",
        model_id="anthropic/claude-sonnet-4.5",
        provider="anthropic", cu_trained=True,
    )


def ClaudeOpus46() -> OpenRouterModel:
    return OpenRouterModel(
        name="claude-opus-4.6",
        model_id="anthropic/claude-opus-4.6",
        provider="anthropic", cu_trained=True,
    )


def GPT4o() -> OpenRouterModel:
    return OpenRouterModel(
        name="gpt-4o",
        model_id="openai/gpt-4o",
        provider="openai", cu_trained=True,
    )


def GPT41() -> OpenRouterModel:
    return OpenRouterModel(
        name="gpt-4.1",
        model_id="openai/gpt-4.1",
        provider="openai", cu_trained=True,
    )


def Gemini25Pro() -> OpenRouterModel:
    return OpenRouterModel(
        name="gemini-2.5-pro",
        model_id="google/gemini-2.5-pro",
        provider="google", cu_trained=False,
    )
