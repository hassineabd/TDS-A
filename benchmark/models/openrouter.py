"""Unified OpenRouter client - OpenAI-compatible API that proxies to
Anthropic, OpenAI, Google and others via a single API key.
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .base import GroundingModel, RawResponse


class OpenRouterModel(GroundingModel):
    """Vision model accessed through OpenRouter."""

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        provider: str,
        cu_trained: bool,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.provider = provider
        self.cu_trained = cu_trained
        self._client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/hassineabd/TDS-A",
                "X-Title": "TDS-A Mobile Grounding Benchmark",
            },
        )

    def call(
        self,
        image_b64: str,
        media_type: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> RawResponse:
        import time
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
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
        except Exception as exc:
            elapsed = int((time.perf_counter() - start) * 1000)
            return RawResponse(
                text="", latency_ms=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = int((time.perf_counter() - start) * 1000)

        text = resp.choices[0].message.content or ""
        meta: dict[str, Any] = {"model_id": self.model_id}
        usage = resp.usage
        in_tok = out_tok = None
        if usage is not None:
            in_tok = usage.prompt_tokens
            out_tok = usage.completion_tokens
            cost = getattr(usage, "cost", None)
            if cost is not None:
                meta["cost_usd"] = cost
        return RawResponse(
            text=text, latency_ms=elapsed,
            input_tokens=in_tok, output_tokens=out_tok,
            metadata=meta,
        )


# Factory builders — one per (model, provider) we benchmark.
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


def ClaudeOpus47() -> OpenRouterModel:
    """Opus 4.7 ships native 1:1 pixel coordinates (no internal downsampling
    rescale required) — included as a comparison point for the article."""
    return OpenRouterModel(
        name="claude-opus-4.7",
        model_id="anthropic/claude-opus-4.7",
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
