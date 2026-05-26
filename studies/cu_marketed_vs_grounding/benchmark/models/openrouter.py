"""Unified OpenRouter client - OpenAI-compatible API that proxies to
Anthropic, OpenAI, Google and others via a single API key.

For each factory, the `source_url` and `cu_training_claim` fields document
what the provider actually says about this specific model's computer-use
capabilities — they are deliberately separated from a (potentially
misleading) boolean "is it CU-trained?", which no public primary source
supports for any of the six models below.

True CU-trained models (OpenAI's `computer-use-preview`, Google's
`gemini-2.5-computer-use-preview-10-2025`) are separate snapshots NOT
exposed on OpenRouter, so they are not part of this study.
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
        marketed_for_cu: bool,
        source_url: str,
        cu_training_claim: str,
        coord_space: str = "normalized_1000",
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.provider = provider
        self.marketed_for_cu = marketed_for_cu
        self.source_url = source_url
        self.cu_training_claim = cu_training_claim
        self.coord_space = coord_space
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


# Factory builders. The `source_url` + `cu_training_claim` together let any
# reader audit whether our marketed_for_cu flag is fair. None of these
# models is publicly claimed by its provider to be "trained on computer
# use / GUI-grounding data"; the truly CU-trained variants are separate
# snapshots not accessible via OpenRouter.

def ClaudeSonnet45() -> OpenRouterModel:
    return OpenRouterModel(
        name="claude-sonnet-4.5",
        model_id="anthropic/claude-sonnet-4.5",
        provider="anthropic",
        marketed_for_cu=True,
        source_url="https://www.anthropic.com/news/claude-sonnet-4-5",
        cu_training_claim=(
            'Marketing: "strengths in coding, agentic tasks, and computer use." '
            "System card describes only general training-data composition and "
            "prompt-injection RL — no claim of CU-specific pre-training."
        ),
    )


def ClaudeOpus46() -> OpenRouterModel:
    return OpenRouterModel(
        name="claude-opus-4.6",
        model_id="anthropic/claude-opus-4.6",
        provider="anthropic",
        marketed_for_cu=True,
        source_url="https://www.anthropic.com/news/claude-opus-4-6",
        cu_training_claim=(
            "Marketed for agentic / computer use via OSWorld benchmark performance "
            "claims; system card mentions only generic training-data composition."
        ),
    )


def ClaudeOpus47() -> OpenRouterModel:
    """Opus 4.7 emits coordinates in pixel-absolute space, not 0-1000 — Anthropic
    documents 1:1 pixel coords for this model regardless of prompt format,
    so we set `coord_space='pixel_yx'`. Other Claude 4.x models still default
    to 'normalized_1000' since their behaviour is undocumented and empirically
    less stable."""
    return OpenRouterModel(
        name="claude-opus-4.7",
        model_id="anthropic/claude-opus-4.7",
        provider="anthropic",
        marketed_for_cu=True,
        source_url="https://www.anthropic.com/news/claude-opus-4-7",
        cu_training_claim=(
            'Marketed via OSWorld-Verified jump (72.7% → 78.0%); no explicit '
            '"trained on computer use" claim in the announcement or system card.'
        ),
        coord_space="pixel_yx",
    )


def GPT4o() -> OpenRouterModel:
    return OpenRouterModel(
        name="gpt-4o",
        model_id="openai/gpt-4o",
        provider="openai",
        marketed_for_cu=False,  # general-purpose VLM; CUA = separate Operator model
        source_url="https://cdn.openai.com/operator_system_card.pdf",
        cu_training_claim=(
            "OpenAI Operator System Card (Jan 2025) explicitly frames "
            "computer-use as a separate model built on top of GPT-4o: "
            '"Operator combines GPT-4o\'s vision with advanced reasoning '
            'through reinforcement learning." GPT-4o itself is not CU-trained.'
        ),
    )


def GPT41() -> OpenRouterModel:
    return OpenRouterModel(
        name="gpt-4.1",
        model_id="openai/gpt-4.1",
        provider="openai",
        marketed_for_cu=False,  # general-purpose VLM; CUA = separate computer-use-preview
        source_url="https://openai.com/index/gpt-4-1/",
        cu_training_claim=(
            "GPT-4.1 announcement (Apr 2025) focuses on coding, long context, "
            "and instruction-following — no CU training claim. The "
            "`computer-use-preview` model is listed as a separate snapshot in "
            "OpenAI's model catalog."
        ),
    )


def Gemini25Pro() -> OpenRouterModel:
    return OpenRouterModel(
        name="gemini-2.5-pro",
        model_id="google/gemini-2.5-pro",
        provider="google",
        marketed_for_cu=False,  # general-purpose VLM; CU model = separate preview
        source_url=(
            "https://blog.google/innovation-and-ai/models-and-research/"
            "google-deepmind/gemini-computer-use-model/"
        ),
        cu_training_claim=(
            'Google describes "Gemini 2.5 Computer Use" as a "specialized model '
            'built on Gemini 2.5 Pro\'s visual understanding and reasoning '
            'capabilities" — so the base 2.5 Pro is NOT CU-trained; the CU '
            "variant is a separate preview snapshot."
        ),
    )
