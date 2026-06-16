"""Provider-agnostic vision LLM client.

A `GroundingModel` is responsible for ONE thing: take an encoded image and a
prompt, return the model's raw text response (plus latency/token metadata).

Prompt construction and response parsing are NOT the model's job — they
belong to `benchmark.experiments`. This separation lets us run multiple
experiments (different prompts, different parsers, different metrics) over
the same set of models without re-touching the API client layer.
"""
from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class RawResponse:
    """A single round-trip with a vision model. No interpretation."""
    text: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def encode_image(
    image: bytes | str | Path,
    jpeg_quality: int = 95,
) -> tuple[str, str]:
    """Encode an image as base64 JPEG.

    Accepts:
      - raw bytes (already in any PIL-readable format) — re-encoded as JPEG
        for consistent size/format across providers
      - a file path (str or Path) — read and re-encoded

    Returns (base64_string, media_type). Always returns 'image/jpeg' so we
    stay under provider size limits and have a single uniform format.

    Resolution is preserved — JPEG re-encoding is lossy on color only.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = Image.open(io.BytesIO(image))

    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return (
        base64.standard_b64encode(buf.getvalue()).decode("utf-8"),
        "image/jpeg",
    )


class GroundingModel(ABC):
    """Subclasses set name/provider/marketed_for_cu/source_url and implement
    `call`.

    On `marketed_for_cu`: this flag captures whether the provider publicly
    markets the model for "computer use" or "agentic UI / GUI" tasks — NOT
    whether the model was demonstrably trained on those tasks. No primary
    source for any of the six models we benchmark explicitly states that
    THIS model was trained on computer-use / GUI-grounding data. The truly
    CU-trained models (OpenAI's `computer-use-preview`, Google's
    `gemini-2.5-computer-use-preview-10-2025`) are SEPARATE model snapshots
    not exposed via OpenRouter. We therefore use a deliberately weaker label
    ("marketed_for_cu") and keep `source_url` next to it so the claim is
    audit-able.

    `coord_space` declares the model's preferred output coordinate convention:
      - "normalized_1000": [y_min, x_min, y_max, x_max] in 0-1000 (Gemini's
        documented native format; default for everyone except Opus 4.7).
      - "pixel_yx": [y_min, x_min, y_max, x_max] in pixel-absolute coords
        (Anthropic documents Claude Opus 4.7 as returning 1:1 pixel coords).
    """

    name: str
    provider: str
    marketed_for_cu: bool
    source_url: str          # primary-source URL backing the marketed_for_cu claim
    cu_training_claim: str   # short quote/paraphrase of what the source actually says
    coord_space: str = "normalized_1000"

    @abstractmethod
    def call(
        self,
        image_b64: str,
        media_type: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> RawResponse:
        """Send (image, prompt) and return the raw text response."""
