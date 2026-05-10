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
    """Subclasses set name/cu_trained/provider and implement `call`."""

    name: str
    cu_trained: bool
    provider: str

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
