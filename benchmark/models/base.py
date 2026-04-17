"""Common interface for mobile grounding models.

Every model receives the SAME prompt and the SAME raw screenshot (no
preprocessing on our side) and must return pixel (x, y) coordinates in the
ORIGINAL screenshot coordinate space. This deliberately tests each model's
*native* grounding ability on real mobile screenshots, including how well they
handle any internal resizing/tiling the provider applies.
"""
from __future__ import annotations

import base64
import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


GROUNDING_PROMPT_TEMPLATE = (
    "You are looking at a mobile screenshot of {width}x{height} pixels "
    "(origin [0,0] is top-left, x goes right, y goes down).\n\n"
    "Return the exact pixel coordinates [x, y] of the CENTER of the following UI element:\n\n"
    '"{description}"\n\n'
    "Respond with ONLY a JSON object in this exact format and nothing else "
    "(no markdown fences, no explanation):\n"
    '{{"x": <integer>, "y": <integer>}}'
)


@dataclass
class Prediction:
    x: int                           # in ORIGINAL screenshot coords
    y: int
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_response: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def encode_image(image_path: str | Path, jpeg_quality: int = 95) -> tuple[str, str]:
    """Encode image at full resolution (no downsampling).

    We convert PNG -> JPEG (quality 95) to stay under Anthropic's 5 MB
    base64 limit on raw PNGs. This is lossy only in color compression, not
    resolution - every pixel is preserved at its original position.

    Returns (base64_string, media_type).
    """
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return (base64.standard_b64encode(buf.getvalue()).decode("utf-8"),
            "image/jpeg")


class GroundingModel(ABC):
    """Base class. Subclasses must set name, cu_trained, provider and
    implement _call(image_b64, prompt)."""

    name: str
    cu_trained: bool
    provider: str

    def predict(self, image_path: str | Path, description: str,
                width: int, height: int) -> Prediction:
        """width/height = original screenshot dimensions; fed verbatim to the model."""
        img_b64, media_type = encode_image(image_path)
        prompt = GROUNDING_PROMPT_TEMPLATE.format(
            width=width, height=height, description=description,
        )
        start = time.perf_counter()
        try:
            raw, meta = self._call(img_b64, prompt, media_type)
        except Exception as exc:
            elapsed = int((time.perf_counter() - start) * 1000)
            return Prediction(x=-1, y=-1, latency_ms=elapsed, raw_response="",
                              error=f"{type(exc).__name__}: {exc}")
        elapsed = int((time.perf_counter() - start) * 1000)

        px, py, parse_err = parse_coordinates(raw)
        return Prediction(
            x=px, y=py,
            latency_ms=elapsed,
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            raw_response=raw[:500],
            error=parse_err,
            metadata=meta,
        )

    @abstractmethod
    def _call(self, image_b64: str, prompt: str,
              media_type: str) -> tuple[str, dict[str, Any]]:
        """Send b64 image + prompt. Return (raw_text_response, meta)."""


def parse_coordinates(raw: str) -> tuple[int, int, str | None]:
    """Extract (x, y) from a JSON-like string. Very tolerant to noise."""
    import json
    import re

    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "x" in obj and "y" in obj:
            return int(obj["x"]), int(obj["y"]), None
    except Exception:
        pass

    m = re.search(r"\{[^{}]*\"x\"[^{}]*\"y\"[^{}]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            return int(obj["x"]), int(obj["y"]), None
        except Exception:
            pass
    m = re.search(r"\{[^{}]*\"y\"[^{}]*\"x\"[^{}]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            return int(obj["x"]), int(obj["y"]), None
        except Exception:
            pass

    nums = re.findall(r"-?\d+", s)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1]), f"loose_parse: {s[:80]}"

    return -1, -1, f"parse_failure: {s[:120]}"
