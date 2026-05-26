"""Grounding, zero-shot.

Tests the H1 hypothesis: do CU-trained VLMs (Claude 4.x, GPT-4o/4.1) produce
more accurate UI grounding than non-CU-trained ones (Gemini 2.5 Pro)?

Per call: ONE target description, ONE expected bbox. The model returns a
single bbox in the coordinate space it was documented to natively output.

Two coord_spaces are supported and dispatched by the runner based on each
model's declared `coord_space` attribute:

  - 'normalized_1000' (default for Gemini, GPT-4o/4.1, Claude Sonnet 4.5,
    Claude Opus 4.6): [y_min, x_min, y_max, x_max] scaled to 0-1000. Matches
    Gemini's documented native format and lets the client denormalise using
    the original screenshot dimensions, sidestepping provider-specific
    internal downsampling.

  - 'pixel_yx' (Claude Opus 4.7 only): [y_min, x_min, y_max, x_max] in
    pixel-absolute coordinates. Anthropic documents Opus 4.7 as returning
    1:1 pixel coordinates and empirical testing confirms it ignores
    normalisation instructions in the prompt.

Each model is therefore asked for the format that maximises its accuracy
under its own training distribution. This is a deliberate methodological
choice: we measure each model's optimal grounding capability, not its
ability to follow non-native formatting instructions.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

from ..coords import BBoxPx, norm_to_px
from ..metrics import grounding_center_error, grounding_hit, grounding_iou
from .base import Call, Experiment, PredictedItem, Prediction, Scene


NORMALIZED_PROMPT_TEMPLATE = (
    "You are looking at a mobile screenshot.\n\n"
    "Locate the following UI element and return its bounding box:\n\n"
    '  "{description}"\n\n'
    "Respond with ONLY a JSON object in this exact format:\n"
    '  {{"box_2d": [y_min, x_min, y_max, x_max]}}\n\n'
    "Coordinates are normalized to 0-1000 (top-left origin). y_min/x_min is "
    "the top-left corner of the element, y_max/x_max is the bottom-right "
    "corner. No markdown fences, no explanation, just the JSON."
)

PIXEL_PROMPT_TEMPLATE = (
    "You are looking at a mobile screenshot of {width}x{height} pixels "
    "(origin [0,0] is top-left, x goes right, y goes down).\n\n"
    "Locate the following UI element and return its bounding box:\n\n"
    '  "{description}"\n\n'
    "Respond with ONLY a JSON object in this exact format:\n"
    '  {{"box_2d": [y_min, x_min, y_max, x_max]}}\n\n'
    "Coordinates are in absolute pixels (NOT normalised). y_min/x_min is "
    "the top-left corner of the element, y_max/x_max is the bottom-right "
    "corner. No markdown fences, no explanation, just the JSON."
)


VALID_COORD_SPACES = {"normalized_1000", "pixel_yx"}


def _check_coord_space(coord_space: str) -> None:
    if coord_space not in VALID_COORD_SPACES:
        raise ValueError(
            f"Unsupported coord_space {coord_space!r}. "
            f"Valid: {sorted(VALID_COORD_SPACES)}"
        )


class GroundingZeroShot(Experiment):
    name = "grounding_zero_shot"

    def iter_calls(
        self, scenes: list[Scene], coord_space: str = "normalized_1000",
    ) -> Iterator[Call]:
        _check_coord_space(coord_space)
        for scene in scenes:
            for target in scene.targets:
                prompt = self._build_prompt(scene, target, coord_space)
                yield Call(
                    scene_name=scene.name,
                    target_id=target["id"],
                    image=scene.image_path,
                    prompt=prompt,
                    metadata={
                        "target_description": target["description"],
                        "ground_truth_bbox_px": list(target["bounds"]),
                        "size_category": target["size_category"],
                        "area": target["area"],
                        "coord_space": coord_space,
                    },
                )

    @staticmethod
    def _build_prompt(scene: Scene, target: dict, coord_space: str) -> str:
        if coord_space == "pixel_yx":
            return PIXEL_PROMPT_TEMPLATE.format(
                description=target["description"],
                width=scene.width, height=scene.height,
            )
        return NORMALIZED_PROMPT_TEMPLATE.format(
            description=target["description"],
        )

    def parse_response(
        self, raw_text: str, scene: Scene, call: Call,
        coord_space: str | None = None,
    ) -> Prediction:
        """Extract a [y,x,y,x] bbox and convert to pixel space.

        Resolves coord_space from (in order): explicit kwarg, `call.metadata`,
        default 'normalized_1000'. This makes parse_response self-contained
        when re-running analysis from a persisted Call.
        """
        if coord_space is None:
            coord_space = call.metadata.get("coord_space", "normalized_1000")
        _check_coord_space(coord_space)

        s = raw_text.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

        bbox = self._extract_bbox(s)
        if bbox is None:
            return Prediction(parse_error=f"parse_failure: {s[:120]}",
                              raw_text=raw_text[:500])

        y1, x1, y2, x2 = bbox
        if coord_space == "pixel_yx":
            bbox_px: BBoxPx = (x1, y1, x2, y2)
        else:  # normalized_1000
            bbox_px = norm_to_px((y1, x1, y2, x2), scene.width, scene.height)

        return Prediction(
            items=[PredictedItem(bbox_px=bbox_px)],
            raw_text=raw_text[:500],
        )

    @staticmethod
    def _extract_bbox(s: str) -> list[int] | None:
        """Find the [y,x,y,x] quadruple in `s`. Robust to noise: tries strict
        JSON first (extracting from known keys), then falls back to scanning
        for a `[a, b, c, d]` substring with four integers."""
        # Strict JSON parse, look for known keys
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for key in ("box_2d", "bbox", "bounding_box", "box"):
                    if key in obj and isinstance(obj[key], list) and len(obj[key]) == 4:
                        return [int(v) for v in obj[key]]
            if isinstance(obj, list) and len(obj) == 4:
                return [int(v) for v in obj]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: first explicit 4-int list `[a,b,c,d]` in the string.
        # We require the bracket form so we don't accidentally pick up the
        # '2' from a key name like "box_2d".
        m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", s)
        if m:
            try:
                return [int(g) for g in m.groups()]
            except ValueError:
                pass
        return None

    def evaluate(
        self, prediction: Prediction, scene: Scene, call: Call,
    ) -> dict[str, Any]:
        gt_bbox: BBoxPx = tuple(call.metadata["ground_truth_bbox_px"])
        if not prediction.parse_ok:
            return {
                "parse_fail": True,
                "center_error_px": None,
                "hit": False,
                "iou": 0.0,
            }
        pred_bbox = prediction.items[0].bbox_px
        return {
            "parse_fail": False,
            "center_error_px": grounding_center_error(pred_bbox, gt_bbox),
            "hit": grounding_hit(pred_bbox, gt_bbox),
            "iou": grounding_iou(pred_bbox, gt_bbox),
        }
