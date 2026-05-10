"""Grounding, zero-shot.

Tests the H1 hypothesis: do CU-trained VLMs (Claude 4.x, GPT-4o/4.1) produce
more accurate UI grounding than non-CU-trained ones (Gemini 2.5 Pro)?

Per call: ONE target description, ONE expected bbox. The model must output
a single bbox in normalized 0-1000 [y_min, x_min, y_max, x_max] space.
We use this protocol (rather than pixel-absolute) for two reasons:
  1) It matches Gemini's documented native format — fair to its training.
  2) It bypasses provider-specific internal downsampling (Claude's 1568-px
     limit, GPT's 768-shortest-side rescale): coordinates round-trip
     correctly regardless of how the provider preprocesses the image.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

from ..coords import BBoxPx, norm_to_px
from ..metrics import grounding_center_error, grounding_hit, grounding_iou
from .base import Call, Experiment, PredictedItem, Prediction, Scene


PROMPT_TEMPLATE = (
    "You are looking at a mobile screenshot.\n\n"
    "Locate the following UI element and return its bounding box:\n\n"
    '  "{description}"\n\n'
    "Respond with ONLY a JSON object in this exact format:\n"
    '  {{"box_2d": [y_min, x_min, y_max, x_max]}}\n\n'
    "Coordinates are normalized to 0-1000 (top-left origin). y_min/x_min is "
    "the top-left corner of the element, y_max/x_max is the bottom-right "
    "corner. No markdown fences, no explanation, just the JSON."
)


class GroundingZeroShot(Experiment):
    name = "grounding_zero_shot"

    def iter_calls(self, scenes: list[Scene]) -> Iterator[Call]:
        for scene in scenes:
            for target in scene.targets:
                prompt = PROMPT_TEMPLATE.format(description=target["description"])
                yield Call(
                    scene_name=scene.name,
                    target_id=target["id"],
                    image=scene.image_path,         # passthrough, no overlay
                    prompt=prompt,
                    metadata={
                        "target_description": target["description"],
                        "ground_truth_bbox_px": list(target["bounds"]),
                        "size_category": target["size_category"],
                        "area": target["area"],
                    },
                )

    def parse_response(
        self, raw_text: str, scene: Scene, call: Call,
    ) -> Prediction:
        """Extract a normalized [y,x,y,x] bbox and convert to pixel space.

        Tolerant: strips markdown fences, accepts the canonical key `box_2d`
        as well as common variants (`bbox`, `bounding_box`, `box`). Falls
        back to the first 4 integers it finds in the response.
        """
        s = raw_text.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

        bbox_norm = self._extract_bbox(s)
        if bbox_norm is None:
            return Prediction(parse_error=f"parse_failure: {s[:120]}",
                              raw_text=raw_text[:500])

        bbox_px: BBoxPx = norm_to_px(
            tuple(bbox_norm), scene.width, scene.height,
        )
        return Prediction(
            items=[PredictedItem(bbox_px=bbox_px)],
            raw_text=raw_text[:500],
        )

    @staticmethod
    def _extract_bbox(s: str) -> list[int] | None:
        """Find the [y,x,y,x] quadruple in `s`. Returns None on failure."""
        # First try strict JSON
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

        # Loose fallback: first 4 integers in the string
        nums = re.findall(r"-?\d+", s)
        if len(nums) >= 4:
            try:
                return [int(n) for n in nums[:4]]
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
