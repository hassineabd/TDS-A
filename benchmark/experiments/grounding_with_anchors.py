"""Grounding, with-anchors (Set-of-Marks-style visual + coordinate priming).

Tests H2: does priming the model with calibration anchors — both drawn on
the screenshot AND given as coordinate text — improve grounding accuracy?

Compared to `grounding_zero_shot`:
  - Image is augmented: 3 anchor bboxes drawn in magenta with labels.
  - Prompt is prefixed with the anchors' coordinates and descriptions.
  - Evaluation set is reduced: anchors are excluded from the targets.

Anchors are selected ONCE per scene (deterministic spread via `select_anchor_indices`)
and shared by every target in that scene. Anchor diversification per target is
left to a future experiment.
"""
from __future__ import annotations

from typing import Any, Iterator

from ..coords import (
    BBoxPx, px_to_norm, select_anchor_indices,
)
from ..render import draw_anchors
from .base import Call, Scene
from .grounding_zero_shot import GroundingZeroShot


N_ANCHORS = 3


ANCHOR_PREAMBLE = (
    "This mobile screenshot has {n} reference elements highlighted with magenta "
    "bounding boxes labeled \"Anchor 1\", \"Anchor 2\", \"Anchor 3\". Their "
    "normalized coordinates [y_min, x_min, y_max, x_max] (0-1000 scale) are:\n\n"
    "{anchor_block}\n\n"
    "Use these as visual and coordinate-space references for the rest of the "
    "screenshot. Do NOT return any of these reference elements as your answer.\n\n"
)


class GroundingWithAnchors(GroundingZeroShot):
    """Reuses zero-shot's parser and evaluator; only overrides call generation."""
    name = "grounding_with_anchors"

    def iter_calls(self, scenes: list[Scene]) -> Iterator[Call]:
        for scene in scenes:
            anchor_idx = select_anchor_indices(scene.targets, n=N_ANCHORS)
            anchors = [scene.targets[i] for i in anchor_idx]
            eval_targets = [t for i, t in enumerate(scene.targets)
                            if i not in set(anchor_idx)]

            # Render scene image once with anchor overlays
            anchor_bboxes_px: list[BBoxPx] = [tuple(a["bounds"]) for a in anchors]
            anchor_labels = [f"Anchor {i + 1}" for i in range(len(anchors))]
            image_bytes = draw_anchors(
                scene.image_path, anchor_bboxes_px, labels=anchor_labels,
            )

            preamble = self._build_preamble(scene, anchors)
            base_template = self._zero_shot_body()

            anchor_meta = [
                {"id": a["id"], "bbox_px": list(a["bounds"]),
                 "description": a["description"]}
                for a in anchors
            ]

            for target in eval_targets:
                prompt = preamble + base_template.format(
                    description=target["description"],
                )
                yield Call(
                    scene_name=scene.name,
                    target_id=target["id"],
                    image=image_bytes,
                    prompt=prompt,
                    metadata={
                        "target_description": target["description"],
                        "ground_truth_bbox_px": list(target["bounds"]),
                        "size_category": target["size_category"],
                        "area": target["area"],
                        "anchors": anchor_meta,
                    },
                )

    @staticmethod
    def _build_preamble(scene: Scene, anchors: list[dict]) -> str:
        lines = []
        for i, a in enumerate(anchors, start=1):
            ymin, xmin, ymax, xmax = px_to_norm(
                tuple(a["bounds"]), scene.width, scene.height,
            )
            lines.append(
                f"  Anchor {i}: [{ymin}, {xmin}, {ymax}, {xmax}]  "
                f"— \"{a['description']}\""
            )
        return ANCHOR_PREAMBLE.format(n=len(anchors), anchor_block="\n".join(lines))

    @staticmethod
    def _zero_shot_body() -> str:
        """Trim PROMPT_TEMPLATE to its target-locating body. We re-emit just
        the actionable instructions because the preamble already framed the
        screenshot; restating 'You are looking at a mobile screenshot' would
        be redundant and might dilute the anchor framing."""
        return (
            "Locate the following UI element and return its bounding box:\n\n"
            '  "{description}"\n\n'
            "Respond with ONLY a JSON object in this exact format:\n"
            '  {{"box_2d": [y_min, x_min, y_max, x_max]}}\n\n'
            "Coordinates are normalized to 0-1000 (top-left origin). "
            "y_min/x_min is the top-left corner of the element, "
            "y_max/x_max is the bottom-right corner. No markdown fences, "
            "no explanation, just the JSON."
        )

    # parse_response and evaluate inherited unchanged from GroundingZeroShot.
