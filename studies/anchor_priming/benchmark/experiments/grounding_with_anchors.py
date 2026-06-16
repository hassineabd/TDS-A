"""Grounding, with-anchors (Set-of-Marks-style visual + coordinate priming).

Tests H2: does priming the model with calibration anchors — both drawn on
the screenshot AND given as coordinate text — improve grounding accuracy?

Compared to `grounding_zero_shot`:
  - Image is augmented: 3 anchor bboxes drawn in magenta with labels.
  - Prompt is prefixed with the anchors' coordinates and descriptions.
  - Evaluation set is reduced: anchors are excluded from the targets.

Anchors are selected ONCE per scene (deterministic spread via
`select_anchor_indices`) and shared by every target in that scene.
The anchor coordinates in the textual preamble are emitted in the same
coord_space as the prompt expects from the model — pixel-absolute for
Claude Opus 4.7, normalised 0-1000 for everyone else.
"""
from __future__ import annotations

from typing import Any, Iterator

from ..coords import (
    BBoxPx, px_to_norm, select_anchor_indices,
)
from ..render import draw_anchors
from .base import Call, Scene
from .grounding_zero_shot import (
    GroundingZeroShot,
    NORMALIZED_PROMPT_TEMPLATE,
    PIXEL_PROMPT_TEMPLATE,
    _check_coord_space,
)


N_ANCHORS = 3


NORMALIZED_PREAMBLE = (
    "This mobile screenshot has {n} reference elements highlighted with magenta "
    "bounding boxes labeled \"Anchor 1\", \"Anchor 2\", \"Anchor 3\". Their "
    "normalized coordinates [y_min, x_min, y_max, x_max] (0-1000 scale) are:\n\n"
    "{anchor_block}\n\n"
    "Use these as visual and coordinate-space references for the rest of the "
    "screenshot. Do NOT return any of these reference elements as your answer.\n\n"
)

PIXEL_PREAMBLE = (
    "This mobile screenshot ({width}x{height} pixels) has {n} reference "
    "elements highlighted with magenta bounding boxes labeled "
    "\"Anchor 1\", \"Anchor 2\", \"Anchor 3\". Their pixel-absolute "
    "coordinates [y_min, x_min, y_max, x_max] are:\n\n"
    "{anchor_block}\n\n"
    "Use these as visual and coordinate-space references for the rest of the "
    "screenshot. Do NOT return any of these reference elements as your answer.\n\n"
)


class GroundingWithAnchors(GroundingZeroShot):
    """Reuses zero-shot's parser and evaluator; only overrides call generation."""
    name = "grounding_with_anchors"

    def iter_calls(
        self, scenes: list[Scene], coord_space: str = "normalized_1000",
    ) -> Iterator[Call]:
        _check_coord_space(coord_space)
        for scene in scenes:
            anchor_idx = select_anchor_indices(scene.targets, n=N_ANCHORS)
            anchors = [scene.targets[i] for i in anchor_idx]
            eval_targets = [t for i, t in enumerate(scene.targets)
                            if i not in set(anchor_idx)]

            # Render scene image once with anchor overlays. The overlays are
            # drawn in pixel space regardless of the prompt's coord_space.
            anchor_bboxes_px: list[BBoxPx] = [tuple(a["bounds"]) for a in anchors]
            anchor_labels = [f"Anchor {i + 1}" for i in range(len(anchors))]
            image_bytes = draw_anchors(
                scene.image_path, anchor_bboxes_px, labels=anchor_labels,
            )

            preamble = self._build_preamble(scene, anchors, coord_space)

            anchor_meta = [
                {"id": a["id"], "bbox_px": list(a["bounds"]),
                 "description": a["description"]}
                for a in anchors
            ]

            for target in eval_targets:
                target_body = self._target_body(scene, target, coord_space)
                prompt = preamble + target_body
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
                        "coord_space": coord_space,
                    },
                )

    @staticmethod
    def _build_preamble(
        scene: Scene, anchors: list[dict], coord_space: str,
    ) -> str:
        lines = []
        for i, a in enumerate(anchors, start=1):
            if coord_space == "pixel_yx":
                x1, y1, x2, y2 = a["bounds"]
                coord_str = f"[{y1}, {x1}, {y2}, {x2}]"
            else:
                ymin, xmin, ymax, xmax = px_to_norm(
                    tuple(a["bounds"]), scene.width, scene.height,
                )
                coord_str = f"[{ymin}, {xmin}, {ymax}, {xmax}]"
            lines.append(
                f"  Anchor {i}: {coord_str}  — \"{a['description']}\""
            )
        block = "\n".join(lines)
        if coord_space == "pixel_yx":
            return PIXEL_PREAMBLE.format(
                n=len(anchors), anchor_block=block,
                width=scene.width, height=scene.height,
            )
        return NORMALIZED_PREAMBLE.format(n=len(anchors), anchor_block=block)

    @staticmethod
    def _target_body(scene: Scene, target: dict, coord_space: str) -> str:
        """Re-emit the actionable instructions only; the preamble already
        framed the screenshot, so we drop the 'You are looking at a mobile
        screenshot' line to avoid redundancy."""
        if coord_space == "pixel_yx":
            template = (
                "Locate the following UI element and return its bounding box:\n\n"
                '  "{description}"\n\n'
                "Respond with ONLY a JSON object in this exact format:\n"
                '  {{"box_2d": [y_min, x_min, y_max, x_max]}}\n\n'
                "Coordinates are in absolute pixels (NOT normalised). "
                "y_min/x_min is the top-left corner of the element, "
                "y_max/x_max is the bottom-right corner. No markdown fences, "
                "no explanation, just the JSON."
            )
        else:
            template = (
                "Locate the following UI element and return its bounding box:\n\n"
                '  "{description}"\n\n'
                "Respond with ONLY a JSON object in this exact format:\n"
                '  {{"box_2d": [y_min, x_min, y_max, x_max]}}\n\n'
                "Coordinates are normalized to 0-1000 (top-left origin). "
                "y_min/x_min is the top-left corner of the element, "
                "y_max/x_max is the bottom-right corner. No markdown fences, "
                "no explanation, just the JSON."
            )
        return template.format(description=target["description"])

    # parse_response and evaluate inherited unchanged from GroundingZeroShot.
