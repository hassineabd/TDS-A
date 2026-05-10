"""Experiment ABC and shared dataclasses.

Architecture:

  An `Experiment` is a fully-specified protocol for evaluating models on a
  benchmark task. It owns:
    - prompt construction (zero-shot, with-anchors, with-examples, ...)
    - image preparation (passthrough, anchor overlay, ...)
    - response parsing (bbox 0-1000, list of bboxes, ...)
    - per-call evaluation (returns a metrics dict)

  The runner is generic: it iterates the experiment's calls, dispatches them
  to models, and stores the parsed predictions + metrics. Adding a new
  experiment = subclassing `Experiment` and registering it in
  `experiments.__init__`.

  Two types of experiments share this base:
    - Grounding: one call per (scene, target). One predicted bbox per call.
    - Detection: one call per scene. List of predicted bboxes per call.

  We keep `Prediction.items` as a list to handle both uniformly.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from ..coords import BBoxPx


# Data primitives

@dataclass
class Scene:
    """A screenshot with its curated targets and dimensions."""
    name: str
    image_path: Path
    width: int
    height: int
    targets: list[dict]                # raw targets.json entries, untouched

    @classmethod
    def load_from(cls, data_dir: Path) -> list["Scene"]:
        """Discover all <name>.png with matching <name>.targets.json."""
        scenes = []
        for png in sorted(data_dir.glob("*.png")):
            name = png.stem
            targets_path = data_dir / f"{name}.targets.json"
            if not targets_path.exists():
                continue
            with Image.open(png) as im:
                w, h = im.size
            targets = json.loads(targets_path.read_text())
            scenes.append(cls(
                name=name, image_path=png, width=w, height=h, targets=targets,
            ))
        return scenes


@dataclass
class Call:
    """One unit of work to dispatch to a model.

    The image is supplied as either a Path (passthrough — the model client
    encodes the file directly) or bytes (already-rendered image, e.g. with
    overlays from a SoM-style experiment). `metadata` carries experiment-
    specific context that should be persisted alongside the prediction.
    """
    scene_name: str
    target_id: str | None              # None for detection (whole-scene call)
    image: Path | bytes
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictedItem:
    """One bbox returned by the model after parsing.

    `description` is only set for detection (model lists what each item is);
    grounding leaves it None because the prompt fixed the target description.
    """
    bbox_px: BBoxPx
    description: str | None = None


@dataclass
class Prediction:
    """The model's parsed output. `items` is empty on parse failure."""
    items: list[PredictedItem] = field(default_factory=list)
    parse_error: str | None = None
    raw_text: str = ""

    @property
    def parse_ok(self) -> bool:
        return self.parse_error is None and len(self.items) > 0


# Experiment ABC

class Experiment(ABC):
    """Base class. Subclasses set `name` and implement the three methods."""

    name: str

    @abstractmethod
    def iter_calls(self, scenes: list[Scene]) -> Iterator[Call]:
        """Yield one Call per unit of work for this experiment."""

    @abstractmethod
    def parse_response(self, raw_text: str, scene: Scene, call: Call) -> Prediction:
        """Convert raw model output into pixel-space PredictedItems."""

    @abstractmethod
    def evaluate(
        self, prediction: Prediction, scene: Scene, call: Call,
    ) -> dict[str, Any]:
        """Per-call metrics. Returned dict is serialized into the result row."""

    def total_calls(self, scenes: list[Scene]) -> int:
        """Default count by materializing iter_calls. Subclasses can override
        for a faster O(1) estimate without building images."""
        return sum(1 for _ in self.iter_calls(scenes))
