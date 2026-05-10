"""Experiment registry. Add new experiments by registering them in EXPERIMENTS."""
from __future__ import annotations

from .base import Call, Experiment, PredictedItem, Prediction, Scene
from .grounding_zero_shot import GroundingZeroShot
from .grounding_with_anchors import GroundingWithAnchors


EXPERIMENTS: dict[str, type[Experiment]] = {
    "grounding_zero_shot":    GroundingZeroShot,
    "grounding_with_anchors": GroundingWithAnchors,
}


def get_experiment(name: str) -> Experiment:
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {name!r}. "
            f"Available: {sorted(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]()


__all__ = [
    "Call", "Experiment", "PredictedItem", "Prediction", "Scene",
    "GroundingZeroShot", "GroundingWithAnchors",
    "EXPERIMENTS", "get_experiment",
]
