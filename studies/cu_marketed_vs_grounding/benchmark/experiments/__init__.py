"""Experiment registry — `cu_marketed_vs_grounding` study.

This study tests one experiment only: how six 'marketed-for-CU' VLMs
perform on zero-shot mobile UI grounding when each is queried in its
documented native coordinate space.
"""
from __future__ import annotations

from .base import Call, Experiment, PredictedItem, Prediction, Scene
from .grounding_zero_shot import GroundingZeroShot


EXPERIMENTS: dict[str, type[Experiment]] = {
    "grounding_zero_shot": GroundingZeroShot,
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
    "GroundingZeroShot",
    "EXPERIMENTS", "get_experiment",
]
