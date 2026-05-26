"""Coordinate-space conversions and geometric primitives for the benchmark.

Coordinate conventions used throughout the codebase:

  - Pixel space: integer (x, y) with origin at top-left, x rightward, y downward.
    Bounding boxes are [x_min, y_min, x_max, y_max] in pixel space.

  - Normalized 0-1000 space: integer coordinates rescaled so the image spans
    [0, 1000] on each axis. Bounding boxes are [y_min, x_min, y_max, x_max]
    (Y-FIRST). This matches Google Gemini's documented coordinate protocol so
    Gemini answers in its native format; other models obey the prompt.

The Y-first ordering applies ONLY to the normalized space. Pixel-space tuples
remain X-first (standard for image processing libraries like PIL).
"""
from __future__ import annotations

import itertools
from typing import Iterable

# Type aliases for readability. Python doesn't enforce these.
BBoxPx = tuple[int, int, int, int]      # [x_min, y_min, x_max, y_max]
BBoxNorm = tuple[int, int, int, int]    # [y_min, x_min, y_max, x_max]
Point = tuple[int, int]                  # (x, y)


# Conversions

def px_to_norm(bbox_px: BBoxPx, width: int, height: int) -> BBoxNorm:
    """Pixel [x_min, y_min, x_max, y_max] -> normalized [y_min, x_min, y_max, x_max]."""
    x1, y1, x2, y2 = bbox_px
    return (
        round(y1 / height * 1000),
        round(x1 / width * 1000),
        round(y2 / height * 1000),
        round(x2 / width * 1000),
    )


def norm_to_px(bbox_norm: BBoxNorm, width: int, height: int) -> BBoxPx:
    """Normalized [y_min, x_min, y_max, x_max] -> pixel [x_min, y_min, x_max, y_max]."""
    y1n, x1n, y2n, x2n = bbox_norm
    return (
        round(x1n / 1000 * width),
        round(y1n / 1000 * height),
        round(x2n / 1000 * width),
        round(y2n / 1000 * height),
    )


# Bounding box utilities (operate on PIXEL bboxes)

def bbox_center(bbox_px: BBoxPx) -> Point:
    """Centroid of a pixel bounding box."""
    x1, y1, x2, y2 = bbox_px
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def bbox_area(bbox_px: BBoxPx) -> int:
    x1, y1, x2, y2 = bbox_px
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a: BBoxPx, b: BBoxPx) -> float:
    """Intersection-over-Union of two pixel bboxes. Returns 0.0 if no overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def point_in_bbox(point: Point, bbox_px: BBoxPx) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox_px
    return x1 <= x <= x2 and y1 <= y <= y2


def euclidean(a: Point, b: Point) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


# Anchor selection — deterministic, runtime, no dataset mutation

def select_anchor_indices(targets: list[dict], n: int = 3) -> list[int]:
    """Pick `n` target indices that maximize the minimum pairwise centroid
    distance. Deterministic given the same input order — no randomness, so
    the same scene always yields the same anchors across experiments.

    Each target dict must expose a 'center' key as [cx, cy] in pixel space.

    If len(targets) <= n, returns list(range(len(targets))) — caller's
    responsibility to handle the degenerate case.
    """
    if len(targets) <= n:
        return list(range(len(targets)))

    centers = [tuple(t["center"]) for t in targets]
    best_combo: tuple[int, ...] | None = None
    best_score = -1.0
    for combo in itertools.combinations(range(len(targets)), n):
        pts = [centers[i] for i in combo]
        # Maximin: maximize the smallest pairwise distance to ensure spread
        min_d = min(euclidean(pts[i], pts[j])
                    for i in range(n) for j in range(i + 1, n))
        if min_d > best_score:
            best_score = min_d
            best_combo = combo
    return list(best_combo) if best_combo else list(range(n))


def split_anchors_and_eval(
    targets: list[dict], n_anchors: int = 3
) -> tuple[list[dict], list[dict]]:
    """Partition `targets` into (anchors, eval_set) using deterministic
    anchor selection. Eval set preserves original order minus anchors."""
    anchor_idx = set(select_anchor_indices(targets, n=n_anchors))
    anchors = [targets[i] for i in sorted(anchor_idx)]
    eval_set = [t for i, t in enumerate(targets) if i not in anchor_idx]
    return anchors, eval_set
