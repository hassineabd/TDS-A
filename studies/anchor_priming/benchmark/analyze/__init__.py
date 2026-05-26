"""Analysis layer — separate per task type because metrics differ.

Grounding analysis aggregates pointwise metrics (center_error, hit, iou)
across (model, size_category, scene). Detection analysis (future) operates
on per-image precision/recall/F1.
"""
