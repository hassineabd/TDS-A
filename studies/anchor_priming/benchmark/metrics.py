"""Metrics for grounding and detection experiments.

Two families, kept separate because the unit of comparison differs:

  Grounding (1 target per call) — metrics computed pointwise over a list of
  (predicted_bbox, ground_truth_bbox) pairs. Aggregation (mean, by-model,
  by-size) is the responsibility of the analysis layer.

  Detection (N targets, M predictions per call) — metrics require a matching
  step (greedy IoU) before precision/recall/IoU can be computed.

All inputs are in PIXEL space. Convert before calling.
"""
from __future__ import annotations

from .coords import BBoxPx, Point, bbox_center, bbox_iou, euclidean, point_in_bbox


# Grounding metrics (pointwise)

def grounding_center_error(pred_bbox: BBoxPx, gt_bbox: BBoxPx) -> float:
    """Euclidean distance between predicted and ground truth centroids (px)."""
    return euclidean(bbox_center(pred_bbox), bbox_center(gt_bbox))


def grounding_hit(pred_bbox: BBoxPx, gt_bbox: BBoxPx) -> bool:
    """True iff the predicted centroid falls inside the ground truth bbox.

    This is the practical 'would the agent click the right element' check —
    an agent acts on the bbox center, so what matters is whether that center
    lands somewhere on the target.
    """
    return point_in_bbox(bbox_center(pred_bbox), gt_bbox)


def grounding_iou(pred_bbox: BBoxPx, gt_bbox: BBoxPx) -> float:
    """IoU between predicted and ground-truth bboxes."""
    return bbox_iou(pred_bbox, gt_bbox)


# Detection metrics (require matching)

def greedy_match(
    predictions: list[BBoxPx],
    ground_truths: list[BBoxPx],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy 1:1 matching between predictions and ground truths by IoU.

    Strategy: compute all (pred_idx, gt_idx, iou) triples, sort by IoU
    descending, and assign greedily — each prediction and each GT can only
    appear in one match. Matches with iou < threshold are rejected.

    Returns:
      matched: list of (pred_idx, gt_idx, iou)
      unmatched_preds: indices of predictions with no match (false positives)
      unmatched_gts: indices of ground truths with no match (missed)
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(predictions):
        for gi, g in enumerate(ground_truths):
            iou = bbox_iou(p, g)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)

    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, pi, gi in candidates:
        if pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        matched.append((pi, gi, iou))

    unmatched_preds = [i for i in range(len(predictions)) if i not in used_preds]
    unmatched_gts = [i for i in range(len(ground_truths)) if i not in used_gts]
    return matched, unmatched_preds, unmatched_gts


def precision_recall_f1(
    n_matched: int, n_predictions: int, n_ground_truths: int,
) -> tuple[float, float, float]:
    """Standard precision/recall/F1 from match counts. All zeros if denominators
    are zero."""
    precision = n_matched / n_predictions if n_predictions > 0 else 0.0
    recall = n_matched / n_ground_truths if n_ground_truths > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def detection_metrics(
    predictions: list[BBoxPx],
    ground_truths: list[BBoxPx],
    iou_threshold: float = 0.5,
) -> dict:
    """All-in-one detection metrics for a single image's predictions.

    Returns:
      {
        precision, recall, f1,
        mean_iou_matched: average IoU over matched pairs (NaN if no matches),
        mean_center_error_matched: average centroid distance over matched
          pairs in pixels (NaN if no matches),
        n_predictions, n_ground_truths, n_matched,
      }
    """
    matched, unmatched_p, unmatched_gt = greedy_match(
        predictions, ground_truths, iou_threshold=iou_threshold,
    )
    p, r, f1 = precision_recall_f1(
        n_matched=len(matched),
        n_predictions=len(predictions),
        n_ground_truths=len(ground_truths),
    )

    if matched:
        mean_iou = sum(m[2] for m in matched) / len(matched)
        mean_ctr_err = sum(
            euclidean(bbox_center(predictions[pi]),
                      bbox_center(ground_truths[gi]))
            for pi, gi, _ in matched
        ) / len(matched)
    else:
        mean_iou = float("nan")
        mean_ctr_err = float("nan")

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "mean_iou_matched": mean_iou,
        "mean_center_error_matched": mean_ctr_err,
        "n_predictions": len(predictions),
        "n_ground_truths": len(ground_truths),
        "n_matched": len(matched),
    }
