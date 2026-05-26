"""Parse UiAutomator2 XML page source into benchmark targets with ground truth bounds.

Material Design guideline: min touch target = 48dp = ~144px on Pixel 7 (xxhdpi).
48 * 48 = 2304 px^2 minimum recommended.
We classify targets by bounding box area:
  - small  : area < 3600 px^2   (below Material recommendation; hard cases)
  - medium : 3600 <= area < 22500 px^2 (typical icons/buttons)
  - large  : area >= 22500 px^2 (images, cards, full-width buttons)
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

SMALL_MAX = 3600
MEDIUM_MAX = 22_500


def parse_bounds(bounds_str: str) -> tuple[int, int, int, int] | None:
    """Parse '[x1,y1][x2,y2]' format -> (x1, y1, x2, y2). Returns None if invalid."""
    m = BOUNDS_RE.match(bounds_str or "")
    if not m:
        return None
    return tuple(int(v) for v in m.groups())  # type: ignore


def size_category(width: int, height: int) -> str:
    area = width * height
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def element_description(elem: ET.Element) -> str:
    """Pick the most informative human-readable description for an element.

    Priority: text > content-desc > resource-id (last segment).
    """
    text = (elem.get("text") or "").strip()
    if text:
        return text
    cd = (elem.get("content-desc") or "").strip()
    if cd:
        return cd
    rid = (elem.get("resource-id") or "").strip()
    if rid and "/" in rid:
        return rid.split("/")[-1].replace("_", " ")
    return ""


def is_useful(elem: ET.Element) -> bool:
    """Keep only elements that a user could reasonably target by description."""
    if not element_description(elem):
        return False
    if elem.get("displayed") == "false":
        return False
    bounds = parse_bounds(elem.get("bounds") or "")
    if not bounds:
        return False
    x1, y1, x2, y2 = bounds
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return False
    return True


def extract_targets(xml_path: Path, screenshot_id: str) -> list[dict]:
    """Parse XML and return a list of candidate targets. Caller should manually
    curate the final targets list for the benchmark."""
    root = ET.parse(xml_path).getroot()
    targets: list[dict] = []
    seen_bounds: set[tuple[int, int, int, int]] = set()

    for idx, elem in enumerate(root.iter()):
        if not is_useful(elem):
            continue
        bounds = parse_bounds(elem.get("bounds"))
        assert bounds
        if bounds in seen_bounds:
            continue  # dedup parent/child with identical bounds
        seen_bounds.add(bounds)

        x1, y1, x2, y2 = bounds
        width, height = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        desc = element_description(elem)
        clickable = elem.get("clickable") == "true"

        targets.append(
            {
                "id": f"{screenshot_id}_{idx}",
                "description": desc,
                "bounds": [x1, y1, x2, y2],
                "center": [cx, cy],
                "width": width,
                "height": height,
                "area": width * height,
                "size_category": size_category(width, height),
                "clickable": clickable,
                "class": elem.get("class", ""),
            }
        )

    return targets


def curate(targets: list[dict], max_targets: int = 8) -> list[dict]:
    """Pick a balanced mix across size categories and prefer clickable elements."""
    by_cat: dict[str, list[dict]] = {"small": [], "medium": [], "large": []}
    for t in targets:
        by_cat[t["size_category"]].append(t)

    # Prefer clickable within each category
    for cat in by_cat:
        by_cat[cat].sort(key=lambda t: (not t["clickable"], -t["area"]))

    picked: list[dict] = []
    per_cat = max(1, max_targets // 3)
    for cat in ("small", "medium", "large"):
        picked.extend(by_cat[cat][:per_cat])

    # Top up if under budget
    for cat in ("medium", "large", "small"):
        while len(picked) < max_targets and len(by_cat[cat]) > per_cat:
            extra = by_cat[cat][per_cat]
            per_cat += 1
            if extra not in picked:
                picked.append(extra)
            if len(picked) >= max_targets:
                break

    return picked[:max_targets]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", type=Path, help="UiAutomator2 page source XML")
    parser.add_argument("--screenshot-id", required=True,
                        help="Short id to prefix target ids, e.g. 'clock_pixel7'")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON path for curated targets")
    parser.add_argument("--raw-out", type=Path, default=None,
                        help="Optional output for ALL parsed targets (pre-curation)")
    parser.add_argument("--max-targets", type=int, default=8)
    args = parser.parse_args()

    all_targets = extract_targets(args.xml, args.screenshot_id)
    print(f"Parsed {len(all_targets)} candidate targets from {args.xml}")
    counts = {c: sum(1 for t in all_targets if t["size_category"] == c)
              for c in ("small", "medium", "large")}
    print(f"  By size: {counts}")

    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(json.dumps(all_targets, indent=2))
        print(f"  Raw candidates -> {args.raw_out}")

    curated = curate(all_targets, max_targets=args.max_targets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curated, indent=2))
    print(f"Curated {len(curated)} targets -> {args.out}")
    for t in curated:
        print(f"  [{t['size_category']:6}] {t['description'][:50]:50} "
              f"center={t['center']} area={t['area']}")


if __name__ == "__main__":
    main()
