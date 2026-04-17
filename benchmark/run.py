"""Benchmark orchestrator.

Loops over (model x screenshot x target x run) and records predictions.
Saves results incrementally to `results/raw_predictions.json` so we can
resume after API failures.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from .models import ALL_MODELS, GroundingModel, available_models


def euclidean(ax: int, ay: int, bx: int, by: int) -> float:
    return math.hypot(ax - bx, ay - by)


def inside_bounds(bounds: list[int], x: int, y: int) -> bool:
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def load_screenshots(data_dir: Path) -> list[dict]:
    """Discover all <name>.{png,xml,targets.json} triples in `data_dir`."""
    scenes = []
    for png in sorted(data_dir.glob("*.png")):
        name = png.stem
        targets_path = data_dir / f"{name}.targets.json"
        if not targets_path.exists():
            print(f"  skip {name}: no targets.json")
            continue
        with Image.open(png) as im:
            w, h = im.size
        targets = json.loads(targets_path.read_text())
        scenes.append(
            {"name": name, "image": png, "width": w, "height": h,
             "targets": targets}
        )
    return scenes


def load_existing(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    return []


def save_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def already_done(results: list[dict], key: tuple) -> bool:
    for r in results:
        k = (r["model"], r["screenshot"], r["target_id"], r["run"])
        if k == key:
            return True
    return False


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/raw_predictions.json"))
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model names; defaults to all available")
    parser.add_argument("--runs", type=int,
                        default=int(os.environ.get("RUNS_PER_TARGET", 3)))
    parser.add_argument("--max-calls", type=int,
                        default=int(os.environ.get("MAX_CALLS", 500)))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without calling APIs")
    args = parser.parse_args()

    if args.models:
        requested = args.models
    else:
        requested = available_models()
        if not requested:
            print("ERROR: no API keys found. Set ANTHROPIC_API_KEY, "
                  "OPENAI_API_KEY, GOOGLE_API_KEY in .env", file=sys.stderr)
            sys.exit(1)

    unknown = [m for m in requested if m not in ALL_MODELS]
    if unknown:
        print(f"ERROR: unknown models: {unknown}", file=sys.stderr)
        print(f"  available: {list(ALL_MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    scenes = load_screenshots(args.data_dir)
    if not scenes:
        print(f"ERROR: no screenshots found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    total_targets = sum(len(s["targets"]) for s in scenes)
    planned = len(requested) * total_targets * args.runs

    print(f"Plan:")
    print(f"  Models ({len(requested)}): {requested}")
    print(f"  Scenes ({len(scenes)}):    {[s['name'] for s in scenes]}")
    print(f"  Targets per scene:          {[len(s['targets']) for s in scenes]}")
    print(f"  Total targets:              {total_targets}")
    print(f"  Runs per target:            {args.runs}")
    print(f"  Total API calls planned:    {planned}")
    print(f"  Max calls allowed:          {args.max_calls}")

    if planned > args.max_calls:
        print(f"ERROR: planned {planned} > max {args.max_calls}. Lower --runs "
              f"or raise --max-calls.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n(dry run - exiting)")
        return

    existing = load_existing(args.out)
    done_keys = {(r["model"], r["screenshot"], r["target_id"], r["run"])
                 for r in existing}
    print(f"\nResuming: {len(existing)} predictions already in {args.out}")

    # Instantiate only needed models lazily
    instances: dict[str, GroundingModel] = {}

    results = list(existing)
    call_count = 0
    failures: dict[str, int] = {m: 0 for m in requested}

    for model_name in requested:
        if model_name not in instances:
            try:
                instances[model_name] = ALL_MODELS[model_name]()
                print(f"\n[model] {model_name} ready")
            except Exception as exc:
                print(f"\n[model] {model_name} FAILED to init: {exc}")
                continue
        model = instances[model_name]

        for scene in scenes:
            for target in scene["targets"]:
                for run_idx in range(args.runs):
                    key = (model_name, scene["name"], target["id"], run_idx)
                    if key in done_keys:
                        continue

                    pred = model.predict(
                        scene["image"], target["description"],
                        scene["width"], scene["height"],
                    )
                    call_count += 1

                    if pred.error:
                        failures[model_name] += 1
                        # Abort a model if it fails 5 times in a row
                        if failures[model_name] >= 5:
                            print(f"  [{model_name}] too many errors, skipping")
                            break

                    err_px = (euclidean(pred.x, pred.y, *target["center"])
                              if pred.x >= 0 else None)
                    ok = (inside_bounds(target["bounds"], pred.x, pred.y)
                          if pred.x >= 0 else False)

                    record = {
                        "model": model_name,
                        "cu_trained": model.cu_trained,
                        "provider": model.provider,
                        "screenshot": scene["name"],
                        "screenshot_w": scene["width"],
                        "screenshot_h": scene["height"],
                        "target_id": target["id"],
                        "target_description": target["description"],
                        "size_category": target["size_category"],
                        "area": target["area"],
                        "ground_truth": target["center"],
                        "bounds": target["bounds"],
                        "predicted": [pred.x, pred.y],
                        "euclidean_error_px": err_px,
                        "inside_bounds": ok,
                        "run": run_idx,
                        "latency_ms": pred.latency_ms,
                        "input_tokens": pred.input_tokens,
                        "output_tokens": pred.output_tokens,
                        "raw_response": pred.raw_response,
                        "error": pred.error,
                    }
                    results.append(record)
                    mark = "OK" if ok else ("  " if err_px is None else "..")
                    err_str = (f"{int(err_px):4}px" if err_px is not None
                               else " fail")
                    print(f"  [{mark}] {model_name:20} {scene['name']:16} "
                          f"{target['id']:28} run{run_idx} -> ({pred.x:4},{pred.y:4}) "
                          f"err={err_str}")

                    if call_count % 10 == 0:
                        save_results(args.out, results)

                    if call_count >= args.max_calls:
                        print(f"\nReached max calls ({args.max_calls}), "
                              f"saving and exiting.")
                        save_results(args.out, results)
                        return
            else:
                continue
            break  # broke inner loop due to model failures
    save_results(args.out, results)
    print(f"\nDone. {len(results)} total predictions in {args.out}")
    print(f"  API calls this run: {call_count}")
    print(f"  Failures per model: {failures}")


if __name__ == "__main__":
    main()
