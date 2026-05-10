"""Generic experiment runner.

Usage:
    python -m benchmark.run --experiment grounding_zero_shot --dry-run
    python -m benchmark.run --experiment grounding_zero_shot
    python -m benchmark.run --experiment grounding_with_anchors --models gemini-2.5-pro

The runner is task-agnostic: it loops over (model x call x run), encodes
the image, dispatches to the model, then asks the experiment to parse and
evaluate the response. Results are streamed to
`results/<experiment>/raw_predictions.json` and the run is resumable —
re-launching skips rows already on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .experiments import EXPERIMENTS, Scene, get_experiment
from .models import ALL_MODELS, GroundingModel, available_models
from .models.base import encode_image


def load_existing(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    return []


def save_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str))


def make_row_key(experiment: str, model: str, scene: str,
                 target_id: str | None, run: int) -> tuple:
    return (experiment, model, scene, target_id or "_scene_", run)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model names; defaults to all available")
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat each call N times for stochasticity "
                             "measurement; default 1 because temperature=0 "
                             "makes additional runs near-deterministic")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-calls", type=int,
                        default=int(os.environ.get("MAX_CALLS", 500)),
                        help="Safety cap; refuses to start if planned > this")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan + sample prompt and exit")
    args = parser.parse_args()

    # Resolve models
    if args.models:
        requested = args.models
    else:
        requested = available_models()
        if not requested:
            print("ERROR: no API keys found. Set OPENROUTER_API_KEY in .env",
                  file=sys.stderr)
            sys.exit(1)
    unknown = [m for m in requested if m not in ALL_MODELS]
    if unknown:
        print(f"ERROR: unknown models: {unknown}", file=sys.stderr)
        print(f"  available: {list(ALL_MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Build the plan
    exp = get_experiment(args.experiment)
    scenes = Scene.load_from(args.data_dir)
    if not scenes:
        print(f"ERROR: no scenes found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    calls = list(exp.iter_calls(scenes))
    planned = len(calls) * len(requested) * args.runs

    out_path = args.out_dir / args.experiment / "raw_predictions.json"

    print(f"Experiment:       {args.experiment}")
    print(f"Models ({len(requested)}):  {requested}")
    print(f"Scenes ({len(scenes)}):   {[s.name for s in scenes]}")
    print(f"Calls per model:  {len(calls)}")
    print(f"Runs per call:    {args.runs}")
    print(f"Total API calls:  {planned}")
    print(f"Max allowed:      {args.max_calls}")
    print(f"Output:           {out_path}")
    if planned > args.max_calls:
        print(f"\nERROR: planned {planned} > max {args.max_calls}. "
              f"Lower --runs or raise --max-calls.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        if calls:
            sample = calls[0]
            print(f"\n--- Sample call ---")
            print(f"  scene = {sample.scene_name}")
            print(f"  target = {sample.target_id}")
            print(f"  image = {type(sample.image).__name__}")
            print(f"  prompt:\n{sample.prompt}")
        print("\n(dry run, exiting)")
        return

    existing = load_existing(out_path)
    done_keys = {tuple(r["_key"]) for r in existing if "_key" in r}
    rows = list(existing)
    print(f"\nResuming: {len(existing)} rows already on disk")

    instances: dict[str, GroundingModel] = {}
    failures: dict[str, int] = {m: 0 for m in requested}
    n_called = 0

    try:
        for model_name in requested:
            try:
                model = instances.setdefault(model_name, ALL_MODELS[model_name]())
            except Exception as exc:
                print(f"[{model_name}] init failed: {exc}", file=sys.stderr)
                continue

            for call in calls:
                # Encode image once per call (no point caching across models —
                # they're independent, and we may have unique images per call)
                try:
                    img_b64, media_type = encode_image(call.image)
                except Exception as exc:
                    print(f"  encode failed for {call.scene_name}/{call.target_id}: {exc}",
                          file=sys.stderr)
                    continue

                for run_idx in range(args.runs):
                    key = make_row_key(args.experiment, model_name,
                                       call.scene_name, call.target_id, run_idx)
                    if key in done_keys:
                        continue

                    resp = model.call(
                        image_b64=img_b64, media_type=media_type,
                        prompt=call.prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    n_called += 1

                    if resp.error:
                        failures[model_name] += 1
                        prediction = exp.parse_response("", _scene_for(scenes, call.scene_name), call)
                        prediction.parse_error = resp.error
                        metrics = exp.evaluate(prediction, _scene_for(scenes, call.scene_name), call)
                        if failures[model_name] >= 5:
                            print(f"  [{model_name}] too many errors, skipping rest")
                            break
                    else:
                        scene = _scene_for(scenes, call.scene_name)
                        prediction = exp.parse_response(resp.text, scene, call)
                        metrics = exp.evaluate(prediction, scene, call)

                    pred_bbox = (prediction.items[0].bbox_px
                                 if prediction.parse_ok else None)
                    row = {
                        "_key": list(key),
                        "experiment": args.experiment,
                        "model": model_name,
                        "cu_trained": model.cu_trained,
                        "provider": model.provider,
                        "scene": call.scene_name,
                        "target_id": call.target_id,
                        "run": run_idx,
                        "predicted_bbox_px": list(pred_bbox) if pred_bbox else None,
                        "raw_response": resp.text[:500],
                        "parse_error": prediction.parse_error,
                        "metrics": metrics,
                        "metadata": call.metadata,
                        "latency_ms": resp.latency_ms,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "api_error": resp.error,
                    }
                    rows.append(row)
                    done_keys.add(key)

                    mark = ("OK" if metrics.get("hit") else
                            "  " if not metrics.get("parse_fail") else "??")
                    err = metrics.get("center_error_px")
                    err_str = f"{int(err):4d}px" if isinstance(err, (int, float)) else "  fail"
                    print(f"  [{mark}] {model_name:20} {call.scene_name:18} "
                          f"{(call.target_id or '_scene_'):28} run{run_idx} -> err={err_str}")

                    if n_called % 10 == 0:
                        save_results(out_path, rows)
                else:
                    continue
                break  # broke out of run loop
            else:
                continue
            # Inner break above came from the model-failure path
    finally:
        save_results(out_path, rows)
        print(f"\nDone. {len(rows)} rows in {out_path}")
        print(f"  API calls this run: {n_called}")
        print(f"  Failures per model: {failures}")


def _scene_for(scenes: list[Scene], name: str) -> Scene:
    for s in scenes:
        if s.name == name:
            return s
    raise KeyError(f"Scene not found: {name}")


if __name__ == "__main__":
    main()
