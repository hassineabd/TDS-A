"""Generic experiment runner.

Usage:
    python -m benchmark.run --experiment grounding_zero_shot --dry-run
    python -m benchmark.run --experiment grounding_zero_shot
    python -m benchmark.run --experiment grounding_with_anchors --models gemini-2.5-pro
    python -m benchmark.run --experiment grounding_zero_shot --run-name no-anchors-with-cot

Output goes to `results/<experiment>/<timestamp>[_<run-name>]/raw_predictions.json`.
Each run is fresh by default (own timestamped directory); pass `--resume <dir>`
to append into an existing run instead.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .experiments import EXPERIMENTS, Scene, get_experiment
from .models import ALL_MODELS, GroundingModel, available_models
from .models.base import encode_image


_RUN_NAME_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def build_run_dir(
    out_dir: Path, experiment: str, run_name: str | None,
    resume_dir: Path | None = None,
) -> Path:
    """Resolve the output directory.

    With --resume: returns the provided path unchanged (creates if missing).
    Without: returns out_dir/experiment/<timestamp>[_<run_name>]/.
    """
    if resume_dir is not None:
        resume_dir.mkdir(parents=True, exist_ok=True)
        return resume_dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = f"{ts}_{run_name}" if run_name else ts
    return out_dir / experiment / folder


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
    parser.add_argument("--run-name", default=None,
                        help="Optional short label appended to the timestamp "
                             "in the output directory name (e.g. 'no-cot', "
                             "'gemini-only'). Allowed chars: A-Z a-z 0-9 . _ -")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume into this exact run directory instead "
                             "of creating a new timestamped one")
    args = parser.parse_args()

    if args.run_name is not None and not _RUN_NAME_OK.match(args.run_name):
        print(f"ERROR: --run-name must match [A-Za-z0-9._-]+; got {args.run_name!r}",
              file=sys.stderr)
        sys.exit(1)

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

    # Build the plan. Number of calls is identical across coord_spaces (we
    # vary the prompt, not the call count), so estimate with the default.
    exp = get_experiment(args.experiment)
    scenes = Scene.load_from(args.data_dir)
    if not scenes:
        print(f"ERROR: no scenes found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    n_calls_per_model = sum(1 for _ in exp.iter_calls(scenes))
    planned = n_calls_per_model * len(requested) * args.runs

    run_dir = build_run_dir(args.out_dir, args.experiment,
                            args.run_name, args.resume)
    out_path = run_dir / "raw_predictions.json"

    print(f"Experiment:       {args.experiment}")
    print(f"Run name:         {args.run_name or '(none)'}")
    print(f"Models ({len(requested)}):  {requested}")
    print(f"Scenes ({len(scenes)}):   {[s.name for s in scenes]}")
    print(f"Calls per model:  {n_calls_per_model}")
    print(f"Runs per call:    {args.runs}")
    print(f"Total API calls:  {planned}")
    print(f"Max allowed:      {args.max_calls}")
    print(f"Output:           {out_path}")
    if planned > args.max_calls:
        print(f"\nERROR: planned {planned} > max {args.max_calls}. "
              f"Lower --runs or raise --max-calls.", file=sys.stderr)
        sys.exit(1)

    # Persist a tiny manifest alongside the predictions so the run is
    # self-describing (when it ran, with what args, on which models).
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "experiment": args.experiment,
            "run_name": args.run_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "models": requested,
            "scenes": [s.name for s in scenes],
            "runs_per_call": args.runs,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "planned_calls": planned,
            "data_dir": str(args.data_dir),
            "resume": str(args.resume) if args.resume else None,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if args.dry_run:
        # Show a sample for both coord_spaces if any model uses each
        print("\n--- Sample prompts ---")
        seen_spaces = set()
        for m in requested:
            cs = getattr(ALL_MODELS[m]().__class__, "coord_space",
                         "normalized_1000")
            try:
                inst = ALL_MODELS[m]()
                cs = inst.coord_space
            except Exception:
                pass
            if cs in seen_spaces:
                continue
            seen_spaces.add(cs)
            sample = next(exp.iter_calls(scenes, coord_space=cs))
            print(f"\ncoord_space = {cs}  (used by {m} and similar)")
            print(f"  scene = {sample.scene_name}  target = {sample.target_id}")
            print(f"  image type = {type(sample.image).__name__}")
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

            # Each model gets its prompts in its own preferred coord_space.
            # We re-materialize calls per model — cheap for grounding_zero_shot
            # and adds ~one PIL render per scene for grounding_with_anchors.
            coord_space = getattr(model, "coord_space", "normalized_1000")
            model_calls = list(exp.iter_calls(scenes, coord_space=coord_space))
            print(f"\n[{model_name}] coord_space={coord_space}  "
                  f"({len(model_calls)} calls)")

            for call in model_calls:
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
                        "marketed_for_cu": model.marketed_for_cu,
                        "cu_source_url": model.source_url,
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
