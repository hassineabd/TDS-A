# Study 2 — Set-of-Marks Anchor Priming on Mobile UI Grounding

Self-contained benchmark for the second TDS article. Tests whether
drawing reference bounding boxes ("anchors") on the screenshot AND
injecting their coordinates into the prompt (Set-of-Marks style) improves
mobile UI grounding accuracy across six frontier VLMs.

## Quick read

| Metric | grounding_zero_shot | grounding_with_anchors | Note |
|---|---|---|---|
| Eval calls per model | 24 | 15 | 3 of the 8 targets per scene become anchors |
| Best model | Claude Opus 4.7 (100%) | Claude Opus 4.7 (100%) | already at ceiling |
| Biggest lift | — | Claude Opus 4.6: 12% → 80% (+68pp) | the headline finding |
| Anchor-resistant | — | Claude Sonnet 4.5: 8% → 7% | bboxes structurally invalid |

## Anchor selection

Per scene we pick 3 anchors by maximin pairwise centroid distance over the
8 manually curated targets (see `benchmark.coords.select_anchor_indices`).
Selection is deterministic — the same scene always produces the same anchors
across re-runs, and the same anchors are shared across all evaluation targets
within a scene. Targets used as anchors are not evaluated.

The anchors are rendered onto the screenshot in solid magenta (`#FF00FF`)
with `Anchor 1/2/3` labels in adjacent badges, and the same coordinates
are also stated textually in the prompt preamble — in the coord_space the
target model expects (pixel-absolute for Opus 4.7, normalized 0-1000 for
the others).

## Models

Same six models as Study 1, with identical `marketed_for_cu` / `source_url`
annotations — see [benchmark/models/openrouter.py](benchmark/models/openrouter.py).

## Dataset

Same as Study 1: `data/{clock,home,youtube}_pixel7.{png,xml,targets.json}`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENROUTER_API_KEY

# Plan
python -m benchmark.run --experiment grounding_with_anchors --dry-run

# Baseline (used as control)
python -m benchmark.run --experiment grounding_zero_shot

# Treatment
python -m benchmark.run --experiment grounding_with_anchors

# Analyze each
python -m benchmark.analyze.grounding --experiment grounding_zero_shot
python -m benchmark.analyze.grounding --experiment grounding_with_anchors
```

## Baseline results bundled

- `results/grounding_zero_shot/20260512-022021_baseline-6models/` — 144 rows
- `results/grounding_with_anchors/20260512-023013_baseline-6models/` — 90 rows

These are the runs used to draft the article. Re-run analysis without
re-calling APIs:

```bash
python -m benchmark.analyze.grounding \
  --run-dir results/grounding_with_anchors/20260512-023013_baseline-6models
```

## Apples-to-apples comparison

Because anchor selection removes 3 targets per scene from the evaluation
set, `grounding_zero_shot` has 24 calls per model and
`grounding_with_anchors` has 15. The pre-print analysis (see article
draft) restricts the zero-shot evaluation to the same 15 targets so the
lift comparison is apples-to-apples — see also the `_key` field on each
result row, which is `(experiment, model, scene, target_id, run)` for
matching.

## Mechanism — why anchors help

For models with a documented and respected coordinate-space contract
(Opus 4.7's pixel-1:1, Gemini's normalized-0-1000), anchors have a small
effect: the median centroid error halves, but hit rate is already near
ceiling.

For models with an *unstable* coordinate output (Claude Opus 4.6 emits a
mix of 0-1000 and pixel-absolute coords from one call to the next),
anchors act as an **implicit contract**: by stating the coordinate format
of three example bboxes, they nudge the model to use the same format
consistently. Opus 4.6's mean IoU multiplies by ~7 (0.06 → 0.44) and its
hit rate jumps from 12% to 80%.

For models with *structurally invalid* output (Sonnet 4.5 returns bboxes
where `y_max < y_min` or `x_max < x_min`), anchors cannot recover the
shape constraint — the model remains broken.

## Limitations

Same as Study 1, plus:
- Anchors are spatially "spread" (maximin); we have not tested whether
  random anchor selection or anchor count (1, 5, 7 instead of 3) matters.
- Visual style (magenta, line width, label position) is fixed; we have
  not tested whether SoM-style numeric badges *without* the bbox outline
  perform similarly.
- Anchor descriptions are short natural-language strings; we have not
  tested whether longer / shorter / no-description variants change the
  effect size.

## Companion article

Draft: [article_draft.md](article_draft.md)
