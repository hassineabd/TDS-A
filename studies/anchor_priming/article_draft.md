# Article 2 — Draft

**Working title (pick one):**
- *"3 magenta rectangles raised a broken VLM from 12% to 80% — and changed nothing for the model next to it."*
- *"Why drawing reference bboxes on a screenshot fixes some VLMs but not others (a Set-of-Marks deep dive on mobile UIs)."*
- *"Anchor priming as an implicit coordinate-space contract."*

**Estimated length:** 1500-2000 words, 4-6 charts.

**Companion article:** [Article 1 — link once published]. Read that first
for the benchmark setup and the surprise that some of these models are
much worse than expected at zero-shot grounding.

---

## 1. The hook (~200 words)

Take a vision model that gets **12% hit rate** on a simple mobile UI
grounding task. Add three magenta rectangles to the screenshot, list
their coordinates in the prompt, and the same model gets **80%** — a
seven-times improvement, on the same screenshots and same targets,
without retraining or fine-tuning anything.

Add the same magenta rectangles to a *different* model from the same
provider — and nothing changes. It stays at 7%.

This is the story of how Set-of-Marks prompting works on mobile UI
grounding, and why it's not the "draw on the image and it gets better"
trick people often assume.

## 2. Setup mini-recap (~150 words)

Two paragraphs and a link to article 1. Just enough to say:
- 6 frontier VLMs, 3 mobile screenshots, 24 targets from UiAutomator2
- Strict hit = predicted centroid falls in the ground-truth bbox
- Article 1 found that only 2 of the 6 work in zero-shot

Then state the question for this article: *can the other 4 be rescued?*

## 3. Set-of-Marks 101 (~250 words)

Quick intro to SoM (cite the Microsoft 2023 paper). The idea: instead
of asking the model to point at things in the raw image, you overlay
numbered markers / bounding boxes on the image first, then ask the
model to refer to them by number.

Our adaptation:
- Pick 3 reference targets per scene by maximin centroid distance
  (deterministic, no randomness — same anchors every time).
- Draw each as a magenta outline + a "Anchor 1/2/3" label badge.
- Inject the anchors into the prompt **with their coordinates** —
  in the same coord_space the model expects (pixel-absolute for
  Opus 4.7, normalized-0-1000 for everyone else).

Show one rendered example screenshot (clock_pixel7 with its 3 anchors).

## 4. Results (~450 words) — *the big chart*

**Hero chart:** side-by-side bars per model — zero-shot vs with-anchors
hit rate.

| Model | ZS hit | WA hit | Lift |
|---|---|---|---|
| claude-opus-4.7 | 100% | 100% | +0pp |
| gemini-2.5-pro | 79% | 87% | +8pp |
| gpt-4.1 | 20% | 47% | **+27pp** |
| gpt-4o | 20% | 33% | +13pp |
| **claude-opus-4.6** | **13%** | **80%** | **+67pp** |
| claude-sonnet-4.5 | 7% | 7% | +0pp |

*(Both columns computed on the same 15 targets per model — apples-to-apples.
The 3 anchor targets are excluded from evaluation in both conditions.)*

Discuss each band:
- **Models at the ceiling (Opus 4.7, Gemini)**: small or zero lift.
  They already had a stable, documented coordinate space. Anchors are
  redundant calibration.
- **Models partially helped (GPT-4o, GPT-4.1)**: +13 to +27pp. Their
  IoU **doubles to triples** (more important than the hit rate jump).
  They were "right neighbourhood, wrong size" — anchors tighten the
  bbox estimation.
- **The transformation case (Opus 4.6)**: +67pp. Median center error
  drops from **902 px to 75 px** — a 12× reduction. Mean IoU goes from
  0.06 to 0.44.
- **The resistant case (Sonnet 4.5)**: zero lift. Why? Open the raw
  responses.

## 5. Why anchors work — the IoU view (~300 words)

The story isn't about hit rate per se. It's about geometric quality
of the bboxes.

Second chart: **mean IoU on matched predictions** before vs after
anchors. Shows IoU rising for every model except Sonnet 4.5.

The mechanism: anchor coordinates in the prompt act as an **implicit
contract**. When a model receives 3 textual examples of "this UI
element has bbox [747, 383, 942, 568]", it imitates the format — even
if it would otherwise alternate between conventions. Opus 4.6 in
zero-shot emits coords above 1000 (pixel-mode) on 15 of 24 calls and
coords below 1000 (normalized-mode) on the others. With anchors all in
normalized-1000, Opus 4.6 sticks to normalized-1000 most of the time.

Show the raw-response contrast (zero-shot Opus 4.6 mixing modes, vs
with-anchors Opus 4.6 mostly clean 0-1000).

## 6. The resistant case — Sonnet 4.5 (~250 words)

Sonnet 4.5 doesn't have a coordinate-space-mixing problem. It has a
**bbox-validity** problem: it returns quadruples where `y_max < y_min`
or `x_max < x_min` — i.e. the bbox is degenerate or inside-out. Three
example raw responses showing this. The provided anchors are all
well-formed rectangles; the model imitates the *coordinate scale* but
keeps emitting structurally invalid outputs.

This is a case where prompt engineering (visual + textual) cannot
recover the issue. The model needs a different intervention — probably
post-hoc validation + retry, or a structured tool-use schema that the
provider enforces (which is not what we tested here).

## 7. What to take away if you're shipping an agent (~200 words)

- **Already using Opus 4.7 or Gemini 2.5 Pro for grounding?** Don't
  bother with anchors. You're already at the ceiling.
- **Stuck with one of the middle-band models** (because of cost,
  context window, latency, or company contract)? Anchors are a
  *cheap, no-retraining* way to roughly halve your error.
- **Considering Sonnet 4.5?** Don't, for grounding. Move to a model
  whose output format is structurally trustworthy.

The wider lesson: when a VLM gives you bad coordinates, the question
isn't "does the model know where the element is?" — it's "did the
model emit coordinates in a format the rest of the pipeline can
interpret?". Anchors fix the second problem.

## 8. Caveats (~150 words)

- Same dataset / device / single-run caveats as article 1.
- We only tested one anchor configuration (3 anchors, maximin spread,
  magenta outline + label). We haven't ablated count / colour / textual
  vs visual.
- The 3 anchors are excluded from the eval set, so anchor-priming is
  evaluated on a different (15-target) eval set than zero-shot
  (24-target). We compensated by restricting zero-shot to the same 15
  targets for comparison, but the apples-to-apples sample stays small.

## 9. What's next (~80 words)

- Anchor count ablation (1, 3, 5, 7).
- Visual-only vs text-only anchors.
- Test on iPhone screenshots (already captured but not in this run).
- Test the actual CU-trained models — does priming still help them, or
  are they post-trained to already produce structurally clean output?

---

## Charts to produce

Already in `results/grounding_with_anchors/20260512-023013_baseline-6models/charts/`:
- `hit_rate_by_model.png` for the WA column
- `iou_by_model.png` for the IoU narrative
- `error_by_model.png` and `scatter_predictions.png` for the geometric
  improvement story

To produce:
- Hero chart: side-by-side ZS vs WA hit rate, sorted by lift (matplotlib
  one-pager script — TODO).
- Annotated screenshot showing the 3 magenta anchors + the target on
  one example scene (use `benchmark.render.draw_anchors`).
- Raw response contrast figure for Opus 4.6: 4 raw outputs ZS vs 4 raw
  outputs WA, highlighting where the mode-mixing disappears.

## TODO before publishing

- [ ] Rotate the OpenRouter key.
- [ ] Get a clean Opus 4.6 raw-response figure showing the
      coordinate-space mode-mixing → uniform 0-1000 transition.
- [ ] Decide if "Set-of-Marks" name is too jargon-y for TDS audience or
      if we just use "anchor priming" throughout.
- [ ] Cross-link article 1 once it's live.
