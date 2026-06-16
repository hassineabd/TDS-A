# Study 1 — Marketed-for-Computer-Use VLMs on Mobile UI Grounding

Self-contained benchmark for the first TDS article. Tests six frontier
vision-language models that are publicly marketed as "computer-use" or
"agentic" capable on a zero-shot mobile UI grounding task, with each model
queried in its documented native coordinate space.

## Key clarification

No model in this study is publicly claimed by its provider to have been
*trained on* computer-use / GUI-grounding data. The truly CU-trained
variants — OpenAI `computer-use-preview`, Google
`gemini-2.5-computer-use-preview-10-2025` — are SEPARATE model snapshots
not exposed via OpenRouter, so they are out of scope here. We use the
deliberately weaker label `marketed_for_cu` and persist a `source_url`
per model citing the primary source for that claim — see
[benchmark/models/openrouter.py](benchmark/models/openrouter.py).

## Models

| Model | marketed_for_cu | coord_space | Source URL |
|---|---|---|---|
| claude-sonnet-4.5 | true | normalized_1000 | https://www.anthropic.com/news/claude-sonnet-4-5 |
| claude-opus-4.6 | true | normalized_1000 | https://www.anthropic.com/news/claude-opus-4-6 |
| claude-opus-4.7 | true | **pixel_yx** | https://www.anthropic.com/news/claude-opus-4-7 |
| gpt-4o | false | normalized_1000 | https://cdn.openai.com/operator_system_card.pdf |
| gpt-4.1 | false | normalized_1000 | https://openai.com/index/gpt-4-1/ |
| gemini-2.5-pro | false | normalized_1000 | https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/ |

## Dataset

Three real mobile screenshots captured from a Google Pixel 7 via BrowserStack
App Automate, with 24 manually-curated target elements total (ground-truth
bounds from UiAutomator2 `page_source`).

- `data/clock_pixel7.{png,xml,targets.json}` — 2400×1080 landscape, 8 targets
- `data/home_pixel7.{png,xml,targets.json}`  — 1080×2400 portrait, 8 targets
- `data/youtube_pixel7.{png,xml,targets.json}` — 2400×1080 landscape, 8 targets

## Run

```bash
# 0. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENROUTER_API_KEY

# 1. Plan only
python -m benchmark.run --experiment grounding_zero_shot --dry-run

# 2. Full run (144 API calls; ~10 min; ~$2 via OpenRouter)
python -m benchmark.run --experiment grounding_zero_shot

# 3. Stats + charts (resolves the latest run automatically)
python -m benchmark.analyze.grounding --experiment grounding_zero_shot
```

Each run goes to its own timestamped directory
(`results/grounding_zero_shot/<YYYYMMDD-HHMMSS>[_<run-name>]/`) so previous
results are preserved. A run can be re-tagged with `--run-name <label>`.

## Baseline results bundled

`results/grounding_zero_shot/20260512-022021_baseline-6models/` contains the
first run used to draft the article. Re-analyzing it without re-running:

```bash
python -m benchmark.analyze.grounding \
  --run-dir results/grounding_zero_shot/20260512-022021_baseline-6models
```

## Metric

Strict hit rate = the predicted bbox's centroid lies inside the ground-truth
bounds from UiAutomator2. This matches what a mobile test agent actually
experiences: tap at the predicted center; if it falls inside the touchable
area of the target, the automation step succeeds; otherwise it fails.

See [benchmark/metrics.py](benchmark/metrics.py) for the secondary metrics
(IoU, mean center error, per-size breakdown).

## Limitations of this study

- N = 144 predictions (6 models × 24 targets × 1 run). Indicative, not
  statistically powered for fine-grained model rankings.
- Single device (Pixel 7), single OS version (Android 13), three apps.
- One run per target with temperature=0 — variance not measured.
- Models accessed via OpenRouter; provider snapshot routing may shift over
  time. Reproducibility on identical inputs in N months is not guaranteed.
- The UiAutomator2 `bounds` is the visual bounding box of the element. On
  Android, a `TouchDelegate` can extend the actual touchable area beyond
  these bounds, so this metric is a slight underestimate of practical
  automation success.

## Companion article

Draft: [article_draft.md](article_draft.md)
