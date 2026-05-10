# Mobile UI Grounding Benchmark

Companion code for the TDS article *"Building a Mobile Test Agent with Robot
Framework and Multimodal AI"*. The benchmark tests how well vision-language
models can locate UI elements on mobile screenshots, which is the core skill
needed to drive a mobile test agent that operates by clicking pixel
coordinates rather than DOM/accessibility selectors.

## Hypotheses

**H1 (zero-shot):** vision-language models that have been trained on Computer
Use tasks (Claude 4.x, GPT-4o/4.1) should locate UI elements more accurately
than models without that training (Gemini 2.5 Pro), even on mobile UIs that
fall outside the desktop training distribution.

**H2 (with anchors):** priming the model with a small number of *calibration
anchors* — a few reference bboxes drawn on the screenshot AND given as
coordinate text — should improve grounding accuracy. This tests whether the
gap between models can be closed (or reversed) with a coordinate-system
priming technique inspired by Set-of-Marks prompting.

## Coordinate protocol

Every model is asked for normalized bounding boxes in `[y_min, x_min, y_max,
x_max]` format, scaled to 0–1000. This:

  - matches Gemini's [native documented format](https://ai.google.dev/gemini-api/docs/image-understanding) so it answers in its training distribution,
  - avoids each provider's internal image downsampling (Claude clamps the long
    edge to 1568 px and returns coordinates in *that* space; GPT-4o/4.1
    rescales to a 768-shortest-side tiled image — both are documented quirks
    that bite naïve "give me pixel coordinates" prompts),
  - leaves the client-side denormalisation as a single, uniform formula:
    `pixel = norm × dim / 1000` using the *original* screenshot dimensions.

The Y-first ordering is Gemini's convention. Pixel-space tuples remain X-first
(standard for image processing).

## Experiments

Each experiment is an independent unit with its own prompt, image-prep
pipeline, parser, evaluator, and results directory. Add a new one by
subclassing `Experiment` and registering it in
[`benchmark/experiments/__init__.py`](benchmark/experiments/__init__.py).

| Experiment                  | Calls | Prompt                                        | Image                       |
|-----------------------------|-------|-----------------------------------------------|-----------------------------|
| `grounding_zero_shot`       | 24    | Locate ONE described element                   | screenshot, untouched       |
| `grounding_with_anchors`    | 15    | Locate ONE element + 3 anchor calibration boxes | screenshot with magenta SoM overlay |

`grounding_with_anchors` excludes the 3 anchors from its eval set, so it
runs over 24 − (3 anchors × 3 scenes) = 15 targets.

## Models

Six VLMs accessed through a single OpenRouter API key:

| Model              | Provider  | CU-trained? | Notes                                                        |
|--------------------|-----------|-------------|--------------------------------------------------------------|
| claude-sonnet-4.5  | Anthropic | yes         | downsamples to 1568 px long edge                             |
| claude-opus-4.6    | Anthropic | yes         | same downsample as Sonnet 4.5                                |
| claude-opus-4.7    | Anthropic | yes         | 2576 px long edge, **no** downsample-space coord quirk        |
| gpt-4o             | OpenAI    | yes (CUA)   | tile pipeline rescales shortest side to 768 px                |
| gpt-4.1            | OpenAI    | yes (CUA)   | same as 4o                                                   |
| gemini-2.5-pro     | Google    | no          | native 0-1000 [y,x,y,x] coordinate output                     |

Total API calls per experiment, single run: 144 (zero-shot) and 90 (anchors).

## Repository layout

```
TDS-A/
├── data/
│   ├── *_pixel7.png          screenshots
│   ├── *_pixel7.xml          UiAutomator2 page sources
│   └── *_pixel7.targets.json curated targets with pixel-space ground-truth bboxes
├── benchmark/
│   ├── coords.py             norm ↔ pixel, bbox math, deterministic anchor pick
│   ├── metrics.py            hit_rate, IoU, center_err, precision/recall/F1
│   ├── render.py             PIL: anchor overlay (magenta bbox + label)
│   ├── models/               OpenRouter client (one class per model)
│   ├── experiments/
│   │   ├── base.py           Experiment ABC + dataclasses (Scene, Call, Prediction)
│   │   ├── grounding_zero_shot.py
│   │   └── grounding_with_anchors.py
│   ├── analyze/
│   │   └── grounding.py      stats + charts for grounding experiments
│   ├── run.py                generic runner: --experiment <name>
│   ├── capture.py            BrowserStack session + screenshot dump
│   └── targets.py            parse XML, classify by Material size category
└── results/<experiment_name>/
    ├── raw_predictions.json
    ├── summary_by_model.csv
    ├── summary_by_model_size.csv
    └── charts/
```

## Reproduce

```bash
git clone https://github.com/hassineabd/TDS-A.git && cd TDS-A
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY (https://openrouter.ai/keys)

# Plan only:
python -m benchmark.run --experiment grounding_zero_shot --dry-run

# Run (resumable; checkpoints every 10 calls):
python -m benchmark.run --experiment grounding_zero_shot
python -m benchmark.run --experiment grounding_with_anchors

# Aggregate stats + charts:
python -m benchmark.analyze.grounding --experiment grounding_zero_shot
python -m benchmark.analyze.grounding --experiment grounding_with_anchors
```

Useful flags on `benchmark.run`:

  - `--models claude-sonnet-4.5 gemini-2.5-pro` — restrict to a subset
  - `--runs 3` — repeat each call N times (default 1; only useful when
    `--temperature > 0`, otherwise outputs are near-deterministic)
  - `--temperature 0.7` — non-zero sampling
  - `--max-calls 1000` — safety cap; runner refuses to start above this

## Methodological caveats

To be honest about the scope of this benchmark:

  - **Sample size is small.** 24 targets × 6 models is illustrative, not a
    statistically powered comparison. We don't claim significance.
  - **Targets only cover medium and large sizes.** The 8-target curation per
    scene happens to skip elements below 3600 px², which means we don't
    evaluate the regime where grounding is hardest.
  - **Target descriptions are humanly annotated and somewhat over-specified.**
    "*The 'Alarm' tab in the bottom navigation bar (leftmost)*" already
    encodes spatial information. A real agent would receive shorter,
    user-style commands. A future experiment could test the gap between
    rich and short descriptions.
  - **OpenRouter is assumed to be a passthrough** to upstream providers.
    Image preprocessing on the OpenRouter side is not officially documented.
  - **Anchor selection is fixed per scene.** The same 3 anchors are reused
    for every target in a scene. Diversifying anchors per target — and
    measuring the variance — is left as a follow-up experiment.

These are deliberate tradeoffs to keep the experiment runnable on a tight
API-cost budget; the article should frame results as directional, not
definitive.

## License

Code: MIT. Screenshots and target annotations: CC-BY-4.0.
