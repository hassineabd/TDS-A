# Mobile UI Grounding Benchmark: CU-trained vs non-CU-trained models

Companion code for the TDS article *"Building a Mobile Test Agent with Robot
Framework and Multimodal AI"*.

## Hypothesis

Vision-language models that have been trained on Computer Use tasks (Claude
4.x, GPT-4o/4.1) should develop a *pixel-counting* skill — the ability to
predict precise click coordinates directly from a screenshot — that transfers
to mobile UIs even though Anthropic/OpenAI only advertise desktop training.
Models without this training (Gemini 2.5 Pro) should rely on weaker general
visual reasoning and produce less accurate coordinates, forcing them to use
an external parser (OmniParser, SoM overlay, accessibility tree) to be
useful for mobile agents.

This benchmark tests that hypothesis on **three real mobile screenshots**
captured from a Google Pixel 7 via BrowserStack App Automate, across five
VLMs accessed through OpenRouter.

## Method

**Screenshots** (no preprocessing — sent at native resolution):
- `clock_pixel7.png` — Android Clock app (landscape 2400×1080)
- `home_pixel7.png`  — Pixel launcher home screen (portrait 1080×2400)
- `youtube_pixel7.png` — YouTube notification permission dialog (2400×1080)

**Targets** (curated manually from UiAutomator2 `page_source` bounds):
24 elements total across the three screens, spanning size categories
*medium* (48dp ≤ area < 150dp²) and *large* (≥ 150dp²), with diverse
positions (corners, center, dock, headers).

**Models** (5 total, 1 API key via OpenRouter):

| Model              | Provider  | CU-trained? |
|--------------------|-----------|-------------|
| claude-sonnet-4.5  | Anthropic | **Yes**     |
| claude-opus-4.6    | Anthropic | **Yes**     |
| gpt-4o             | OpenAI    | **Yes** (via CUA) |
| gpt-4.1            | OpenAI    | **Yes** (via CUA) |
| gemini-2.5-pro     | Google    | No          |

**Prompt** (identical for all models):

```
You are looking at a mobile screenshot of <W>x<H> pixels (origin [0,0] is
top-left, x goes right, y goes down).

Return the exact pixel coordinates [x, y] of the CENTER of the following UI
element:

"<element description>"

Respond with ONLY a JSON object in this exact format and nothing else
(no markdown fences, no explanation):
{"x": <integer>, "y": <integer>}
```

No tools, no schema, no Computer-Use-mode activation. Every model gets the
raw screenshot (JPEG 95, full resolution — no downsampling) and a plain text
prompt. We measure how close their predicted `(x, y)` lands to the ground
truth center of each element.

**Runs**: 3 runs per target per model (to capture stochasticity) →
`5 × 24 × 3 = 360 API calls`.

**Metrics**:
- Mean Euclidean error in pixels (native screenshot coordinates)
- Hit rate: % of predictions that land inside the target's bounding box
- Breakdown by element size category
- Latency per model

## Repository layout

```
TDS-A/
├── data/                      Screenshots + UiAutomator2 XML + curated targets
│   ├── clock_pixel7.png       Clock app (2400×1080, landscape)
│   ├── clock_pixel7.xml       UiAutomator2 page source
│   ├── clock_pixel7.targets.json  Curated targets with ground-truth bounds
│   ├── home_pixel7.*
│   └── youtube_pixel7.*
├── benchmark/
│   ├── capture.py             BrowserStack session + screenshot + XML dump
│   ├── targets.py             Parse XML, classify targets by Material size
│   ├── models/
│   │   ├── base.py            GroundingModel ABC + prompt + JPEG encoder
│   │   └── openrouter.py      Unified OpenAI-compatible client for 5 models
│   ├── run.py                 Orchestrator: model × scene × target × run → JSON
│   └── analyze.py             Pandas stats + matplotlib charts
├── results/
│   ├── raw_predictions.json   All 360 predictions with metadata
│   ├── summary_by_model.csv   Per-model stats
│   ├── summary.csv            Per-model × size-category stats
│   └── charts/*.png           Error, hit rate, latency, scatter
└── requirements.txt
```

## Reproduce

```bash
# 1. Clone + deps
git clone https://github.com/hassineabd/TDS-A.git
cd TDS-A
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env and add OPENROUTER_API_KEY (https://openrouter.ai/keys)
# Optional: BROWSERSTACK_* if you want to re-capture screenshots

# 3. (Optional) Re-capture apps via BrowserStack
python -m benchmark.capture --app com.google.android.deskclock --out data/clock_pixel7
python -m benchmark.capture --app com.google.android.dialer    --out data/home_pixel7
python -m benchmark.capture --app com.google.android.youtube   --out data/youtube_pixel7
python -m benchmark.targets data/clock_pixel7.xml --screenshot-id clock_pixel7 --out data/clock_pixel7.targets.json

# 4. Run the benchmark (~$1-2 via OpenRouter, 15-30 min)
python -m benchmark.run --dry-run         # plan
python -m benchmark.run                   # full run (resumable)

# 5. Stats + charts
python -m benchmark.analyze
open results/charts/error_by_model.png
```

## Results

See `results/charts/` and `results/summary_by_model.csv` after running the
benchmark. A high-resolution version of the key chart is exported to
`article_snippets/figure_grounding.png` for inclusion in the TDS article.

## License

Code: MIT. Screenshots and target annotations: CC-BY-4.0.
