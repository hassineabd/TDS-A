# Article 1 — Draft

**Working title (pick one):**
- *"6 frontier VLMs walk into a mobile UI grounding test. Only 2 are usable. Here's why."*
- *"Why none of the 'computer-use-capable' VLMs are actually trained for it — and which ones still work for a mobile test agent."*
- *"What 'computer-use-trained' actually means (and why your favourite VLM probably isn't)."*

**Estimated length:** 2000-2500 words, 5-7 charts.

---

## 1. Setup (~250 words)

Hook: I was building a mobile test agent backed by a VLM. The obvious
question: which model? "Computer-use-trained ones, surely."

Quickly explain the test agent setup (Robot Framework + Appium + a VLM
acting as the grounding step) and link to the GitHub repo. State the
research question: **on real mobile screenshots, which frontier VLMs
return pixel coordinates accurate enough that a tap at the predicted
center actually triggers the right UI element?**

## 2. The honest framing (~350 words) — *the part most articles get wrong*

> *"Use a computer-use-trained model"* sounds like the obvious answer.
> So I went looking for them in the OpenRouter catalogue.

Walk through the audit:
- Anthropic Claude 4.x: marketed via OSWorld scores, but no system card
  ever claims "trained on computer-use data". The CU functionality is a
  **tool** (`computer-use-2025-11-24`) gated by a beta header — not a
  training claim about the model.
- OpenAI: the actual CU-trained model is `computer-use-preview`, a
  separate snapshot built *on top of* GPT-4o (Operator System Card,
  Jan 2025). GPT-4o and GPT-4.1 are general-purpose VLMs.
- Google: same story. `gemini-2.5-computer-use-preview-10-2025` is a
  separate "specialized model built on Gemini 2.5 Pro's visual
  understanding". Base Gemini 2.5 Pro is general-purpose.

→ **All six "obvious" choices are general-purpose VLMs sold as
agentic-capable** but not trained on GUI grounding per their providers'
own documentation. The truly CU-trained variants are gated behind
preview waitlists.

Include the sourced table from `benchmark/models/openrouter.py` here.

## 3. The benchmark (~300 words)

- 3 Android screenshots (Pixel 7), 24 manually curated targets with
  ground-truth bbox bounds from UiAutomator2.
- One prompt per (model, target). The model returns a bbox in the
  coordinate space it's documented to natively use (normalized 0-1000
  for everyone, pixel-absolute for Opus 4.7 — a finding I had to
  reverse-engineer; cite the Anthropic docs section that confirms it).
- Metric: **strict hit** — predicted bbox's centroid falls inside the
  ground-truth UiAutomator2 bounds. This is what an agent actually
  experiences: tap at the center, hope it triggers the right element.

## 4. Results (~500 words) — the plot twist

**Hero chart:** hit rate per model, bar chart, models ordered by
performance.

| Model | Hit rate |
|---|---|
| claude-opus-4.7 | 100% (24/24) |
| gemini-2.5-pro | 83% (19/23) |
| gpt-4.1 | 17% (4/24) |
| gpt-4o | 12% (3/24) |
| claude-opus-4.6 | 12% (3/24) |
| claude-sonnet-4.5 | 8% (2/24) |

Narrative beats:
- Gemini 2.5 Pro — not marketed as CU-capable — beats four of the five
  Claude/GPT models that ARE marketed as CU-capable.
- The two models at the top (Opus 4.7, Gemini) share one thing: each
  has a **single, publicly documented, and respected** coordinate
  convention. Opus 4.7 returns 1:1 pixels (Anthropic doc); Gemini
  returns `[y_min, x_min, y_max, x_max]` in 0-1000 (Google doc).
- The middle four don't have that. Their raw outputs alternate
  unpredictably between conventions — and that's what breaks them.

Forensic deep-dive (3 raw response examples per model, contrasting "good"
and "bad" cases). Show that GPT-4o actually localizes correctly but
returns over-large bboxes whose centroid lands a few pixels past the
target edge.

## 5. So what predicts grounding skill? (~300 words)

Not "computer-use training" (none of these are CU-trained anyway).
What separates the top two from the rest is:
1. **A publicly documented coordinate-space contract.**
2. **The model honoring that contract reliably across calls.**

When a model lacks (1), it falls back to whatever statistical mix of
conventions its general training saw — and the resulting bboxes are
geometrically reasonable but in the wrong scale or mixed scales.

This is also why the original "CU-trained vs not" hypothesis fails: the
training data composition isn't the bottleneck. The bottleneck is the
contract between the model's training distribution and the prompt format
at inference time.

## 6. What you should do for your mobile test agent (~250 words)

Concrete decision tree:
- **Zero-shot grounding works**: use Opus 4.7 (100%, 24 px median error)
  or Gemini 2.5 Pro (83%, 60 px median error).
- **Want one of the middle four anyway?** They need extra scaffolding
  to behave (see follow-up article on anchor priming — link it).
- **Stay away from Sonnet 4.5** for grounding; it returns structurally
  invalid bboxes (`y_max < y_min`) that no prompt engineering recovers.

## 7. Caveats (~150 words)

- N=144 predictions. Indicative, not statistically powered.
- Three apps, one device (Pixel 7), one orientation pair (landscape +
  portrait). Don't generalize.
- One run per target at temperature=0. Variance not measured.
- OpenRouter routing may shift over time; reproducing the exact numbers
  in 6 months is not guaranteed. Pinning would require direct
  provider-API calls with pinned snapshots.
- "Strict hit" is a slight underestimate because Android `TouchDelegate`
  can extend the touchable area beyond the visual bbox.

## 8. Future work / what's next (~100 words)

- Test the *actual* CU-trained models (`computer-use-preview`,
  `gemini-2.5-computer-use-preview-10-2025`) once we get API access. Are
  they meaningfully better than Opus 4.7 and Gemini 2.5 Pro on the same
  test? Or do general-purpose VLMs with the right coord-space contract
  already saturate this task?
- Larger benchmark: 100+ apps, 1000+ targets, 2-3 devices.
- Live validation on BrowserStack: confirm that strict-hit predicts
  actual automation success rate.

---

## Charts to produce

Already in `results/grounding_zero_shot/20260512-022021_baseline-6models/charts/`:
- `hit_rate_by_model.png` ← **hero chart for section 4**
- `error_by_model.png`
- `iou_by_model.png`
- `hit_rate_by_size.png`
- `latency_by_model.png`
- `scatter_predictions.png` ← good for section 5 (shows the "wrong scale" pattern)

To produce additionally:
- Side-by-side raw response examples (3 from a "good" model + 3 from a
  "bad" model) annotated on the actual screenshot.
- Honest CU-training table (already in README).

## TODO before publishing

- [ ] Revoke and rotate the OpenRouter key
- [ ] Confirm Opus 4.7 pricing on OpenRouter at time of publication (cost
      figure in section 3)
- [ ] Take a fresh screenshot showing the Opus 4.7 raw response (for the
      "pixel-1:1 surprise" callout)
- [ ] Decide title and write the lead paragraph (300 chars max for TDS
      preview)
