# BiasLens — End-to-End LLM Pipeline for Vietnamese Generated Text Bias

Generative AI now drafts Vietnamese content at scale — chat replies, social copy, articles — but the
safety tooling that exists to audit that output (Google's Perspective API, OpenAI's Moderation API,
Detoxify) is English-centric and tuned for overt toxicity, not the culturally-coded bias that shows up
in Vietnamese: hierarchical pronouns, North/Central/South regional stereotyping, ethnic and religious
framing that doesn't survive literal translation from an English benchmark.

**BiasLens** closes that gap: paste Vietnamese (or English) text in, get it scored across **14
bias categories** on a 1–5 severity scale, and get an AI-rewritten, bias-reduced version of any flagged
text — all through a browser UI backed by a locally-hosted, LoRA-fine-tuned model.

## Live pipeline at a glance

```
Browser (localhost:3000)
  └─ Next.js UI  (Input → Detect → Mitigate → Evaluate)
        ├─ /api/detect-severity → FastAPI :8000 /detect-severity
        ├─ /api/mitigate        → FastAPI :8000 /mitigate
        └─ /api/metrics         → FastAPI :8000 /metrics
              FastAPI (src/training/serve.py)
                ├─ inference_llm_lora.py   Qwen2.5-1.5B-Instruct + LoRA   (local CUDA GPU)
                ├─ mitigate.py             GPT-4o via RMIT's VAL gateway + Bitwarden
                └─ evaluate snapshot       checkpoints/.../eval_metrics.json
```

Detection runs fully offline once the base model is cached; only mitigation needs network access
(a call to the hosted rewrite model).

## Features

* 14-category, multi-label bias detection with a 1–5 severity rating per category (not single-label —
  a sentence that's both sexist and classist gets both flagged, not forced into one bucket)
* LoRA-fine-tuned Qwen2.5-1.5B-Instruct as the production detector, trained on the human-annotated
  **VNFairness** dataset
* A comparable zero-shot GPT-4o baseline sharing the identical category rubric, for a fair
  apples-to-apples comparison against the fine-tuned model
* AI-powered mitigation: rewrites flagged text into a neutral version while preserving meaning, with
  **per-category rewrite guidance** (not just a generic "remove bias" instruction) so a sentence
  flagged for multiple categories gets a targeted instruction for each one in a single rewrite call
* A four-stage web UI (Input → Detect → Mitigate → Evaluate) with a plain-language results summary and
  a 14-axis radar chart (colour **and** marker-shape coded, since a colour-vision-deficiency check
  flagged colour alone as insufficient)
* Multi-label iterative stratification for the train/val/test split, so rare bias categories get a
  proportionally fair share of data in every split instead of being left almost absent by chance
* Optional 5-fold cross-validation tooling for a more statistically robust per-category estimate

## Tech stack

| Layer | Choice |
|---|---|
| Detector / trainer | Python, PyTorch (CUDA), Hugging Face Transformers, PEFT (LoRA), Datasets, Accelerate |
| Detection API | FastAPI + Uvicorn + Pydantic |
| Mitigation | `openai` SDK against an OpenAI-compatible gateway (RMIT VAL by default), key resolved from Bitwarden |
| Analysis UI | Next.js 15, React 19, TypeScript, Tailwind CSS |
| Data wrangling | pandas, scikit-learn, `iterative-stratification` (multi-label stratified splits) |
| Legacy chat/auth backend (separate product surface, see below) | Express, Prisma, SQLite, JWT, bcrypt |

## Repository layout

```
dataset/
  raw_v2/                Label Studio JSON exports (VNFairness) -- not committed, private
  processed_v2/          train/val/test CSVs built by convert_labelstudio.py
  convert_labelstudio.py Raw JSON -> stratified, leak-free train/val/test CSVs
  build_cv_folds.py      Optional: 5-fold multi-label-stratified CV data prep
  raw/, build_splits.py  Legacy -- only relevant to the abandoned PhoBERT experiment (see below)

src/training/
  config_llm.py          LoRA/Qwen2.5 hyperparameters and data/checkpoint paths
  train_llm_lora.py      Trains the production detector
  evaluate_llm_lora.py   Generation-based evaluation (severity + binary-presence views)
  inference_llm_lora.py  Loads the trained adapter for inference (used by serve.py)
  inference_severity_llm.py  Category rubric (shared with the LoRA model) + zero-shot GPT-4o baseline
  mitigate.py             AI rewrite of flagged text, with per-category guidance
  serve.py                FastAPI service tying it all together
  run_cv.py               Orchestrates the optional 5-fold cross-validation
  config.py, train_phobert.py, evaluate.py, inference_test.py  Legacy PhoBERT path (see below)

UI/                      Next.js analysis app (Input -> Detect -> Mitigate -> Evaluate)
UI/backend/              Separate Express/Prisma chat+auth backend (not wired to the detector)

test_cases/manual_test_cases.csv  55 hand-written QA sentences (single- and multi-category) for manual testing
poster/, report/         Showcase poster and capstone report drafts
```

## Getting started

All commands below are run from the repo root unless noted otherwise.

### 1. Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA-capable GPU with ≥6GB VRAM for training/inference. On Windows, watch for a second
Python install shadowing the one with `torch`/`transformers`/`peft` — always invoke `python -m uvicorn`
rather than a bare `uvicorn` on PATH (see step 6).

### 2. Build the dataset splits

Needs `dataset/raw_v2/` populated with VNFairness Label Studio export JSON (private, not committed).

```bash
python dataset/convert_labelstudio.py
```

Parses each task's human_input/model_output pair and its 14-category ratings, drops Label Studio QC
probes, keeps both Vietnamese and English variants (the base model is natively multilingual), and
splits into `dataset/processed_v2/{train,val,test}.csv`. The split is **grouped by task** (a task's two
rows never land in different splits) and **multi-label-stratified** (`MultilabelStratifiedShuffleSplit`)
so every category's share of the data stays consistent across train/val/test — a plain random split can
leave a rare category almost entirely out of val/test by chance. Prints a per-category balance report
so a skewed split is visible immediately.

### 3. Train

```bash
python src/training/train_llm_lora.py
```

LoRA-fine-tunes Qwen2.5-1.5B-Instruct (r=16, alpha=32, targeting all attention + MLP projections —
only ~1.2% of parameters trainable) with the rubric prompt masked out of the loss, so only the JSON
rating completion is learned from. Saves to `checkpoints/qwen_severity_lora_stratified/` (~2 hours on
a 6GB GPU with a clear system — background GPU load from other apps can slow this down significantly,
worth checking `nvidia-smi` if a run looks unexpectedly slow).

### 4. Evaluate

```bash
python src/training/evaluate_llm_lora.py            # test split (default)
python src/training/evaluate_llm_lora.py --split train --limit 500   # overfitting check
```

Doesn't score from teacher-forced logits — actually **generates** a completion per row and parses it,
measuring what the model produces at real inference time. Reports two views: exact 1–5 severity match,
and binary presence (rating > 1) with per-category precision/recall/F1. Writes
`checkpoints/qwen_severity_lora_stratified/eval_metrics.json`, which `serve.py`'s `/metrics` endpoint
(and the UI's Evaluate stage) reads.

### 5. (Optional) 5-fold cross-validation

For a more robust per-category estimate than a single split gives (useful for the rarest categories,
which a single split's small test set scores noisily):

```bash
python dataset/build_cv_folds.py     # once, builds dataset/processed_v2/cv/fold{0-4}/
python src/training/run_cv.py        # trains + evaluates one adapter per fold, ~2hr/fold
```

Resumable — a fold whose `eval_metrics.json` already exists is skipped, so an interrupted run only
costs the folds not yet finished. Aggregates mean±std F1 per category across folds into
`checkpoints/qwen_severity_lora_cv/cv_summary.json`.

### 6. Serve over HTTP

```bash
python -m uvicorn serve:app --reload --port 8000 --app-dir src/training
```

Exposes `POST /detect-severity` (the production 14-category detector), `POST /mitigate`, `GET /metrics`,
`GET /health`, plus the legacy single-label endpoints (`POST /detect`, `POST /detect/batch`,
`GET /metrics/legacy`) kept for the abandoned PhoBERT path. `--app-dir src/training` is required so
uvicorn can import `serve.py` without changing the working directory — `config_llm.py`'s
repo-root-relative paths depend on that.

**Use `python -m uvicorn`, not a bare `uvicorn`** — on a machine with more than one Python install
(e.g. Anaconda alongside python.org), a bare `uvicorn` on PATH can resolve to an interpreter missing
`torch`/`peft` and fail with `ModuleNotFoundError`.

### 7. Mitigation setup (Bitwarden)

`mitigate.py` talks to **RMIT's VAL gateway** by default (`https://val.rmit.edu.au/api/`, model
`openai-gpt-4o`), not OpenAI's own API — a plain OpenAI-format key/URL will fail auth against it. The
key is never stored in `.env`; it's pulled at request time from Bitwarden via the `bw` CLI:

```bash
npm install -g @bitwarden/cli
bw login
bw unlock                       # prints an export command with a session key
export BW_SESSION="<session key from the line above>"
```

Looks up the `VAL-Balam-Key3` item's password field by default (override with `OPENAI_BW_ITEM`), or
set `OPENAI_API_KEY` directly to skip Bitwarden entirely (e.g. in CI). `bw unlock` needs your master
password once per shell — `serve.py` never prompts for it, so keep that shell's `BW_SESSION` exported
while the API is running.

Test the key resolves (no API call, no tokens spent):

```bash
python src/training/mitigate.py
```

Test a full rewrite, with per-category guidance:

```bash
python src/training/mitigate.py "Phụ nữ thì nên ở nhà lo bếp núc." --categories sexism
```

### 8. Run the web UI

Two terminals, both from the repo root:

**Terminal 1** — detection API (step 6, keep it running):

```bash
python -m uvicorn serve:app --reload --port 8000 --app-dir src/training
```

**Terminal 2** — frontend:

```bash
cd UI
npm install   # first time only
npm run dev
```

Open **http://localhost:3000**. The UI's `/api/detect-severity`, `/api/mitigate`, and `/api/metrics`
routes proxy to `http://localhost:8000` and show a clear "service unreachable" error if it isn't
running yet.

`UI/backend/` (Express + Prisma, its own `npm run dev` on port 3001) is a **separate, unrelated**
chat/login backend (`/auth`, `/dashboard` pages) — not linked from the home page and not wired to the
detector. You don't need it for the analysis tool above.

## Results

On 288 held-out examples never seen during training (multi-label-stratified split):

| Metric | All 14 categories | 7 categories with ≥10 test examples |
|---|---|---|
| Binary bias-presence macro F1 | **54.7%** | ~70.0% |
| Exact 1–5 severity macro F1 | ~31% | — |

Two categories (ageism, linguistic-hierarchical bias) score near zero regardless of split methodology
— their total labelled example count (2–9 tasks across the whole dataset) is a data-collection gap, not
a modelling failure.

## Known limitations

* Two categories (ageism, linguistic-hierarchical bias) have too few labelled examples for any split
  strategy to produce a trustworthy score — needs targeted data collection, not a modeling fix.
* Mitigation quality (does the rewrite preserve meaning while removing bias?) is not measured
  quantitatively — no BLEU/ROUGE/BERTScore or human evaluation is wired in yet.
* Named-entity span extraction (which words/phrases carry the bias, and toward which target group) is
  not implemented — detection is sentence-level only.
* The zero-shot GPT-4o baseline (`inference_severity_llm.py`) shares the detector's rubric but isn't
  yet shown side-by-side with the LoRA model in the UI's Evaluate stage.
* The legacy chat/auth backend (`UI/backend/`) stores whatever `biasData` a client sends rather than
  calling the detector itself — it can display stale or client-invented analysis if used directly.

## Legacy / leftover components

The repository keeps two earlier, abandoned attempts rather than deleting them, since they're part of
the project's actual technical history:

* **PhoBERT 13-class classifier** (`train_phobert.py`, `evaluate.py`, `inference_test.py`, `config.py`,
  `POST /detect`, `POST /detect/batch`, `GET /metrics/legacy`) — trained on an older, template-generated
  corpus. Scores near-perfect (macro F1 ≈ 0.99), but that reflects in-template phrasing, not general
  Vietnamese text — treat it as a pipeline-correctness check, not a production accuracy number. **Not**
  the model the UI or production API path uses.
* **A from-scratch PhoBERT trained directly on the real VNFairness data** (`checkpoints/phobert_severity_classifier/`)
  collapsed to macro F1 ≈ 0.33 — the concrete evidence that motivated the pivot to LoRA-fine-tuning an
  instruction-tuned decoder instead of training a classifier from scratch (several categories have far
  too few labelled examples for a randomly-initialised head to learn from).

## Team

**Student team:** Balam — Nguyen Hai Long, Ho Gia Bao, Huynh Bao Khang, Nguyen Trieu Duong, Pham Minh Hoang
**Academic supervisor:** Tom Huynh · **Industry supervisor:** Ginel Dorleon · **Company:** SSET

RMIT University — Engineering & IT Capstone Project.

## License

MIT.
