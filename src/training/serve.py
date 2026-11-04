"""HTTP wrapper around the detection module, for the UI to call.

Thin layer over inference_test.predict()/predict_batch() and
inference_llm_lora.predict() -- no inference logic duplicated here. config.py
and config_llm.py's paths are all relative to the repo root, so run this
from the repo root with --app-dir (not from src/training/, which would
resolve those paths wrong).

If this machine has more than one Python installation (e.g. Anaconda plus a
plain python.org install), make sure `uvicorn` here resolves to the same
interpreter that has torch/transformers/peft installed -- a bare `uvicorn`
on PATH can silently pick up a different one and fail with
"ModuleNotFoundError: No module named 'peft'" on startup. `python -m
uvicorn` sidesteps that by using whichever `python` you already have on PATH:

    python -m uvicorn serve:app --reload --port 8000 --app-dir src/training
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import OUTPUT_DIR
from config_llm import OUTPUT_DIR as SEVERITY_OUTPUT_DIR
from inference_llm_lora import predict as predict_severity
from inference_test import DEFAULT_THRESHOLD, predict, predict_batch
from mitigate import mitigate as run_mitigation

# Legacy single-label PhoBERT metrics -- kept only because /detect (legacy) is
# still exposed below; the UI's Evaluate stage now reads SEVERITY_METRICS_PATH.
METRICS_PATH = Path(OUTPUT_DIR) / "eval_metrics.json"
SEVERITY_METRICS_PATH = Path(SEVERITY_OUTPUT_DIR) / "eval_metrics.json"

app = FastAPI(title="BiasLens detection service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    text: str
    threshold: float = DEFAULT_THRESHOLD


class DetectBatchRequest(BaseModel):
    texts: list[str]
    threshold: float = DEFAULT_THRESHOLD


class SeverityDetectRequest(BaseModel):
    text: str


class MitigateRequest(BaseModel):
    text: str
    label: str | None = None
    # Raw detector category keys (e.g. ["sexism", "ageism"]), preferred over
    # `label` when present -- lets mitigate.py look up specific per-category
    # rewrite guidance instead of just a human-readable label string.
    categories: list[str] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect")
def detect(req: DetectRequest) -> dict:
    return predict(req.text, threshold=req.threshold)


@app.post("/detect/batch")
def detect_batch(req: DetectBatchRequest) -> list[dict]:
    return predict_batch(req.texts, threshold=req.threshold)


@app.post("/detect-severity")
def detect_severity(req: SeverityDetectRequest) -> dict:
    """14-category multi-label detector (LoRA-fine-tuned Qwen2.5, see
    src/training/train_llm_lora.py), replacing the single-label /detect
    model above for the UI. First call loads the model onto the GPU and is
    slower; inference_llm_lora caches it in-process after that.
    """
    return predict_severity(req.text)


@app.post("/mitigate")
def mitigate(req: MitigateRequest) -> dict:
    try:
        return run_mitigation(req.text, label=req.label, categories=req.categories)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/metrics")
def metrics() -> dict:
    """Eval snapshot for the model the UI actually uses (LoRA severity
    detector), written by evaluate_llm_lora.py.
    """
    if not SEVERITY_METRICS_PATH.exists():
        raise HTTPException(
            status_code=404, detail="No eval_metrics.json yet -- run evaluate_llm_lora.py first"
        )
    with open(SEVERITY_METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/metrics/legacy")
def metrics_legacy() -> dict:
    """Eval snapshot for the old single-label PhoBERT classifier (/detect,
    /detect/batch above) -- not used by the current UI.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No eval_metrics.json yet -- run evaluate.py first")
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)
