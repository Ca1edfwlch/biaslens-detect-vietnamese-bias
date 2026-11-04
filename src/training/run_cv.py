"""Runs 5-fold multi-label-stratified cross-validation for the LoRA severity
detector: trains one adapter per fold into its own checkpoint directory,
evaluates it on that fold's held-out test set, and aggregates per-category
F1 (mean +/- std) across all 5 folds -- a more trustworthy estimate for
low-support categories than any single train/val/test split, since every
task is used as held-out test data in exactly one fold.

Each fold is a full LoRA training run (~2 hours on a 6GB GPU), so this
script takes on the order of 10 hours end to end. Resumable: a fold whose
eval_metrics.json already exists is skipped, so a crash partway through only
costs the folds not yet finished -- delete a fold's checkpoint dir to force
it to rerun.

Prerequisite: python dataset/build_cv_folds.py (once, to build the fold CSVs)

Run from the repo root: python src/training/run_cv.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

N_FOLDS = 5
CV_DATA_DIR = Path("dataset/processed_v2/cv")
CV_CHECKPOINT_DIR = Path("checkpoints/qwen_severity_lora_cv")
CATEGORIES_PATH = Path("dataset/metadata/category_mapping_v2.json")


def _run(cmd: list[str], env: dict) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def run_fold(fold_idx: int) -> dict:
    fold_data_dir = CV_DATA_DIR / f"fold{fold_idx}"
    fold_output_dir = CV_CHECKPOINT_DIR / f"fold{fold_idx}"
    metrics_path = fold_output_dir / "eval_metrics.json"

    if metrics_path.exists():
        print(f"fold{fold_idx}: {metrics_path} already exists, skipping (delete it to force a rerun)")
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    env = os.environ.copy()
    env["BIASLENS_TRAIN_PATH"] = str(fold_data_dir / "train.csv")
    env["BIASLENS_VAL_PATH"] = str(fold_data_dir / "val.csv")
    env["BIASLENS_TEST_PATH"] = str(fold_data_dir / "test.csv")
    env["BIASLENS_OUTPUT_DIR"] = str(fold_output_dir)

    _run([sys.executable, "src/training/train_llm_lora.py"], env=env)
    _run([sys.executable, "src/training/evaluate_llm_lora.py", "--split", "test"], env=env)

    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def aggregate(fold_metrics: list[dict], categories: list[str]) -> dict:
    binary_f1s = {c: [] for c in categories}
    severity_f1s = {c: [] for c in categories}
    overall_binary, overall_severity = [], []

    for m in fold_metrics:
        overall_binary.append(m["binary_presence"]["overall"]["macro_f1_across_categories"])
        overall_severity.append(m["severity"]["overall"]["macro_f1_across_categories"])
        for row in m["binary_presence"]["per_category"]:
            binary_f1s[row["category"]].append(row["f1"])
        for row in m["severity"]["per_category"]:
            severity_f1s[row["category"]].append(row["macro_f1"])

    per_category = [
        {
            "category": c,
            "binary_f1_mean": round(float(np.mean(binary_f1s[c])), 4),
            "binary_f1_std": round(float(np.std(binary_f1s[c])), 4),
            "severity_f1_mean": round(float(np.mean(severity_f1s[c])), 4),
            "severity_f1_std": round(float(np.std(severity_f1s[c])), 4),
        }
        for c in categories
    ]

    return {
        "n_folds": len(fold_metrics),
        "overall": {
            "binary_macro_f1_mean": round(float(np.mean(overall_binary)), 4),
            "binary_macro_f1_std": round(float(np.std(overall_binary)), 4),
            "severity_macro_f1_mean": round(float(np.mean(overall_severity)), 4),
            "severity_macro_f1_std": round(float(np.std(overall_severity)), 4),
        },
        "per_category": per_category,
    }


def main() -> None:
    if not CV_DATA_DIR.exists():
        raise FileNotFoundError(f"{CV_DATA_DIR} does not exist -- run dataset/build_cv_folds.py first.")
    with open(CATEGORIES_PATH, encoding="utf-8") as f:
        cat_to_id = json.load(f)
    categories = sorted(cat_to_id, key=lambda c: cat_to_id[c])

    fold_metrics = []
    for fold_idx in range(N_FOLDS):
        print(f"\n{'=' * 60}\nFold {fold_idx}/{N_FOLDS - 1}\n{'=' * 60}", flush=True)
        fold_metrics.append(run_fold(fold_idx))

    summary = aggregate(fold_metrics, categories)
    CV_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = CV_CHECKPOINT_DIR / "cv_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"\n5-fold CV overall binary macro F1: "
        f"{summary['overall']['binary_macro_f1_mean']:.4f} +/- {summary['overall']['binary_macro_f1_std']:.4f}"
    )
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
