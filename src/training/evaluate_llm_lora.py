"""Evaluates the LoRA-fine-tuned Qwen2.5 severity model on a dataset split.
Unlike a classification head, a generative model can't be scored from
teacher-forced logits alone -- this actually generates a completion per row
(via inference_llm_lora.predict_batch) and parses it, so it's slower than the
earlier classifier eval scripts but measures what the model actually produces
at inference time.

Reports two views of the same predictions:
  - "severity" -- exact match against the raw 1-5 ratings per category.
  - "binary_presence" -- collapsed to rating>1 (biased) vs. rating==1 (not),
    to see how much of the severity task's difficulty is "wrong by one
    level" vs. "missed the category entirely." Both are computed from a
    single generation pass; raw predictions are also saved to a CSV so this
    comparison (or others) can be re-run without hitting the GPU again.

Run against test (default, what serve.py's /metrics endpoint reads):
    python evaluate_llm_lora.py

Run against train or val -- e.g. to compare train vs. test performance as an
overfitting check (a much smaller train-vs-test gap than the train-loss-vs-
eval-loss gap seen during training would suggest the loss-based overfitting
signal didn't translate into a real accuracy gap):
    python evaluate_llm_lora.py --split train
    python evaluate_llm_lora.py --split train --limit 500   # faster, random subsample
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from config_llm import NUM_SEVERITY_LEVELS, OUTPUT_DIR, SEED, TEST_PATH, TRAIN_PATH, VAL_PATH
from inference_llm_lora import predict_batch
from inference_severity_llm import CATEGORIES

SPLIT_PATHS = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}


def _severity_report(true_ratings: np.ndarray, pred_ratings: np.ndarray) -> tuple[list[dict], float, float]:
    level_labels = list(range(1, NUM_SEVERITY_LEVELS + 1))
    per_category = []
    f1s = []
    exact_matches = []
    for i, name in enumerate(CATEGORIES):
        report = classification_report(
            true_ratings[:, i], pred_ratings[:, i], labels=level_labels, zero_division=0, output_dict=True
        )
        f1s.append(report["macro avg"]["f1-score"])
        per_category.append(
            {
                "category": name,
                "macro_precision": round(report["macro avg"]["precision"], 4),
                "macro_recall": round(report["macro avg"]["recall"], 4),
                "macro_f1": round(report["macro avg"]["f1-score"], 4),
                "exact_match_rate": round(float(np.mean(true_ratings[:, i] == pred_ratings[:, i])), 4),
                "support": len(true_ratings),
            }
        )
        exact_matches.append(true_ratings[:, i] == pred_ratings[:, i])

    all_exact_match = float(np.mean(np.all(np.stack(exact_matches, axis=1), axis=1)))
    return per_category, round(float(np.mean(f1s)), 4), round(all_exact_match, 4)


def _binary_report(true_ratings: np.ndarray, pred_ratings: np.ndarray) -> tuple[list[dict], float, float]:
    true_flagged = (true_ratings > 1).astype(int)
    pred_flagged = (pred_ratings > 1).astype(int)

    per_category = []
    f1s = []
    for i, name in enumerate(CATEGORIES):
        report = classification_report(
            true_flagged[:, i], pred_flagged[:, i], labels=[0, 1], zero_division=0, output_dict=True
        )
        f1s.append(report["1"]["f1-score"])
        per_category.append(
            {
                "category": name,
                "precision": round(report["1"]["precision"], 4),
                "recall": round(report["1"]["recall"], 4),
                "f1": round(report["1"]["f1-score"], 4),
                "support": int(report["1"]["support"]),
            }
        )

    exact_match = float(np.mean(np.all(true_flagged == pred_flagged, axis=1)))
    return per_category, round(float(np.mean(f1s)), 4), round(exact_match, 4)


def evaluate_split(split: str, limit: int | None = None) -> dict:
    df = pd.read_csv(SPLIT_PATHS[split], encoding="utf-8-sig")
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=SEED).reset_index(drop=True)

    texts = df["text"].tolist()
    true_ratings = df[CATEGORIES].to_numpy(dtype="int64")

    print(f"Generating predictions for {len(texts)} {split} rows...")
    results = predict_batch(texts)
    pred_ratings = np.array(
        [[results[i]["categories"][c]["rating"] for c in CATEGORIES] for i in range(len(texts))]
    )

    predictions_path = Path(OUTPUT_DIR) / f"predictions_{split}.csv"
    pred_df = pd.DataFrame(pred_ratings, columns=[f"pred_{c}" for c in CATEGORIES])
    true_df = pd.DataFrame(true_ratings, columns=[f"true_{c}" for c in CATEGORIES])
    pd.concat([df[["uid", "source", "text"]], true_df, pred_df], axis=1).to_csv(
        predictions_path, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote raw predictions -> {predictions_path}")

    severity_per_category, severity_macro_f1, severity_exact_match = _severity_report(true_ratings, pred_ratings)
    binary_per_category, binary_macro_f1, binary_exact_match = _binary_report(true_ratings, pred_ratings)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen2.5-1.5B-Instruct + LoRA",
        "split": split,
        "dataset": {name: len(pd.read_csv(path, encoding="utf-8-sig")) for name, path in SPLIT_PATHS.items()},
        "rows_evaluated": len(df),
        "severity": {
            "overall": {
                "macro_f1_across_categories": severity_macro_f1,
                "all_categories_exact_match": severity_exact_match,
            },
            "per_category": severity_per_category,
        },
        "binary_presence": {
            "overall": {
                "macro_f1_across_categories": binary_macro_f1,
                "all_categories_exact_match": binary_exact_match,
            },
            "per_category": binary_per_category,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--limit", type=int, default=None, help="Randomly subsample to this many rows (faster, e.g. for train)"
    )
    args = parser.parse_args()

    snapshot = evaluate_split(args.split, limit=args.limit)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))

    # test keeps the original filename so serve.py's /metrics endpoint (which
    # the UI reads) doesn't need to change; other splits get their own file.
    filename = "eval_metrics.json" if args.split == "test" else f"eval_metrics_{args.split}.json"
    metrics_path = Path(OUTPUT_DIR) / filename
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\nWrote metrics snapshot -> {metrics_path}")


if __name__ == "__main__":
    main()
