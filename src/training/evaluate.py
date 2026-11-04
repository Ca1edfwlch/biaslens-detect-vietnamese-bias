"""Evaluates the fine-tuned checkpoint on the held-out test set.

Reports the full metric set docs/evaluation_plan.md asks for (accuracy,
precision, recall, macro F1, weighted F1, confusion matrix) plus a
per-class breakdown, since macro F1 alone hides which classes are weak.

Also writes a compact JSON snapshot (checkpoints/.../eval_metrics.json) so
other tools (e.g. the UI's Evaluate stage, served via serve.py's /metrics)
can show real numbers from the last run instead of hardcoded ones.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from config import OUTPUT_DIR, TEST_PATH, TRAIN_PATH, VAL_PATH
from metrics import compute_metrics
from trainer_utils import load_csv, tokenize

COLUMNS_TO_DROP = ["text", "label", "bias_terms", "target_group", "severity", "safer_text"]
METRICS_PATH = Path(OUTPUT_DIR) / "eval_metrics.json"


def load_test_split():
    ds = load_csv(TEST_PATH)
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label_id", "labels")
    ds = ds.remove_columns([c for c in COLUMNS_TO_DROP if c in ds.column_names])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def main():
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    id_to_label = model.config.id2label
    label_names = [id_to_label[i] for i in range(len(id_to_label))]

    test_ds = load_test_split()
    # eval-only Trainer -- output_dir is required but nothing is written there
    # (no save_strategy), so this just avoids Trainer's "tmp_trainer" default
    # littering the repo root.
    args = TrainingArguments(output_dir=f"{OUTPUT_DIR}/_eval_scratch", report_to=[])
    trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics)
    output = trainer.predict(test_ds)

    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids

    report = classification_report(labels, preds, target_names=label_names, zero_division=0)
    print(report)
    report_dict = classification_report(
        labels, preds, target_names=label_names, zero_division=0, output_dict=True
    )

    cm = confusion_matrix(labels, preds)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "".join(f"{name[:10]:>12}" for name in label_names)
    print(f"{'':>22}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{label_names[i][:20]:>22}{row_str}")

    metrics = {k: float(v) for k, v in output.metrics.items()} if output.metrics else {}
    print("\nAggregate metrics:", json.dumps(metrics, indent=2))

    _write_metrics_snapshot(label_names, report_dict, metrics)


def _write_metrics_snapshot(label_names, report_dict, aggregate_metrics) -> None:
    dataset_sizes = {
        "train": len(pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")),
        "val": len(pd.read_csv(VAL_PATH, encoding="utf-8-sig")),
        "test": len(pd.read_csv(TEST_PATH, encoding="utf-8-sig")),
    }

    per_category = [
        {
            "label": name,
            "precision": round(report_dict[name]["precision"], 4),
            "recall": round(report_dict[name]["recall"], 4),
            "f1": round(report_dict[name]["f1-score"], 4),
            "support": int(report_dict[name]["support"]),
        }
        for name in label_names
    ]

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_sizes,
        "overall": {
            "accuracy": round(report_dict["accuracy"], 4),
            "macro_precision": round(report_dict["macro avg"]["precision"], 4),
            "macro_recall": round(report_dict["macro avg"]["recall"], 4),
            "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
        },
        "per_category": per_category,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\nWrote metrics snapshot -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
