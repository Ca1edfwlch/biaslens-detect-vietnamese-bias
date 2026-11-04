"""Inference against the fine-tuned checkpoint: single text or a batch.

CLI usage:
    Single line:
        python inference_test.py "Người già thường khó tiếp thu."

    Mass detect over a file (one sentence per line, or a CSV with a "text" column):
        python inference_test.py --file sentences.txt
        python inference_test.py --file input.csv --out results.csv

    Flag every category above a confidence threshold, not just the top one:
        python inference_test.py "..." --threshold 0.2
"""

import argparse
import csv
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import OUTPUT_DIR
from trainer_utils import segment

DEFAULT_THRESHOLD = 0.2

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
        _model.eval()
    return _tokenizer, _model


def predict(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    return predict_batch([text], threshold=threshold)[0]


def predict_batch(
    texts: list[str], batch_size: int = 16, threshold: float = DEFAULT_THRESHOLD
) -> list[dict]:
    """Runs detection over many texts, batching the forward passes for speed
    rather than calling predict() in a loop.

    The model is trained single-label (one softmax over 13 competing
    categories, matching docs/annotation_guideline.md's "assign only ONE
    primary label" rule), so "labels" below is a threshold heuristic on top
    of that -- it surfaces every category whose softmax score clears
    `threshold`, not an independently-calibrated multi-label probability.
    Because softmax scores compete (they sum to 1), a very confident top
    label pushes every other score down, so this works best for genuinely
    ambiguous/mixed sentences rather than clear-cut ones.
    """
    tokenizer, model = _load()
    id_to_label = model.config.id2label
    results = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        segmented = [segment(t) for t in chunk]
        inputs = tokenizer(segmented, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        for text, row in zip(chunk, probs):
            pred_id = int(torch.argmax(row).item())
            above_threshold = sorted(
                (
                    {"label": id_to_label[i], "confidence": round(p.item(), 4)}
                    for i, p in enumerate(row)
                    if p.item() >= threshold
                ),
                key=lambda x: x["confidence"],
                reverse=True,
            )
            # the top label always counts, even if it's under threshold,
            # so callers always get at least one label back.
            if not above_threshold:
                above_threshold = [
                    {"label": id_to_label[pred_id], "confidence": round(row[pred_id].item(), 4)}
                ]

            results.append(
                {
                    "text": text,
                    "label": id_to_label[pred_id],
                    "confidence": round(row[pred_id].item(), 4),
                    "labels": above_threshold,
                    "probabilities": {
                        id_to_label[i]: round(p.item(), 4) for i, p in enumerate(row)
                    },
                }
            )

    return results


def _read_texts(path: str) -> list[str]:
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "text" not in (reader.fieldnames or []):
                raise ValueError(f"{path} has no 'text' column")
            return [row["text"] for row in reader if row["text"].strip()]

    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_results(results: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "confidence", "labels"])
        for r in results:
            labels_str = "|".join(f"{l['label']}:{l['confidence']}" for l in r["labels"])
            writer.writerow([r["text"], r["label"], r["confidence"], labels_str])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="A single sentence to classify")
    parser.add_argument("--file", help="Path to a .txt (one sentence per line) or .csv (with a 'text' column) to mass-detect")
    parser.add_argument("--out", help="Write results to this CSV instead of printing them (only with --file)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for flagging additional labels beyond the top one (default {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    if args.file:
        texts = _read_texts(args.file)
        results = predict_batch(texts, threshold=args.threshold)
        if args.out:
            _write_results(results, args.out)
            print(f"Wrote {len(results)} results -> {args.out}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        text = args.text or "Người già thường khó tiếp thu."
        print(json.dumps(predict(text, threshold=args.threshold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
