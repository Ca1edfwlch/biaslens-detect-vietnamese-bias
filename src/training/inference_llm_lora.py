"""Inference against the LoRA-fine-tuned Qwen2.5 severity model: single text
or a batch. Same output shape as inference_severity_llm.py (the zero-shot
Tier-1 module) -- {"rating": 1-5, "flagged": bool} per category -- so the
fine-tuned model and the zero-shot prompted model are directly comparable.

CLI usage:
    Single line:
        python inference_llm_lora.py "Nguoi gia thuong kho tiep thu."

    Mass score over a file (one sentence per line, or a CSV with a "text" column):
        python inference_llm_lora.py --file sentences.txt
        python inference_llm_lora.py --file input.csv --out results.json
"""

import argparse
import csv
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_llm import MODEL_NAME, OUTPUT_DIR
from inference_severity_llm import CATEGORIES, SYSTEM_PROMPT

_model = None
_tokenizer = None

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Instruct models -- especially a lightly-trained LoRA adapter -- often
    wrap JSON output in markdown code fences out of habit even when the
    system prompt asks for raw JSON. Training on more real data should mostly
    train this habit out, but parsing defensively is cheap insurance either way.
    """
    return _CODE_FENCE_RE.sub("", text.strip()).strip()


def _load():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
        _model = PeftModel.from_pretrained(base, OUTPUT_DIR).to("cuda")
        _model.eval()
    return _model, _tokenizer


def predict(text: str) -> dict:
    return predict_batch([text])[0]


def predict_batch(texts: list[str]) -> list[dict]:
    """One generate() call per text (matching inference_severity_llm.py's
    per-text API call pattern) -- simpler and more robust than batched
    generation, which needs left-padding and per-sequence completion
    trimming to get right.
    """
    model, tokenizer = _load()
    results = []

    for text in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        # apply_chat_template(tokenize=True, return_tensors="pt") returns a
        # BatchEncoding dict ({"input_ids": tensor, "attention_mask": tensor}),
        # not a raw tensor -- unpack it into generate() rather than passing
        # the dict itself as input_ids.
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        completion = _strip_code_fence(completion)

        try:
            ratings = json.loads(completion)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model returned invalid JSON for text={text!r}: {completion!r}") from exc

        categories_out = {}
        for name in CATEGORIES:
            rating = ratings.get(name)
            if not isinstance(rating, int) or not (1 <= rating <= 5):
                raise ValueError(f"Model returned an invalid rating for {name!r}: {rating!r} (text={text!r})")
            categories_out[name] = {"rating": rating, "flagged": rating > 1}

        results.append({"text": text, "categories": categories_out})

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="A single sentence to score")
    parser.add_argument("--file", help="Path to a .txt (one sentence per line) or .csv (with a 'text' column) to mass-score")
    parser.add_argument("--out", help="Write results to this JSON file instead of printing them (only with --file)")
    args = parser.parse_args()

    if args.file:
        texts = _read_texts(args.file)
        results = predict_batch(texts)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Wrote {len(results)} results -> {args.out}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        text = args.text or "Nguoi gia thuong kho tiep thu."
        print(json.dumps(predict(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
