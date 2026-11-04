"""LoRA-fine-tunes Qwen2.5-1.5B-Instruct to rate Vietnamese/English text
across the 14 bias categories, instead of training a classifier from scratch.

Rationale: a from-scratch encoder classifier (PhoBERT, mDeBERTa, etc.) has to
learn the entire concept of "what sexism looks like" from ~900-2000 labeled
rows split across 14 categories -- several categories end up with single-digit
support. An instruction-tuned decoder model already has a working notion of
bias/stereotyping from its pretraining (the same reason the zero-shot prompted
approach in inference_severity_llm.py works reasonably well); fine-tuning it
only has to calibrate that existing understanding to this dataset's specific
rubric and labeling conventions, which should need much less data.

Reuses inference_severity_llm.py's category rubric (SYSTEM_PROMPT/CATEGORIES)
as the training prompt template, so the fine-tuned model and the zero-shot
Tier-1 module are directly comparable -- same instructions, same output format.

Trains with standard supervised fine-tuning: the prompt (rubric + input text)
is masked out of the loss, so only the JSON completion tokens are learned
from. LoRA + bf16 (no 4-bit quantization) keeps this within a 6GB GPU without
bitsandbytes' Windows-compatibility risk.

Run:
    python dataset/convert_labelstudio.py   (once, to build processed_v2/)
    python src/training/train_llm_lora.py   (from the repo root)
"""

import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from config_llm import (
    BATCH_SIZE,
    CATEGORY_MAPPING_PATH,
    EPOCHS,
    GRAD_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_SEVERITY_LEVELS,
    OUTPUT_DIR,
    SAMPLE_WEIGHT_CAP,
    TRAIN_PATH,
    VAL_PATH,
    WARMUP_RATIO,
)
from inference_severity_llm import CATEGORIES, SYSTEM_PROMPT


def load_categories() -> list[str]:
    with open(CATEGORY_MAPPING_PATH, encoding="utf-8") as f:
        cat_to_id = json.load(f)
    dataset_categories = sorted(cat_to_id, key=lambda c: cat_to_id[c])
    if set(dataset_categories) != set(CATEGORIES):
        raise ValueError(
            f"dataset/metadata/category_mapping_v2.json categories {dataset_categories} don't match "
            f"inference_severity_llm.CATEGORIES {CATEGORIES} -- the rubric and the real data have drifted "
            "apart, update one to match the other before training."
        )
    return CATEGORIES


def compute_sample_weights(df: pd.DataFrame, categories: list[str], cap: float) -> np.ndarray:
    """Per-row oversampling weight: the inverse frequency (within `df`) of the
    row's rarest flagged category (rating > 1, matching the binary_presence
    convention in evaluate_llm_lora.py), capped at `cap`. Rows with no flagged
    category (all-1 ratings) get the baseline weight 1.0 -- negatives are
    already the majority class and don't need oversampling.
    """
    flagged = df[categories].gt(1)
    freq = flagged.mean()  # fraction of rows flagged, per category
    freq = freq.replace(0, np.nan)  # a category with zero positives can't set a finite weight

    inv_freq = 1.0 / freq
    # For each row, the weight is the max inverse-frequency among its flagged
    # categories (rarest category dominates), or 1.0 if none are flagged.
    weighted = flagged.mul(inv_freq, axis=1).where(flagged)
    row_weight = weighted.max(axis=1).fillna(1.0)
    return row_weight.clip(lower=1.0, upper=cap).to_numpy(dtype="float64")


def build_examples(
    tokenizer, path: str, categories: list[str], weight_cap: float | None = None
) -> tuple[list[dict], np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    raw_ratings = df[categories].to_numpy(dtype="int64")
    if raw_ratings.min() < 1 or raw_ratings.max() > NUM_SEVERITY_LEVELS:
        raise ValueError(
            f"Ratings out of expected 1..{NUM_SEVERITY_LEVELS} range in {path}: "
            f"min={raw_ratings.min()} max={raw_ratings.max()}. Check config_llm.NUM_SEVERITY_LEVELS."
        )

    row_weights = compute_sample_weights(df, categories, weight_cap) if weight_cap is not None else None

    examples = []
    kept_weights = []
    for pos, (_, row) in enumerate(df.iterrows()):
        completion = json.dumps({c: int(row[c]) for c in categories}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["text"]},
        ]
        # apply_chat_template(tokenize=True) returns a BatchEncoding dict
        # ({"input_ids": [...], "attention_mask": [...]}), not a flat token
        # list -- pull "input_ids" out explicitly.
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)["input_ids"]
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}],
            tokenize=True,
            add_generation_prompt=False,
        )["input_ids"]

        if len(full_ids) > MAX_LENGTH:
            # Drop rather than silently truncate into the middle of the JSON
            # completion, which would teach the model to emit malformed JSON.
            print(f"Skipping a row: {len(full_ids)} tokens > MAX_LENGTH={MAX_LENGTH}")
            continue

        prompt_len = len(prompt_ids)
        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len

        examples.append({"input_ids": full_ids, "labels": labels})
        if row_weights is not None:
            kept_weights.append(row_weights[pos])

    return examples, np.array(kept_weights, dtype="float64") if row_weights is not None else np.ones(len(examples))


class SFTCollator:
    """Dynamic padding to the batch's own max length (not a fixed MAX_LENGTH),
    since most rows are much shorter than the worst case and static padding
    would waste both memory and compute on a 6GB GPU.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    """Trainer that oversamples training rows by a fixed per-example weight
    instead of the default uniform-random sampler. Reweighting the loss
    directly isn't practical here (loss is per-token cross-entropy over a
    generated JSON completion, not a per-category logit), so imbalance is
    instead addressed by how often a row is *seen* during training.
    """

    def __init__(self, *args, sample_weights: np.ndarray | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def _get_train_sampler(self, *args, **kwargs):
        if self.sample_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        return WeightedRandomSampler(
            weights=self.sample_weights.tolist(), num_samples=len(self.sample_weights), replacement=True
        )


def main():
    categories = load_categories()
    print(f"Training on {len(categories)} categories: {categories}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building training examples...")
    train_examples, train_weights = build_examples(tokenizer, TRAIN_PATH, categories, weight_cap=SAMPLE_WEIGHT_CAP)
    val_examples, _ = build_examples(tokenizer, VAL_PATH, categories)
    print(f"  train: {len(train_examples)} examples, val: {len(val_examples)} examples")
    print(
        f"  train sample weights: min={train_weights.min():.2f} max={train_weights.max():.2f} "
        f"mean={train_weights.mean():.2f} (cap={SAMPLE_WEIGHT_CAP})"
    )

    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=10,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        sample_weights=train_weights,
    )

    trainer.train()

    # peft's save_pretrained on a LoRA-wrapped model writes only the small
    # adapter weights (a few MB), not the full 1.5B-parameter base model.
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
