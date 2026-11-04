import json

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_LABELS,
    OUTPUT_DIR,
    TRAIN_PATH,
    VAL_PATH,
    MODEL_NAME,
)
from metrics import compute_metrics
from trainer_utils import load_csv, tokenize, tokenizer

LABEL_MAPPING_PATH = "dataset/metadata/label_mapping.json"

COLUMNS_TO_DROP = ["text", "label", "bias_terms", "target_group", "severity", "safer_text"]


def load_split(path):
    ds = load_csv(path)
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label_id", "labels")
    ds = ds.remove_columns([c for c in COLUMNS_TO_DROP if c in ds.column_names])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def load_label_maps():
    with open(LABEL_MAPPING_PATH, encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def compute_class_weights(train_ds, num_labels):
    """Inverse-frequency weights so the loss doesn't just favor majority classes.

    The dataset is ~4:1 imbalanced (Non-bias/Class-Socioeconomic vs. the
    smallest categories), and evaluation_plan.md's success criterion is
    macro F1, which weights every class equally regardless of frequency.
    """
    counts = torch.zeros(num_labels)
    for label in train_ds["labels"]:
        counts[int(label)] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_labels * counts)
    return weights


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    label_to_id, id_to_label = load_label_maps()

    train_ds = load_split(TRAIN_PATH)
    val_ds = load_split(VAL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    class_weights = compute_class_weights(train_ds, NUM_LABELS)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
