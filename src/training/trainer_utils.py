import pandas as pd
from datasets import Dataset
from pyvi import ViTokenizer
from transformers import AutoTokenizer

from config import MAX_LENGTH, MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return Dataset.from_pandas(df)


def segment(text: str) -> str:
    """Word-segments Vietnamese text (e.g. "trung tâm" -> "trung_tâm").

    PhoBERT's BPE vocabulary was built on RDRSegmenter-segmented text, so
    feeding it raw, unsegmented text at train or inference time measurably
    hurts accuracy. pyvi.ViTokenizer is a lightweight, pure-Python stand-in
    for that segmentation step.
    """
    return ViTokenizer.tokenize(text)


def tokenize(batch):
    segmented = [segment(text) for text in batch["text"]]
    return tokenizer(
        segmented,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
