"""Converts dataset/raw_v2/*.json (Label Studio VNFairness task exports) into
dataset/processed_v2/{train,val,test}.csv for the LoRA fine-tuning pipeline
(train_llm_lora.py).

Each JSON file is one task: a human_input/model_output pair rated across N
bias categories on an ordinal severity scale (see DATASET_CARD.md). The
single predictions[0].result block is the adjudicated gold annotation for
this export -- despite the Label Studio key name "predictions", it is not a
raw pre-label to second-guess (confirmed with the data owner).

Unlike the earlier from-scratch classifier pipelines, this keeps BOTH
uid_lang variants (_vi and _en) -- the base model (Qwen2.5-1.5B-Instruct) is
natively multilingual, so there's no need to throw away half the dataset the
way a Vietnamese-only encoder would have required. Tasks with sys_ref !=
"n000" (attention checks / calibration anchors) are still dropped -- they're
synthetic QC probes, not organic training examples.

The category set is discovered from the data itself (every "<prefix>_score"
rating field, prefix stripped) rather than hardcoded, so this script doesn't
silently drift from whatever categories the real batch actually contains.

Splits are grouped by uid, not by row: each task produces two rows (one for
human_input, one for model_output), and both must land in the same split or
a held-out split could leak related content from a task also seen in train.

Within that grouping constraint, which uids land in which split is chosen by
multi-label iterative stratification (Sechidis et al.) over each task's
per-category "is this category flagged anywhere in this task" vector, not a
uniform random shuffle. Several categories are rare (a handful of positive
tasks out of the whole set), and a plain random grouped split can starve
val/test of a rare category by chance -- stratification keeps each split's
share of every category close to its share in the full dataset instead. It
does not manufacture examples for rare categories; a category with very few
labeled tasks overall still ends up thin everywhere, just proportionally so.

Run from the repo root: python dataset/convert_labelstudio.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

RAW_DIR = Path("dataset/raw_v2")
PROCESSED_DIR = Path("dataset/processed_v2")
CATEGORY_MAPPING_PATH = Path("dataset/metadata/category_mapping_v2.json")
SEED = 42

RATING_SUFFIX = "_score"


def _category_from_field(from_name: str, prefix: str) -> str | None:
    if not from_name.startswith(prefix) or not from_name.endswith(RATING_SUFFIX):
        return None
    return from_name[len(prefix) : -len(RATING_SUFFIX)]


def _parse_task(path: Path) -> dict | None:
    with open(path, encoding="utf-8") as f:
        task = json.load(f)

    data = task["data"]
    uid_lang = data["uid_lang"]
    if data.get("sys_ref", "n000") != "n000":
        return None

    predictions = task.get("predictions") or []
    if not predictions:
        return None
    result = predictions[0]["result"]

    human_scores: dict[str, int] = {}
    model_scores: dict[str, int] = {}
    human_reasoning = ""
    model_reasoning = ""
    for entry in result:
        from_name = entry["from_name"]
        if entry["type"] == "rating":
            rating = entry["value"]["rating"]
            cat = _category_from_field(from_name, "human_")
            if cat is not None:
                human_scores[cat] = rating
                continue
            cat = _category_from_field(from_name, "model_")
            if cat is not None:
                model_scores[cat] = rating
        elif entry["type"] == "textarea":
            text = " ".join(entry["value"].get("text", []))
            if from_name == "human_reasoning":
                human_reasoning = text
            elif from_name == "model_reasoning":
                model_reasoning = text

    return {
        "uid": data["uid"],
        "uid_lang": uid_lang,
        "human_input": data["human_input"],
        "model_output": data["model_output"],
        "human_scores": human_scores,
        "model_scores": model_scores,
        "human_reasoning": human_reasoning,
        "model_reasoning": model_reasoning,
    }


def _load_tasks() -> list[dict]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR} does not exist -- populate it with the VNFairness task JSON files first."
        )
    paths = sorted(RAW_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {RAW_DIR}.")

    tasks = []
    skipped_filtered = 0
    skipped_errors = 0
    for path in paths:
        try:
            task = _parse_task(path)
        except (KeyError, IndexError, TypeError) as exc:
            print(f"Skipping {path.name}: malformed task ({exc})")
            skipped_errors += 1
            continue
        if task is None:
            skipped_filtered += 1
            continue
        tasks.append(task)

    print(
        f"Loaded {len(tasks)} tasks from {len(paths)} files "
        f"({skipped_filtered} filtered out [QC task / no predictions], "
        f"{skipped_errors} skipped due to malformed JSON)."
    )
    return tasks


def _category_set(tasks: list[dict]) -> list[str]:
    categories: set[str] = set()
    for task in tasks:
        categories.update(task["human_scores"])
        categories.update(task["model_scores"])
    if not categories:
        raise ValueError("No rating categories found across any task -- check the JSON schema.")
    return sorted(categories)


def _rows_for_task(task: dict, categories: list[str]) -> list[dict] | None:
    rows = []
    for source, text, scores, reasoning in (
        ("human_input", task["human_input"], task["human_scores"], task["human_reasoning"]),
        ("model_output", task["model_output"], task["model_scores"], task["model_reasoning"]),
    ):
        missing = [c for c in categories if c not in scores]
        if missing:
            print(f"Skipping uid={task['uid']} ({source}): missing ratings for {missing}")
            return None
        row = {
            "uid": task["uid"],
            "uid_lang": task["uid_lang"],
            "source": source,
            "text": text.strip(),
            "reasoning": reasoning,
        }
        row.update({c: scores[c] for c in categories})
        rows.append(row)
    return rows


def _task_label_matrix(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """One row per uid: 1 per category if either its human_input or
    model_output row rates that category >1 (biased, per the binary_presence
    convention in evaluate_llm_lora.py), 0 otherwise. This is the vector
    multi-label stratification balances across splits -- not the raw 1-5
    ratings, since stratification needs a fixed, comparable label space and
    "does this task exhibit this category" is what the imbalance is about.
    """
    return df.groupby("uid")[categories].max().gt(1).astype(int)


def _stratified_group_split(
    df: pd.DataFrame, categories: list[str], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_flags = _task_label_matrix(df, categories)
    uids = task_flags.index.to_numpy()
    X = uids.reshape(-1, 1)
    y = task_flags.to_numpy()

    train_pos, temp_pos = next(
        MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y)
    )
    train_uids, temp_uids = uids[train_pos], uids[temp_pos]

    temp_y = task_flags.loc[temp_uids].to_numpy()
    val_pos, test_pos = next(
        MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(
            temp_uids.reshape(-1, 1), temp_y
        )
    )
    val_uids, test_uids = temp_uids[val_pos], temp_uids[test_pos]

    train_df = df[df["uid"].isin(train_uids)].reset_index(drop=True)
    val_df = df[df["uid"].isin(val_uids)].reset_index(drop=True)
    test_df = df[df["uid"].isin(test_uids)].reset_index(drop=True)
    return train_df, val_df, test_df


def _report_category_balance(
    df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, categories: list[str]
) -> None:
    """Prints each category's flagged-task share per split, so a skewed
    stratification result is visible immediately instead of only showing up
    later as a surprising eval_metrics.json number."""
    full_flags = _task_label_matrix(df, categories)
    rows = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_flags = full_flags.loc[full_flags.index.isin(split_df["uid"])]
        rows.append(split_flags.mean().rename(name))
    balance = pd.concat(rows, axis=1)
    balance["full"] = full_flags.mean()
    print("Per-category flagged-task share by split (target: close to 'full' in every row):")
    print(balance.round(3).to_string())


def _assert_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_uids, val_uids, test_uids = set(train_df["uid"]), set(val_df["uid"]), set(test_df["uid"])
    overlaps = {
        "train/val": train_uids & val_uids,
        "train/test": train_uids & test_uids,
        "val/test": val_uids & test_uids,
    }
    for name, overlap in overlaps.items():
        if overlap:
            raise AssertionError(f"Leakage detected between {name}: {len(overlap)} shared task uids")
    print("No leakage between splits (grouped by uid): confirmed.")


def main() -> None:
    tasks = _load_tasks()
    categories = _category_set(tasks)
    print(f"Found {len(categories)} rating categories: {categories}")

    rows = []
    for task in tasks:
        task_rows = _rows_for_task(task, categories)
        if task_rows is not None:
            rows.extend(task_rows)

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} rows from {df['uid'].nunique()} tasks (human_input + model_output).")
    print(df["uid_lang"].str.rsplit("_", n=1).str[-1].value_counts().to_string())

    train_df, val_df, test_df = _stratified_group_split(df, categories, SEED)
    _assert_no_leakage(train_df, val_df, test_df)
    _report_category_balance(df, train_df, val_df, test_df, categories)

    columns = ["uid", "uid_lang", "source", "text", "reasoning"] + categories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = PROCESSED_DIR / f"{name}.csv"
        split_df[columns].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(split_df):5d} rows -> {out_path}")

    CATEGORY_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATEGORY_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump({cat: idx for idx, cat in enumerate(categories)}, f, ensure_ascii=False, indent=2)
    print(f"Wrote category mapping -> {CATEGORY_MAPPING_PATH}")


if __name__ == "__main__":
    main()
