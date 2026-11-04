"""Builds 5 multi-label-stratified cross-validation folds from the same
VNFairness task pool convert_labelstudio.py uses, for a more robust
per-category F1 estimate than a single train/val/test split can give.

A single split leaves the rarest categories (2-9 tasks total) with only a
handful of test examples, so that category's reported F1 is mostly noise.
Under 5-fold CV every task is used as held-out test data in exactly one
fold, so a rare category's full available support gets evaluated across the
5 folds combined instead of just whatever landed in one fixed test set.

Each fold's test set is one of 5 disjoint, stratified partitions of the
whole task pool (MultilabelStratifiedKFold, balanced the same way as
convert_labelstudio.py's single split: per-category "flagged anywhere in
this task" vector). The other 4 folds are further split 90/10 into that
fold's train/val -- val exists only so LoRA training's
load_best_model_at_end has something to pick a checkpoint against, it is
not part of the held-out metric.

Run from the repo root: python dataset/build_cv_folds.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from convert_labelstudio import _category_set, _load_tasks, _rows_for_task, _task_label_matrix

N_FOLDS = 5
SEED = 42
CV_DIR = Path("dataset/processed_v2/cv")
CATEGORY_MAPPING_PATH = Path("dataset/metadata/category_mapping_v2.json")


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
    print(f"Built {len(df)} rows from {df['uid'].nunique()} tasks.")

    task_flags = _task_label_matrix(df, categories)
    uids = task_flags.index.to_numpy()
    X = uids.reshape(-1, 1)
    y = task_flags.to_numpy()

    columns = ["uid", "uid_lang", "source", "text", "reasoning"] + categories
    kfold = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (trainval_pos, test_pos) in enumerate(kfold.split(X, y)):
        trainval_uids = uids[trainval_pos]
        test_uids = uids[test_pos]

        trainval_y = task_flags.loc[trainval_uids].to_numpy()
        train_pos, val_pos = next(
            MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(
                trainval_uids.reshape(-1, 1), trainval_y
            )
        )
        train_uids = trainval_uids[train_pos]
        val_uids = trainval_uids[val_pos]

        assert not (set(train_uids) & set(val_uids))
        assert not (set(train_uids) & set(test_uids))
        assert not (set(val_uids) & set(test_uids))

        fold_dir = CV_DIR / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, split_uids in [("train", train_uids), ("val", val_uids), ("test", test_uids)]:
            split_df = df[df["uid"].isin(split_uids)].reset_index(drop=True)
            out_path = fold_dir / f"{name}.csv"
            split_df[columns].to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"fold{fold_idx} {name}: {len(split_df):5d} rows ({len(split_uids)} tasks) -> {out_path}")

    CATEGORY_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATEGORY_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump({cat: idx for idx, cat in enumerate(categories)}, f, ensure_ascii=False, indent=2)
    print(f"Wrote category mapping -> {CATEGORY_MAPPING_PATH}")


if __name__ == "__main__":
    main()
