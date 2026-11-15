"""Validation checks shared by preprocessing pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_split_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    user_column: str,
    item_column: str,
) -> dict[str, Any]:
    def pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
        return set(zip(frame[user_column].astype(str), frame[item_column].astype(str)))

    sets = {"train": pairs(train), "val": pairs(val), "test": pairs(test)}
    return {
        "train_val_overlap": len(sets["train"] & sets["val"]),
        "train_test_overlap": len(sets["train"] & sets["test"]),
        "val_test_overlap": len(sets["val"] & sets["test"]),
        "duplicate_train_pairs": int(train.duplicated([user_column, item_column]).sum()),
        "duplicate_val_pairs": int(val.duplicated([user_column, item_column]).sum()),
        "duplicate_test_pairs": int(test.duplicated([user_column, item_column]).sum()),
    }


def validate_temporal_order(
    frame: pd.DataFrame,
    time_column: str,
) -> dict[str, Any]:
    times = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    return {
        "missing_timestamps": int(times.isna().sum()),
        "min_timestamp": times.min().isoformat() if times.notna().any() else None,
        "max_timestamp": times.max().isoformat() if times.notna().any() else None,
    }
