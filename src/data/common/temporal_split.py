"""Leakage-safe temporal splitting utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd


def temporal_split(
    frame: pd.DataFrame,
    time_column: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split chronologically while keeping equal timestamps together."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")
    if frame.empty:
        raise ValueError("Cannot split an empty frame")

    data = frame.copy()
    data[time_column] = pd.to_datetime(data[time_column], utc=True, errors="raise")
    if data[time_column].isna().any():
        raise ValueError(f"Cannot split rows with missing {time_column}")
    data = data.sort_values([time_column], kind="mergesort").reset_index(drop=True)
    grouped = data.groupby(time_column, sort=True, dropna=False).size().reset_index(name="count")
    if len(grouped) < 3:
        raise ValueError("At least three distinct timestamps are required for train/val/test")
    grouped["cumulative"] = grouped["count"].cumsum()

    def nearest_boundary(target: float, lower: int, upper: int) -> int:
        candidates = grouped.index.to_series().loc[lower:upper]
        return int((grouped.loc[candidates, "cumulative"] - target).abs().idxmin())

    train_target = len(data) * ratios[0]
    val_target = len(data) * (ratios[0] + ratios[1])
    train_end = nearest_boundary(train_target, 0, len(grouped) - 3)
    val_end = nearest_boundary(val_target, train_end + 1, len(grouped) - 2)

    train_cutoff = grouped.loc[train_end, time_column]
    val_cutoff = grouped.loc[val_end, time_column]
    train = data[data[time_column] <= train_cutoff].copy()
    val = data[(data[time_column] > train_cutoff) & (data[time_column] <= val_cutoff)].copy()
    test = data[data[time_column] > val_cutoff].copy()

    metadata = {
        "time_column": time_column,
        "ratios_target": list(ratios),
        "rows": {"all": len(data), "train": len(train), "val": len(val), "test": len(test)},
        "cutoffs": {
            "train_end": train_cutoff.isoformat(),
            "validation_start": (train_cutoff + pd.Timedelta(nanoseconds=1)).isoformat(),
            "validation_end": val_cutoff.isoformat(),
            "test_start": (val_cutoff + pd.Timedelta(nanoseconds=1)).isoformat(),
        },
        "ranges": {
            "train": {"min": train[time_column].min().isoformat(), "max": train[time_column].max().isoformat()},
            "val": {"min": val[time_column].min().isoformat(), "max": val[time_column].max().isoformat()},
            "test": {"min": test[time_column].min().isoformat(), "max": test[time_column].max().isoformat()},
        },
        "timestamp_group_count": int(len(grouped)),
        "same_timestamp_kept_together": True,
        "shuffle": False,
    }
    return train, val, test, metadata
