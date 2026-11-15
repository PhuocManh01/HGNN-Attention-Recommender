"""Small leakage-aware utilities for feature preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class NumericStats:
    """Statistics fitted only on an explicitly supplied training frame."""

    columns: tuple[str, ...]
    means: dict[str, float]
    scales: dict[str, float]
    fit_rows: int
    fit_scope: str


def fit_numeric_stats(
    train_frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    fit_scope: str = "train",
) -> NumericStats:
    """Fit means/scales from train only; callers must label the fit scope."""
    columns = tuple(columns)
    if not fit_scope.lower().startswith("train"):
        raise ValueError("Numeric statistics must be fitted on a train-scoped frame")
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in columns:
        if column not in train_frame:
            raise KeyError(f"Missing numeric training column: {column}")
        values = pd.to_numeric(train_frame[column], errors="coerce")
        if values.notna().sum() == 0:
            means[column], scales[column] = 0.0, 1.0
            continue
        means[column] = float(values.mean())
        scale = float(values.std(ddof=0))
        scales[column] = scale if scale > 0 else 1.0
    return NumericStats(columns, means, scales, int(len(train_frame)), fit_scope)


def transform_numeric(frame: pd.DataFrame, stats: NumericStats) -> pd.DataFrame:
    """Transform with frozen train statistics and preserve missing indicators."""
    result = frame.copy()
    for column in stats.columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_missing"] = values.isna()
        result[f"{column}_zscore"] = (values - stats.means[column]) / stats.scales[column]
    return result


def save_numeric_stats(stats: NumericStats, path) -> None:
    path.write_text(
        pd.Series({
            "columns": list(stats.columns),
            "means": stats.means,
            "scales": stats.scales,
            "fit_rows": stats.fit_rows,
            "fit_scope": stats.fit_scope,
        }).to_json(indent=2),
        encoding="utf-8",
    )
