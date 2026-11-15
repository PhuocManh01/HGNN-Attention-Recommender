"""Temporal-safe uniform negative sampling."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplingConfig:
    negative_ratio: int = 1
    seed: int = 2026
    unique_per_positive: bool = True
    resample_each_epoch: bool = True

    def __post_init__(self) -> None:
        if self.negative_ratio < 1:
            raise ValueError("negative_ratio must be >= 1")


def _rng(config: SamplingConfig, epoch: int) -> np.random.Generator:
    effective_epoch = int(epoch) if config.resample_each_epoch else 0
    return np.random.default_rng(np.random.SeedSequence([config.seed, effective_epoch]))


def _normalise_pool(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values})


def sample_bpr_triplets(
    positives: pd.DataFrame,
    *,
    entity_column: str,
    item_column: str,
    candidate_pool: Callable[[Any, Any], Iterable[Any]],
    cutoff: Any,
    config: SamplingConfig | None = None,
    epoch: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample BPR triplets from an eligibility pool at one fixed cutoff."""
    config = config or SamplingConfig()
    rng = _rng(config, epoch)
    rows: list[dict[str, Any]] = []
    skipped = 0
    shortfall = 0
    pool_sizes: list[int] = []
    for _, positive in positives.iterrows():
        pool = sorted(set(_normalise_pool(candidate_pool(positive[entity_column], cutoff))) - {str(positive[item_column])})
        pool_sizes.append(len(pool))
        if not pool:
            skipped += 1
            shortfall += config.negative_ratio
            continue
        if config.unique_per_positive:
            count = min(config.negative_ratio, len(pool))
            shortfall += config.negative_ratio - count
            selected = rng.choice(np.asarray(pool, dtype=object), size=count, replace=False).tolist()
        else:
            selected = rng.choice(np.asarray(pool, dtype=object), size=config.negative_ratio, replace=True).tolist()
        for negative in selected:
            rows.append({
                entity_column: positive[entity_column],
                "positive_item": positive[item_column],
                "negative_item": negative,
                "is_assumed_negative": True,
                "source_positive_id": positive.get("interaction_id", positive.name),
                "epoch": int(epoch),
            })
    result = pd.DataFrame(rows, columns=[entity_column, "positive_item", "negative_item", "is_assumed_negative", "source_positive_id", "epoch"])
    stats = {
        "input_positive_rows": int(len(positives)),
        "output_triplets": int(len(result)),
        "negative_ratio": config.negative_ratio,
        "seed": config.seed,
        "epoch": epoch,
        "unique_per_positive": config.unique_per_positive,
        "resample_each_epoch": config.resample_each_epoch,
        "skipped_positive_rows": skipped,
        "negative_shortfall": shortfall,
        "pool_size_min": min(pool_sizes) if pool_sizes else 0,
        "pool_size_max": max(pool_sizes) if pool_sizes else 0,
        "pool_size_mean": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "false_negative_risk": "unobserved pairs are assumed negatives; future or unrecorded positives may be false negatives",
        "future_labels_used": False,
        "evaluation_protocol": "training sampling only; not a full-ranking evaluation metric",
    }
    return result, stats


def sample_evaluation_candidates(
    positives: pd.DataFrame,
    *,
    entity_column: str,
    item_column: str,
    candidate_pool: Callable[[Any, Any], Iterable[Any]],
    cutoff: Any,
    negative_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a fixed sampled evaluation set; it is not full-ranking."""
    config = SamplingConfig(negative_ratio=negative_count, seed=seed, unique_per_positive=True, resample_each_epoch=False)
    result, stats = sample_bpr_triplets(positives, entity_column=entity_column, item_column=item_column, candidate_pool=candidate_pool, cutoff=cutoff, config=config)
    stats.update({"evaluation_type": "sampled", "fixed_seed": seed, "negative_candidates_per_positive": negative_count})
    return result, stats


def sample_negatives(
    positives: pd.DataFrame,
    users: Iterable[object],
    items: Iterable[object],
    user_column: str,
    item_column: str,
    observed: dict[str, set[str]] | Callable[[pd.Series], Iterable[object]],
    ratio: int,
    seed: int,
    eligible_items: Callable[[pd.Series], Iterable[object]] | None = None,
) -> pd.DataFrame:
    """Backward-compatible sampler for unobserved pairs."""
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    rng = np.random.default_rng(seed)
    all_items = [str(item) for item in items]
    rows: list[dict[str, object]] = []

    for _, positive in positives.iterrows():
        user = str(positive[user_column])
        blocked = (
            set(observed(positive))
            if callable(observed)
            else observed.get(user, set())
        )
        pool = sorted(set(str(item) for item in (eligible_items(positive) if eligible_items else all_items)) - {str(value) for value in blocked} - {str(positive[item_column])})
        count = min(ratio, len(pool))
        if count == 0:
            continue
        for item in rng.choice(np.asarray(pool, dtype=object), size=count, replace=False).tolist():
            rows.append({
                user_column: positive[user_column],
                item_column: item,
                "is_positive": False,
                "source_positive_id": positive.get("interaction_id", positive.name),
            })
    return pd.DataFrame(rows)
