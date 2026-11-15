"""Prepare negative-sampling configurations and leakage checks.

Actual training code should call :func:`sample_bpr_triplets` with the same
candidate-pool getter and cutoff.  This module deliberately writes metadata
and a bounded pilot statistic, not a full negative-sample table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.candidate_pool import JTHCandidatePool, MovieLensCandidatePool
from src.data.common.negative_sampler import SamplingConfig, sample_bpr_triplets
from src.paths import DATA_ROOT


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _cutoff(output_dir: Path) -> str:
    metadata = json.loads((output_dir / "split_metadata.json").read_text(encoding="utf-8"))
    return metadata["split"]["cutoffs"]["train_end"]


def _pilot(positives: pd.DataFrame, entity: str, item: str, getter: Any, cutoff: str) -> dict[str, Any]:
    # Bound the metadata run; training can resample the complete frame lazily.
    pilot = positives.head(128).copy()
    started = time.perf_counter()
    _, stats = sample_bpr_triplets(
        pilot,
        entity_column=entity,
        item_column=item,
        candidate_pool=getter,
        cutoff=cutoff,
        config=SamplingConfig(negative_ratio=1, seed=2026, unique_per_positive=True, resample_each_epoch=True),
        epoch=0,
    )
    stats["pilot_rows"] = len(pilot)
    stats["full_training_positive_rows"] = len(positives)
    elapsed = time.perf_counter() - started
    stats["pilot_elapsed_seconds"] = round(elapsed, 4)
    stats["pilot_rows_per_second"] = round(len(pilot) / max(elapsed, 1e-9), 2)
    return stats


def prepare_movielens() -> dict[str, Any]:
    output_dir = DATA_ROOT / "movielens" / "processed"
    pool = MovieLensCandidatePool.from_processed(output_dir)
    positives = pd.read_csv(output_dir / "train.csv", usecols=["user_id", "movie_id"])
    cutoff = _cutoff(output_dir)
    config = {
        "dataset": "movielens",
        "sampler": "uniform",
        "negative_ratio": 1,
        "seed": 2026,
        "resample_each_epoch": True,
        "unique_per_positive": True,
        "cutoff": cutoff,
        "candidate_pool": "MovieLensCandidatePool.for_user; all ratings <= cutoff are blocked",
        "observed_non_positive_blocked": True,
        "future_labels_used": False,
        "false_negative_risk": "unobserved ratings are assumed negative; future/unrecorded positives can be false negatives",
        "evaluation": "prefer full ranking over the fixed candidate pool; sampled evaluation is not equivalent",
    }
    stats = _pilot(positives, "user_id", "movie_id", pool.for_user, cutoff)
    stats["dataset"] = "movielens"
    _write_json(output_dir / "negative_sampling_config.json", config)
    _write_json(output_dir / "negative_sampling_statistics.json", stats)
    return stats


def prepare_jth() -> dict[str, Any]:
    output_dir = DATA_ROOT / "jth" / "processed"
    pool = JTHCandidatePool.from_processed(output_dir)
    positives = pd.read_csv(output_dir / "train.csv", dtype={"candidate_id": "string", "job_id": "string"}, usecols=["candidate_id", "job_id"])
    cutoff = _cutoff(output_dir)
    config = {
        "dataset": "jth",
        "sampler": "uniform",
        "negative_ratio": 1,
        "seed": 2026,
        "resample_each_epoch": True,
        "unique_per_positive": True,
        "cutoff": cutoff,
        "candidate_pool": "JTHCandidatePool.for_candidate; job create_date <= cutoff and prior pairs blocked",
        "future_stage_outcomes_used": False,
        "future_labels_used": False,
        "false_negative_risk": "unobserved applications are assumed negative; future/unrecorded applications can be false negatives",
        "evaluation": "prefer full ranking over the fixed candidate pool; sampled evaluation is not equivalent",
    }
    stats = _pilot(positives, "candidate_id", "job_id", pool.for_candidate, cutoff)
    stats["dataset"] = "jth"
    _write_json(output_dir / "negative_sampling_config.json", config)
    _write_json(output_dir / "negative_sampling_statistics.json", stats)
    return stats


def prepare() -> dict[str, Any]:
    return {"movielens": prepare_movielens(), "jth": prepare_jth()}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, ensure_ascii=False, default=str))
