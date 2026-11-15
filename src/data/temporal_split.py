"""Create chronological train/validation/test tables for both datasets.

This module only splits observed positive ground-truth interactions. Negative
sampling and graph construction are intentionally separate later stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.common.temporal_split import temporal_split
from src.paths import DATA_ROOT


RATIOS = (0.8, 0.1, 0.1)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _pair_set(frame: pd.DataFrame, user_column: str, item_column: str) -> set[tuple[str, str]]:
    return set(zip(frame[user_column].astype(str), frame[item_column].astype(str)))


def _cold_start(train: pd.DataFrame, current: pd.DataFrame, user_column: str, item_column: str) -> dict[str, int]:
    train_users, train_items = set(train[user_column].astype(str)), set(train[item_column].astype(str))
    return {
        "new_users": int(current.loc[~current[user_column].astype(str).isin(train_users), user_column].astype(str).nunique()),
        "new_items": int(current.loc[~current[item_column].astype(str).isin(train_items), item_column].astype(str).nunique()),
    }


def _stats(
    splits: dict[str, pd.DataFrame],
    time_column: str,
    user_column: str,
    item_column: str,
    target_rows: int,
) -> dict[str, Any]:
    train, val, test = splits["train"], splits["val"], splits["test"]
    train_pairs, val_pairs, test_pairs = (_pair_set(frame, user_column, item_column) for frame in (train, val, test))
    history_for_test = pd.concat([train, val], ignore_index=True)
    counts = {name: int(len(frame)) for name, frame in splits.items()}
    return {
        "rows": counts,
        "ratios_target": list(RATIOS),
        "ratios_actual": {name: count / target_rows for name, count in counts.items()},
        "ratio_deviation": {name: count / target_rows - RATIOS[index] for index, (name, count) in enumerate(counts.items())},
        "timestamp_ranges": {
            name: {"min": frame[time_column].min().isoformat(), "max": frame[time_column].max().isoformat()}
            for name, frame in splits.items()
        },
        "unique_entities": {
            name: {"users": int(frame[user_column].nunique()), "items": int(frame[item_column].nunique())}
            for name, frame in splits.items()
        },
        "cold_start_vs_history": {
            "val": _cold_start(train, val, user_column, item_column),
            "test": _cold_start(history_for_test, test, user_column, item_column),
        },
        "pair_overlap": {
            "train_val": len(train_pairs & val_pairs),
            "train_test": len(train_pairs & test_pairs),
            "val_test": len(val_pairs & test_pairs),
        },
        "timestamp_overlap": {
            "train_val": int(set(train[time_column]) & set(val[time_column]) != set()),
            "train_test": int(set(train[time_column]) & set(test[time_column]) != set()),
            "val_test": int(set(val[time_column]) & set(test[time_column]) != set()),
        },
    }


def _write_split_outputs(output_dir: Path, splits: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def _split_dataset(
    *,
    name: str,
    frame: pd.DataFrame,
    output_dir: Path,
    time_column: str,
    user_column: str,
    item_column: str,
    source_rows: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = frame.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    train, val, test, split_meta = temporal_split(frame, time_column, RATIOS)
    splits = {"train": train, "val": val, "test": test}
    stats = _stats(splits, time_column, user_column, item_column, len(frame))
    metadata: dict[str, Any] = {
        "dataset": name,
        "source_rows": source_rows,
        "ground_truth_rows": len(frame),
        "time_column": time_column,
        "split": split_meta,
        "graph_history_protocol": {
            "train": "train only",
            "validation": "train only; validation positives are future labels",
            "test": "train plus validation history, applied consistently across models",
            "future_edges_in_train_graph": False,
            "validation_edges_added_before_test": True,
        },
        "negative_sampling": "not performed in this stage",
        "statistics_file": "split_statistics.json",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_split_outputs(output_dir, splits)
    _write_json(output_dir / "split_metadata.json", metadata)
    _write_json(output_dir / "split_statistics.json", stats)
    return {"dataset": name, **stats["rows"], "metadata": str(output_dir / "split_metadata.json")}


def split_movielens() -> dict[str, Any]:
    output_dir = DATA_ROOT / "movielens" / "processed"
    interactions = pd.read_csv(output_dir / "interactions.csv", parse_dates=["interaction_time"])
    positive = interactions[interactions["is_positive"]].copy()
    return _split_dataset(
        name="movielens",
        frame=positive,
        output_dir=output_dir,
        time_column="interaction_time",
        user_column="user_id",
        item_column="movie_id",
        source_rows=len(interactions),
        extra_metadata={
            "ground_truth_definition": "is_positive == True (rating >= configured threshold)",
            "observed_non_positive_rows_excluded_from_ground_truth": int((~interactions["is_positive"]).sum()),
            "observed_non_positive_source": "interactions.csv; reserved for negative sampling exclusion",
            "future_rating_features_used": False,
        },
    )


def split_jth() -> dict[str, Any]:
    output_dir = DATA_ROOT / "jth" / "processed"
    interactions = pd.read_csv(output_dir / "interactions.csv", dtype={"candidate_id": "string", "job_id": "string", "application_id": "string"}, parse_dates=["interaction_time"])
    candidates = pd.read_csv(output_dir / "candidates_clean.csv", dtype={"candidate_id": "string"}, usecols=["candidate_id", "create_date"])
    jobs = pd.read_csv(output_dir / "jobs_clean.csv", dtype={"job_id": "string"}, usecols=["job_id", "create_date"])
    candidates["create_date"] = pd.to_datetime(candidates["create_date"], utc=True, errors="coerce")
    jobs["create_date"] = pd.to_datetime(jobs["create_date"], utc=True, errors="coerce")
    candidate_times, job_times = dict(zip(candidates["candidate_id"], candidates["create_date"])), dict(zip(jobs["job_id"], jobs["create_date"]))
    before_candidate = sum(pd.notna(candidate_times.get(row.candidate_id)) and row.interaction_time < candidate_times[row.candidate_id] for row in interactions.itertuples())
    before_job = sum(pd.notna(job_times.get(row.job_id)) and row.interaction_time < job_times[row.job_id] for row in interactions.itertuples())
    anomalies = pd.read_csv(output_dir / "temporal_anomalies.csv")
    return _split_dataset(
        name="jth",
        frame=interactions,
        output_dir=output_dir,
        time_column="interaction_time",
        user_column="candidate_id",
        item_column="job_id",
        source_rows=42288,
        extra_metadata={
            "ground_truth_definition": "first interaction per candidate-job pair from preprocessing",
            "future_stage_outcomes_used": False,
            "unreliable_timestamp_anomaly_rows": int(len(anomalies)),
            "interaction_before_candidate_create_date": int(before_candidate),
            "interaction_before_job_create_date": int(before_job),
            "proxy_timestamp_source": "earliest valid stage timestamp; end_of_process_date excluded",
        },
    )


def prepare() -> list[dict[str, Any]]:
    return [split_movielens(), split_jth()]


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
