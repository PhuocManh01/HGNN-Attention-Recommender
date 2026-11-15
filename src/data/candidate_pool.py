"""Leakage-safe, reproducible candidate pools for MovieLens and JTH.

The pools are lazy: a query returns one entity's eligible items and is cached
in memory.  This stage does not create negatives and does not materialize a
full user-item or candidate-job matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.paths import DATA_ROOT


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _cutoffs(output_dir: Path) -> dict[str, pd.Timestamp]:
    metadata = json.loads((output_dir / "split_metadata.json").read_text(encoding="utf-8"))
    raw = metadata["split"]["cutoffs"]
    return {"validation": _utc(raw["train_end"]), "test": _utc(raw["validation_end"])}


@dataclass
class MovieLensCandidatePool:
    """Static-catalog MovieLens pool excluding ratings known at cutoff."""

    catalog_movie_ids: tuple[int, ...]
    observed_by_user: dict[int, tuple[tuple[pd.Timestamp, int], ...]]
    cache: dict[tuple[int, str], tuple[int, ...]] = field(default_factory=dict)
    max_cache_entries: int = 1024

    @classmethod
    def from_processed(cls, output_dir: Path | None = None) -> "MovieLensCandidatePool":
        output_dir = output_dir or DATA_ROOT / "movielens" / "processed"
        movies = pd.read_csv(output_dir / "movies_clean.csv", usecols=["movie_id"])
        interactions = pd.read_csv(
            output_dir / "interactions.csv",
            usecols=["user_id", "movie_id", "interaction_time"],
            dtype={"user_id": "int64", "movie_id": "int64"},
            parse_dates=["interaction_time"],
        )
        interactions["interaction_time"] = pd.to_datetime(interactions["interaction_time"], utc=True)
        observed: dict[int, list[tuple[pd.Timestamp, int]]] = {}
        for row in interactions.itertuples(index=False):
            observed.setdefault(int(row.user_id), []).append((_utc(row.interaction_time), int(row.movie_id)))
        return cls(
            catalog_movie_ids=tuple(sorted(movies["movie_id"].astype(int).unique())),
            observed_by_user={key: tuple(value) for key, value in observed.items()},
        )

    def for_user(self, user_id: int, cutoff: Any) -> tuple[int, ...]:
        cutoff = _utc(cutoff)
        key = (int(user_id), cutoff.isoformat())
        if key not in self.cache:
            blocked = {movie_id for time, movie_id in self.observed_by_user.get(int(user_id), ()) if time <= cutoff}
            self.cache[key] = tuple(movie_id for movie_id in self.catalog_movie_ids if movie_id not in blocked)
            if len(self.cache) > self.max_cache_entries:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


@dataclass
class JTHCandidatePool:
    """JTH pool based on job creation and first observed pair interaction."""

    job_create_dates: dict[str, pd.Timestamp]
    observed_by_candidate: dict[str, tuple[tuple[pd.Timestamp, str], ...]]
    all_candidate_ids: tuple[str, ...]
    cache: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)
    eligible_job_cache: dict[str, tuple[str, ...]] = field(default_factory=dict)
    max_cache_entries: int = 256

    @classmethod
    def from_processed(cls, output_dir: Path | None = None) -> "JTHCandidatePool":
        output_dir = output_dir or DATA_ROOT / "jth" / "processed"
        candidates = pd.read_csv(output_dir / "candidates_clean.csv", usecols=["candidate_id"], dtype={"candidate_id": "string"})
        jobs = pd.read_csv(output_dir / "jobs_clean.csv", usecols=["job_id", "create_date"], dtype={"job_id": "string"})
        interactions = pd.read_csv(
            output_dir / "interactions.csv",
            usecols=["candidate_id", "job_id", "interaction_time"],
            dtype={"candidate_id": "string", "job_id": "string"},
            parse_dates=["interaction_time"],
        )
        jobs["create_date"] = pd.to_datetime(jobs["create_date"], utc=True, errors="coerce")
        interactions["interaction_time"] = pd.to_datetime(interactions["interaction_time"], utc=True, errors="coerce")
        observed: dict[str, list[tuple[pd.Timestamp, str]]] = {}
        for row in interactions.itertuples(index=False):
            observed.setdefault(str(row.candidate_id), []).append((_utc(row.interaction_time), str(row.job_id)))
        return cls(
            job_create_dates={str(row.job_id): _utc(row.create_date) for row in jobs.itertuples(index=False) if pd.notna(row.create_date)},
            observed_by_candidate={key: tuple(value) for key, value in observed.items()},
            all_candidate_ids=tuple(sorted(candidates["candidate_id"].astype(str).unique())),
        )

    def for_candidate(self, candidate_id: str, cutoff: Any) -> tuple[str, ...]:
        cutoff = _utc(cutoff)
        candidate_id = str(candidate_id)
        key = (candidate_id, cutoff.isoformat())
        if key not in self.cache:
            cutoff_key = cutoff.isoformat()
            if cutoff_key not in self.eligible_job_cache:
                self.eligible_job_cache[cutoff_key] = tuple(sorted(job_id for job_id, create_date in self.job_create_dates.items() if create_date <= cutoff))
            blocked = {job_id for time, job_id in self.observed_by_candidate.get(candidate_id, ()) if time <= cutoff}
            self.cache[key] = tuple(job_id for job_id in self.eligible_job_cache[cutoff_key] if job_id not in blocked)
            if len(self.cache) > self.max_cache_entries:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


def _movie_ground_truth_stats(pool: MovieLensCandidatePool, output_dir: Path, cutoffs: dict[str, pd.Timestamp]) -> dict[str, Any]:
    observed = pd.read_csv(output_dir / "interactions.csv", usecols=["user_id", "movie_id", "interaction_time"])
    observed["interaction_time"] = pd.to_datetime(observed["interaction_time"], utc=True)
    result: dict[str, Any] = {}
    catalog = set(pool.catalog_movie_ids)
    for split, cutoff in cutoffs.items():
        targets = pd.read_csv(output_dir / f"{('val' if split == 'validation' else 'test')}.csv", usecols=["user_id", "movie_id"])
        prior = observed[observed["interaction_time"] <= cutoff]
        prior_pairs = set(zip(prior["user_id"].astype(int), prior["movie_id"].astype(int)))
        excluded: dict[str, int] = {"observed_before_cutoff": 0, "not_in_static_catalog": 0}
        examples: dict[str, list[dict[str, Any]]] = {key: [] for key in excluded}
        for row in targets.itertuples(index=False):
            movie_id = int(row.movie_id)
            pair = (int(row.user_id), movie_id)
            reason = "not_in_static_catalog" if movie_id not in catalog else ("observed_before_cutoff" if pair in prior_pairs else None)
            if reason:
                excluded[reason] += 1
                if len(examples[reason]) < 5:
                    examples[reason].append({"user_id": int(row.user_id), "movie_id": movie_id})
        result[split] = {"ground_truth_rows": len(targets), "excluded_rows": excluded, "examples": examples}
    return result


def _jth_ground_truth_stats(pool: JTHCandidatePool, output_dir: Path, cutoffs: dict[str, pd.Timestamp]) -> dict[str, Any]:
    observed = pd.read_csv(
        output_dir / "interactions.csv",
        dtype={"candidate_id": "string", "job_id": "string"},
        usecols=["candidate_id", "job_id", "interaction_time"],
    )
    observed["interaction_time"] = pd.to_datetime(observed["interaction_time"], utc=True)
    result: dict[str, Any] = {}
    for split, cutoff in cutoffs.items():
        targets = pd.read_csv(output_dir / f"{('val' if split == 'validation' else 'test')}.csv", dtype={"candidate_id": "string", "job_id": "string"}, usecols=["candidate_id", "job_id"])
        prior = observed[observed["interaction_time"] <= cutoff]
        prior_pairs = set(zip(prior["candidate_id"].astype(str), prior["job_id"].astype(str)))
        excluded = {"job_created_after_cutoff": 0, "observed_before_cutoff": 0}
        examples = {key: [] for key in excluded}
        for row in targets.itertuples(index=False):
            job_id, candidate_id = str(row.job_id), str(row.candidate_id)
            reason = "job_created_after_cutoff" if job_id not in pool.job_create_dates or pool.job_create_dates[job_id] > cutoff else ("observed_before_cutoff" if (candidate_id, job_id) in prior_pairs else None)
            if reason:
                excluded[reason] += 1
                if len(examples[reason]) < 5:
                    examples[reason].append({"candidate_id": candidate_id, "job_id": job_id})
        result[split] = {"ground_truth_rows": len(targets), "excluded_rows": excluded, "examples": examples}
    return result


def _prepare_movielens() -> dict[str, Any]:
    output_dir = DATA_ROOT / "movielens" / "processed"
    pool = MovieLensCandidatePool.from_processed(output_dir)
    cutoffs = _cutoffs(output_dir)
    interactions = pd.read_csv(output_dir / "interactions.csv", usecols=["user_id", "movie_id", "interaction_time"])
    interactions["interaction_time"] = pd.to_datetime(interactions["interaction_time"], utc=True)
    stats: dict[str, Any] = {
        "dataset": "movielens",
        "catalog_movies": len(pool.catalog_movie_ids),
        "catalog_assumption": "static_catalog: every catalog movie is eligible at every cutoff; no reliable availability timestamp exists",
        "release_year_used_for_eligibility": False,
        "pool_generation": "lazy_per_user_with_in_memory_cache",
        "cutoffs": {name: value.isoformat() for name, value in cutoffs.items()},
        "ground_truth_eligibility": _movie_ground_truth_stats(pool, output_dir, cutoffs),
        "unrated_vs_new": {},
    }
    for name, cutoff in cutoffs.items():
        rated = set(interactions.loc[interactions["interaction_time"] <= cutoff, "movie_id"].astype(int))
        future_first = set(interactions.loc[interactions["interaction_time"] > cutoff, "movie_id"].astype(int)) - rated
        stats["unrated_vs_new"][name] = {
            "catalog_unrated_at_cutoff": len(set(pool.catalog_movie_ids) - rated),
            "items_first_seen_after_cutoff": len(future_first),
            "new_item_status": "not_identifiable_without availability history; future first rating is not treated as item creation",
        }
    config = {
        "dataset": "movielens",
        "entity_key": "user_id",
        "item_key": "movie_id",
        "ground_truth": "positive interactions only",
        "observed_blocking": "ratings with interaction_time <= query cutoff, including non-positive ratings",
        "future_events_used_for_pool": False,
        "catalog_availability": "static catalog assumption",
        "full_matrix_written": False,
        "query_api": "MovieLensCandidatePool.for_user(user_id, cutoff)",
    }
    _write_json(output_dir / "candidate_pool_config.json", config)
    _write_json(output_dir / "candidate_pool_statistics.json", stats)
    return stats


def _prepare_jth() -> dict[str, Any]:
    output_dir = DATA_ROOT / "jth" / "processed"
    pool = JTHCandidatePool.from_processed(output_dir)
    cutoffs = _cutoffs(output_dir)
    stats: dict[str, Any] = {
        "dataset": "jth",
        "candidates": len(pool.all_candidate_ids),
        "jobs": len(pool.job_create_dates),
        "pool_generation": "lazy_per_candidate_with_in_memory_cache",
        "cutoffs": {name: value.isoformat() for name, value in cutoffs.items()},
        "job_eligibility": "job create_date <= cutoff",
        "job_open_status": "unknown; no reliable closing date/open-state field is available",
        "future_stage_outcomes_used": False,
        "candidates_without_history": {},
        "ground_truth_eligibility": _jth_ground_truth_stats(pool, output_dir, cutoffs),
    }
    interactions = pd.read_csv(output_dir / "interactions.csv", dtype={"candidate_id": "string"}, usecols=["candidate_id", "interaction_time"])
    interactions["interaction_time"] = pd.to_datetime(interactions["interaction_time"], utc=True)
    for name, cutoff in cutoffs.items():
        eligible_jobs = sum(create_date <= cutoff for create_date in pool.job_create_dates.values())
        history_candidates = set(interactions.loc[interactions["interaction_time"] <= cutoff, "candidate_id"].astype(str))
        active_candidates = set(pool.all_candidate_ids)
        sizes = []
        blocked_counts = interactions[interactions["interaction_time"] <= cutoff].groupby("candidate_id").size()
        for candidate_id in active_candidates:
            sizes.append(max(0, eligible_jobs - int(blocked_counts.get(candidate_id, 0))))
        stats["candidates_without_history"][name] = {
            "count": len(active_candidates - history_candidates),
            "eligible_jobs_at_cutoff": int(eligible_jobs),
            "pool_size_min": min(sizes),
            "pool_size_max": max(sizes),
            "pool_size_mean": sum(sizes) / len(sizes),
            "zero_pool_candidates": sum(size == 0 for size in sizes),
        }
    config = {
        "dataset": "jth",
        "entity_key": "candidate_id",
        "item_key": "job_id",
        "ground_truth": "first candidate-job interaction only",
        "job_eligibility": "create_date <= query cutoff",
        "pair_blocking": "first interaction_time <= query cutoff",
        "job_open_state_used": False,
        "future_stage_outcomes_used": False,
        "future_events_used_for_pool": False,
        "full_matrix_written": False,
        "query_api": "JTHCandidatePool.for_candidate(candidate_id, cutoff)",
    }
    _write_json(output_dir / "candidate_pool_config.json", config)
    _write_json(output_dir / "candidate_pool_statistics.json", stats)
    return stats


def prepare() -> dict[str, Any]:
    return {"movielens": _prepare_movielens(), "jth": _prepare_jth()}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, ensure_ascii=False, default=str))
