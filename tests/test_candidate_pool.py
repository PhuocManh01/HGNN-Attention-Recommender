import json
from pathlib import Path

import pandas as pd

from src.data.candidate_pool import JTHCandidatePool, MovieLensCandidatePool, prepare


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_pool_outputs_and_ground_truth_reports():
    prepare()
    for dataset in ("movielens", "jth"):
        output_dir = ROOT / "datasets" / dataset / "processed"
        assert (output_dir / "candidate_pool_config.json").exists()
        assert (output_dir / "candidate_pool_statistics.json").exists()
        stats = json.loads((output_dir / "candidate_pool_statistics.json").read_text(encoding="utf-8"))
        assert stats["ground_truth_eligibility"]["validation"]["ground_truth_rows"] > 0
        assert stats["ground_truth_eligibility"]["test"]["ground_truth_rows"] > 0


def test_movielens_pool_is_deterministic_and_blocks_only_prior_ratings():
    output_dir = ROOT / "datasets" / "movielens" / "processed"
    pool = MovieLensCandidatePool.from_processed(output_dir)
    metadata = json.loads((output_dir / "split_metadata.json").read_text(encoding="utf-8"))
    cutoff = metadata["split"]["cutoffs"]["train_end"]
    first = pool.for_user(1, cutoff)
    second = pool.for_user(1, cutoff)
    assert first == second
    assert len(first) == len(set(first))
    interactions = pd.read_csv(output_dir / "interactions.csv", parse_dates=["interaction_time"])
    interactions["interaction_time"] = pd.to_datetime(interactions["interaction_time"], utc=True)
    prior = interactions[(interactions.user_id == 1) & (interactions.interaction_time <= pd.Timestamp(cutoff))]
    assert not set(prior.movie_id.astype(int)).intersection(first)
    assert len(first) <= 9742


def test_jth_pool_respects_create_date_and_handles_no_history_candidate():
    output_dir = ROOT / "datasets" / "jth" / "processed"
    pool = JTHCandidatePool.from_processed(output_dir)
    cutoff = "2024-03-08T00:00:00Z"
    no_history = next(candidate for candidate in pool.all_candidate_ids if candidate not in pool.observed_by_candidate)
    first = pool.for_candidate(no_history, cutoff)
    second = pool.for_candidate(no_history, cutoff)
    assert first == second
    assert len(first) == len(set(first))
    assert all(pool.job_create_dates[job_id] <= pd.Timestamp(cutoff) for job_id in first)
    assert len(first) > 0


def test_jth_ground_truth_exclusions_are_explicitly_reported():
    stats = json.loads((ROOT / "datasets" / "jth" / "processed" / "candidate_pool_statistics.json").read_text(encoding="utf-8"))
    for split in ("validation", "test"):
        excluded = stats["ground_truth_eligibility"][split]["excluded_rows"]
        assert excluded["job_created_after_cutoff"] >= 0
        assert excluded["observed_before_cutoff"] == 0
