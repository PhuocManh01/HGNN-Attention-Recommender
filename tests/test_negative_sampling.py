import pandas as pd
from pathlib import Path

from src.data.candidate_pool import JTHCandidatePool, MovieLensCandidatePool
from src.data.common.negative_sampler import SamplingConfig, sample_bpr_triplets, sample_evaluation_candidates
from src.data.common.negative_sampler import sample_negatives


ROOT = Path(__file__).resolve().parents[1]


def test_negative_samples_are_not_observed():
    positives = pd.DataFrame({"user_id": [1], "item_id": [10], "interaction_id": ["p1"]})
    negatives = sample_negatives(
        positives,
        users=[1],
        items=[10, 11, 12, 13],
        user_column="user_id",
        item_column="item_id",
        observed={"1": {"10", "11"}},
        ratio=2,
        seed=42,
    )
    assert len(negatives) == 2
    assert not set(negatives["item_id"].astype(str)).intersection({"10", "11"})


def test_bpr_sampling_is_reproducible_and_unique():
    positives = pd.DataFrame({"user_id": [1, 2], "movie_id": [10, 20]})
    pool = lambda user, cutoff: {1: [10, 11, 12, 13], 2: [20, 21, 22, 23]}[int(user)]
    config = SamplingConfig(negative_ratio=2, seed=7, unique_per_positive=True)
    first, stats = sample_bpr_triplets(positives, entity_column="user_id", item_column="movie_id", candidate_pool=pool, cutoff="cutoff", config=config)
    second, _ = sample_bpr_triplets(positives, entity_column="user_id", item_column="movie_id", candidate_pool=pool, cutoff="cutoff", config=config)
    assert first.equals(second)
    assert len(first) == 4
    assert not set(first.negative_item).intersection({"10", "20"})
    assert first.groupby("user_id").negative_item.nunique().eq(2).all()
    assert stats["false_negative_risk"]


def test_epoch_resampling_and_empty_pool_are_safe():
    positives = pd.DataFrame({"user_id": [1, 2], "movie_id": [10, 20]})
    pool = lambda user, cutoff: [10] if int(user) == 1 else []
    config = SamplingConfig(negative_ratio=1, seed=11, unique_per_positive=True, resample_each_epoch=True)
    first, stats = sample_bpr_triplets(positives, entity_column="user_id", item_column="movie_id", candidate_pool=pool, cutoff=0, config=config)
    assert first.empty
    assert stats["skipped_positive_rows"] == 2
    assert stats["negative_shortfall"] == 2


def test_epoch_seed_changes_only_when_resampling_is_enabled():
    positives = pd.DataFrame({"user_id": [1], "movie_id": [10]})
    pool = lambda user, cutoff: [11, 12, 13, 14]
    stable = SamplingConfig(negative_ratio=1, seed=11, resample_each_epoch=False)
    first, _ = sample_bpr_triplets(positives, entity_column="user_id", item_column="movie_id", candidate_pool=pool, cutoff=0, config=stable, epoch=0)
    second, _ = sample_bpr_triplets(positives, entity_column="user_id", item_column="movie_id", candidate_pool=pool, cutoff=0, config=stable, epoch=1)
    assert first["negative_item"].tolist() == second["negative_item"].tolist()


def test_fixed_sampled_evaluation_is_distinct_from_full_ranking():
    positives = pd.DataFrame({"user_id": [1], "movie_id": [10]})
    pool = lambda user, cutoff: [10, 11, 12, 13, 14]
    result, stats = sample_evaluation_candidates(positives, entity_column="user_id", item_column="movie_id", candidate_pool=pool, cutoff=0, negative_count=3, seed=5)
    assert len(result) == 3
    assert stats["evaluation_type"] == "sampled"
    assert stats["negative_candidates_per_positive"] == 3


def test_real_movielens_training_negatives_respect_cutoff_pool():
    output = ROOT / "datasets" / "movielens" / "processed"
    positives = pd.read_csv(output / "train.csv", nrows=16, usecols=["user_id", "movie_id"])
    cutoff = "2016-02-04T21:34:50Z"
    pool = MovieLensCandidatePool.from_processed(output)
    result, stats = sample_bpr_triplets(positives, entity_column="user_id", item_column="movie_id", candidate_pool=pool.for_user, cutoff=cutoff, config=SamplingConfig(negative_ratio=2, seed=3))
    assert stats["output_triplets"] == 32
    for row in result.itertuples(index=False):
        assert int(row.negative_item) in pool.for_user(int(row.user_id), cutoff)


def test_real_jth_training_negatives_are_not_prior_pairs():
    output = ROOT / "datasets" / "jth" / "processed"
    positives = pd.read_csv(output / "train.csv", nrows=16, dtype={"candidate_id": "string", "job_id": "string"}, usecols=["candidate_id", "job_id"])
    cutoff = "2024-03-08T00:00:00Z"
    pool = JTHCandidatePool.from_processed(output)
    result, stats = sample_bpr_triplets(positives, entity_column="candidate_id", item_column="job_id", candidate_pool=pool.for_candidate, cutoff=cutoff, config=SamplingConfig(negative_ratio=2, seed=4))
    assert stats["output_triplets"] == 32
    for row in result.itertuples(index=False):
        assert str(row.negative_item) in pool.for_candidate(str(row.candidate_id), cutoff)


def test_sampling_outputs_are_generated():
    for dataset in ("movielens", "jth"):
        output = ROOT / "datasets" / dataset / "processed"
        assert (output / "negative_sampling_config.json").exists()
        assert (output / "negative_sampling_statistics.json").exists()
