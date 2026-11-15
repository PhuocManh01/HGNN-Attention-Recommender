import json

import pandas as pd
import pytest

from src.data.common.feature_utils import fit_numeric_stats, transform_numeric
from src.data.common.id_mapping import build_mapping, mapping_object, reverse_mapping


def test_id_mapping_is_deterministic_and_reversible():
    mapping = build_mapping(["b", "a", "b", "c"], "raw_id")
    assert mapping["id"].tolist() == [0, 1, 2]
    assert mapping_object(mapping, "raw_id").encode(["c", "a"]) == [2, 0]
    assert mapping_object(mapping, "raw_id").decode([2, 0]) == ["c", "a"]
    assert reverse_mapping(mapping, "raw_id")["raw_id"].tolist() == ["a", "b", "c"]
    with pytest.raises(KeyError):
        mapping_object(mapping, "raw_id").encode(["unknown"])


def test_numeric_fit_is_train_only():
    train = pd.DataFrame({"salary": [10.0, 20.0]})
    stats = fit_numeric_stats(train, ["salary"], fit_scope="train_temporal_prefix")
    transformed = transform_numeric(pd.DataFrame({"salary": [30.0, None]}), stats)
    assert transformed.loc[0, "salary_zscore"] == pytest.approx(3.0)
    assert transformed.loc[1, "salary_missing"]
    with pytest.raises(ValueError):
        fit_numeric_stats(train, ["salary"], fit_scope="validation")


def test_generated_feature_schemas_are_leakage_safe():
    movie_schema = json.loads(open("datasets/movielens/processed/feature_schema.json", encoding="utf-8").read())
    jth_schema = json.loads(open("datasets/jth/processed/feature_schema.json", encoding="utf-8").read())
    jth_availability = json.loads(open("datasets/jth/processed/feature_availability.json", encoding="utf-8").read())
    movie_features = pd.read_csv("datasets/movielens/processed/movie_features.csv")
    jth_candidates = pd.read_csv("datasets/jth/processed/candidate_features.csv")
    jth_jobs = pd.read_csv("datasets/jth/processed/job_features.csv")

    assert len(movie_features) == 9742
    assert "rating" not in movie_features.columns
    assert movie_schema["train_fit_required_for_learned_transforms"]
    assert len(jth_candidates) == 37554
    assert len(jth_jobs) == 6011
    assert "last_stage_reached" in jth_schema["excluded_features"]
    assert "llm_*" in jth_schema["excluded_features"]
    assert jth_availability["future_outcomes_not_used"] is True
    assert (pd.read_csv("datasets/jth/processed/job_skill_features.csv")["is_rare_skill"]).any()
