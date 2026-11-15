import pandas as pd
import json
import pytest

from src.data.jth.preprocess import prepare


@pytest.fixture(scope="module")
def prepared_jth():
    import src.data.jth.preprocess as module

    yield module.OUT_DIR


def test_jth_first_pair_and_temporal_outputs(prepared_jth):
    result = {"dataset": "jth", **json.loads((prepared_jth / "data_quality.json").read_text())["counts"]}
    assert result["dataset"] == "jth"
    assert result["application_processes"] == 42288
    assert result["unique_candidate_job_pairs"] == 41308
    assert result["repeated_candidate_job_rows"] == 980
    assert result["candidates"] == 37554
    assert result["jobs"] == 6011
    quality = json.loads((prepared_jth / "data_quality.json").read_text())
    assert quality["temporal_anomalies"] == {}
    events = pd.read_csv(prepared_jth / "application_events.csv", dtype="string")
    interactions = pd.read_csv(prepared_jth / "interactions.csv", dtype="string")
    assert len(events) == 42288
    assert len(interactions) == 41308
    assert events["application_id"].notna().all()
    assert not interactions.duplicated(["candidate_id", "job_id"]).any()
    assert not events["excluded_from_interactions"].eq("True").any()
    assert quality["errors"]["orphan_candidates"] == 0
    assert quality["errors"]["orphan_jobs"] == 0
    assert quality["errors"]["duplicate_history_rows"] == 0


def test_jth_outputs_preserve_future_outcome_policy(prepared_jth):
    policy = json.loads((prepared_jth / "feature_availability.json").read_text())
    assert policy["future_outcomes_not_used"] is True
    assert "last_stage_reached" in policy["features_not_used_for_top_k_input"]
    assert policy["skill_policy"]["rare_skill_node"] is False
