import json
from pathlib import Path

import pandas as pd


ROOT = Path("datasets")


def _assert_temporal_outputs(dataset: str, user_column: str, item_column: str, expected: tuple[int, int, int]):
    output = ROOT / dataset / "processed"
    splits = {name: pd.read_csv(output / f"{name}.csv", parse_dates=["interaction_time"]) for name in ["train", "val", "test"]}
    assert tuple(len(splits[name]) for name in ["train", "val", "test"]) == expected
    assert not (set(splits["train"]["interaction_time"]) & set(splits["val"]["interaction_time"]))
    assert not (set(splits["val"]["interaction_time"]) & set(splits["test"]["interaction_time"]))
    assert splits["train"]["interaction_time"].max() < splits["val"]["interaction_time"].min()
    assert splits["val"]["interaction_time"].max() < splits["test"]["interaction_time"].min()
    pairs = [set(zip(frame[user_column].astype(str), frame[item_column].astype(str))) for frame in splits.values()]
    assert not pairs[0] & pairs[1]
    assert not pairs[0] & pairs[2]
    assert not pairs[1] & pairs[2]
    metadata = json.loads((output / "split_metadata.json").read_text(encoding="utf-8"))
    statistics = json.loads((output / "split_statistics.json").read_text(encoding="utf-8"))
    assert metadata["split"]["same_timestamp_kept_together"] is True
    assert metadata["split"]["shuffle"] is False
    assert metadata["graph_history_protocol"]["future_edges_in_train_graph"] is False
    assert all(value > 0 for value in statistics["rows"].values())


def test_movielens_temporal_outputs():
    _assert_temporal_outputs("movielens", "user_id", "movie_id", (38864, 4858, 4858))
    train = pd.read_csv(ROOT / "movielens/processed/train.csv")
    assert train["is_positive"].all()


def test_jth_temporal_outputs():
    _assert_temporal_outputs("jth", "candidate_id", "job_id", (33041, 4140, 4127))
