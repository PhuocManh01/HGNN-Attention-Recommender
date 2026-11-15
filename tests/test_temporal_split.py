import pandas as pd

from src.data.common.temporal_split import temporal_split


def test_equal_timestamps_stay_in_one_split():
    frame = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "item_id": [1, 1, 1, 1],
        "event_time": pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"
        ], utc=True),
    })
    train, val, test, metadata = temporal_split(frame, "event_time")
    assert metadata["same_timestamp_kept_together"] is True
    assert set(train["event_time"]).isdisjoint(set(val["event_time"]))
    assert set(val["event_time"]).isdisjoint(set(test["event_time"]))
