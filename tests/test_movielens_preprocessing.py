import json
import pandas as pd

from src.data.movielens.preprocess import prepare


def test_movielens_preprocessing_outputs(tmp_path, monkeypatch):
    import src.data.movielens.preprocess as module

    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "movielens")
    result = prepare()
    assert result["dataset"] == "movielens"
    interactions = pd.read_csv(tmp_path / "movielens/interactions.csv")
    movies = pd.read_csv(tmp_path / "movielens/movies_clean.csv")
    movie_genres = pd.read_csv(tmp_path / "movielens/movie_genres.csv")
    quality = json.loads((tmp_path / "movielens/data_quality.json").read_text())
    assert len(interactions) == 100836
    assert int(interactions["is_positive"].sum()) == 48580
    assert interactions["user_id"].nunique() == 610
    assert len(movies) == 9742
    assert not interactions.duplicated(["user_id", "movie_id"]).any()
    assert interactions["rating"].between(0.5, 5.0).all()
    assert (interactions.loc[interactions["rating"] < 4, "interaction_status"] == "observed_non_positive").all()
    assert movie_genres["genre"].eq("(no genres listed)").sum() == 34
    assert quality["errors"]["invalid_ratings"] == 0
    assert quality["errors"]["ratings_duplicate_user_movie"] == 0


def test_timestamp_is_utc_and_low_rating_is_not_unobserved(tmp_path, monkeypatch):
    import src.data.movielens.preprocess as module

    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "movielens")
    module.prepare()
    interactions = pd.read_csv(tmp_path / "movielens/interactions.csv", parse_dates=["interaction_time"])
    first = interactions.iloc[0]
    assert str(first["interaction_time"].tz) == "UTC"
    assert first["interaction_time"] == pd.Timestamp(first["timestamp"], unit="s", tz="UTC")
    assert not (interactions["interaction_status"] == "unobserved").any()
