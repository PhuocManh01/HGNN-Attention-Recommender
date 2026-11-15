"""Leakage-safe MovieLens catalog feature preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.common.id_mapping import mapping_object, reverse_mapping
from src.paths import DATA_ROOT


DATA_DIR = DATA_ROOT / "movielens" / "processed"
MAPPING_DIR = DATA_DIR / "mappings"


def _write_mapping_views(frame: pd.DataFrame, raw_name: str, stem: str) -> None:
    mapping_object(frame, raw_name)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MAPPING_DIR / f"{stem}_mapping.csv", index=False)
    reverse_mapping(frame, raw_name).to_csv(MAPPING_DIR / f"{stem}_reverse.csv", index=False)


def prepare() -> dict[str, object]:
    movies = pd.read_csv(DATA_DIR / "movies_clean.csv", dtype={"raw_movie_id": "string"})
    movie_genres = pd.read_csv(DATA_DIR / "movie_genres.csv")
    users = pd.read_csv(DATA_DIR / "users.csv", dtype={"raw_user_id": "string"})
    movie_map = pd.read_csv(DATA_DIR / "movies.csv", dtype={"raw_movie_id": "string"})
    genres = pd.read_csv(DATA_DIR / "genres.csv")
    _write_mapping_views(users.rename(columns={"user_id": "id"}), "raw_user_id", "users")
    _write_mapping_views(movie_map.rename(columns={"movie_id": "id"}), "raw_movie_id", "movies")
    _write_mapping_views(genres.rename(columns={"genre_id": "id"}), "genre", "genres")

    genre_ids = movie_genres.sort_values(["movie_id", "genre_id"]).groupby("movie_id")["genre_id"].apply(
        lambda values: ";".join(str(int(value)) for value in values)
    )
    features = movies[["movie_id", "raw_movie_id", "release_year"]].copy()
    features["release_year_missing"] = features["release_year"].isna()
    features["title_length"] = movies["title"].fillna("").astype("string").str.len().astype("int64")
    features["genre_ids"] = features["movie_id"].map(genre_ids).fillna("")
    features["genre_count"] = features["genre_ids"].map(lambda value: len(value.split(";")) if value else 0)
    features["no_genres_listed"] = features["genre_ids"].eq(str(int(genres.loc[genres["genre"].eq("(no genres listed)"), "genre_id"].iloc[0])))
    features.to_csv(DATA_DIR / "movie_features.csv", index=False)

    schema = {
        "dataset": "movielens",
        "id_mappings": {"user": "catalog", "movie": "catalog", "genre": "catalog"},
        "features": {
            "genre_ids": {"type": "multi_index", "source": "movie_genres.csv", "fit_scope": "catalog"},
            "release_year": {"type": "numeric", "missing_indicator": "release_year_missing", "fit_scope": "deterministic_metadata"},
            "title_length": {"type": "numeric", "fit_scope": "deterministic_metadata"},
            "genre_count": {"type": "numeric", "fit_scope": "deterministic_metadata"},
        },
        "excluded_features": {"rating": "interaction/label, never a movie feature", "timestamp": "interaction time, not catalog metadata"},
        "leakage_policy": "No rating, tag, or future interaction statistic is used in movie_features.csv.",
        "train_fit_required_for_learned_transforms": True,
    }
    (DATA_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"dataset": "movielens", "movies": len(features), "genres": len(genres), "output": str(DATA_DIR)}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
