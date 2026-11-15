"""Validate and normalize MovieLens data for later recommendation stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.common.id_mapping import apply_mapping, build_mapping, save_mapping
from src.paths import DATA_ROOT


RAW_DIR = DATA_ROOT / "movielens"
OUT_DIR = RAW_DIR / "processed"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "movielens.yaml"
SOURCE_SCHEMAS = {
    "ratings": {"userId", "movieId", "rating", "timestamp"},
    "movies": {"movieId", "title", "genres"},
    "tags": {"userId", "movieId", "tag", "timestamp"},
    "links": {"movieId", "imdbId", "tmdbId"},
}


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _require_columns(name: str, frame: pd.DataFrame) -> None:
    missing = sorted(SOURCE_SCHEMAS[name] - set(frame.columns))
    if missing:
        raise ValueError(f"{name}.csv is missing required columns: {missing}")


def _normalize_genres(movies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = movies[["movie_id", "genres"]].copy()
    values["genres"] = values["genres"].fillna("").astype("string")
    rows = values.assign(genre=values["genres"].str.split("|")).explode("genre", ignore_index=True)
    rows["genre"] = rows["genre"].fillna("").astype("string").str.strip()
    rows["genre"] = rows["genre"].replace("", "(no genres listed)")
    genre_map = build_mapping(rows["genre"], "genre")
    rows["genre_id"] = apply_mapping(rows["genre"], genre_map, "genre")
    rows["is_no_genres_listed"] = rows["genre"].eq("(no genres listed)")
    return genre_map.rename(columns={"id": "genre_id"}), rows[[
        "movie_id", "genre_id", "genre", "is_no_genres_listed"
    ]]


def prepare() -> dict[str, Any]:
    """Validate and write only MovieLens preprocessing outputs."""
    config = _load_config()
    threshold = float(config.get("preprocessing", {}).get("positive_threshold", 4.0))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {name: pd.read_csv(RAW_DIR / f"{name}.csv") for name in SOURCE_SCHEMAS}
    for name, frame in sources.items():
        _require_columns(name, frame)

    ratings, movies = sources["ratings"].copy(), sources["movies"].copy()
    quality: dict[str, Any] = {
        "rows_before": {name: int(len(frame)) for name, frame in sources.items()},
        "errors": {}, "threshold": threshold,
        "links_used_for_interactions": False, "tags_used_for_interactions": False,
        "unobserved_is_not_materialized": True,
    }
    quality["errors"].update({
        "ratings_null_ids": int(ratings[["userId", "movieId"]].isna().any(axis=1).sum()),
        "movies_null_ids": int(movies["movieId"].isna().sum()),
        "ratings_duplicate_user_movie": int(ratings.duplicated(["userId", "movieId"]).sum()),
        "movies_duplicate_movie_id": int(movies["movieId"].duplicated().sum()),
        "tags_duplicate_rows": int(sources["tags"].duplicated().sum()),
        "links_duplicate_movie_id": int(sources["links"]["movieId"].duplicated().sum()),
    })
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    parsed_time = pd.to_datetime(ratings["timestamp"], unit="s", utc=True, errors="coerce")
    quality["errors"].update({
        "invalid_ratings": int((ratings["rating"].isna() | ~ratings["rating"].between(0.5, 5.0)).sum()),
        "invalid_timestamps": int(parsed_time.isna().sum()),
        "unknown_movie_references": int((~ratings["movieId"].isin(movies["movieId"])).sum()),
        "links_missing_tmdb_id": int(sources["links"]["tmdbId"].isna().sum()),
    })
    fatal = {key: value for key, value in quality["errors"].items() if value and key not in {
        "tags_duplicate_rows", "links_duplicate_movie_id", "links_missing_tmdb_id"
    }}
    if fatal:
        raise ValueError(f"MovieLens validation failed: {fatal}")

    ratings["interaction_time"] = parsed_time
    ratings["raw_user_id"], ratings["raw_movie_id"] = ratings["userId"].astype("string"), ratings["movieId"].astype("string")
    movies["raw_movie_id"] = movies["movieId"].astype("string")
    user_map, movie_map = build_mapping(ratings["raw_user_id"], "raw_user_id"), build_mapping(movies["raw_movie_id"], "raw_movie_id")
    ratings["user_id"] = apply_mapping(ratings["raw_user_id"], user_map, "raw_user_id")
    ratings["movie_id"] = apply_mapping(ratings["raw_movie_id"], movie_map, "raw_movie_id")
    movies["movie_id"] = apply_mapping(movies["raw_movie_id"], movie_map, "raw_movie_id")
    ratings["is_positive"] = ratings["rating"] >= threshold
    ratings["interaction_status"] = ratings["is_positive"].map({True: "positive", False: "observed_non_positive"})
    ratings["interaction_id"] = ratings.index.astype("int64")
    ratings[["interaction_id", "user_id", "movie_id", "rating", "is_positive", "interaction_status", "interaction_time", "timestamp"]].to_csv(OUT_DIR / "interactions.csv", index=False)

    movies_clean = movies[["movie_id", "raw_movie_id", "title", "genres"]].copy()
    movies_clean["release_year"] = pd.to_numeric(movies_clean["title"].astype("string").str.extract(r"\((\d{4})\)")[0], errors="coerce").astype("Int64")
    movies_clean.to_csv(OUT_DIR / "movies_clean.csv", index=False)
    genre_map, movie_genres = _normalize_genres(movies)
    movie_genres.to_csv(OUT_DIR / "movie_genres.csv", index=False)
    save_mapping(user_map.rename(columns={"id": "user_id"}), OUT_DIR / "users.csv")
    save_mapping(movie_map.rename(columns={"id": "movie_id"}), OUT_DIR / "movies.csv")
    genre_map.to_csv(OUT_DIR / "genres.csv", index=False)

    quality["rows_after"] = {"interactions": int(len(ratings)), "movies_clean": int(len(movies_clean)), "movie_genres": int(len(movie_genres))}
    quality["counts"] = {
        "users": int(ratings["user_id"].nunique()), "catalog_movies": int(movies_clean["movie_id"].nunique()),
        "rated_movies": int(ratings["movie_id"].nunique()), "positive_ratings": int(ratings["is_positive"].sum()),
        "observed_non_positive_ratings": int((~ratings["is_positive"]).sum()),
        "unobserved_user_movie_pairs": int(ratings["user_id"].nunique() * len(movies_clean) - len(ratings)),
    }
    quality["timestamp_range_utc"] = {"min": ratings["interaction_time"].min().isoformat(), "max": ratings["interaction_time"].max().isoformat()}
    quality["rating_range"] = {"min": float(ratings["rating"].min()), "max": float(ratings["rating"].max())}
    (OUT_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"dataset": "movielens", "output": str(OUT_DIR), **quality["counts"], "rows_before": quality["rows_before"], "errors": quality["errors"]}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, default=str))
