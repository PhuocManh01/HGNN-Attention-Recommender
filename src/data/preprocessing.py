from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_THRESHOLD = 4.0
SEED = 42


def load_movielens(raw_dir: Path):
    raw_dir = Path(raw_dir)

    ratings = pd.read_csv(raw_dir / "ratings.csv")
    movies = pd.read_csv(raw_dir / "movies.csv")

    return ratings, movies


def validate_data(ratings, movies):
    required_ratings = {
        "userId", "movieId", "rating", "timestamp"
    }
    required_movies = {
        "movieId", "title", "genres"
    }

    assert required_ratings.issubset(ratings.columns)
    assert required_movies.issubset(movies.columns)

    assert not ratings.empty
    assert not movies.empty

    assert not ratings[
        ["userId", "movieId", "rating", "timestamp"]
    ].isnull().any().any()

    assert ratings["rating"].between(0.5, 5.0).all()
    assert ratings["rating"].mul(2).mod(1).eq(0).all()

    assert not ratings.duplicated(
        ["userId", "movieId"]
    ).any(), "Duplicate user-movie interactions"

    assert not movies["movieId"].duplicated().any()

    assert ratings["movieId"].isin(
        movies["movieId"]
    ).all(), "Unknown movie references"

    assert pd.api.types.is_integer_dtype(
        ratings["timestamp"]
    )

    assert ratings["timestamp"].ge(0).all()


def build_mappings(ratings, movies):
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(movies["movieId"].unique())

    user_mapping = pd.DataFrame({
        "userId": user_ids,
        "user_idx": np.arange(len(user_ids))
    })

    movie_mapping = pd.DataFrame({
        "movieId": movie_ids,
        "movie_idx": np.arange(len(movie_ids))
    })

    return user_mapping, movie_mapping


def prepare_interactions(
    ratings,
    user_mapping,
    movie_mapping
):
    interactions = ratings.merge(
        user_mapping,
        on="userId",
        how="left",
        validate="many_to_one"
    ).merge(
        movie_mapping,
        on="movieId",
        how="left",
        validate="many_to_one"
    )

    interactions["is_positive"] = (
        interactions["rating"] >= POSITIVE_THRESHOLD
    ).astype("int8")

    interactions = interactions.sort_values(
        ["timestamp", "userId", "movieId"],
        kind="stable"
    ).reset_index(drop=True)

    interactions.insert(
        0,
        "interaction_id",
        np.arange(len(interactions), dtype=np.int64)
    )

    return interactions


def prepare_movie_genres(movies, movie_mapping):
    movie_table = movies.merge(
        movie_mapping,
        on="movieId",
        how="left",
        validate="one_to_one"
    )

    genre_rows = []

    for row in movie_table.itertuples(index=False):
        raw_genres = (
            row.genres
            if isinstance(row.genres, str)
            else "(no genres listed)"
        )

        for genre in raw_genres.split("|"):
            genre_rows.append({
                "movie_idx": int(row.movie_idx),
                "genre": genre
            })

    movie_genres = pd.DataFrame(
        genre_rows
    ).drop_duplicates()

    # "(no genres listed)" is missing metadata,
    # not a real genre node.
    movie_genres = movie_genres[
        movie_genres["genre"] != "(no genres listed)"
    ].copy()

    genre_names = sorted(
        movie_genres["genre"].unique()
    )

    genre_mapping = pd.DataFrame({
        "genre": genre_names,
        "genre_idx": np.arange(len(genre_names))
    })

    movie_genres = movie_genres.merge(
        genre_mapping,
        on="genre",
        how="left",
        validate="many_to_one"
    )

    return movie_table, movie_genres, genre_mapping


def temporal_split(
    interactions,
    train_ratio=0.8,
    val_ratio=0.1
):
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1

    df = interactions.sort_values(
        ["timestamp", "interaction_id"]
    ).reset_index(drop=True)

    n = len(df)

    # Group identical timestamps together.
    timestamp_counts = (
        df.groupby("timestamp", sort=True)
        .size()
        .to_numpy()
    )

    cumulative = np.cumsum(timestamp_counts)

    assert len(cumulative) >= 3, (
        "At least 3 distinct timestamps required"
    )

    target_train = n * train_ratio
    target_train_val = n * (train_ratio + val_ratio)

    # Choose non-empty, chronological partitions
    # with boundaries closest to target ratios.
    train_end = int(
        np.argmin(
            np.abs(cumulative[:-2] - target_train)
        )
    )

    train_count = int(cumulative[train_end])

    possible_val_ends = np.arange(
        train_end + 1,
        len(cumulative) - 1
    )

    val_end_idx = possible_val_ends[
        np.argmin(
            np.abs(
                cumulative[possible_val_ends]
                - target_train_val
            )
        )
    ]

    val_end = int(cumulative[val_end_idx])

    train = df.iloc[:train_count].copy()
    val = df.iloc[train_count:val_end].copy()
    test = df.iloc[val_end:].copy()

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()

    return train, val, test


def get_evaluation_users(train, val, test):
    # Evaluation cohort is fixed by predeclared
    # warm-start eligibility rules.
    train_positive = set(
        train.loc[
            train["is_positive"] == 1,
            "user_idx"
        ]
    )

    val_positive = set(
        val.loc[
            val["is_positive"] == 1,
            "user_idx"
        ]
    )

    test_positive = set(
        test.loc[
            test["is_positive"] == 1,
            "user_idx"
        ]
    )

    val_users = sorted(
        train_positive & val_positive
    )

    train_val = pd.concat(
        [train, val],
        ignore_index=True
    )

    train_val_positive = set(
        train_val.loc[
            train_val["is_positive"] == 1,
            "user_idx"
        ]
    )

    test_users = sorted(
        train_val_positive & test_positive
    )

    all_users = sorted(
        set(train["user_idx"])
        | set(val["user_idx"])
        | set(test["user_idx"])
    )

    result = pd.DataFrame({
        "user_idx": all_users
    })

    result["eligible_val"] = (
        result["user_idx"].isin(val_users)
    )

    result["eligible_test"] = (
        result["user_idx"].isin(test_users)
    )

    return result


def validate_splits(train, val, test, interactions):
    all_ids = pd.concat([
        train["interaction_id"],
        val["interaction_id"],
        test["interaction_id"]
    ])

    assert len(all_ids) == len(interactions)
    assert all_ids.is_unique

    assert set(all_ids) == set(
        interactions["interaction_id"]
    )

    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()

    for split in [train, val, test]:
        assert not split.duplicated(
            ["user_idx", "movie_idx"]
        ).any()


def run_pipeline(raw_dir, output_root):
    raw_dir = Path(raw_dir)
    output_root = Path(output_root)

    processed_dir = output_root / "processed"
    splits_dir = output_root / "splits"

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies = load_movielens(raw_dir)
    validate_data(ratings, movies)

    user_mapping, movie_mapping = build_mappings(
        ratings, movies
    )

    interactions = prepare_interactions(
        ratings,
        user_mapping,
        movie_mapping
    )

    movie_table, movie_genres, genre_mapping = (
        prepare_movie_genres(
            movies,
            movie_mapping
        )
    )

    train, val, test = temporal_split(interactions)

    validate_splits(
        train, val, test, interactions
    )

    train_val = pd.concat(
        [train, val],
        ignore_index=True
    ).sort_values(
        "interaction_id"
    ).reset_index(drop=True)

    evaluation_users = get_evaluation_users(
        train, val, test
    )

    processed_files = {
        "interactions.csv": interactions,
        "movies.csv": movie_table,
        "movie_genres.csv": movie_genres,
        "user_mapping.csv": user_mapping,
        "movie_mapping.csv": movie_mapping,
        "genre_mapping.csv": genre_mapping
    }

    for filename, dataframe in processed_files.items():
        dataframe.to_csv(
            processed_dir / filename,
            index=False
        )

    split_files = {
        "train.csv": train,
        "val.csv": val,
        "test.csv": test,
        "train_val.csv": train_val,
        "evaluation_users.csv": evaluation_users
    }

    for filename, dataframe in split_files.items():
        dataframe.to_csv(
            splits_dir / filename,
            index=False
        )

    metadata = {
        "dataset": "MovieLens Latest Small",
        "split_method": "global_temporal",
        "timestamp_tie_policy": "keep_together",
        "train_ratio_target": 0.8,
        "val_ratio_target": 0.1,
        "test_ratio_target": 0.1,
        "positive_threshold": POSITIVE_THRESHOLD,
        "seed": SEED,
        "num_users": len(user_mapping),
        "num_movies": len(movie_mapping),
        "num_genres": len(genre_mapping),
        "num_interactions": len(interactions),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "train_ratio_actual": len(train) / len(interactions),
        "val_ratio_actual": len(val) / len(interactions),
        "test_ratio_actual": len(test) / len(interactions),
        "train_end_timestamp": int(train["timestamp"].max()),
        "val_start_timestamp": int(val["timestamp"].min()),
        "val_end_timestamp": int(val["timestamp"].max()),
        "test_start_timestamp": int(test["timestamp"].min()),
        "num_eligible_val_users": int(
            evaluation_users["eligible_val"].sum()
        ),
        "num_eligible_test_users": int(
            evaluation_users["eligible_test"].sum()
        ),
        "evaluation_policy": (
            "positive history before cutoff "
            "and positive label in target window"
        ),
        "node_mapping_policy": (
            "global ID vocabulary; no future "
            "interaction statistics as features"
        )
    }

    with open(
        splits_dir / "split_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    return metadata
