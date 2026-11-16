
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from src.data.preprocessing import validate_splits

RATES = ("user", "rates", "movie")
REV_RATES = ("movie", "rev_rates", "user")
HAS_GENRE = ("movie", "has_genre", "genre")
REV_HAS_GENRE = ("genre", "rev_has_genre", "movie")


def make_edge_index(
    dataframe: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> torch.Tensor:
    if dataframe.empty:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(
        dataframe[[source_column, target_column]]
        .to_numpy(dtype="int64")
        .T.copy(),
        dtype=torch.long,
    )


def build_heterogeneous_graph(
    interactions: pd.DataFrame,
    movie_genres: pd.DataFrame,
    num_users: int,
    num_movies: int,
    num_genres: int,
) -> HeteroData:
    """Build one graph from a cutoff-safe interaction snapshot."""

    if min(num_users, num_movies, num_genres) < 0:
        raise ValueError("Node counts must be nonnegative")

    required_interactions = {
        "user_idx", "movie_idx", "is_positive"
    }
    required_genres = {"movie_idx", "genre_idx"}

    if not required_interactions.issubset(interactions.columns):
        raise ValueError("Missing interaction columns")

    if not required_genres.issubset(movie_genres.columns):
        raise ValueError("Missing movie-genre columns")

    positive = interactions.loc[
        interactions["is_positive"] == 1,
        ["user_idx", "movie_idx"],
    ].copy()

    if positive.duplicated().any():
        raise ValueError("Duplicate positive user-movie edges")

    genres = movie_genres[
        ["movie_idx", "genre_idx"]
    ].drop_duplicates()

    user_movie = make_edge_index(
        positive, "user_idx", "movie_idx"
    )

    movie_genre = make_edge_index(
        genres, "movie_idx", "genre_idx"
    )

    graph = HeteroData()

    graph["user"].num_nodes = num_users
    graph["movie"].num_nodes = num_movies
    graph["genre"].num_nodes = num_genres

    graph[RATES].edge_index = user_movie
    graph[REV_RATES].edge_index = (
        user_movie.flip(0).contiguous()
    )

    graph[HAS_GENRE].edge_index = movie_genre
    graph[REV_HAS_GENRE].edge_index = (
        movie_genre.flip(0).contiguous()
    )

    validate_graph(graph)

    return graph


def validate_graph(graph: HeteroData) -> None:
    expected_node_types = {"user", "movie", "genre"}

    expected_edge_types = {
        RATES,
        REV_RATES,
        HAS_GENRE,
        REV_HAS_GENRE,
    }

    assert set(graph.node_types) == expected_node_types
    assert set(graph.edge_types) == expected_edge_types

    assert graph.validate(raise_on_error=True)

    for edge_type in graph.edge_types:
        edge_index = graph[edge_type].edge_index

        assert edge_index.dtype == torch.long
        assert edge_index.ndim == 2
        assert edge_index.shape[0] == 2

        source_type, _, target_type = edge_type

        if edge_index.numel() > 0:
            assert int(edge_index[0].min()) >= 0
            assert int(edge_index[1].min()) >= 0

            assert int(edge_index[0].max()) < (
                graph[source_type].num_nodes
            )
            assert int(edge_index[1].max()) < (
                graph[target_type].num_nodes
            )

        # Check duplicate directed edges.
        if edge_index.shape[1] > 0:
            assert torch.unique(
                edge_index.T, dim=0
            ).shape[0] == edge_index.shape[1]

    assert torch.equal(
        graph[REV_RATES].edge_index,
        graph[RATES].edge_index.flip(0),
    )

    assert torch.equal(
        graph[REV_HAS_GENRE].edge_index,
        graph[HAS_GENRE].edge_index.flip(0),
    )


def edge_pairs(graph: HeteroData, edge_type):
    return set(
        map(
            tuple,
            graph[edge_type].edge_index.T.tolist(),
        )
    )


def validate_snapshot_integrity(
    train_graph: HeteroData,
    train_val_graph: HeteroData,
    train: pd.DataFrame,
    train_val: pd.DataFrame,
) -> None:
    expected_train = set(
        map(
            tuple,
            train.loc[
                train["is_positive"] == 1,
                ["user_idx", "movie_idx"],
            ].to_numpy().tolist(),
        )
    )

    expected_train_val = set(
        map(
            tuple,
            train_val.loc[
                train_val["is_positive"] == 1,
                ["user_idx", "movie_idx"],
            ].to_numpy().tolist(),
        )
    )

    actual_train = edge_pairs(train_graph, RATES)
    actual_train_val = edge_pairs(train_val_graph, RATES)

    assert actual_train == expected_train
    assert actual_train_val == expected_train_val

    assert actual_train.issubset(actual_train_val)

    assert edge_pairs(
        train_graph, HAS_GENRE
    ) == edge_pairs(train_val_graph, HAS_GENRE)


def run_graph_pipeline(
    data_root: str | Path,
) -> dict:
    data_root = Path(data_root)

    processed = data_root / "processed"
    splits = data_root / "splits"
    graphs_dir = data_root / "graphs"

    graphs_dir.mkdir(parents=True, exist_ok=True)

    user_mapping = pd.read_csv(
        processed / "user_mapping.csv"
    )
    movie_mapping = pd.read_csv(
        processed / "movie_mapping.csv"
    )
    genre_mapping = pd.read_csv(
        processed / "genre_mapping.csv"
    )
    movie_genres = pd.read_csv(
        processed / "movie_genres.csv"
    )

    train = pd.read_csv(splits / "train.csv")
    val = pd.read_csv(splits / "val.csv")
    test = pd.read_csv(splits / "test.csv")
    train_val = pd.read_csv(splits / "train_val.csv")

    # Read the active Phase 2 split protocol.
    with open(
        splits / "split_metadata.json",
        "r",
        encoding="utf-8",
    ) as f:
        split_metadata = json.load(f)

    split_method = split_metadata["split_method"]

    if split_method not in {
        "global_temporal",
        "per_user_temporal",
    }:
        raise ValueError(
            f"Unsupported split method: {split_method}"
        )

    # Validate chronology according to the active protocol.
    all_interactions = pd.concat(
        [train, val, test],
        ignore_index=True,
    )

    validate_splits(
        train=train,
        val=val,
        test=test,
        interactions=all_interactions,
        split_method=split_method,
    )

    assert set(train["interaction_id"]).isdisjoint(
        val["interaction_id"]
    )
    assert set(train["interaction_id"]).isdisjoint(
        test["interaction_id"]
    )
    assert set(val["interaction_id"]).isdisjoint(
        test["interaction_id"]
    )

    assert set(train_val["interaction_id"]) == (
        set(train["interaction_id"])
        | set(val["interaction_id"])
    )


    expected_train_val = pd.concat(
        [train, val],
        ignore_index=True,
    )

    comparison_columns = [
        "interaction_id",
        "user_idx",
        "movie_idx",
        "timestamp",
        "is_positive",
    ]

    pd.testing.assert_frame_equal(
        train_val[comparison_columns]
        .sort_values("interaction_id")
        .reset_index(drop=True),

        expected_train_val[comparison_columns]
        .sort_values("interaction_id")
        .reset_index(drop=True),

        check_dtype=False,
    )

    
    num_users = len(user_mapping)
    num_movies = len(movie_mapping)
    num_genres = len(genre_mapping)

    args = {
        "movie_genres": movie_genres,
        "num_users": num_users,
        "num_movies": num_movies,
        "num_genres": num_genres,
    }

    train_graph = build_heterogeneous_graph(
        interactions=train,
        **args,
    )

    train_val_graph = build_heterogeneous_graph(
        interactions=train_val,
        **args,
    )

    validate_snapshot_integrity(
        train_graph,
        train_val_graph,
        train,
        train_val,
    )

    torch.save(
        train_graph,
        graphs_dir / "train_graph.pt",
    )

    torch.save(
        train_val_graph,
        graphs_dir / "train_val_graph.pt",
    )

    metadata = {
        "phase": 3,
        "graph_type": "heterogeneous",
        "node_types": train_graph.node_types,
        "edge_types": [
            list(edge_type)
            for edge_type in train_graph.edge_types
        ],
        "num_users": num_users,
        "num_movies": num_movies,
        "num_genres": num_genres,
        "train_positive_edges": int(
            train_graph[RATES].edge_index.shape[1]
        ),
        "train_val_positive_edges": int(
            train_val_graph[RATES].edge_index.shape[1]
        ),
        "movie_genre_edges": int(
            train_graph[HAS_GENRE].edge_index.shape[1]
        ),
        "train_snapshot": "train_positive_only",
        "test_interactions_in_graph": False,
        "node_features": "none",
        "genre_metadata_policy": (
            "static metadata from MovieLens movies.csv"
        ),
        "split_method": split_method,
        "split_seed": split_metadata["seed"],
        "positive_threshold": split_metadata["positive_threshold"],
        "num_eligible_val_users": split_metadata["num_eligible_val_users"],
        "num_eligible_test_users": split_metadata["num_eligible_test_users"],
    }

    with open(
        graphs_dir / "graph_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metadata
