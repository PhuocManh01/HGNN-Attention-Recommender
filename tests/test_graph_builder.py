
import unittest

import pandas as pd
import torch
import json
import tempfile
from pathlib import Path

from src.data.graph_builder import run_graph_pipeline

from src.data.graph_builder import (
    RATES,
    REV_RATES,
    HAS_GENRE,
    REV_HAS_GENRE,
    build_heterogeneous_graph,
    validate_graph,
    validate_snapshot_integrity,
)


class TestGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.train = pd.DataFrame({
            "user_idx": [0, 0, 1, 2],
            "movie_idx": [0, 1, 1, 2],
            "is_positive": [1, 0, 1, 1],
        })

        self.train_val = pd.concat([
            self.train,
            pd.DataFrame({
                "user_idx": [0, 1],
                "movie_idx": [3, 3],
                "is_positive": [1, 0],
            }),
        ], ignore_index=True)

        self.movie_genres = pd.DataFrame({
            "movie_idx": [0, 1, 2, 3],
            "genre_idx": [0, 1, 0, 1],
        })

        def build(interactions):
            return build_heterogeneous_graph(
                interactions=interactions,
                movie_genres=self.movie_genres,
                num_users=3,
                num_movies=4,
                num_genres=2,
            )

        self.train_graph = build(self.train)
        self.train_val_graph = build(self.train_val)

    def test_node_types(self):
        self.assertEqual(
            set(self.train_graph.node_types),
            {"user", "movie", "genre"},
        )

    def test_edge_types(self):
        self.assertEqual(
            set(self.train_graph.edge_types),
            {
                RATES,
                REV_RATES,
                HAS_GENRE,
                REV_HAS_GENRE,
            },
        )

    def test_node_counts(self):
        self.assertEqual(
            self.train_graph["user"].num_nodes, 3
        )
        self.assertEqual(
            self.train_graph["movie"].num_nodes, 4
        )
        self.assertEqual(
            self.train_graph["genre"].num_nodes, 2
        )

    def test_positive_edges_only(self):
        self.assertEqual(
            self.train_graph[RATES].edge_index.shape[1],
            3,
        )

    def test_reverse_edges(self):
        self.assertTrue(
            torch.equal(
                self.train_graph[REV_RATES].edge_index,
                self.train_graph[RATES]
                .edge_index.flip(0),
            )
        )

    def test_movie_genre_edges(self):
        self.assertEqual(
            self.train_graph[HAS_GENRE]
            .edge_index.shape[1],
            4,
        )

    def test_snapshot_integrity(self):
        validate_snapshot_integrity(
            self.train_graph,
            self.train_val_graph,
            self.train,
            self.train_val,
        )

    def test_graph_validation(self):
        validate_graph(self.train_graph)
        validate_graph(self.train_val_graph)

    def test_no_node_features(self):
        for node_type in self.train_graph.node_types:
            self.assertNotIn(
                "x",
                self.train_graph[node_type],
            )


class TestPerUserGraphPipeline(unittest.TestCase):

    def test_per_user_split_builds_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            splits = root / "splits"

            processed.mkdir()
            splits.mkdir()

            pd.DataFrame({
                "user_idx": [0, 1],
            }).to_csv(
                processed / "user_mapping.csv",
                index=False,
            )

            pd.DataFrame({
                "movie_idx": [0, 1, 2, 3, 4, 5],
            }).to_csv(
                processed / "movie_mapping.csv",
                index=False,
            )

            pd.DataFrame({
                "genre_idx": [0, 1],
            }).to_csv(
                processed / "genre_mapping.csv",
                index=False,
            )

            pd.DataFrame({
                "movie_idx": [0, 1, 2, 3, 4, 5],
                "genre_idx": [0, 1, 0, 1, 0, 1],
            }).to_csv(
                processed / "movie_genres.csv",
                index=False,
            )

            # Per-user chronology is valid, but
            # global chronology overlaps.
            train = pd.DataFrame({
                "interaction_id": [0, 1],
                "user_idx": [0, 1],
                "movie_idx": [0, 1],
                "timestamp": [10, 100],
                "is_positive": [1, 1],
            })

            val = pd.DataFrame({
                "interaction_id": [2, 3],
                "user_idx": [0, 1],
                "movie_idx": [2, 3],
                "timestamp": [20, 110],
                "is_positive": [1, 1],
            })

            test = pd.DataFrame({
                "interaction_id": [4, 5],
                "user_idx": [0, 1],
                "movie_idx": [4, 5],
                "timestamp": [30, 120],
                "is_positive": [1, 1],
            })

            train.to_csv(
                splits / "train.csv", index=False
            )
            val.to_csv(
                splits / "val.csv", index=False
            )
            test.to_csv(
                splits / "test.csv", index=False
            )

            pd.concat(
                [train, val], ignore_index=True
            ).to_csv(
                splits / "train_val.csv",
                index=False,
            )

            split_metadata = {
                "split_method": "per_user_temporal",
                "seed": 42,
                "positive_threshold": 4.0,
                "num_eligible_val_users": 2,
                "num_eligible_test_users": 2,
            }

            with open(
                splits / "split_metadata.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(split_metadata, f)

            metadata = run_graph_pipeline(root)

            self.assertEqual(
                metadata["split_method"],
                "per_user_temporal",
            )
            self.assertEqual(
                metadata["train_positive_edges"],
                2,
            )
            self.assertEqual(
                metadata["train_val_positive_edges"],
                4,
            )

            self.assertTrue(
                (root / "graphs" / "train_graph.pt").exists()
            )
            self.assertTrue(
                (root / "graphs" / "train_val_graph.pt").exists()
            )


if __name__ == "__main__":
    unittest.main()
