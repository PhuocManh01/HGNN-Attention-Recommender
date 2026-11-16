
import unittest

import numpy as np
import pandas as pd
import torch

from src.models.baselines import (
    PopularityModel,
    MFBPR,
    LightGCN
)
from src.evaluation.ranking import (
    ranking_metrics,
    evaluate_full_ranking
)
from src.training.baseline_trainer import (
    sample_negatives,
    bpr_loss
)


class TestBaselines(unittest.TestCase):

    def test_popularity(self):
        train = pd.DataFrame({
            "movie_idx": [0, 0, 1, 2],
            "is_positive": [1, 1, 1, 0]
        })

        model = PopularityModel(4).fit(train)

        self.assertEqual(
            float(model.scores[0]), 2.0
        )
        self.assertEqual(
            float(model.scores[2]), 0.0
        )

    def test_mf_output_shape(self):
        model = MFBPR(3, 5, 8)

        scores = model(
            torch.tensor([0, 1]),
            torch.tensor([2, 3])
        )

        self.assertEqual(tuple(scores.shape), (2,))

    def test_lightgcn_output_shape(self):
        edges = torch.tensor([
            [0, 1, 2],
            [1, 2, 3]
        ])

        model = LightGCN(
            num_users=3,
            num_movies=5,
            edge_index=edges,
            embedding_dim=8,
            num_layers=2
        )

        scores = model(
            torch.tensor([0, 1]),
            torch.tensor([1, 2])
        )

        self.assertEqual(tuple(scores.shape), (2,))

    def test_bpr_loss(self):
        positive = torch.tensor([3.0])
        negative = torch.tensor([1.0])

        loss = bpr_loss(positive, negative)

        self.assertGreater(float(loss), 0)
        self.assertLess(float(loss), 0.2)

    def test_negative_sampling(self):
        rng = np.random.default_rng(42)

        history = {
            0: {0, 1, 2},
            1: {1, 3}
        }

        users = torch.tensor([0, 1, 0, 1])

        negatives = sample_negatives(
            users, history, 5, rng
        )

        for user, movie in zip(
            users.tolist(), negatives.tolist()
        ):
            self.assertNotIn(movie, history[user])

    def test_ranking_metrics(self):
        metrics = ranking_metrics(
            recommended=[1, 2, 3, 4],
            relevant={1, 3},
            k=10
        )

        self.assertAlmostEqual(
            metrics["Precision@10"], 0.2
        )
        self.assertAlmostEqual(
            metrics["Recall@10"], 1.0
        )
        self.assertGreater(
            metrics["NDCG@10"], 0
        )

    def test_full_ranking_filters_history(self):
        train = pd.DataFrame({
            "user_idx": [0],
            "movie_idx": [0],
            "is_positive": [1]
        })

        val = pd.DataFrame({
            "user_idx": [0],
            "movie_idx": [1],
            "is_positive": [1]
        })

        model = PopularityModel(3)
        model.scores = torch.tensor(
            [100.0, 10.0, 1.0]
        )

        summary, _ = evaluate_full_ranking(
            model, train, val, [0], 3
        )

        self.assertAlmostEqual(
            summary["Recall@10"], 1.0
        )
        
    def test_ranking_excludes_observed_even_when_candidates_below_k(self):
        train = pd.DataFrame({
            "user_idx": [0, 0, 0],
            "movie_idx": [0, 1, 2],
            "is_positive": [1, 1, 1],
        })

        val = pd.DataFrame({
            "user_idx": [0],
            "movie_idx": [3],
            "is_positive": [1],
        })

        model = PopularityModel(4)
        model.scores = torch.tensor(
            [100.0, 90.0, 80.0, 1.0]
        )

        summary, _ = evaluate_full_ranking(
            model, train, val, [0], 4, k=10
        )

        self.assertEqual(summary["users_evaluated"], 1)
        self.assertAlmostEqual(summary["Recall@10"], 1.0)
        self.assertAlmostEqual(summary["Precision@10"], 0.1)


    def test_negative_sampling_never_uses_observed_low_ratings(self):
        rng = np.random.default_rng(42)

        # Both positive and low-rated observed items
        # must be excluded from negative sampling.
        history = {0: {0, 1, 2, 3}}
        users = torch.tensor([0] * 20)

        negatives = sample_negatives(
            users, history, 5, rng
        )

        self.assertTrue(
            all(movie == 4 for movie in negatives.tolist())
        )


if __name__ == "__main__":
    unittest.main()
