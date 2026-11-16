
import unittest

import torch
from torch_geometric.data import HeteroData

from src.models.hgnn_attention import HGNNWithAttention
from src.training.baseline_trainer import bpr_loss


class TestHGNNWithAttention(unittest.TestCase):

    def setUp(self):
        self.graph = HeteroData()

        self.graph["user"].num_nodes = 3
        self.graph["movie"].num_nodes = 4
        self.graph["genre"].num_nodes = 2

        rates = torch.tensor([
            [0, 0, 1, 2],
            [0, 1, 2, 3],
        ])

        has_genre = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 0, 1],
        ])

        self.graph[
            "user", "rates", "movie"
        ].edge_index = rates

        self.graph[
            "movie", "rev_rates", "user"
        ].edge_index = rates.flip(0).contiguous()

        self.graph[
            "movie", "has_genre", "genre"
        ].edge_index = has_genre

        self.graph[
            "genre", "rev_has_genre", "movie"
        ].edge_index = has_genre.flip(0).contiguous()

        self.model = HGNNWithAttention(
            metadata=self.graph.metadata(),
            num_users=3,
            num_movies=4,
            num_genres=2,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )

    def test_embedding_shapes(self):
        self.model.eval()

        with torch.no_grad():
            result = self.model.propagate(self.graph)

        self.assertEqual(
            result["user"].shape, (3, 16)
        )
        self.assertEqual(
            result["movie"].shape, (4, 16)
        )
        self.assertEqual(
            result["genre"].shape, (2, 16)
        )

    def test_forward_shape(self):
        scores = self.model(
            self.graph,
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
        )

        self.assertEqual(scores.shape, (2,))
        self.assertTrue(
            torch.isfinite(scores).all()
        )

    def test_backward(self):
        users = torch.tensor([0, 1])
        positives = torch.tensor([0, 2])
        negatives = torch.tensor([3, 1])

        embeddings = self.model.propagate(self.graph)

        pos_scores = (
            self.model.score_pairs_from_embeddings(
                users, positives, embeddings
            )
        )

        neg_scores = (
            self.model.score_pairs_from_embeddings(
                users, negatives, embeddings
            )
        )

        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()

        self.assertTrue(
            torch.isfinite(loss).item()
        )

        self.assertIsNotNone(
            self.model.embeddings["user"].weight.grad
        )

        self.assertTrue(
            any(
                p.grad is not None
                for conv in self.model.convs
                for p in conv.parameters()
            )
        )

    def test_invalid_heads(self):
        with self.assertRaises(ValueError):
            HGNNWithAttention(
                metadata=self.graph.metadata(),
                num_users=3,
                num_movies=4,
                num_genres=2,
                hidden_dim=15,
                num_heads=4,
            )
            
    


if __name__ == "__main__":
    unittest.main()
