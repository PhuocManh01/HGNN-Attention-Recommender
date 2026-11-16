
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv


class HGNNWithAttention(nn.Module):
    def __init__(
        self,
        metadata,
        num_users: int,
        num_movies: int,
        num_genres: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.embeddings = nn.ModuleDict({
            "user": nn.Embedding(num_users, hidden_dim),
            "movie": nn.Embedding(num_movies, hidden_dim),
            "genre": nn.Embedding(num_genres, hidden_dim),
        })

        for embedding in self.embeddings.values():
            nn.init.normal_(
                embedding.weight,
                mean=0.0,
                std=0.01,
            )

        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Learnable residual strength for each layer.
        self.residual_weights = nn.Parameter(
            torch.zeros(num_layers)
        )

    def propagate(self, graph):
        x_dict = {
            node_type: embedding.weight
            for node_type, embedding
            in self.embeddings.items()
        }

        for layer_idx, conv in enumerate(self.convs):
            updated = conv(
                x_dict,
                graph.edge_index_dict,
            )

            alpha = torch.sigmoid(
                self.residual_weights[layer_idx]
            )

            next_dict = {}

            for node_type, previous in x_dict.items():
                message = updated.get(node_type)

                if message is None:
                    next_dict[node_type] = previous
                    continue

                # Residual path preserves trainable
                # information for sparse nodes.
                next_dict[node_type] = (
                    (1.0 - alpha) * previous
                    + alpha * self.dropout(message)
                )

                next_dict[node_type] = self.norms[layer_idx](
                    next_dict[node_type]
                )

            x_dict = next_dict

        return x_dict

    def score_pairs_from_embeddings(
        self,
        users,
        movies,
        embeddings,
    ):
        user_emb = embeddings["user"][users]
        movie_emb = embeddings["movie"][movies]

        return (user_emb * movie_emb).sum(dim=-1)

    def forward(self, graph, users, movies):
        embeddings = self.propagate(graph)

        return self.score_pairs_from_embeddings(
            users,
            movies,
            embeddings,
        )

    @torch.no_grad()
    def score_all(self, graph, user_idx):
        self.eval()
        embeddings = self.propagate(graph)

        scores = (
            embeddings["movie"]
            @ embeddings["user"][int(user_idx)]
        )

        return scores.detach().cpu()
