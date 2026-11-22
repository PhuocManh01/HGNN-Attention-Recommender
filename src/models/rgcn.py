"""Relation-aware graph recommender without attention."""

import torch
from torch import nn
from torch_geometric.nn import GraphConv, HeteroConv


class RGCNRecommender(nn.Module):
    """A relation-specific GCN baseline for the user/movie/genre graph.

    Each edge type has its own GraphConv operator. This preserves the
    heterogeneous relations while intentionally removing HGT attention, so it
    is a direct non-attention comparator for ``HGNNWithAttention``.
    """

    def __init__(
        self,
        metadata,
        num_users,
        num_movies,
        num_genres,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dim < 1 or num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.ModuleDict({
            "user": nn.Embedding(num_users, hidden_dim),
            "movie": nn.Embedding(num_movies, hidden_dim),
            "genre": nn.Embedding(num_genres, hidden_dim),
        })
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)

        edge_types = metadata[1]
        self.convs = nn.ModuleList([
            HeteroConv(
                {
                    edge_type: GraphConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in metadata[0]
            })
            for _ in range(num_layers)
        ])

    def propagate(self, graph):
        x_dict = {
            node_type: embedding.weight
            for node_type, embedding in self.embeddings.items()
        }

        for conv, norms in zip(self.convs, self.norms):
            updated = conv(x_dict, graph.edge_index_dict)
            x_dict = {
                node_type: norms[node_type](
                    previous + self.dropout(updated.get(node_type, previous))
                )
                for node_type, previous in x_dict.items()
            }
        return x_dict

    def score_pairs_from_embeddings(self, users, movies, embeddings):
        return (
            embeddings["user"][users]
            * embeddings["movie"][movies]
        ).sum(dim=-1)

    def forward(self, graph, users, movies):
        return self.score_pairs_from_embeddings(
            users, movies, self.propagate(graph)
        )

    @torch.no_grad()
    def score_all(self, graph, user_idx):
        self.eval()
        embeddings = self.propagate(graph)
        return (
            embeddings["movie"] @ embeddings["user"][int(user_idx)]
        ).detach().cpu()
