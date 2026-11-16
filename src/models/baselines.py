
import torch
from torch import nn


class PopularityModel:
    def __init__(self, num_movies):
        self.num_movies = num_movies
        self.scores = torch.zeros(
            num_movies, dtype=torch.float32
        )

    def fit(self, train):
        positive = train[
            train["is_positive"] == 1
        ]

        counts = (
            positive.groupby("movie_idx")
            .size()
        )

        self.scores.zero_()

        for movie_idx, count in counts.items():
            self.scores[int(movie_idx)] = float(count)

        return self

    def score_all(self, user_idx=None):
        return self.scores.clone()


class MFBPR(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=64
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users, embedding_dim
        )
        self.movie_embedding = nn.Embedding(
            num_movies, embedding_dim
        )

        nn.init.normal_(
            self.user_embedding.weight, std=0.01
        )
        nn.init.normal_(
            self.movie_embedding.weight, std=0.01
        )

    def forward(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)

        return (user_emb * movie_emb).sum(dim=-1)

    def score_all(self, user_idx):
        with torch.no_grad():
            user = self.user_embedding.weight[user_idx]
            return (
                self.movie_embedding.weight @ user
            ).detach().cpu()


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        edge_index,
        embedding_dim=64,
        num_layers=2
    ):
        super().__init__()

        self.num_users = num_users
        self.num_movies = num_movies
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_users + num_movies,
            embedding_dim
        )

        nn.init.normal_(
            self.embedding.weight, std=0.01
        )

        # User IDs: [0, num_users)
        # Movie IDs: [num_users, num_users + num_movies)
        users = edge_index[0].long()
        movies = edge_index[1].long() + num_users

        src = torch.cat([users, movies])
        dst = torch.cat([movies, users])

        n = num_users + num_movies

        degree = torch.bincount(
            src, minlength=n
        ).float().clamp(min=1)

        values = (
            degree[src].rsqrt()
            * degree[dst].rsqrt()
        )

        adjacency = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            values,
            (n, n),
            check_invariants=True
        ).coalesce()

        self.register_buffer(
            "adjacency", adjacency
        )

    def propagate(self):
        x = self.embedding.weight
        layers = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(
                self.adjacency, x
            )
            layers.append(x)

        final = torch.stack(layers).mean(dim=0)

        users = final[:self.num_users]
        movies = final[self.num_users:]

        return users, movies

    def forward(self, users, movies):
        user_emb, movie_emb = self.propagate()

        return self.score_pairs_from_embeddings(
            users, movies, user_emb, movie_emb
        )

    def score_pairs_from_embeddings(
        self, users, movies, user_emb, movie_emb
    ):
        return (
            user_emb[users] * movie_emb[movies]
        ).sum(dim=-1)

    def score_all(self, user_idx):
        with torch.no_grad():
            user_emb, movie_emb = self.propagate()

            return (
                movie_emb @ user_emb[user_idx]
            ).detach().cpu()
