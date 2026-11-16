
import numpy as np
import pandas as pd
import torch


def build_history(interactions):
    return {
        int(user): set(group["movie_idx"].astype(int))
        for user, group in interactions.groupby("user_idx")
    }


def build_positive_labels(interactions):
    positive = interactions[
        interactions["is_positive"] == 1
    ]

    return {
        int(user): set(group["movie_idx"].astype(int))
        for user, group in positive.groupby("user_idx")
    }


def ranking_metrics(recommended, relevant, k=10):
    if k <= 0:
        raise ValueError("k must be positive")
    
    if not relevant:
        raise ValueError(
            "Relevant set must not be empty"
        )

    recommended = list(recommended)[:k]

    hits = np.array(
        [int(movie in relevant) for movie in recommended],
        dtype=np.float64
    )

    precision = hits.sum() / k
    recall = hits.sum() / len(relevant)

    discounts = 1.0 / np.log2(
        np.arange(2, len(hits) + 2)
    )

    dcg = float((hits * discounts).sum())

    ideal_length = min(k, len(relevant))

    idcg = float(
        (
            1.0 / np.log2(
                np.arange(2, ideal_length + 2)
            )
        ).sum()
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "Precision@10": float(precision),
        "Recall@10": float(recall),
        "NDCG@10": float(ndcg)
    }


def evaluate_full_ranking(
    model,
    history_interactions,
    target_interactions,
    eligible_users,
    num_movies,
    k=10,
    score_fn=None
):
    history = build_history(history_interactions)
    labels = build_positive_labels(target_interactions)

    records = []

    cached_user_embeddings = None
    cached_movie_embeddings = None

    if hasattr(model, "propagate"):
        model.eval()

        with torch.no_grad():
            cached_user_embeddings, cached_movie_embeddings = (
                model.propagate()
            )

    for user_idx in eligible_users:
        user_idx = int(user_idx)

        observed = history.get(user_idx, set())

        # Fixed movie universe: all mapped movies.
        # Only past observed interactions are filtered.
        candidates = np.ones(
            num_movies, dtype=bool
        )

        if observed:
            candidates[list(observed)] = False

        relevant = labels.get(user_idx, set())

        # Held-out positives must remain rankable.
        relevant = relevant & set(
            np.flatnonzero(candidates).tolist()
        )

        if not relevant:
            continue

        if cached_user_embeddings is not None:
            scores = (
                cached_movie_embeddings
                @ cached_user_embeddings[user_idx]
            )
        elif score_fn is None:
            scores = model.score_all(user_idx)
        else:
            scores = score_fn(user_idx)

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        scores = np.asarray(
            scores, dtype=np.float64
        ).copy()

        assert len(scores) == num_movies
        assert np.isfinite(scores[candidates]).all()

        candidate_ids = np.flatnonzero(candidates)

        ranked_candidates = candidate_ids[
            np.lexsort((
                candidate_ids,
                -scores[candidate_ids],
            ))
        ]

        top_k = ranked_candidates[:k].tolist()

        metrics = ranking_metrics(
            top_k, relevant, k
        )

        records.append({
            "user_idx": user_idx,
            **metrics
        })

    if not records:
        raise ValueError(
            "No eligible users with rankable positives"
        )

    details = pd.DataFrame(records)

    summary = {
        "users_evaluated": len(details),
        "Precision@10": float(
            details["Precision@10"].mean()
        ),
        "Recall@10": float(
            details["Recall@10"].mean()
        ),
        "NDCG@10": float(
            details["NDCG@10"].mean()
        )
    }

    return summary, details
