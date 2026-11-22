
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.ranking import (
    build_history,
    evaluate_full_ranking
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_training_data(train):
    positive = train[
        train["is_positive"] == 1
    ]

    users = torch.tensor(
        positive["user_idx"].to_numpy(),
        dtype=torch.long
    )

    movies = torch.tensor(
        positive["movie_idx"].to_numpy(),
        dtype=torch.long
    )

    return users, movies


def sample_negatives(
    users,
    history,
    num_movies,
    rng,
    num_negatives=1,
):
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")

    negatives = []

    for user in users.tolist():
        observed = history.get(int(user), set())

        if len(observed) >= num_movies:
            raise ValueError(
                "User has no unobserved candidate"
            )

        user_negatives = []
        while len(user_negatives) < num_negatives:
            movie = int(rng.integers(num_movies))

            if movie not in observed:
                user_negatives.append(movie)

        negatives.append(user_negatives)

    negative_tensor = torch.tensor(
        negatives,
        dtype=torch.long
    )

    if num_negatives == 1:
        return negative_tensor[:, 0]

    return negative_tensor


def bpr_loss(positive_scores, negative_scores):
    return F.softplus(
        negative_scores - positive_scores
    ).mean()


def train_bpr_model(
    model,
    train,
    val,
    eligible_val_users,
    num_movies,
    epochs=50,
    batch_size=1024,
    learning_rate=0.001,
    weight_decay=0.0001,
    patience=7,
    seed=42,
    device="cpu"
):
    set_seed(seed)

    device = torch.device(device)
    model = model.to(device)

    users, positives = make_training_data(train)

    history = build_history(train)
    rng = np.random.default_rng(seed)

    # Remove users with no valid negative candidates.
    valid_mask = torch.tensor(
        [
            len(history[int(u)]) < num_movies
            for u in users.tolist()
        ],
        dtype=torch.bool
    )

    users = users[valid_mask]
    positives = positives[valid_mask]

    if len(users) == 0:
        raise ValueError("No trainable positive pairs")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_ndcg = float("-inf")
    best_state = None
    best_epoch = 0
    wait = 0
    stopped_early = False
    logs = []

    is_lightgcn = (
        model.__class__.__name__ == "LightGCN"
    )

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()

        permutation = torch.randperm(len(users))
        epoch_losses = []

        if is_lightgcn:
            # Full-graph propagation is computed once
            # per optimizer step, not once per sample.
            # On small graphs, use one full batch.
            batches = [permutation]
        else:
            batches = permutation.split(batch_size)

        for batch in batches:
            batch_users = users[batch]
            batch_pos = positives[batch]

            batch_neg = sample_negatives(
                batch_users,
                history,
                num_movies,
                rng
            )

            batch_users = batch_users.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            optimizer.zero_grad()

            if is_lightgcn:
                user_emb, movie_emb = model.propagate()

                pos_scores = model.score_pairs_from_embeddings(
                    batch_users, batch_pos,
                    user_emb, movie_emb
                )

                neg_scores = model.score_pairs_from_embeddings(
                    batch_users, batch_neg,
                    user_emb, movie_emb
                )
            else:
                pos_scores = model(
                    batch_users, batch_pos
                )

                neg_scores = model(
                    batch_users, batch_neg
                )

            loss = bpr_loss(
                pos_scores, neg_scores
            )

            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        model.eval()

        summary, _ = evaluate_full_ranking(
            model=model,
            history_interactions=train,
            target_interactions=val,
            eligible_users=eligible_val_users,
            num_movies=num_movies,
            k=10
        )

        current_ndcg = summary["NDCG@10"]

        logs.append({
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "val_precision": summary["Precision@10"],
            "val_recall": summary["Recall@10"],
            "val_ndcg": current_ndcg,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "optimizer_steps": len(batches),
            "positive_pairs": int(len(users)),
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {logs[-1]['loss']:.5f} | "
            f"NDCG@10: {current_ndcg:.5f}"
        )

        if current_ndcg > best_ndcg + 1e-8:
            best_ndcg = current_ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping triggered")
            stopped_early = True
            break

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_ndcg": best_ndcg,
        "history": logs,
        "epochs_ran": len(logs),
        "stopped_early": stopped_early,
    }
