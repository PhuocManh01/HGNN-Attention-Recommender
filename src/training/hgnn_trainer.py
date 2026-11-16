
import copy

import numpy as np
import torch

from src.evaluation.ranking import (
    build_history,
    evaluate_full_ranking,
)
from src.training.baseline_trainer import (
    bpr_loss,
    make_training_data,
    sample_negatives,
    set_seed,
)


@torch.no_grad()
def evaluate_hgnn(
    model,
    graph,
    history_interactions,
    target_interactions,
    eligible_users,
    num_movies,
    k=10,
):
    model.eval()
    embeddings = model.propagate(graph)

    user_emb = embeddings["user"]
    movie_emb = embeddings["movie"]

    def score_fn(user_idx):
        return (
            movie_emb @ user_emb[int(user_idx)]
        ).detach().cpu()

    return evaluate_full_ranking(
        model=model,
        history_interactions=history_interactions,
        target_interactions=target_interactions,
        eligible_users=eligible_users,
        num_movies=num_movies,
        k=k,
        score_fn=score_fn,
    )


def train_hgnn(
    model,
    graph,
    train,
    val,
    eligible_val_users,
    num_movies,
    *,
    epochs=50,
    batch_size=1024,
    num_negatives=5,
    learning_rate=0.001,
    weight_decay=0.0001,
    patience=7,
    seed=42,
    device="cpu",
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")

    set_seed(seed)
    device = torch.device(device)

    model = model.to(device)
    graph = graph.to(device)

    users, positives = make_training_data(train)
    history = build_history(train)

    valid_mask = torch.tensor([
        len(history[int(u)]) < num_movies
        for u in users.tolist()
    ], dtype=torch.bool)

    users = users[valid_mask].to(device)
    positives = positives[valid_mask].to(device)

    if len(users) == 0:
        raise ValueError("No trainable positive pairs")

    rng = np.random.default_rng(seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_ndcg = float("-inf")
    best_epoch = 0
    best_state = None
    wait = 0
    logs = []

    for epoch in range(1, epochs + 1):
        model.train()

        permutation = torch.randperm(len(users))
        epoch_losses = []

        for batch in permutation.split(batch_size):
            batch_users = users[batch]
            batch_pos = positives[batch]
            batch_neg = sample_negatives(
                batch_users.cpu(),
                history,
                num_movies,
                rng,
                num_negatives=num_negatives,
            ).to(device)

            optimizer.zero_grad()
            embeddings = model.propagate(graph)

            positive_scores = model.score_pairs_from_embeddings(
                batch_users,
                batch_pos,
                embeddings,
            )

            if batch_neg.ndim == 1:
                negative_scores = model.score_pairs_from_embeddings(
                    batch_users,
                    batch_neg,
                    embeddings,
                )
                loss = bpr_loss(positive_scores, negative_scores)
            else:
                repeated_users = batch_users[:, None].expand_as(batch_neg)
                negative_scores = model.score_pairs_from_embeddings(
                    repeated_users.reshape(-1),
                    batch_neg.reshape(-1),
                    embeddings,
                ).reshape(batch_neg.shape)
                loss = torch.nn.functional.softplus(
                    negative_scores - positive_scores[:, None]
                ).mean()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        summary, _ = evaluate_hgnn(
            model,
            graph,
            train,
            val,
            eligible_val_users,
            num_movies,
        )

        ndcg = summary["NDCG@10"]

        logs.append({
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "val_precision": summary["Precision@10"],
            "val_recall": summary["Recall@10"],
            "val_ndcg": ndcg,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={loss.item():.5f} | "
            f"NDCG@10={ndcg:.5f}"
        )

        if ndcg > best_ndcg + 1e-8:
            best_ndcg = ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_ndcg": best_ndcg,
        "history": logs,
    }
