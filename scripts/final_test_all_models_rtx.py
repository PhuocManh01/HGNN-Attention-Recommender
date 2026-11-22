"""Final Test evaluation for all recommender models on an RTX GPU.

Protocol:
  1. Keep the existing split and mappings fixed.
  2. Use epoch counts selected from Validation only.
  3. Retrain on Train + Validation.
  4. Build/use the Train + Validation graph only.
  5. Evaluate once on Test with full ranking.

No Test metric is used for early stopping, tuning, or model selection.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.ranking import evaluate_full_ranking
from src.models.baselines import LightGCN, MFBPR, PopularityModel
from src.models.hgnn_attention import HGNNWithAttention
from src.models.rgcn import RGCNRecommender
from src.training.baseline_trainer import (
    bpr_loss,
    make_training_data,
    sample_negatives,
    set_seed,
)


# These values are frozen from the Validation runs before Test evaluation.
FINAL_EPOCHS = {
    "MF-BPR": 2,
    "LightGCN": 24,
    "Hetero-GCN-No-Attention": 33,
    "HGT-Attention": 45,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain and evaluate all models on the final Test split."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/final_test_rtx",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def require_cuda(device_name):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            f"torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda!r}"
        )
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device does not exist: {device_name}")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_bpr_fixed(
    model,
    train,
    num_movies,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    seed,
    device,
    *,
    lightgcn=False,
):
    """Train MF-BPR or LightGCN for a validation-frozen epoch count."""
    set_seed(seed)
    device = torch.device(device)
    model = model.to(device)
    users, positives = make_training_data(train)
    history = {
        int(user): set(group["movie_idx"].astype(int))
        for user, group in train.groupby("user_idx")
    }
    valid_mask = torch.tensor(
        [len(history[int(user)]) < num_movies for user in users.tolist()],
        dtype=torch.bool,
    )
    users = users[valid_mask]
    positives = positives[valid_mask]
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(users))
        batches = [permutation] if lightgcn else permutation.split(batch_size)
        losses = []
        for batch in batches:
            batch_users = users[batch]
            batch_pos = positives[batch]
            batch_neg = sample_negatives(
                batch_users, history, num_movies, rng
            )
            batch_users = batch_users.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            optimizer.zero_grad()
            if lightgcn:
                user_emb, movie_emb = model.propagate()
                pos_scores = model.score_pairs_from_embeddings(
                    batch_users, batch_pos, user_emb, movie_emb
                )
                neg_scores = model.score_pairs_from_embeddings(
                    batch_users, batch_neg, user_emb, movie_emb
                )
            else:
                pos_scores = model(batch_users, batch_pos)
                neg_scores = model(batch_users, batch_neg)
            loss = bpr_loss(pos_scores, neg_scores)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite BPR loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        synchronize(device)
        print(
            f"{model.__class__.__name__} final epoch {epoch:02d}/{epochs} | "
            f"loss={np.mean(losses):.5f} | "
            f"seconds={time.perf_counter() - started:.2f}"
        )
    return model


def fit_gnn_fixed(
    model,
    graph,
    train,
    num_movies,
    epochs,
    batch_size,
    num_negatives,
    learning_rate,
    weight_decay,
    seed,
    device,
):
    """Train either heterogeneous GNN for a frozen epoch count."""
    set_seed(seed)
    device = torch.device(device)
    model = model.to(device)
    graph = graph.to(device)
    users, positives = make_training_data(train)
    history = {
        int(user): set(group["movie_idx"].astype(int))
        for user, group in train.groupby("user_idx")
    }
    valid_mask = torch.tensor(
        [len(history[int(user)]) < num_movies for user in users.tolist()],
        dtype=torch.bool,
    )
    users = users[valid_mask].to(device)
    positives = positives[valid_mask].to(device)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(users), device=device)
        optimizer.zero_grad()
        embeddings = model.propagate(graph)
        losses = []
        for batch in permutation.split(batch_size):
            batch_users = users[batch]
            batch_pos = positives[batch]
            batch_neg = sample_negatives(
                batch_users.cpu(), history, num_movies, rng, num_negatives
            ).to(device)
            positive_scores = model.score_pairs_from_embeddings(
                batch_users, batch_pos, embeddings
            )
            if batch_neg.ndim == 1:
                negative_scores = model.score_pairs_from_embeddings(
                    batch_users, batch_neg, embeddings
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
                raise FloatingPointError(f"Non-finite GNN loss at epoch {epoch}")
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        synchronize(device)
        print(
            f"{model.__class__.__name__} final epoch {epoch:02d}/{epochs} | "
            f"loss={loss.item():.5f} | "
            f"seconds={time.perf_counter() - started:.2f}"
        )
    return model


def evaluate_model(model, history, test, eligible, num_movies, device, score_fn=None):
    if hasattr(model, "eval"):
        model.eval()
    return evaluate_full_ranking(
        model=model,
        history_interactions=history,
        target_interactions=test,
        eligible_users=eligible,
        num_movies=num_movies,
        k=10,
        score_fn=score_fn,
    )


def main():
    args = parse_args()
    device = require_cuda(args.device)
    with open(ROOT / "configs/hgnn_attention.yaml", encoding="utf-8") as file:
        hgnn_cfg = yaml.safe_load(file)
    with open(ROOT / "configs/baselines.yaml", encoding="utf-8") as file:
        baseline_cfg = yaml.safe_load(file)

    data_root = ROOT / "data"
    split_dir = data_root / "splits"
    graph_dir = data_root / "graphs"
    with open(split_dir / "split_metadata.json", encoding="utf-8") as file:
        split_meta = json.load(file)
    with open(graph_dir / "graph_metadata.json", encoding="utf-8") as file:
        graph_meta = json.load(file)
    if split_meta["dataset"] != "ml-1m":
        raise ValueError("Final Test script requires MovieLens 1M")
    if split_meta["split_method"] != "per_user_temporal":
        raise ValueError("Unexpected split method")
    if graph_meta["split_method"] != split_meta["split_method"]:
        raise ValueError("Graph/split protocol mismatch")

    train_val = pd.read_csv(split_dir / "train_val.csv")
    test = pd.read_csv(split_dir / "test.csv")
    users = pd.read_csv(split_dir / "evaluation_users.csv")
    eligible = users.loc[
        users["eligible_test"] == True, "user_idx"
    ].astype(int).tolist()
    graph = torch.load(
        graph_dir / "train_val_graph.pt", map_location="cpu", weights_only=False
    )
    num_users = int(graph["user"].num_nodes)
    num_movies = int(graph["movie"].num_nodes)
    num_genres = int(graph["genre"].num_nodes)
    expected_edges = int(train_val["is_positive"].sum())
    if graph_meta["train_val_positive_edges"] != expected_edges:
        raise ValueError("Train+Validation graph edge mismatch")

    seed = int(hgnn_cfg["project"]["seed"])
    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    details_dir = output_dir / "per_user"
    details_dir.mkdir(parents=True, exist_ok=True)

    print(f"Final Test device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Train+Validation positive edges: {expected_edges}")
    print(f"Eligible Test users: {len(eligible)}")

    # 1. Non-learned popularity baseline.
    started = time.perf_counter()
    popularity = PopularityModel(num_movies).fit(train_val)
    pop_summary, pop_details = evaluate_model(
        popularity, train_val, test, eligible, num_movies, device
    )
    pop_details.to_csv(details_dir / "popularity_test_users.csv", index=False)
    torch.save(popularity.scores, checkpoint_dir / "popularity_final.pt")
    results.append({"model": "Popularity", **pop_summary, "epochs": 0})
    print(f"Popularity complete in {time.perf_counter() - started:.2f}s")

    # 2. MF-BPR.
    set_seed(seed)
    mf = MFBPR(
        num_users,
        num_movies,
        baseline_cfg["training"]["embedding_dim"],
    )
    mf = fit_bpr_fixed(
        mf,
        train_val,
        num_movies,
        FINAL_EPOCHS["MF-BPR"],
        baseline_cfg["training"]["batch_size"],
        baseline_cfg["training"]["learning_rate"],
        baseline_cfg["training"]["weight_decay"],
        seed,
        device,
    )
    mf_summary, mf_details = evaluate_model(
        mf, train_val, test, eligible, num_movies, device
    )
    mf_details.to_csv(details_dir / "mf_bpr_test_users.csv", index=False)
    torch.save(mf.state_dict(), checkpoint_dir / "mf_bpr_final.pt")
    results.append({
        "model": "MF-BPR",
        **mf_summary,
        "epochs": FINAL_EPOCHS["MF-BPR"],
    })

    # 3. LightGCN.
    edge_index = graph[("user", "rates", "movie")].edge_index
    set_seed(seed)
    lightgcn = LightGCN(
        num_users=num_users,
        num_movies=num_movies,
        edge_index=edge_index,
        embedding_dim=baseline_cfg["training"]["embedding_dim"],
        num_layers=baseline_cfg["lightgcn"]["num_layers"],
    )
    lightgcn = fit_bpr_fixed(
        lightgcn,
        train_val,
        num_movies,
        FINAL_EPOCHS["LightGCN"],
        baseline_cfg["training"]["batch_size"],
        baseline_cfg["lightgcn"]["learning_rate"],
        baseline_cfg["lightgcn"]["weight_decay"],
        seed,
        device,
        lightgcn=True,
    )
    lg_summary, lg_details = evaluate_model(
        lightgcn, train_val, test, eligible, num_movies, device
    )
    lg_details.to_csv(details_dir / "lightgcn_test_users.csv", index=False)
    torch.save(lightgcn.state_dict(), checkpoint_dir / "lightgcn_final.pt")
    results.append({
        "model": "LightGCN",
        **lg_summary,
        "epochs": FINAL_EPOCHS["LightGCN"],
    })

    # 4. Hetero-GCN without attention.
    set_seed(seed)
    gnn = RGCNRecommender(
        metadata=graph.metadata(),
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        hidden_dim=hgnn_cfg["rgcn"]["hidden_dim"],
        num_layers=hgnn_cfg["rgcn"]["num_layers"],
        dropout=hgnn_cfg["rgcn"]["dropout"],
    )
    gnn = fit_gnn_fixed(
        gnn,
        graph,
        train_val,
        num_movies,
        FINAL_EPOCHS["Hetero-GCN-No-Attention"],
        hgnn_cfg["training"]["batch_size"],
        hgnn_cfg["training"].get("num_negatives", 1),
        hgnn_cfg["training"]["learning_rate"],
        hgnn_cfg["training"]["weight_decay"],
        seed,
        device,
    )
    gnn_embeddings = gnn.propagate(graph)
    gnn_summary, gnn_details = evaluate_model(
        gnn, train_val, test, eligible, num_movies, device,
        score_fn=lambda user_idx: (
            gnn_embeddings["movie"]
            @ gnn_embeddings["user"][int(user_idx)]
        ).detach().cpu(),
    )
    gnn_details.to_csv(details_dir / "gnn_no_attention_test_users.csv", index=False)
    torch.save(gnn.state_dict(), checkpoint_dir / "gnn_no_attention_final.pt")
    results.append({
        "model": "Hetero-GCN-No-Attention",
        **gnn_summary,
        "epochs": FINAL_EPOCHS["Hetero-GCN-No-Attention"],
    })

    # 5. HGT attention.
    set_seed(seed)
    hgt = HGNNWithAttention(
        metadata=graph.metadata(),
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        hidden_dim=hgnn_cfg["model"]["hidden_dim"],
        num_layers=hgnn_cfg["model"]["num_layers"],
        num_heads=hgnn_cfg["model"]["num_heads"],
        dropout=hgnn_cfg["model"]["dropout"],
    )
    hgt = fit_gnn_fixed(
        hgt,
        graph,
        train_val,
        num_movies,
        FINAL_EPOCHS["HGT-Attention"],
        hgnn_cfg["training"]["batch_size"],
        hgnn_cfg["training"].get("num_negatives", 1),
        hgnn_cfg["training"]["learning_rate"],
        hgnn_cfg["training"]["weight_decay"],
        seed,
        device,
    )
    hgt_embeddings = hgt.propagate(graph)
    hgt_summary, hgt_details = evaluate_model(
        hgt, train_val, test, eligible, num_movies, device,
        score_fn=lambda user_idx: (
            hgt_embeddings["movie"]
            @ hgt_embeddings["user"][int(user_idx)]
        ).detach().cpu(),
    )
    hgt_details.to_csv(details_dir / "hgt_attention_test_users.csv", index=False)
    torch.save(hgt.state_dict(), checkpoint_dir / "hgt_attention_final.pt")
    results.append({
        "model": "HGT-Attention",
        **hgt_summary,
        "epochs": FINAL_EPOCHS["HGT-Attention"],
    })

    result_table = pd.DataFrame(results)
    result_table.to_csv(output_dir / "final_test_results.csv", index=False)
    report = {
        "dataset": split_meta["dataset"],
        "split_method": split_meta["split_method"],
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "train_positive_edges": int(train_val["is_positive"].sum()),
        "eligible_test_users": len(eligible),
        "evaluation": "full-ranking@10",
        "history_for_test": "Train+Validation",
        "test_labels_used_only_for_metrics": True,
        "frozen_validation_epochs": FINAL_EPOCHS,
        "results": results,
    }
    with open(output_dir / "final_test_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print("\nFINAL TEST RESULTS")
    print(result_table.to_string(index=False))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
