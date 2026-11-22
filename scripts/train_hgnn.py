
import json
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.hgnn_attention import HGNNWithAttention
from src.training.hgnn_trainer import (
    evaluate_hgnn,
    train_hgnn,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoint and experiment outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(
        ROOT / "configs/hgnn_attention.yaml",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    data_root = ROOT / cfg["data"]["root"]
    split_dir = data_root / "splits"
    graph_dir = data_root / "graphs"

    with open(
        split_dir / "split_metadata.json",
        encoding="utf-8",
    ) as f:
        split_meta = json.load(f)

    with open(
        graph_dir / "graph_metadata.json",
        encoding="utf-8",
    ) as f:
        graph_meta = json.load(f)

    expected_split = cfg["data"]["split_method"]

    if (
        split_meta["split_method"] != expected_split
        or graph_meta["split_method"] != expected_split
    ):
        raise ValueError("Split protocol mismatch")

    if graph_meta["split_seed"] != split_meta["seed"]:
        raise ValueError("Graph seed mismatch")

    if (
        float(split_meta["positive_threshold"])
        != float(cfg["data"]["positive_threshold"])
    ):
        raise ValueError("Positive threshold mismatch")

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    evaluation_users = pd.read_csv(
        split_dir / "evaluation_users.csv"
    )

    eligible = evaluation_users.loc[
        evaluation_users["eligible_val"] == True,
        "user_idx",
    ].astype(int).tolist()

    graph = torch.load(
        graph_dir / "train_graph.pt",
        map_location="cpu",
        weights_only=False,
    )

    edge_type = ("user", "rates", "movie")

    actual = set(
        map(tuple, graph[edge_type].edge_index.T.tolist())
    )
    expected = set(
        map(
            tuple,
            train.loc[
                train["is_positive"] == 1,
                ["user_idx", "movie_idx"],
            ].to_numpy().tolist(),
        )
    )

    if actual != expected:
        raise ValueError("Train graph edge mismatch")

    num_users = graph["user"].num_nodes
    num_movies = graph["movie"].num_nodes
    num_genres = graph["genre"].num_nodes

    requested_device = cfg["project"]["device"]

    if requested_device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = requested_device

    torch.manual_seed(cfg["project"]["seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            cfg["project"]["seed"]
        )

    model = HGNNWithAttention(
        metadata=graph.metadata(),
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    )

    print("Device:", device)
    print("Split:", expected_split)
    print("Eligible Validation users:", len(eligible))

    result = train_hgnn(
        model=model,
        graph=graph,
        train=train,
        val=val,
        eligible_val_users=eligible,
        num_movies=num_movies,
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        num_negatives=cfg["training"].get("num_negatives", 1),
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        seed=cfg["project"]["seed"],
        device=device,
    )

    model = result["model"]

    summary, details = evaluate_hgnn(
        model,
        graph.to(device),
        train,
        val,
        eligible,
        num_movies,
        k=cfg["project"]["top_k"],
    )

    if summary["users_evaluated"] != len(eligible):
        raise ValueError(
            "Evaluation user count mismatch"
        )

    result_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            ROOT / "experiments/phase5_smoke"
            if int(cfg["training"]["epochs"]) <= 2
            else ROOT / "experiments/phase5"
        )
    )
    checkpoint_dir = (
        result_dir / "checkpoints"
        if args.output_dir is not None
        else ROOT / "checkpoints/hgnn"
    )

    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )
    result_dir.mkdir(
        parents=True, exist_ok=True
    )

    torch.save({
        "state_dict": model.state_dict(),
        "model_config": cfg["model"],
        "num_users": num_users,
        "num_movies": num_movies,
        "num_genres": num_genres,
        "best_epoch": result["best_epoch"],
        "seed": cfg["project"]["seed"],
        "split_method": expected_split,
    }, checkpoint_dir / "hgnn_attention.pt")

    details.to_csv(
        result_dir / "hgnn_val_users.csv",
        index=False,
    )

    with open(
        result_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result["history"], f, indent=2)

    report = {
        "phase": 5,
        "model": "HGNN-HGT-Attention",
        "dataset": split_meta.get("dataset", "unknown"),
        "split_method": expected_split,
        "seed": cfg["project"]["seed"],
        "device": device,
        "best_epoch": result["best_epoch"],
        "best_validation_ndcg": result["best_ndcg"],
        "metrics": summary,
        "test_evaluated": False,
    }

    with open(
        result_dir / "phase5_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\nPHASE 5 VALIDATION RESULTS")
    print(json.dumps(report, indent=2))
    print("\nPhase 5 training completed.")


if __name__ == "__main__":
    main()
