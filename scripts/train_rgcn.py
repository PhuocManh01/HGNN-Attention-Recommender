"""Train the non-attention R-GCN comparator using the Phase 5 protocol."""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.rgcn import RGCNRecommender
from src.training.hgnn_trainer import evaluate_hgnn, train_hgnn


def main():
    with open(ROOT / "configs/hgnn_attention.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = ROOT / cfg["data"]["root"]
    split_dir = data_root / "splits"
    graph_dir = data_root / "graphs"
    with open(split_dir / "split_metadata.json", encoding="utf-8") as f:
        split_meta = json.load(f)
    with open(graph_dir / "graph_metadata.json", encoding="utf-8") as f:
        graph_meta = json.load(f)

    if split_meta.get("dataset") != "ml-1m":
        raise ValueError("R-GCN requires MovieLens 1M artifacts")
    if split_meta["split_method"] != "per_user_temporal":
        raise ValueError("R-GCN requires the per_user_temporal split")
    if graph_meta["train_positive_edges"] != int(
        pd.read_csv(split_dir / "train.csv")["is_positive"].sum()
    ):
        raise ValueError("Train graph does not match Train split")

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    evaluation_users = pd.read_csv(split_dir / "evaluation_users.csv")
    eligible = evaluation_users.loc[
        evaluation_users["eligible_val"] == True, "user_idx"
    ].astype(int).tolist()
    graph = torch.load(
        graph_dir / "train_graph.pt", map_location="cpu", weights_only=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RGCNRecommender(
        metadata=graph.metadata(),
        num_users=graph["user"].num_nodes,
        num_movies=graph["movie"].num_nodes,
        num_genres=graph["genre"].num_nodes,
        hidden_dim=cfg["rgcn"]["hidden_dim"],
        num_layers=cfg["rgcn"]["num_layers"],
        dropout=cfg["rgcn"]["dropout"],
    )
    result = train_hgnn(
        model=model,
        graph=graph,
        train=train,
        val=val,
        eligible_val_users=eligible,
        num_movies=graph["movie"].num_nodes,
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        num_negatives=cfg["training"].get("num_negatives", 1),
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        seed=cfg["project"]["seed"],
        device=device,
    )
    summary, details = evaluate_hgnn(
        result["model"], graph.to(device), train, val, eligible,
        graph["movie"].num_nodes, k=cfg["project"]["top_k"]
    )

    result_dir = ROOT / cfg["rgcn"]["output_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(result_dir / "rgcn_val_users.csv", index=False)
    with open(result_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(result["history"], f, indent=2)
    report = {
        "phase": 5,
        "model": "R-GCN",
        "dataset": split_meta["dataset"],
        "split_method": split_meta["split_method"],
        "best_epoch": result["best_epoch"],
        "best_validation_ndcg": result["best_ndcg"],
        "metrics": summary,
        "test_evaluated": False,
    }
    with open(result_dir / "phase5_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
