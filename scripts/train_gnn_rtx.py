"""Train the heterogeneous GNN baseline without attention on an NVIDIA RTX GPU.

The model is ``RGCNRecommender``: relation-specific ``GraphConv`` layers
wrapped by ``HeteroConv``. It intentionally does not use HGTConv or attention.

Run from the repository root:

    python scripts/train_gnn_rtx.py
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the non-attention heterogeneous GNN on RTX."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/rtx_gnn_no_attention",
        help="Directory for the checkpoint and evaluation reports.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device, for example cuda:0.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs from the YAML configuration.",
    )
    return parser.parse_args()


def require_cuda(device_name):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build. "
            f"Current torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda!r}."
        )

    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device, such as cuda:0")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {device_name}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )
    return device


def main():
    args = parse_args()
    device = require_cuda(args.device)

    with open(ROOT / "configs/hgnn_attention.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_root = ROOT / cfg["data"]["root"]
    split_dir = data_root / "splits"
    graph_dir = data_root / "graphs"
    with open(split_dir / "split_metadata.json", encoding="utf-8") as file:
        split_meta = json.load(file)
    with open(graph_dir / "graph_metadata.json", encoding="utf-8") as file:
        graph_meta = json.load(file)

    if split_meta.get("dataset") != "ml-1m":
        raise ValueError("This GNN script requires MovieLens 1M artifacts")
    if split_meta["split_method"] != "per_user_temporal":
        raise ValueError("This GNN script requires the per_user_temporal split")
    if graph_meta["split_method"] != split_meta["split_method"]:
        raise ValueError("Graph and split protocols do not match")

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    evaluation_users = pd.read_csv(split_dir / "evaluation_users.csv")
    eligible = evaluation_users.loc[
        evaluation_users["eligible_val"] == True, "user_idx"
    ].astype(int).tolist()

    graph = torch.load(
        graph_dir / "train_graph.pt", map_location="cpu", weights_only=False
    )
    expected_edges = int(train["is_positive"].sum())
    if graph_meta["train_positive_edges"] != expected_edges:
        raise ValueError("Train graph does not match the Train split")

    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = RGCNRecommender(
        metadata=graph.metadata(),
        num_users=graph["user"].num_nodes,
        num_movies=graph["movie"].num_nodes,
        num_genres=graph["genre"].num_nodes,
        hidden_dim=cfg["rgcn"]["hidden_dim"],
        num_layers=cfg["rgcn"]["num_layers"],
        dropout=cfg["rgcn"]["dropout"],
    )

    epochs = args.epochs or cfg["training"]["epochs"]
    print(f"Model: Hetero-GCN / RGCN without attention")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"CUDA: {torch.version.cuda}; split: {split_meta['split_method']}")
    print(f"Eligible validation users: {len(eligible)}")

    result = train_hgnn(
        model=model,
        graph=graph,
        train=train,
        val=val,
        eligible_val_users=eligible,
        num_movies=graph["movie"].num_nodes,
        epochs=epochs,
        batch_size=cfg["training"]["batch_size"],
        num_negatives=cfg["training"].get("num_negatives", 1),
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        seed=seed,
        device=str(device),
    )

    model = result["model"]
    summary, details = evaluate_hgnn(
        model,
        graph.to(device),
        train,
        val,
        eligible,
        graph["movie"].num_nodes,
        k=cfg["project"]["top_k"],
    )

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "hetero_gnn_without_attention",
            "model_config": cfg["rgcn"],
            "num_users": graph["user"].num_nodes,
            "num_movies": graph["movie"].num_nodes,
            "num_genres": graph["genre"].num_nodes,
            "best_epoch": result["best_epoch"],
            "seed": seed,
            "split_method": split_meta["split_method"],
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        checkpoint_dir / "gnn_no_attention_rtx.pt",
    )
    details.to_csv(output_dir / "gnn_val_users.csv", index=False)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(result["history"], file, indent=2)

    report = {
        "model": "Hetero-GCN-RGCN-No-Attention",
        "dataset": split_meta["dataset"],
        "split_method": split_meta["split_method"],
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "best_epoch": result["best_epoch"],
        "best_validation_ndcg": result["best_ndcg"],
        "metrics": summary,
        "test_evaluated": False,
    }
    with open(output_dir / "gnn_no_attention_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nGNN WITHOUT ATTENTION RESULTS")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
