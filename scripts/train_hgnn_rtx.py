"""Train the HGNN attention recommender on an NVIDIA RTX GPU.

This is a GPU-specific entry point. It reuses the existing model and trainer,
but refuses to silently fall back to CPU when the PyTorch installation does
not have CUDA support.

Run from the repository root with:

    python scripts/train_hgnn_rtx.py

The installed PyTorch must be a CUDA build (not ``+cpu``).
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HGNN-HGT attention on an NVIDIA RTX GPU."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/rtx_hgnn",
        help="Directory for the RTX checkpoint and reports.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device, for example cuda:0 or cuda:1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs from configs/hgnn_attention.yaml.",
    )
    return parser.parse_args()


def require_cuda(device_name: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build "
            "matching the NVIDIA driver, then rerun this script. "
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

    # Import PyG only after the CUDA check so a CPU-only or incomplete
    # environment reports the actionable RTX/CUDA prerequisite first.
    from src.models.hgnn_attention import HGNNWithAttention
    from src.training.hgnn_trainer import evaluate_hgnn, train_hgnn

    with open(ROOT / "configs/hgnn_attention.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_root = ROOT / cfg["data"]["root"]
    split_dir = data_root / "splits"
    graph_dir = data_root / "graphs"

    with open(split_dir / "split_metadata.json", encoding="utf-8") as file:
        split_meta = json.load(file)
    with open(graph_dir / "graph_metadata.json", encoding="utf-8") as file:
        graph_meta = json.load(file)

    expected_split = cfg["data"]["split_method"]
    if split_meta["split_method"] != expected_split:
        raise ValueError("Split protocol mismatch")
    if graph_meta["split_method"] != expected_split:
        raise ValueError("Graph protocol mismatch")
    if graph_meta["split_seed"] != split_meta["seed"]:
        raise ValueError("Graph seed mismatch")
    if float(split_meta["positive_threshold"]) != float(
        cfg["data"]["positive_threshold"]
    ):
        raise ValueError("Positive threshold mismatch")

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    evaluation_users = pd.read_csv(split_dir / "evaluation_users.csv")
    eligible = evaluation_users.loc[
        evaluation_users["eligible_val"] == True, "user_idx"
    ].astype(int).tolist()

    graph = torch.load(
        graph_dir / "train_graph.pt", map_location="cpu", weights_only=False
    )
    edge_type = ("user", "rates", "movie")
    actual_edges = set(map(tuple, graph[edge_type].edge_index.T.tolist()))
    expected_edges = set(
        map(
            tuple,
            train.loc[
                train["is_positive"] == 1, ["user_idx", "movie_idx"]
            ].to_numpy().tolist(),
        )
    )
    if actual_edges != expected_edges:
        raise ValueError("Train graph edge mismatch")

    num_users = graph["user"].num_nodes
    num_movies = graph["movie"].num_nodes
    num_genres = graph["genre"].num_nodes
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    epochs = args.epochs or cfg["training"]["epochs"]
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"CUDA: {torch.version.cuda}; split: {expected_split}")
    print(f"Eligible validation users: {len(eligible)}")

    result = train_hgnn(
        model=model,
        graph=graph,
        train=train,
        val=val,
        eligible_val_users=eligible,
        num_movies=num_movies,
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
        model, graph.to(device), train, val, eligible, num_movies,
        k=cfg["project"]["top_k"],
    )
    if summary["users_evaluated"] != len(eligible):
        raise ValueError("Evaluation user count mismatch")

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": cfg["model"],
            "num_users": num_users,
            "num_movies": num_movies,
            "num_genres": num_genres,
            "best_epoch": result["best_epoch"],
            "seed": seed,
            "split_method": expected_split,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        checkpoint_dir / "hgnn_attention_rtx.pt",
    )
    details.to_csv(output_dir / "hgnn_val_users.csv", index=False)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(result["history"], file, indent=2)

    report = {
        "model": "HGNN-HGT-Attention",
        "dataset": split_meta.get("dataset", "unknown"),
        "split_method": expected_split,
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
    with open(output_dir / "rtx_hgnn_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nRTX HGNN TRAINING RESULTS")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
