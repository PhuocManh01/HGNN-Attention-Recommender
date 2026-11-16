
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baselines import (
    PopularityModel,
    MFBPR,
    LightGCN
)
from src.training.baseline_trainer import (
    set_seed,
    train_bpr_model
)
from src.evaluation.ranking import (
    evaluate_full_ranking
)


def main():
    with open(
        ROOT / "configs" / "baselines.yaml",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    seed = config["project"]["seed"]
    set_seed(seed)

    data_root = ROOT / config["data"]["root"]
    split_dir = data_root / "splits"
    processed_dir = data_root / "processed"
    
    with open(
        split_dir / "split_metadata.json",
        encoding="utf-8",
    ) as f:
        split_metadata = json.load(f)

    with open(
        data_root / "graphs" / "graph_metadata.json",
        encoding="utf-8",
    ) as f:
        graph_metadata = json.load(f)

    expected_split = "per_user_temporal"

    if split_metadata["split_method"] != expected_split:
        raise ValueError("Phase 2 split protocol mismatch")

    if graph_metadata["split_method"] != expected_split:
        raise ValueError("Phase 3 graph protocol mismatch")

    if (
        graph_metadata["split_seed"]
        != split_metadata["seed"]
    ):
        raise ValueError("Split seed mismatch")

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    
    expected_edges = int(train["is_positive"].sum())

    if graph_metadata["train_positive_edges"] != expected_edges:
        raise ValueError("Train graph edge count mismatch")

    users = pd.read_csv(
        split_dir / "evaluation_users.csv"
    )

    eligible_val_users = users.loc[
        users["eligible_val"] == True,
        "user_idx"
    ].astype(int).tolist()

    num_users = len(
        pd.read_csv(processed_dir / "user_mapping.csv")
    )
    num_movies = len(
        pd.read_csv(processed_dir / "movie_mapping.csv")
    )

    checkpoint_dir = ROOT / "checkpoints" / "baselines"
    result_dir = ROOT / "experiments" / "phase4"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_logs = {}

    # 1. Popularity
    popularity = PopularityModel(num_movies).fit(train)

    pop_summary, pop_details = evaluate_full_ranking(
        popularity, train, val,
        eligible_val_users, num_movies
    )

    results.append({
        "model": "Popularity",
        **pop_summary
    })

    torch.save(
        popularity.scores,
        checkpoint_dir / "popularity.pt"
    )

    pop_details.to_csv(
        result_dir / "popularity_val_users.csv",
        index=False
    )

    # 2. MF-BPR
    mf = MFBPR(
        num_users,
        num_movies,
        config["training"]["embedding_dim"]
    )

    mf_result = train_bpr_model(
        model=mf,
        train=train,
        val=val,
        eligible_val_users=eligible_val_users,
        num_movies=num_movies,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["patience"],
        seed=seed
    )

    mf = mf_result["model"]
    all_logs["MF-BPR"] = mf_result["history"]

    mf_summary, mf_details = evaluate_full_ranking(
        mf, train, val,
        eligible_val_users, num_movies
    )

    results.append({
        "model": "MF-BPR",
        **mf_summary
    })

    torch.save(
        {
            "state_dict": mf.state_dict(),
            "num_users": num_users,
            "num_movies": num_movies,
            "embedding_dim": config["training"]["embedding_dim"],
            "best_epoch": mf_result["best_epoch"],
            "seed": seed
        },
        checkpoint_dir / "mf_bpr.pt"
    )

    mf_details.to_csv(
        result_dir / "mf_bpr_val_users.csv",
        index=False
    )

    # 3. LightGCN
    graph = torch.load(
        data_root / "graphs" / "train_graph.pt",
        map_location="cpu",
        weights_only=False
    )

    edge_index = graph[
        ("user", "rates", "movie")
    ].edge_index
    
    actual_edges = set(
        map(tuple, edge_index.T.tolist())
    )

    expected_edges_set = set(
        map(
            tuple,
            train.loc[
                train["is_positive"] == 1,
                ["user_idx", "movie_idx"],
            ].to_numpy().tolist(),
        )
    )

    if actual_edges != expected_edges_set:
        raise ValueError("Train graph edges do not match Train split")

    lightgcn = LightGCN(
        num_users=num_users,
        num_movies=num_movies,
        edge_index=edge_index,
        embedding_dim=config["training"]["embedding_dim"],
        num_layers=config["lightgcn"]["num_layers"]
    )

    lg_result = train_bpr_model(
        model=lightgcn,
        train=train,
        val=val,
        eligible_val_users=eligible_val_users,
        num_movies=num_movies,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["patience"],
        seed=seed
    )

    lightgcn = lg_result["model"]
    all_logs["LightGCN"] = lg_result["history"]

    lg_summary, lg_details = evaluate_full_ranking(
        lightgcn, train, val,
        eligible_val_users, num_movies
    )

    results.append({
        "model": "LightGCN",
        **lg_summary
    })

    torch.save(
        {
            "state_dict": lightgcn.state_dict(),
            "num_users": num_users,
            "num_movies": num_movies,
            "embedding_dim": config["training"]["embedding_dim"],
            "num_layers": config["lightgcn"]["num_layers"],
            "best_epoch": lg_result["best_epoch"],
            "seed": seed
        },
        checkpoint_dir / "lightgcn.pt"
    )

    lg_details.to_csv(
        result_dir / "lightgcn_val_users.csv",
        index=False
    )

    result_table = pd.DataFrame(results)

    result_table.to_csv(
        result_dir / "validation_results.csv",
        index=False
    )

    with open(
        result_dir / "training_history.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(all_logs, f, indent=2)

    print("\nVALIDATION RESULTS")
    print(result_table.to_string(index=False))

    print("\nPhase 4 completed successfully.")


if __name__ == "__main__":
    main()
