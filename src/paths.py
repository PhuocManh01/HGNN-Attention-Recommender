import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    os.environ.get(
        "HGNN_DATA_ROOT",
        str(PROJECT_ROOT / "datasets")
    )
)

CHECKPOINT_ROOT = Path(
    os.environ.get(
        "HGNN_CHECKPOINT_ROOT",
        str(PROJECT_ROOT / "checkpoints")
    )
)

RESULTS_ROOT = Path(
    os.environ.get(
        "HGNN_RESULTS_ROOT",
        str(PROJECT_ROOT / "results")
    )
)


def create_output_dirs():
    for root in [CHECKPOINT_ROOT, RESULTS_ROOT]:
        root.mkdir(parents=True, exist_ok=True)

        for dataset in ["movielens", "medajob"]:
            (root / dataset).mkdir(
                parents=True,
                exist_ok=True
            )
