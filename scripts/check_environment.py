import sys
from src.config import load_config
from src.paths import (
    PROJECT_ROOT,
    DATA_ROOT,
    CHECKPOINT_ROOT,
    RESULTS_ROOT,
    create_output_dirs,
)


def main():
    print("=== HGNN-Attention-Recommender ===")
    print("Python:", sys.version.split()[0])
    print("Project:", PROJECT_ROOT)

    for dataset in ["movielens", "jth"]:
        config = load_config(dataset)
        print("Config loaded:", config["dataset"]["name"])

    create_output_dirs()

    print("Data:", DATA_ROOT)
    print("Checkpoints:", CHECKPOINT_ROOT)
    print("Results:", RESULTS_ROOT)

    print("Local environment: PASSED")


if __name__ == "__main__":
    main()
