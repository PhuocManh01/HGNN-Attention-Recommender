import importlib
import sys
from src.config import load_config
from src.paths import (
    PROJECT_ROOT,
    DATA_ROOT,
    CHECKPOINT_ROOT,
    RESULTS_ROOT,
    create_output_dirs,
)


REQUIRED_MODULES = {
    "torch": "PyTorch",
    "torch_geometric": "PyTorch Geometric",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "sklearn": "Scikit-learn",
    "xgboost": "XGBoost",
    "networkx": "NetworkX",
    "pyvis": "PyVis",
}


def check_imports():
    versions = {}

    for module_name, display_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"Missing required dependency: {display_name} "
                f"({module_name})"
            ) from exc

        versions[display_name] = getattr(module, "__version__", "available")

    return versions


def main():
    print("=== HGNN-Attention-Recommender ===")
    print("Python:", sys.version.split()[0])
    print("Project:", PROJECT_ROOT)

    for name, version in check_imports().items():
        print(f"Dependency: {name} {version}")

    import torch

    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("GPU status: local CPU environment; run the Colab notebook for GPU verification")

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
