import random
import numpy as np
import yaml

from src.paths import PROJECT_ROOT


def load_config(dataset_name: str):
    config_path = (
        PROJECT_ROOT / "configs" / f"{dataset_name}.yaml"
    )

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
