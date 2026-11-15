"""Check the Python runtime used by local notebooks or Google Colab."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    "torch",
    "torch_geometric",
)


def _is_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Colab detected: {_is_colab()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    missing = [
        package
        for package in REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    if missing:
        print("Missing packages: " + ", ".join(missing))
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
