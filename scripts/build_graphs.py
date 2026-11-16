
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_builder import run_graph_pipeline


def main():
    data_root = PROJECT_ROOT / "data"

    metadata = run_graph_pipeline(data_root)

    print("Phase 3 graph construction: PASSED")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
