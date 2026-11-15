import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="MovieLens Phase 2 preprocessing"
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    metadata = run_pipeline(
        raw_dir=args.raw_dir,
        output_root=args.output_root
    )

    print("\nPhase 2 completed successfully!")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
