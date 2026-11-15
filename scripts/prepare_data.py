"""Prepare MovieLens and JTH for temporal Top-K recommendation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.jth.preprocess import prepare as prepare_jth
from src.data.movielens.preprocess import prepare as prepare_movielens


def main() -> None:
    results = [prepare_movielens(), prepare_jth()]
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
