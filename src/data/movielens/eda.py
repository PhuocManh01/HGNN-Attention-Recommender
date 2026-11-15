import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import DATA_ROOT, RESULTS_ROOT


DATA_DIR = DATA_ROOT / "movielens"
OUT_DIR = RESULTS_ROOT / "movielens" / "eda"


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=160)
    plt.close()


def inspect_dataframe(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "missing_values": {
            k: int(v) for k, v in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        "ratings.csv", "movies.csv",
        "tags.csv", "links.csv"
    ]

    for filename in required:
        if not (DATA_DIR / filename).exists():
            raise FileNotFoundError(
                f"Missing {DATA_DIR / filename}"
            )

    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")

    tables = {
        "ratings": ratings,
        "movies": movies,
        "tags": tags,
        "links": links,
    }

    report = {
        "tables": {
            name: inspect_dataframe(df)
            for name, df in tables.items()
        }
    }

    n_users = ratings["userId"].nunique()
    n_rated_movies = ratings["movieId"].nunique()
    n_catalog_movies = movies["movieId"].nunique()

    interactions = len(ratings)
    positive = int((ratings["rating"] >= 4).sum())

    report["statistics"] = {
        "users": int(n_users),
        "rated_movies": int(n_rated_movies),
        "catalog_movies": int(n_catalog_movies),
        "ratings": int(interactions),
        "positive_ratings_ge_4": positive,
        "positive_ratio": positive / interactions,
        "sparsity_rated_items": (
            1 - interactions / (n_users * n_rated_movies)
        ),
        "sparsity_catalog": (
            1 - interactions / (n_users * n_catalog_movies)
        ),
        "duplicate_user_movie_pairs": int(
            ratings.duplicated(
                subset=["userId", "movieId"]
            ).sum()
        ),
        "unknown_movie_references": int(
            (~ratings["movieId"].isin(movies["movieId"])).sum()
        ),
    }

    ratings["rating"].value_counts().sort_index().plot(
        kind="bar", figsize=(8, 5)
    )
    plt.title("MovieLens - Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    save_plot("rating_distribution.png")

    ratings.groupby("userId").size().plot(
        kind="hist", bins=30, figsize=(8, 5)
    )
    plt.title("Ratings per User")
    plt.xlabel("Number of Ratings")
    save_plot("ratings_per_user.png")

    ratings.groupby("movieId").size().plot(
        kind="hist", bins=40, figsize=(8, 5)
    )
    plt.title("Ratings per Movie")
    plt.xlabel("Number of Ratings")
    save_plot("ratings_per_movie.png")

    genre_counts = (
        movies["genres"]
        .fillna("")
        .str.split("|")
        .explode()
        .value_counts()
        .drop(labels=["", "(no genres listed)"], errors="ignore")
    )

    genre_counts.to_csv(
        OUT_DIR / "genre_counts.csv",
        header=["count"]
    )

    genre_counts.head(20).sort_values().plot(
        kind="barh", figsize=(9, 7)
    )
    plt.title("MovieLens - Top Genres")
    plt.xlabel("Number of Movies")
    save_plot("genre_distribution.png")

    report_path = OUT_DIR / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(
        report["statistics"], indent=2
    ))
    print(f"EDA saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
