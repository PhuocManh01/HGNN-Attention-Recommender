import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import DATA_ROOT, RESULTS_ROOT


def inspect_table(name: str, df: pd.DataFrame) -> dict:
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:")
    print(df.isna().sum().to_string())
    print("Duplicate rows:", int(df.duplicated().sum()))
    print(df.head(3).to_string(index=False))

    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "missing": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


def save_histogram(series, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=40, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    raw_dir = DATA_ROOT / "movielens"
    output_dir = RESULTS_ROOT / "movielens" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "movies.csv",
        "ratings.csv",
        "tags.csv",
        "links.csv",
    ]

    for filename in required:
        if not (raw_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing {filename} in {raw_dir}"
            )

    movies = pd.read_csv(raw_dir / "movies.csv")
    ratings = pd.read_csv(raw_dir / "ratings.csv")
    tags = pd.read_csv(raw_dir / "tags.csv")
    links = pd.read_csv(raw_dir / "links.csv")

    tables = {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
    }

    report = {
        "tables": {
            name: inspect_table(name, df)
            for name, df in tables.items()
        }
    }

    users = ratings["userId"].nunique()
    items = ratings["movieId"].nunique()
    interactions = len(ratings)

    # Độ thưa trong ma trận user-item có tương tác
    possible = users * items
    sparsity = (
        1 - interactions / possible
        if possible else None
    )

    positive = ratings[ratings["rating"] >= 4.0]

    ratings_per_user = ratings.groupby("userId").size()
    ratings_per_movie = ratings.groupby("movieId").size()

    genre_lists = movies["genres"].fillna("").str.split("|")
    genres = genre_lists.explode()
    genres = genres[
        genres.notna()
        & (genres != "")
        & (genres != "(no genres listed)")
    ]

    missing_movie_ids = set(ratings["movieId"]) - set(
        movies["movieId"]
    )

    report["statistics"] = {
        "num_users": int(users),
        "num_rated_movies": int(items),
        "num_movies_in_catalog": int(movies["movieId"].nunique()),
        "num_ratings": int(interactions),
        "num_positive_ratings_threshold_4": int(len(positive)),
        "positive_ratio": float(len(positive) / interactions)
            if interactions else None,
        "interaction_sparsity": sparsity,
        "num_genres": int(genres.nunique()),
        "num_tags": int(len(tags)),
        "num_unique_tag_texts": int(tags["tag"].nunique()),
        "ratings_per_user": {
            "min": int(ratings_per_user.min()),
            "median": float(ratings_per_user.median()),
            "max": int(ratings_per_user.max()),
        },
        "ratings_per_movie": {
            "min": int(ratings_per_movie.min()),
            "median": float(ratings_per_movie.median()),
            "max": int(ratings_per_movie.max()),
        },
        "ratings_with_missing_movie_metadata": int(
            ratings["movieId"].isin(missing_movie_ids).sum()
        ),
        "duplicate_user_movie_pairs": int(
            ratings.duplicated(["userId", "movieId"]).sum()
        ),
    }

    print("\n=== DATASET STATISTICS ===")
    print(json.dumps(
        report["statistics"],
        indent=2,
        ensure_ascii=False,
    ))

    # Rating distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ratings["rating"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("MovieLens Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "rating_distribution.png", dpi=150)
    plt.close(fig)

    save_histogram(
        ratings_per_user,
        "Ratings per User",
        "Number of ratings",
        output_dir / "ratings_per_user.png",
    )

    save_histogram(
        ratings_per_movie,
        "Ratings per Movie",
        "Number of ratings",
        output_dir / "ratings_per_movie.png",
    )

    # Genre frequency
    genre_counts = genres.value_counts()
    genre_counts.to_csv(
        output_dir / "genre_counts.csv",
        header=["count"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    genre_counts.sort_values().plot.barh(ax=ax)
    ax.set_title("Movie Genre Frequency")
    ax.set_xlabel("Movies")
    fig.tight_layout()
    fig.savefig(output_dir / "genre_distribution.png", dpi=150)
    plt.close(fig)

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nEDA completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
