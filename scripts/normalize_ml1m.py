"""Convert the MovieLens 1M DAT files to the CSV schema used by this project."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_dat(path: Path):
    # MovieLens 1M is distributed as ISO-8859-1 text with ``::`` separators.
    with path.open("r", encoding="latin-1", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.rstrip("\r\n").split("::")
            yield line_number, fields


def write_csv(path: Path, header: list[str], rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def normalize_movies(source: Path, target: Path) -> int:
    rows = []
    for line_number, fields in read_dat(source):
        if len(fields) != 3:
            raise ValueError(f"{source}:{line_number}: expected 3 fields, got {len(fields)}")
        movie_id, title, genres = fields
        # Match MovieLens latest-small's genre spelling and preserve the pipe format.
        genres = genres.replace("Children's", "Children")
        rows.append((int(movie_id), title, genres))
    write_csv(target, ["movieId", "title", "genres"], rows)
    return len(rows)


def normalize_ratings(source: Path, target: Path) -> int:
    rows = []
    for line_number, fields in read_dat(source):
        if len(fields) != 4:
            raise ValueError(f"{source}:{line_number}: expected 4 fields, got {len(fields)}")
        user_id, movie_id, rating, timestamp = fields
        rows.append((int(user_id), int(movie_id), float(rating), int(timestamp)))
    write_csv(target, ["userId", "movieId", "rating", "timestamp"], rows)
    return len(rows)


def normalize_users(source: Path, target: Path) -> int:
    rows = []
    for line_number, fields in read_dat(source):
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number}: expected 5 fields, got {len(fields)}")
        user_id, gender, age, occupation, zip_code = fields
        rows.append((int(user_id), gender, int(age), int(occupation), zip_code))
    write_csv(target, ["userId", "gender", "age", "occupation", "zipCode"], rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ml-1m"))
    args = parser.parse_args()

    movie_count = normalize_movies(args.raw_dir / "movies.dat", args.raw_dir / "movies.csv")
    rating_count = normalize_ratings(args.raw_dir / "ratings.dat", args.raw_dir / "ratings.csv")
    user_count = normalize_users(args.raw_dir / "users.dat", args.raw_dir / "users.csv")
    print(
        f"wrote {movie_count:,} movies, {rating_count:,} ratings, "
        f"and {user_count:,} users to {args.raw_dir}"
    )


if __name__ == "__main__":
    main()
