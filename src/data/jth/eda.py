import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import DATA_ROOT, RESULTS_ROOT


DATA_DIR = DATA_ROOT / "jth"
OUT_DIR = RESULTS_ROOT / "jth" / "eda"


def inspect_table(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_percent": {
            col: round(float(value), 2)
            for col, value in (
                df.isna().mean() * 100
            ).items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


def parse_semicolon(series):
    return (
        series.dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=160)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(
        DATA_DIR / "candidates.csv",
        dtype={"candidate_id": "string"}
    )
    jobs = pd.read_csv(
        DATA_DIR / "jobs.csv",
        dtype={"job_id": "string"}
    )
    history = pd.read_csv(
        DATA_DIR / "history.csv",
        dtype={
            "candidate_id": "string",
            "job_id": "string"
        }
    )

    tables = {
        "candidates": candidates,
        "jobs": jobs,
        "history": history
    }

    report = {
        "tables": {
            name: inspect_table(df)
            for name, df in tables.items()
        }
    }

    n_candidates = candidates["candidate_id"].nunique()
    n_jobs = jobs["job_id"].nunique()

    report["statistics"] = {
        "candidates": int(n_candidates),
        "jobs": int(n_jobs),
        "application_processes": int(len(history)),
        "active_candidates": int(
            history["candidate_id"].nunique()
        ),
        "jobs_with_interactions": int(
            history["job_id"].nunique()
        ),
        "duplicate_candidate_job_pairs": int(
            history.duplicated(
                ["candidate_id", "job_id"]
            ).sum()
        ),
        "unknown_candidates": int(
            (~history["candidate_id"].isin(
                candidates["candidate_id"]
            )).sum()
        ),
        "unknown_jobs": int(
            (~history["job_id"].isin(
                jobs["job_id"]
            )).sum()
        ),
    }

    # Missing values
    for name, df in tables.items():
        missing = (
            df.isna().mean().mul(100)
            .sort_values(ascending=False)
        )
        missing.to_csv(
            OUT_DIR / f"{name}_missing.csv",
            header=["missing_percent"]
        )

        missing.head(20).sort_values().plot(
            kind="barh", figsize=(10, 7)
        )
        plt.title(f"{name} - Missing Values (%)")
        plt.xlabel("Missing (%)")
        save_plot(f"{name}_missing.png")

    # Applications per candidate
    history.groupby("candidate_id").size().plot(
        kind="hist", bins=40, figsize=(8, 5)
    )
    plt.title("JTH - Applications per Candidate")
    plt.xlabel("Number of Application Processes")
    save_plot("applications_per_candidate.png")

    # Applications per job
    history.groupby("job_id").size().plot(
        kind="hist", bins=40, figsize=(8, 5)
    )
    plt.title("JTH - Applications per Job")
    plt.xlabel("Number of Application Processes")
    save_plot("applications_per_job.png")

    # Last stage reached - descriptive analysis only
    if "last_stage_reached" in history.columns:
        counts = (
            history["last_stage_reached"]
            .fillna("Unknown")
            .value_counts()
        )
        report["last_stage_reached"] = {
            str(k): int(v)
            for k, v in counts.items()
        }

        counts.sort_values().plot(
            kind="barh", figsize=(10, 7)
        )
        plt.title("JTH - Last Stage Reached")
        save_plot("recruitment_stages.png")

    # Manual skills
    for name, df in [
        ("candidate", candidates),
        ("job", jobs)
    ]:
        if "skills" not in df.columns:
            continue

        skill_counts = (
            parse_semicolon(df["skills"])
            .value_counts()
        )

        skill_counts.to_csv(
            OUT_DIR / f"{name}_skill_counts.csv",
            header=["count"]
        )

        skill_counts.head(20).sort_values().plot(
            kind="barh", figsize=(10, 7)
        )
        plt.title(f"JTH - Top {name.title()} Skills")
        save_plot(f"{name}_top_skills.png")

        report[f"{name}_unique_skills"] = int(
            len(skill_counts)
        )

    # Temporal coverage
    for name, df in tables.items():
        date_columns = [
            c for c in df.columns
            if c == "create_date" or c.endswith("_date")
        ]

        report[f"{name}_date_ranges"] = {}

        for col in date_columns:
            parsed = pd.to_datetime(
                df[col], errors="coerce"
            )

            report[f"{name}_date_ranges"][col] = {
                "min": (
                    str(parsed.min().date())
                    if parsed.notna().any() else None
                ),
                "max": (
                    str(parsed.max().date())
                    if parsed.notna().any() else None
                ),
                "valid_dates": int(parsed.notna().sum())
            }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(
        report["statistics"], indent=2
    ))
    print(f"EDA saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()