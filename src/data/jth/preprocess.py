"""Prepare JTH for temporal candidate-to-job Top-K recommendation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.common.id_mapping import apply_mapping, build_mapping, save_mapping
from src.paths import DATA_ROOT


RAW_DIR = DATA_ROOT / "jth"
OUT_DIR = RAW_DIR / "processed"

STAGE_DATE_COLUMNS = [
    "spontaneous_application_date",
    "shortlist_date",
    "qualification_date",
    "resume_sent_to_company_date",
    "1st_interview_date",
    "2nd_interview_date",
    "3rd_interview_date",
    "4th_interview_date",
    "job_offer_proposed_date",
    "job_offer_accepted_date",
]
ALL_DATE_COLUMNS = STAGE_DATE_COLUMNS + ["end_of_process_date"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def normalize_skill(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def skill_tokens(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype(str).str.split(";").explode().map(normalize_skill)
    )


def _stage_order_violations(history: pd.DataFrame) -> int:
    parsed = history[ALL_DATE_COLUMNS].apply(pd.to_datetime, errors="coerce")
    violations = 0
    for _, row in parsed.iterrows():
        values = row.dropna().tolist()
        violations += int(any(left > right for left, right in zip(values, values[1:])))
    return violations


def _make_negatives(
    positive: pd.DataFrame,
    history: pd.DataFrame,
    jobs: pd.DataFrame,
    ratio: int,
    seed: int,
) -> pd.DataFrame:
    job_create = jobs[["job_id", "job_create_time"]].dropna().sort_values("job_create_time")
    job_times = job_create["job_create_time"].tolist()
    job_values = job_create["job_id"].astype(str).tolist()
    grouped_history = {
        candidate: group[["interaction_time", "job_id"]].sort_values("interaction_time")
        for candidate, group in history.groupby("candidate_id")
    }

    def observed_until(row: pd.Series) -> set[str]:
        prior = grouped_history.get(row["candidate_id"])
        if prior is None:
            return set()
        return set(prior.loc[
            prior["interaction_time"] <= row["interaction_time"], "job_id"
        ].astype(str))

    def eligible(row: pd.Series):
        query_time = row["interaction_time"]
        return job_values[:bisect_right(job_times, query_time)]

    return sample_negatives(
        positive,
        users=positive["candidate_id"].unique(),
        items=jobs["job_id"].unique(),
        user_column="candidate_id",
        item_column="job_id",
        observed=observed_until,
        ratio=ratio,
        seed=seed,
        eligible_items=eligible,
    )


def _write_features(candidates: pd.DataFrame, jobs: pd.DataFrame, skill_map: pd.DataFrame) -> None:
    skill_lookup = dict(zip(skill_map["raw_skill"], skill_map["id"]))

    candidate_features = candidates[[
        "candidate_id", "create_date", "skills", "expertise_area", "job_category",
        "years_experience", "contract_type", "zipcode",
    ]].copy()
    job_features = jobs[[
        "job_id", "create_date", "skills", "expertise_area", "job_category",
        "years_experience", "contract_type", "zipcode", "salary", "daily_rate",
    ]].copy()

    for frame in (candidate_features, job_features):
        frame["skills_missing"] = frame["skills"].isna()
        frame["skills_normalized"] = frame["skills"].fillna("").map(
            lambda value: ";".join(normalize_skill(token) for token in str(value).split(";") if normalize_skill(token))
        )
        frame["skill_ids"] = frame["skills_normalized"].map(
            lambda value: ";".join(str(skill_lookup[token]) for token in value.split(";") if token in skill_lookup)
        )

    candidate_features.to_csv(OUT_DIR / "features/candidates.csv", index=False)
    job_features.to_csv(OUT_DIR / "features/jobs.csv", index=False)


def _legacy_prepare() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(RAW_DIR / "candidates.csv", dtype={"candidate_id": "string"})
    jobs = pd.read_csv(RAW_DIR / "jobs.csv", dtype={"job_id": "string"})
    history = pd.read_csv(
        RAW_DIR / "history.csv",
        dtype={"candidate_id": "string", "job_id": "string", "application_id": "string"},
    )

    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    jobs["job_id"] = jobs["job_id"].astype(str)
    history["candidate_id"] = history["candidate_id"].astype(str)
    history["job_id"] = history["job_id"].astype(str)
    history["application_id"] = history["application_id"].astype(str)
    jobs["job_create_time"] = pd.to_datetime(jobs["create_date"], utc=True, errors="coerce")
    for column in ALL_DATE_COLUMNS:
        history[column] = pd.to_datetime(history[column], utc=True, errors="coerce")
    history["interaction_time"] = history[STAGE_DATE_COLUMNS].min(axis=1)
    history = history.sort_values(
        ["interaction_time", "candidate_id", "job_id", "application_id"]
    ).reset_index(drop=True)
    history["is_first_pair_interaction"] = ~history.duplicated(["candidate_id", "job_id"])
    history["interaction_id"] = history.index.astype(str)

    candidate_map = build_mapping(candidates["candidate_id"], "raw_candidate_id")
    job_map = build_mapping(jobs["job_id"], "raw_job_id")
    all_skills = pd.concat([skill_tokens(candidates["skills"]), skill_tokens(jobs["skills"])])
    all_skills = all_skills[all_skills.ne("")]
    skill_map = build_mapping(all_skills, "raw_skill")
    save_mapping(candidate_map, OUT_DIR / "mappings/candidates.csv")
    save_mapping(job_map, OUT_DIR / "mappings/jobs.csv")
    save_mapping(skill_map, OUT_DIR / "mappings/skills.csv")

    candidates["candidate_id"] = apply_mapping(candidates["candidate_id"], candidate_map, "raw_candidate_id")
    jobs["job_id"] = apply_mapping(jobs["job_id"], job_map, "raw_job_id")
    history["candidate_id"] = apply_mapping(history["candidate_id"], candidate_map, "raw_candidate_id")
    history["job_id"] = apply_mapping(history["job_id"], job_map, "raw_job_id")
    output_columns = [
        "interaction_id", "application_id", "candidate_id", "job_id",
        *ALL_DATE_COLUMNS, "interaction_time", "last_stage_reached",
        "application_source", "is_first_pair_interaction",
    ]
    output = history[output_columns].copy()
    for column in [*ALL_DATE_COLUMNS, "interaction_time"]:
        output[column] = output[column].astype(str).replace("NaT", "")
    output.to_csv(OUT_DIR / "interactions.csv", index=False)

    positive = history[history["is_first_pair_interaction"]].copy()
    train, val, test, split_meta = temporal_split(positive, "interaction_time")
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        out = frame[["interaction_id", "application_id", "candidate_id", "job_id", "interaction_time", "last_stage_reached"]].copy()
        out["interaction_time"] = out["interaction_time"].astype(str)
        out.to_csv(OUT_DIR / f"{name}.csv", index=False)

    train_neg = _make_negatives(train, history, jobs, NEGATIVE_RATIO_TRAIN, SEED)
    val_neg = _make_negatives(val, history, jobs, NEGATIVE_RATIO_EVAL, SEED + 1)
    test_neg = _make_negatives(test, history, jobs, NEGATIVE_RATIO_EVAL, SEED + 2)
    for name, frame in [("train", train_neg), ("val", val_neg), ("test", test_neg)]:
        frame.to_csv(OUT_DIR / f"negative_{name}.csv", index=False)

    _write_features(candidates, jobs, skill_map)
    _write_json(OUT_DIR / "metadata/candidate_pools.json", {
        "task": "candidate_to_job_top_k",
        "positive_rule": "first interaction event per candidate-job pair",
        "available_job_rule": "job create_date <= query cutoff; no assumption that old jobs remain open",
        "observed_rule": "all history events at or before query timestamp block the pair",
        "negative_rule": "unobserved eligible job; not confirmed negative feedback",
        "negative_ratio": {"train": NEGATIVE_RATIO_TRAIN, "val": NEGATIVE_RATIO_EVAL, "test": NEGATIVE_RATIO_EVAL},
    })
    _write_json(OUT_DIR / "metadata/split_metadata.json", split_meta)
    _write_json(OUT_DIR / "metadata/feature_policy.json", {
        "safe_original_features": ["skills", "expertise_area", "job_category", "years_experience", "contract_type", "zipcode", "salary", "daily_rate"],
        "llm_features": "excluded from temporal-safe feature exports until observation timestamps are verified",
        "rare_skill": "kept as a bucket in the mapping; do not interpret as a semantic skill",
    })

    validation = {
        "temporal": validate_temporal_order(history, "interaction_time"),
        "split": validate_split_overlap(train, val, test, "candidate_id", "job_id"),
        "history_rows": int(len(history)),
        "unique_candidate_job_pairs": int(len(positive)),
        "repeated_candidate_job_rows": int((~history["is_first_pair_interaction"]).sum()),
        "stage_order_violations": _stage_order_violations(history),
        "missing_interaction_time": int(history["interaction_time"].isna().sum()),
        "future_llm_features_used": False,
    }
    _write_json(OUT_DIR / "metadata/validation_report.json", validation)
    _write_json(OUT_DIR / "metadata/data_quality.json", {
        "rows": {"candidates": len(candidates), "jobs": len(jobs), "history": len(history)},
        "history_missing": {name: int(value) for name, value in history.isna().sum().items()},
        "stage_order_columns": STAGE_DATE_COLUMNS,
    })
    return {"dataset": "jth", "output": str(OUT_DIR), **validation}


if False and __name__ == "__main__":
    print(json.dumps(_legacy_prepare(), indent=2, default=str))


def _iso(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def _record_anomaly(rows: list[dict[str, object]], history_row: pd.Series, anomaly_type: str, **details: object) -> None:
    rows.append({
        "application_id": history_row.get("application_id", ""),
        "candidate_id": history_row.get("candidate_id", ""),
        "job_id": history_row.get("job_id", ""),
        "anomaly_type": anomaly_type,
        **details,
    })


def _skill_rows(frame: pd.DataFrame, entity_column: str) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_variants: defaultdict[str, set[str]] = defaultdict(set)
    for entity_id, value in frame[[entity_column, "skills"]].itertuples(index=False):
        if pd.isna(value) or not str(value).strip():
            continue
        counts: defaultdict[str, int] = defaultdict(int)
        raw_by_normalized: defaultdict[str, str] = defaultdict(str)
        for token in re.split(r";", str(value)):
            raw_token = token.strip()
            normalized = raw_token.lower()
            if not normalized:
                continue
            counts[normalized] += 1
            raw_by_normalized.setdefault(normalized, raw_token)
            raw_variants[normalized].add(raw_token)
        rare_count = counts.get("_rare_skill_", 0)
        for skill, count in sorted(counts.items()):
            rows.append({
                entity_column: str(entity_id),
                "skill": skill,
                "raw_skill": raw_by_normalized[skill],
                "skill_occurrence": count,
                "rare_skill_count": rare_count,
                "is_rare_skill": skill == "_rare_skill_",
                "is_semantic_skill": skill != "_rare_skill_",
            })
    result = pd.DataFrame(rows, columns=[
        entity_column, "skill", "raw_skill", "skill_occurrence", "rare_skill_count",
        "is_rare_skill", "is_semantic_skill",
    ])
    variants = {
        skill: sorted(values)
        for skill, values in raw_variants.items()
        if len(values) > 1
    }
    summary = {
        "rows": int(len(result)),
        "unique_normalized_skills": int(result["skill"].nunique()) if not result.empty else 0,
        "rare_skill_rows": int(result["is_rare_skill"].sum()) if not result.empty else 0,
        "formatting_variants": variants,
    }
    return result, summary


def _missing_summary(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(value) for column, value in frame.isna().sum().items() if value}


def prepare() -> dict[str, object]:
    """Create application-level and first-pair JTH preprocessing outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(RAW_DIR / "candidates.csv", dtype={"candidate_id": "string"})
    jobs = pd.read_csv(RAW_DIR / "jobs.csv", dtype={"job_id": "string"})
    history = pd.read_csv(RAW_DIR / "history.csv", dtype={
        "application_id": "string", "candidate_id": "string", "job_id": "string",
    })
    required = {
        "candidates": {"candidate_id", "create_date", "skills"},
        "jobs": {"job_id", "create_date", "skills"},
        "history": {"application_id", "candidate_id", "job_id", *STAGE_DATE_COLUMNS, "end_of_process_date", "last_stage_reached"},
    }
    for name, frame in (("candidates", candidates), ("jobs", jobs), ("history", history)):
        missing = sorted(required[name] - set(frame.columns))
        if missing:
            raise ValueError(f"{name}.csv is missing required columns: {missing}")

    for frame, columns in ((candidates, ["candidate_id"]), (jobs, ["job_id"]), (history, ["application_id", "candidate_id", "job_id"])):
        for column in columns:
            frame[column] = frame[column].astype("string")

    quality: dict[str, Any] = {
        "rows_before": {"candidates": int(len(candidates)), "jobs": int(len(jobs)), "history": int(len(history))},
        "errors": {},
        "excluded_from_interactions": {},
        "raw_data_preserved": True,
        "interaction_time_definition": "earliest valid stage date; end_of_process_date is excluded",
        "interaction_time_is_proxy": True,
    }
    quality["errors"].update({
        "candidate_null_ids": int(candidates["candidate_id"].isna().sum()),
        "job_null_ids": int(jobs["job_id"].isna().sum()),
        "history_null_application_ids": int(history["application_id"].isna().sum()),
        "history_null_candidate_ids": int(history["candidate_id"].isna().sum()),
        "history_null_job_ids": int(history["job_id"].isna().sum()),
        "duplicate_candidate_rows": int(candidates.duplicated().sum()),
        "duplicate_job_rows": int(jobs.duplicated().sum()),
        "duplicate_history_rows": int(history.duplicated().sum()),
        "duplicate_candidate_ids": int(candidates["candidate_id"].duplicated().sum()),
        "duplicate_job_ids": int(jobs["job_id"].duplicated().sum()),
        "duplicate_application_ids": int(history["application_id"].duplicated().sum()),
        "duplicate_candidate_job_pairs": int(history.duplicated(["candidate_id", "job_id"]).sum()),
        "orphan_candidates": int((~history["candidate_id"].isin(candidates["candidate_id"])).sum()),
        "orphan_jobs": int((~history["job_id"].isin(jobs["job_id"])).sum()),
    })

    candidate_create = pd.to_datetime(candidates["create_date"], utc=True, errors="coerce")
    job_create = pd.to_datetime(jobs["create_date"], utc=True, errors="coerce")
    candidate_create_map = dict(zip(candidates["candidate_id"], candidate_create))
    job_create_map = dict(zip(jobs["job_id"], job_create))
    for column in ["create_date"]:
        candidates[column] = candidate_create.map(_iso)
        jobs[column] = job_create.map(_iso)

    parsed_dates = history[ALL_DATE_COLUMNS].apply(pd.to_datetime, utc=True, errors="coerce")
    anomaly_rows: list[dict[str, object]] = []
    invalid_date_counts: dict[str, int] = {}
    for column in ALL_DATE_COLUMNS:
        invalid = history[column].notna() & parsed_dates[column].isna()
        invalid_date_counts[column] = int(invalid.sum())
        for index in history.index[invalid]:
            _record_anomaly(anomaly_rows, history.loc[index], "invalid_timestamp", field=column, value=str(history.loc[index, column]))
    history[ALL_DATE_COLUMNS] = parsed_dates
    history["interaction_time"] = parsed_dates[STAGE_DATE_COLUMNS].min(axis=1)
    history["interaction_time_source"] = parsed_dates[STAGE_DATE_COLUMNS].idxmin(axis=1).fillna("")

    for index, row in history.iterrows():
        non_null_stages = [(column, row[column]) for column in STAGE_DATE_COLUMNS if pd.notna(row[column])]
        for (left_name, left), (right_name, right) in zip(non_null_stages, non_null_stages[1:]):
            if left > right:
                _record_anomaly(anomaly_rows, row, "stage_order_violation", field=left_name, next_field=right_name, value=_iso(left), next_value=_iso(right))
        interaction_time = row["interaction_time"]
        if pd.isna(interaction_time):
            _record_anomaly(anomaly_rows, row, "missing_interaction_time", field="interaction_time")
        else:
            candidate_time, job_time = candidate_create_map.get(row["candidate_id"]), job_create_map.get(row["job_id"])
            if pd.notna(candidate_time) and interaction_time < candidate_time:
                _record_anomaly(anomaly_rows, row, "interaction_before_candidate_create_date", field="interaction_time", value=_iso(interaction_time), reference_value=_iso(candidate_time))
            if pd.notna(job_time) and interaction_time < job_time:
                _record_anomaly(anomaly_rows, row, "interaction_before_job_create_date", field="interaction_time", value=_iso(interaction_time), reference_value=_iso(job_time))

    history = history.sort_values(["candidate_id", "job_id", "interaction_time", "application_id"], na_position="last").reset_index(drop=True)
    valid_pair = history["candidate_id"].notna() & history["job_id"].notna() & history["interaction_time"].notna()
    history["is_first_pair_interaction"] = False
    history.loc[valid_pair, "is_first_pair_interaction"] = ~history.loc[valid_pair].duplicated(["candidate_id", "job_id"])
    history["excluded_from_interactions"] = ~valid_pair | ~history["candidate_id"].isin(candidates["candidate_id"]) | ~history["job_id"].isin(jobs["job_id"])
    history["exclusion_reason"] = ""
    history.loc[history["interaction_time"].isna(), "exclusion_reason"] = "missing_or_invalid_interaction_time"
    history.loc[~history["candidate_id"].isin(candidates["candidate_id"]), "exclusion_reason"] = "orphan_candidate_id"
    history.loc[~history["job_id"].isin(jobs["job_id"]), "exclusion_reason"] = "orphan_job_id"
    quality["excluded_from_interactions"] = {str(key): int(value) for key, value in history.loc[history["excluded_from_interactions"], "exclusion_reason"].value_counts().items()}

    event_output = history.copy()
    for column in ALL_DATE_COLUMNS + ["interaction_time"]:
        event_output[column] = event_output[column].map(_iso)
    event_output.to_csv(OUT_DIR / "application_events.csv", index=False)
    interaction_output = history[history["is_first_pair_interaction"] & ~history["excluded_from_interactions"]][[
        "application_id", "candidate_id", "job_id", "interaction_time", "interaction_time_source", "application_source",
    ]].copy()
    interaction_output["interaction_time"] = interaction_output["interaction_time"].map(_iso)
    interaction_output.to_csv(OUT_DIR / "interactions.csv", index=False)
    candidates.to_csv(OUT_DIR / "candidates_clean.csv", index=False)
    jobs.to_csv(OUT_DIR / "jobs_clean.csv", index=False)

    candidate_skills, candidate_skill_summary = _skill_rows(candidates, "candidate_id")
    job_skills, job_skill_summary = _skill_rows(jobs, "job_id")
    candidate_skills.to_csv(OUT_DIR / "candidate_skills.csv", index=False)
    job_skills.to_csv(OUT_DIR / "job_skills.csv", index=False)
    pd.DataFrame(anomaly_rows, columns=["application_id", "candidate_id", "job_id", "anomaly_type", "field", "next_field", "value", "next_value", "reference_value"]).to_csv(OUT_DIR / "temporal_anomalies.csv", index=False)

    llm_columns = sorted(
        column for column in set(candidates.columns).union(jobs.columns)
        if column.startswith("llm_")
    )
    feature_policy = {
        "safe_features_pending_as_of_validation": ["create_date", "skills", "expertise_area", "job_category", "years_experience", "contract_type", "zipcode", "salary", "daily_rate"],
        "features_requiring_source_or_snapshot_verification": llm_columns,
        "features_not_used_for_top_k_input": ["last_stage_reached", "end_of_process_date", *STAGE_DATE_COLUMNS],
        "future_outcomes_not_used": True,
        "missing_value_policy": "preserve missing values and add masks later; do not invent recruitment dates",
        "skill_policy": {"delimiter": ";", "normalization": "lowercase and trim", "synonym_merging": False, "rare_skill_node": False},
    }
    (OUT_DIR / "feature_availability.json").write_text(json.dumps(feature_policy, indent=2, ensure_ascii=False), encoding="utf-8")

    quality["rows_after"] = {"application_events": int(len(event_output)), "interactions": int(len(interaction_output)), "candidates_clean": int(len(candidates)), "jobs_clean": int(len(jobs)), "candidate_skills": int(len(candidate_skills)), "job_skills": int(len(job_skills))}
    quality["counts"] = {"candidates": int(len(candidates)), "jobs": int(len(jobs)), "application_processes": int(len(history)), "unique_candidate_job_pairs": int(len(interaction_output)), "repeated_candidate_job_rows": int(len(history) - len(interaction_output))}
    quality["invalid_timestamp_counts"] = invalid_date_counts
    quality["temporal_anomalies"] = {str(key): int(value) for key, value in pd.Series([row["anomaly_type"] for row in anomaly_rows]).value_counts().items()} if anomaly_rows else {}
    quality["temporal_anomaly_examples"] = anomaly_rows[:20]
    quality["missing_values"] = {"candidates": _missing_summary(candidates), "jobs": _missing_summary(jobs), "history": _missing_summary(history)}
    quality["skill_summary"] = {"candidates": candidate_skill_summary, "jobs": job_skill_summary, "rare_skill_is_not_semantic_node": True}
    (OUT_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {"dataset": "jth", "output": str(OUT_DIR), **quality["counts"], "errors": quality["errors"], "temporal_anomalies": quality["temporal_anomalies"]}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, default=str))
