"""Leakage-aware JTH feature preparation.

Catalog mappings are deterministic and may cover the complete catalog. Learned
vocabularies, imputers, and scalers must be fitted later through
``src.data.common.feature_utils`` using a train-scoped frame.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from src.data.common.id_mapping import build_mapping, mapping_object, reverse_mapping
from src.paths import DATA_ROOT


DATA_DIR = DATA_ROOT / "jth" / "processed"
MAPPING_DIR = DATA_DIR / "mappings"


def _normalize(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().lower()


def _write_mapping(values: pd.Series, raw_name: str, stem: str) -> pd.DataFrame:
    values = values.astype("string")
    mapping = build_mapping(values[values.notna() & values.str.strip().ne("")], raw_name)
    mapping_object(mapping, raw_name)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_DIR / f"{stem}_mapping.csv", index=False)
    reverse_mapping(mapping, raw_name).to_csv(MAPPING_DIR / f"{stem}_reverse.csv", index=False)
    return mapping


def _list_features(frame: pd.DataFrame, entity_column: str, category_map: pd.DataFrame, expertise_map: pd.DataFrame) -> pd.DataFrame:
    result = frame[[entity_column, "expertise_area", "job_category"]].copy()
    for column, mapping, raw_name in [("expertise_area", expertise_map, "expertise_area"), ("job_category", category_map, "category")]:
        lookup = dict(zip(mapping[raw_name], mapping["id"]))
        result[f"{column}_normalized"] = result[column].map(_normalize)
        result[f"{column}_ids"] = result[f"{column}_normalized"].map(
            lambda value: ";".join(str(lookup[token]) for token in re.split(r";", value) if token in lookup)
        )
        result[f"{column}_missing"] = result[column].isna()
    result["contract_type"] = frame["contract_type"]
    result["zipcode"] = frame["zipcode"]
    result["contract_type_missing"] = frame["contract_type"].isna()
    result["zipcode_missing"] = frame["zipcode"].isna()
    return result


def _numeric_features(frame: pd.DataFrame, entity_column: str, numeric_columns: list[str]) -> pd.DataFrame:
    result = frame[[entity_column]].copy()
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result[column] = values
        result[f"{column}_missing"] = values.isna()
        result[f"{column}_log1p"] = values.where(values.ge(0)).map(lambda value: None if pd.isna(value) else float(math.log1p(value)))
    years = frame["years_experience"].astype("string")
    result["years_experience_lower_bound"] = pd.to_numeric(years.str.extract(r"(\d+)")[0], errors="coerce")
    result["years_experience_missing"] = years.isna()
    return result


def prepare() -> dict[str, object]:
    candidates = pd.read_csv(DATA_DIR / "candidates_clean.csv", dtype={"candidate_id": "string"})
    jobs = pd.read_csv(DATA_DIR / "jobs_clean.csv", dtype={"job_id": "string"})
    candidate_skills = pd.read_csv(DATA_DIR / "candidate_skills.csv", dtype={"candidate_id": "string"})
    job_skills = pd.read_csv(DATA_DIR / "job_skills.csv", dtype={"job_id": "string"})

    candidate_map = _write_mapping(candidates["candidate_id"], "raw_candidate_id", "candidates")
    job_map = _write_mapping(jobs["job_id"], "raw_job_id", "jobs")
    skill_map = _write_mapping(pd.concat([candidate_skills["skill"], job_skills["skill"]]), "skill", "skills")
    skill_lookup = dict(zip(skill_map["skill"], skill_map["id"]))
    for skill_frame in [candidate_skills, job_skills]:
        skill_frame["skill_catalog_id"] = skill_frame["skill"].map(skill_lookup)
    candidate_skills.to_csv(DATA_DIR / "candidate_skill_features.csv", index=False)
    job_skills.to_csv(DATA_DIR / "job_skill_features.csv", index=False)
    category_values = pd.concat([candidates["job_category"], jobs["job_category"]]).dropna().map(_normalize).str.split(";").explode()
    expertise_values = pd.concat([candidates["expertise_area"], jobs["expertise_area"]]).dropna().map(_normalize).str.split(";").explode()
    category_map = _write_mapping(category_values, "category", "categories")
    expertise_map = _write_mapping(expertise_values, "expertise_area", "expertise_areas")

    candidate_features = _list_features(candidates, "candidate_id", category_map, expertise_map)
    candidate_features = candidate_features.merge(_numeric_features(candidates, "candidate_id", []), on="candidate_id")
    candidate_features = candidate_features.merge(candidate_skills.groupby("candidate_id").agg(skill_count=("skill", "nunique"), rare_skill_count=("is_rare_skill", "sum")).reset_index(), on="candidate_id", how="left")
    candidate_features["skills_missing"] = candidates["skills"].isna().to_numpy()
    candidate_features[["skill_count", "rare_skill_count"]] = candidate_features[["skill_count", "rare_skill_count"]].fillna(0).astype("int64")
    candidate_features.to_csv(DATA_DIR / "candidate_features.csv", index=False)

    job_features = _list_features(jobs, "job_id", category_map, expertise_map)
    job_features = job_features.merge(_numeric_features(jobs, "job_id", ["salary", "daily_rate"]), on="job_id")
    job_features = job_features.merge(job_skills.groupby("job_id").agg(skill_count=("skill", "nunique"), rare_skill_count=("is_rare_skill", "sum")).reset_index(), on="job_id", how="left")
    job_features["skills_missing"] = jobs["skills"].isna().to_numpy()
    job_features[["skill_count", "rare_skill_count"]] = job_features[["skill_count", "rare_skill_count"]].fillna(0).astype("int64")
    job_features.to_csv(DATA_DIR / "job_features.csv", index=False)

    schema = {
        "dataset": "jth",
        "id_mappings": {"candidate": "catalog", "job": "catalog", "skill": "catalog", "category": "catalog", "expertise_area": "catalog"},
        "selected_features": {"candidate": ["skills", "expertise_area", "job_category", "years_experience", "contract_type", "zipcode"], "job": ["skills", "expertise_area", "job_category", "salary", "daily_rate", "contract_type", "zipcode"]},
        "missing_indicators": ["*_missing", "rare_skill_count"],
        "deterministic_numeric_transforms": ["*_log1p", "years_experience_lower_bound"],
        "train_fit_only": ["zscore", "imputer", "learned_feature_vocabulary", "frequency_encoding"],
        "excluded_features": {"last_stage_reached": "future recruitment outcome", "end_of_process_date": "future process outcome", "actual_salary": "candidate outcome/current value timing is not verified", "actual_daily_salary": "candidate outcome/current value timing is not verified", "llm_*": "creation time and source snapshot unverified"},
        "skill_policy": {"rare_skill_is_semantic_node": False, "synonym_merging": False, "catalog_mapping_is_not_train_vocabulary": True},
        "temporal_policy": "Use selected features only from snapshots available at the query cutoff; this export does not infer snapshots.",
    }
    (DATA_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    availability = json.loads((DATA_DIR / "feature_availability.json").read_text(encoding="utf-8"))
    availability.update({"selected_features": schema["selected_features"], "excluded_features": schema["excluded_features"], "train_fit_only": schema["train_fit_only"]})
    (DATA_DIR / "feature_availability.json").write_text(json.dumps(availability, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"dataset": "jth", "candidates": len(candidate_features), "jobs": len(job_features), "skills": len(skill_map), "categories": len(category_map), "expertise_areas": len(expertise_map), "output": str(DATA_DIR)}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
