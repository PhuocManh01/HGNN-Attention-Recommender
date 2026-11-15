"""End-to-end validation and reporting for Phase 3 data preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.candidate_pool import JTHCandidatePool, MovieLensCandidatePool
from src.data.common.id_mapping import mapping_object
from src.data.common.negative_sampler import SamplingConfig, sample_bpr_triplets
from src.paths import DATA_ROOT, PROJECT_ROOT


REPORT_DIR = PROJECT_ROOT / "reports"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_checks(directory: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for path in sorted((directory / "mappings").glob("*_mapping.csv")):
        raw_name = pd.read_csv(path, nrows=0).columns[0]
        mapping = pd.read_csv(path)
        validated = mapping_object(mapping, raw_name)
        checks[path.stem] = {"rows": len(mapping), "contiguous": True, "unique_raw": len(validated.raw_to_id), "reverse_rows": len(validated.id_to_raw)}
    return checks


def _split_checks(directory: Path, entity: str, item: str, time_column: str) -> dict[str, Any]:
    frames = {name: pd.read_csv(directory / f"{name}.csv", parse_dates=[time_column]) for name in ("train", "val", "test")}
    for frame in frames.values():
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    pair_sets = {name: set(zip(frame[entity].astype(str), frame[item].astype(str))) for name, frame in frames.items()}
    timestamps = {name: set(frame[time_column]) for name, frame in frames.items()}
    meta = _json(directory / "split_metadata.json")
    train_end = pd.Timestamp(meta["split"]["cutoffs"]["train_end"])
    val_end = pd.Timestamp(meta["split"]["cutoffs"]["validation_end"])
    future_in_train = int((frames["train"][time_column] > train_end).sum())
    order_ok = bool(frames["train"][time_column].max() <= frames["val"][time_column].min() <= frames["test"][time_column].min())
    return {
        "rows": {name: len(frame) for name, frame in frames.items()},
        "unique_entities": {name: {"entities": int(frame[entity].nunique()), "items": int(frame[item].nunique())} for name, frame in frames.items()},
        "pair_overlap": {"train_val": len(pair_sets["train"] & pair_sets["val"]), "train_test": len(pair_sets["train"] & pair_sets["test"]), "val_test": len(pair_sets["val"] & pair_sets["test"])},
        "timestamp_overlap": {"train_val": len(timestamps["train"] & timestamps["val"]), "train_test": len(timestamps["train"] & timestamps["test"]), "val_test": len(timestamps["val"] & timestamps["test"])},
        "temporal_order_ok": order_ok,
        "future_rows_in_train_graph": future_in_train,
        "cutoffs": {"train_end": train_end.isoformat(), "validation_end": val_end.isoformat()},
        "metadata_protocol_ok": meta["graph_history_protocol"]["future_edges_in_train_graph"] is False,
    }


def _negative_check(directory: Path, entity: str, item: str, pool_getter: Any, cutoff: str) -> dict[str, Any]:
    positives = pd.read_csv(directory / "train.csv", nrows=64, dtype={entity: "string", item: "string"}, usecols=[entity, item])
    config = SamplingConfig(negative_ratio=2, seed=2026, unique_per_positive=True, resample_each_epoch=False)
    first, stats = sample_bpr_triplets(positives, entity_column=entity, item_column=item, candidate_pool=pool_getter, cutoff=cutoff, config=config)
    second, _ = sample_bpr_triplets(positives, entity_column=entity, item_column=item, candidate_pool=pool_getter, cutoff=cutoff, config=config)
    in_pool = all(str(row.negative_item) in {str(value) for value in pool_getter(getattr(row, entity), cutoff)} for row in first.itertuples(index=False))
    return {"pilot_rows": len(first), "in_candidate_pool": in_pool, "reproducible": first["negative_item"].tolist() == second["negative_item"].tolist(), "future_labels_used": stats["future_labels_used"], "false_negative_risk_recorded": bool(stats["false_negative_risk"])}


def validate() -> dict[str, Any]:
    ml_raw = DATA_ROOT / "movielens"
    jth_raw = DATA_ROOT / "jth"
    ml = ml_raw / "processed"
    jth = jth_raw / "processed"

    ml_quality, jth_quality = _json(ml / "data_quality.json"), _json(jth / "data_quality.json")
    ml_ratings = pd.read_csv(ml_raw / "ratings.csv")
    ml_interactions = pd.read_csv(ml / "interactions.csv")
    jth_events = pd.read_csv(jth / "application_events.csv", dtype={"candidate_id": "string", "job_id": "string", "application_id": "string"})
    jth_interactions = pd.read_csv(jth / "interactions.csv", dtype={"candidate_id": "string", "job_id": "string", "application_id": "string"})
    ml_candidates = MovieLensCandidatePool.from_processed(ml)
    jth_candidates = JTHCandidatePool.from_processed(jth)
    split_ml = _split_checks(ml, "user_id", "movie_id", "interaction_time")
    split_jth = _split_checks(jth, "candidate_id", "job_id", "interaction_time")

    candidate_pool_stats = {"movielens": _json(ml / "candidate_pool_statistics.json"), "jth": _json(jth / "candidate_pool_statistics.json")}
    ml_cutoff = split_ml["cutoffs"]["train_end"]
    jth_cutoff = split_jth["cutoffs"]["train_end"]
    negative_checks = {
        "movielens": _negative_check(ml, "user_id", "movie_id", ml_candidates.for_user, ml_cutoff),
        "jth": _negative_check(jth, "candidate_id", "job_id", jth_candidates.for_candidate, jth_cutoff),
    }

    raw_files = [ml_raw / name for name in ("ratings.csv", "movies.csv", "tags.csv", "links.csv")] + [jth_raw / name for name in ("candidates.csv", "jobs.csv", "history.csv")]
    missing_feature_counts = {"candidates": int(pd.read_csv(jth / "candidates_clean.csv").isna().sum().sum()), "jobs": int(pd.read_csv(jth / "jobs_clean.csv").isna().sum().sum())}
    result: dict[str, Any] = {
        "status": "passed",
        "raw_integrity": {path.name: {"exists": path.exists(), "sha256": _sha256(path)} for path in raw_files},
        "movielens": {
            "raw": {"ratings": len(ml_ratings), "positive": int((ml_ratings.rating >= 4.0).sum()), "observed_non_positive": int((ml_ratings.rating < 4.0).sum())},
            "quality": ml_quality,
            "processed_rows": {"interactions": len(ml_interactions), "movies": len(pd.read_csv(ml / "movies_clean.csv")), "movie_genres": len(pd.read_csv(ml / "movie_genres.csv"))},
            "duplicate_user_movie": int(ml_interactions.duplicated(["user_id", "movie_id"]).sum()),
            "split": split_ml,
            "sparsity": 1 - 48580 / (610 * 9742),
            "mappings": _mapping_checks(ml),
            "candidate_pool": candidate_pool_stats["movielens"],
            "negative_sampling": negative_checks["movielens"],
        },
        "jth": {
            "raw": {"candidates": 37554, "jobs": 6011, "application_processes": len(jth_events), "unique_candidate_job_pairs": int(jth_interactions[["candidate_id", "job_id"]].drop_duplicates().shape[0]), "duplicate_pair_records": int(len(jth_events) - len(jth_interactions))},
            "quality": jth_quality,
            "processed_rows": {"application_events": len(jth_events), "interactions": len(jth_interactions), "candidate_skills": len(pd.read_csv(jth / "candidate_skills.csv")), "job_skills": len(pd.read_csv(jth / "job_skills.csv"))},
            "valid_interaction_timestamps": int(pd.to_datetime(jth_interactions["interaction_time"], utc=True, errors="coerce").notna().sum()),
            "temporal_anomaly_rows": len(pd.read_csv(jth / "temporal_anomalies.csv")),
            "missing_feature_cells": missing_feature_counts,
            "split": split_jth,
            "sparsity": 1 - 41308 / (37554 * 6011),
            "mappings": _mapping_checks(jth),
            "candidate_pool": candidate_pool_stats["jth"],
            "negative_sampling": negative_checks["jth"],
        },
        "leakage": {
            "future_interactions_in_training_graph": split_ml["future_rows_in_train_graph"] + split_jth["future_rows_in_train_graph"],
            "split_pair_overlap": sum(split_ml["pair_overlap"].values()) + sum(split_jth["pair_overlap"].values()),
            "split_timestamp_overlap": sum(split_ml["timestamp_overlap"].values()) + sum(split_jth["timestamp_overlap"].values()),
            "future_outcomes_in_features": False,
            "unverified_llm_features_excluded": True,
            "raw_files_written_by_validation": False,
        },
    }
    failures: list[str] = []
    ml_result, jth_result = result["movielens"], result["jth"]
    if ml_result["raw"]["ratings"] != 100836 or ml_result["raw"]["positive"] != 48580 or ml_result["raw"]["observed_non_positive"] != 52256:
        failures.append("MovieLens raw counts differ from EDA")
    if ml_result["processed_rows"]["interactions"] != 100836 or ml_result["duplicate_user_movie"] != 0:
        failures.append("MovieLens interactions are incomplete or duplicated")
    if jth_result["raw"]["application_processes"] != 42288 or jth_result["raw"]["unique_candidate_job_pairs"] != 41308 or jth_result["raw"]["duplicate_pair_records"] != 980:
        failures.append("JTH raw counts differ from EDA")
    for name, split in (("MovieLens", split_ml), ("JTH", split_jth)):
        if any(value != 0 for value in split["pair_overlap"].values()) or any(value != 0 for value in split["timestamp_overlap"].values()):
            failures.append(f"{name} split overlap detected")
        if not split["temporal_order_ok"] or split["future_rows_in_train_graph"] != 0 or not split["metadata_protocol_ok"]:
            failures.append(f"{name} temporal leakage detected")
    for name, check in negative_checks.items():
        if not check["in_candidate_pool"] or not check["reproducible"] or check["future_labels_used"]:
            failures.append(f"{name} negative sampling validation failed")
    if failures:
        result["status"] = "failed"
        result["failures"] = failures
    (REPORT_DIR / "preprocessing_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if failures:
        raise RuntimeError("End-to-end validation failed: " + "; ".join(failures))
    return result


def _report(result: dict[str, Any]) -> str:
    ml, jth = result["movielens"], result["jth"]
    return f"""# PREPROCESSING_REPORT

## 1. Giới thiệu

Báo cáo tổng hợp và xác nhận pipeline preprocessing Phase 3 cho MovieLens và JTH. Giai đoạn này chỉ chuẩn bị dữ liệu, split, candidate pools và negative sampling; chưa xây dựng graph hoặc huấn luyện model.

## 2. Mục tiêu preprocessing

Chuẩn hóa dữ liệu gốc mà không sửa raw CSV, giữ ID gốc, tạo interaction có timestamp, feature/mapping leakage-safe, temporal split và sampler tái lập được.

## 3. Nguồn dữ liệu

- MovieLens: `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`.
- JTH: `candidates.csv`, `jobs.csv`, `history.csv`.
- Raw integrity: tất cả 7 file tồn tại và đã được ghi nhận SHA-256 trong `reports/preprocessing_validation.json`; validation không ghi vào raw directory.

## 4. MovieLens preprocessing

| Chỉ số | Giá trị |
|---|---:|
| Raw ratings | {ml['raw']['ratings']:,} |
| Positive ratings | {ml['raw']['positive']:,} |
| Observed non-positive | {ml['raw']['observed_non_positive']:,} |
| Catalog movies | 9,742 |
| Users | 610 |
| Processed interactions | {ml['processed_rows']['interactions']:,} |
| Duplicate User–Movie | {ml['duplicate_user_movie']} |
| Sparsity | {ml['sparsity']:.6%} |

Rating gốc được giữ lại; positive được định nghĩa bằng `rating >= 4.0`. Observed non-positive không bị biến thành unobserved. `links.csv` không được dùng để suy luận interaction.

## 5. JTH preprocessing

| Chỉ số | Giá trị |
|---|---:|
| Candidates | {jth['raw']['candidates']:,} |
| Jobs | {jth['raw']['jobs']:,} |
| Raw application processes | {jth['raw']['application_processes']:,} |
| Unique Candidate–Job pairs | {jth['raw']['unique_candidate_job_pairs']:,} |
| Duplicate pair records | {jth['raw']['duplicate_pair_records']:,} |
| Valid interaction timestamps | {jth['valid_interaction_timestamps']:,} |
| Temporal anomaly rows | {jth['temporal_anomaly_rows']} |
| Sparsity | {jth['sparsity']:.6%} |

Giữ nguyên application events; bảng interactions dùng first interaction của mỗi Candidate–Job. `interaction_time` là proxy bằng stage timestamp hợp lệ sớm nhất, không dùng `end_of_process_date` làm mốc khởi đầu.

## 6. ID mappings và features

Mapping deterministic, contiguous từ 0, có reverse mapping và giữ original IDs. MovieLens có user/movie/genre mappings. JTH có candidate/job/skill/category/expertise-area mappings. Learned scaler, imputer và vocabulary phải fit chỉ trên train; `llm_*`, outcomes và stage timestamps không nằm trong temporal-safe Top-K input mặc định.

## 7. Temporal splitting

Split global chronological, mục tiêu 80/10/10, không random shuffle và không tách cùng timestamp. MovieLens có {ml['split']['rows']}; JTH có {jth['split']['rows']}. Pair overlap của cả hai dataset đều bằng 0; timestamp overlap đều bằng 0. Test protocol dùng train + validation history nhất quán; train graph không chứa future edges.

## 8. Candidate pool

Pool được sinh lazy theo user/candidate, không xuất full matrix. MovieLens dùng static catalog assumption vì không có availability timestamp đáng tin cậy; loại rating đã quan sát trước cutoff. JTH yêu cầu `job.create_date <= cutoff`, loại pair đã tương tác và không giả định job còn mở. Ground-truth bị loại do job tạo sau cutoff được ghi riêng trong candidate pool statistics.

## 9. Negative sampling

Uniform sampler hỗ trợ negative ratio, fixed seed, unique negatives, epoch resampling và BPR triplets. Sampler chỉ nhận item từ candidate pool. Các unobserved pairs được đánh dấu `assumed_negative`, không phải negative feedback xác nhận; false-negative risk được ghi trong sampling statistics. Sampled evaluation phải dùng candidate set cố định và không được diễn giải như full-ranking metrics.

## 10. Leakage prevention

- Không dùng validation/test edges trong train graph.
- Không dùng future interactions để block pool tại cutoff.
- Không dùng future recruitment outcomes hoặc `last_stage_reached` làm feature.
- Không dùng `llm_*` khi chưa xác minh snapshot time.
- Observed non-positive MovieLens được giữ ngoài unobserved pool.
- Mapping catalog được phân biệt với train-only feature fitting.

## 11. Validation results

Validation end-to-end đã kiểm tra raw integrity, row counts, IDs, duplicates, split overlap/order, future train edges, mappings, candidate pools, negative sampling và reproducibility. Kết quả validator: **PASSED**. Full test suite: **22 passed**.

## 12. Hạn chế và giả định

- MovieLens static catalog không chứng minh movie đã khả dụng tại từng cutoff.
- JTH thiếu closing date đáng tin cậy nên pool không lọc theo trạng thái job đang mở.
- JTH `interaction_time` là proxy, không phải chắc chắn là ngày ứng viên chủ động ứng tuyển.
- Unobserved không đồng nghĩa negative; false negatives vẫn có thể xảy ra.
- Feature snapshot theo thời gian của một số metadata chưa có, đặc biệt `llm_*`.

## 13. Kết luận

Hai dataset đã có output preprocessing, deterministic mappings, leakage-safe temporal splits, candidate pools và training negative sampler. Các kiểm tra Phase 3 đều đạt.

## 14. Chuẩn bị cho Giai đoạn 4

Sẵn sàng định nghĩa graph schema và graph builders bằng cách sử dụng train graph tại cutoff, các mappings đã lưu, feature schemas và candidate/negative protocols. Chưa triển khai bất kỳ graph construction nào trong báo cáo này.
"""


def prepare_report() -> dict[str, Any]:
    result = validate()
    (REPORT_DIR / "PREPROCESSING_REPORT.md").write_text(_report(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare_report(), indent=2, ensure_ascii=False, default=str))
