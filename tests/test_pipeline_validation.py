import json
from pathlib import Path

from src.data.validate_pipeline import prepare_report, validate


ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_validation_passes():
    result = validate()
    assert result["status"] == "passed"
    assert result["leakage"]["future_interactions_in_training_graph"] == 0
    assert result["leakage"]["split_pair_overlap"] == 0
    assert result["leakage"]["split_timestamp_overlap"] == 0
    assert result["leakage"]["future_outcomes_in_features"] is False


def test_validation_report_is_generated_with_required_sections():
    result = prepare_report()
    report = (ROOT / "reports" / "PREPROCESSING_REPORT.md").read_text(encoding="utf-8")
    validation = json.loads((ROOT / "reports" / "preprocessing_validation.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert validation["status"] == "passed"
    for section in ("MovieLens preprocessing", "JTH preprocessing", "Temporal splitting", "Candidate pool", "Negative sampling", "Leakage prevention", "Chuẩn bị cho Giai đoạn 4"):
        assert section in report
