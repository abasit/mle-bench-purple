import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solver.task_classify import profile_task  # noqa: E402


def test_profile_task_detects_nlp_from_long_text_columns(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "review_text": [
                "This product was excellent and I would buy it again.",
                "Terrible experience with multiple defects and poor support.",
                "Decent value overall but packaging and delivery were disappointing.",
            ],
            "label": [1, 0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "id": [4, 5],
            "review_text": [
                "The story was moving and beautifully written.",
                "I regret the purchase and would not recommend it.",
            ],
        }
    )
    sample = pd.DataFrame({"id": [4, 5], "label": [0, 0]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample.to_csv(data_dir / "sample_submission.csv", index=False)

    profile = profile_task("Predict sentiment from review text.", data_dir)

    assert profile.category == "nlp"
    assert profile.objective in {"binary_classification", "classification"}


def test_profile_task_detects_timeseries_from_datetime_and_description(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sensor_id": [1, 1, 1],
            "target": [0.1, 0.2, 0.3],
        }
    )
    test = pd.DataFrame(
        {
            "timestamp": ["2024-01-04", "2024-01-05"],
            "sensor_id": [1, 1],
        }
    )
    sample = pd.DataFrame({"timestamp": ["2024-01-04", "2024-01-05"], "target": [0.0, 0.0]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample.to_csv(data_dir / "sample_submission.csv", index=False)

    profile = profile_task("Forecast the next horizon of this time series.", data_dir)

    assert profile.category == "timeseries"
    assert profile.objective == "forecasting"


def test_profile_task_detects_audio_from_file_evidence(tmp_path: Path):
    data_dir = tmp_path / "data"
    (data_dir / "train_audio").mkdir(parents=True)
    (data_dir / "test_audio").mkdir(parents=True)
    for i in range(12):
        (data_dir / "train_audio" / f"{i}.wav").write_bytes(b"RIFF")
    for i in range(4):
        (data_dir / "test_audio" / f"{i}.wav").write_bytes(b"RIFF")

    profile = profile_task("Classify the audio recordings.", data_dir)

    assert profile.category == "audio"


def test_profile_task_detects_multiclass_probability_route(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "species": ["cat", "dog", "fox", "dog"],
            "feature": [0.1, 0.4, 0.2, 0.9],
        }
    )
    test = pd.DataFrame({"id": [5, 6], "feature": [0.3, 0.7]})
    sample = pd.DataFrame(
        {
            "id": [5, 6],
            "cat": [0.0, 0.0],
            "dog": [0.0, 0.0],
            "fox": [0.0, 0.0],
        }
    )
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample.to_csv(data_dir / "sample_submission.csv", index=False)

    profile = profile_task("Predict the animal species with multiclass logloss.", data_dir)

    assert profile.category == "tabular"
    assert profile.objective == "multiclass_classification"
    assert profile.prediction_mode == "multiclass_probabilities"
    assert profile.validation_mode == "stratified"
    assert profile.route_id == "tabular_multiclass_prob"


def test_profile_task_prefers_grouped_validation_when_group_col_exists(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame({"patient_id": [3, 3], "feature": [0.5, 0.6]})
    sample = pd.DataFrame({"patient_id": [3, 3], "target": [0, 0]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample.to_csv(data_dir / "sample_submission.csv", index=False)

    profile = profile_task("Predict the binary target for each patient.", data_dir)

    assert profile.validation_mode == "stratified_group"
    assert "patient_id" in profile.likely_group_cols
