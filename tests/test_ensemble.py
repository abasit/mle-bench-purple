import sys
from io import BytesIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solver.ensemble import blend_submissions  # noqa: E402


def test_blend_submissions_uses_mean_for_regression(tmp_path: Path):
    p1 = tmp_path / "s1.csv"
    p2 = tmp_path / "s2.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [10.0, 20.0, 30.0]}).to_csv(p1, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": [12.0, 18.0, 33.0]}).to_csv(p2, index=False)

    blended = blend_submissions(
        [p1, p2],
        objective_hint="regression",
    )
    assert blended is not None

    df = pd.read_csv(BytesIO(blended))
    assert df["target"].tolist() == [11.0, 19.0, 31.5]


def test_blend_submissions_majority_votes_string_labels(tmp_path: Path):
    p1 = tmp_path / "s1.csv"
    p2 = tmp_path / "s2.csv"
    p3 = tmp_path / "s3.csv"
    pd.DataFrame({"id": [1, 2, 3], "label": ["cat", "dog", "cat"]}).to_csv(p1, index=False)
    pd.DataFrame({"id": [1, 2, 3], "label": ["dog", "dog", "cat"]}).to_csv(p2, index=False)
    pd.DataFrame({"id": [1, 2, 3], "label": ["dog", "fox", "cat"]}).to_csv(p3, index=False)

    blended = blend_submissions(
        [p1, p2, p3],
        objective_hint="multiclass_classification",
    )
    assert blended is not None

    df = pd.read_csv(BytesIO(blended))
    assert df["label"].tolist() == ["dog", "dog", "cat"]


def test_blend_submissions_uses_arithmetic_for_binary_logloss(tmp_path: Path):
    p1 = tmp_path / "s1.csv"
    p2 = tmp_path / "s2.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(p1, index=False)
    pd.DataFrame({"id": [1, 2], "target": [0.6, 0.4]}).to_csv(p2, index=False)

    blended = blend_submissions(
        [p1, p2],
        objective_hint="binary_classification",
        metric_hint="logloss",
        prediction_mode="single_probability",
    )
    assert blended is not None

    df = pd.read_csv(BytesIO(blended))
    assert [round(value, 6) for value in df["target"].tolist()] == [0.4, 0.6]


def test_blend_submissions_normalizes_multiclass_probabilities(tmp_path: Path):
    p1 = tmp_path / "s1.csv"
    p2 = tmp_path / "s2.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "cat": [1.2, 0.1],
            "dog": [0.0, 0.6],
            "fox": [-0.2, 0.6],
        }
    ).to_csv(p1, index=False)
    pd.DataFrame(
        {
            "id": [1, 2],
            "cat": [0.8, 0.4],
            "dog": [0.4, 0.4],
            "fox": [0.0, 0.2],
        }
    ).to_csv(p2, index=False)

    blended = blend_submissions(
        [p1, p2],
        objective_hint="multiclass_classification",
        metric_hint="logloss",
        prediction_mode="multiclass_probabilities",
    )
    assert blended is not None

    df = pd.read_csv(BytesIO(blended))
    row_sums = (df[["cat", "dog", "fox"]].sum(axis=1)).round(6).tolist()
    assert row_sums == [1.0, 1.0]
    assert df[["cat", "dog", "fox"]].min().min() >= 0.0
    assert df[["cat", "dog", "fox"]].max().max() <= 1.0
