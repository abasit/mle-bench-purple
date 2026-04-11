"""Tests for the fake-success detector."""

from pathlib import Path

import pandas as pd

from purple_next.exec import detect_fake_success


def test_detect_fake_success_flags_constant_submission(tmp_path: Path):
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"id": range(5), "target": [0.5] * 5}).to_csv(sub, index=False)
    reason = detect_fake_success(sub, None)
    assert reason is not None
    assert "constant prediction" in reason


def test_detect_fake_success_flags_byte_identical_sample(tmp_path: Path):
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": range(5), "target": [0.0] * 5}).to_csv(sample, index=False)
    sub = tmp_path / "submission.csv"
    sub.write_bytes(sample.read_bytes())
    reason = detect_fake_success(sub, sample)
    assert reason is not None
    # Constant check catches it first (lookalike), but both paths are acceptable.
    assert "constant prediction" in reason or "byte-identical" in reason


def test_detect_fake_success_passes_on_honest_submission(tmp_path: Path):
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": range(5), "target": [0.0] * 5}).to_csv(sample, index=False)
    sub = tmp_path / "submission.csv"
    pd.DataFrame({"id": range(5), "target": [0.1, 0.4, 0.6, 0.8, 0.2]}).to_csv(sub, index=False)
    assert detect_fake_success(sub, sample) is None
