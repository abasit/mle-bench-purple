"""Tests for the runner-owned split protocol and task contract."""

from pathlib import Path

import pandas as pd

from purple_next.protocol import (
    PROTOCOL_JSON,
    SPLIT_CSV,
    infer_contract,
    prepare_splits,
)


def _write_basic_competition(data_dir: Path, n: int = 40) -> None:
    pd.DataFrame(
        {"id": range(n), "feature": range(n), "target": [0, 1] * (n // 2)}
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": range(10), "feature": range(10)}).to_csv(
        data_dir / "test.csv", index=False
    )
    pd.DataFrame({"id": range(10), "target": [0] * 10}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    (data_dir / "description.md").write_text(
        "Binary classification. Metric: AUC. Higher is better.", encoding="utf-8"
    )


def test_infer_contract_reads_description_and_sample(tmp_path: Path):
    _write_basic_competition(tmp_path)
    contract = infer_contract(tmp_path, n_folds=5, holdout_fraction=0.2, seed=42)
    assert contract.maximize is True
    assert contract.metric == "auc"
    assert contract.target_col == "target"
    assert contract.id_col == "id"


def test_prepare_splits_writes_dev_and_holdout(tmp_path: Path):
    _write_basic_competition(tmp_path, n=50)
    contract = infer_contract(tmp_path, n_folds=4, holdout_fraction=0.2, seed=42)
    artifact = prepare_splits(tmp_path, contract)
    assert artifact is not None
    assert artifact.n_holdout == 10
    assert artifact.n_dev == 40
    splits = pd.read_csv(tmp_path / SPLIT_CSV)
    assert len(splits) == 50
    assert set(splits["split"]) == {"dev", "holdout"}
    # Folds are assigned only for dev rows, holdout rows carry -1.
    assert (splits.loc[splits["split"] == "holdout", "fold"] == -1).all()
    dev_folds = splits.loc[splits["split"] == "dev", "fold"]
    assert set(dev_folds.unique()) == {0, 1, 2, 3}
    # Protocol JSON is written.
    assert (tmp_path / PROTOCOL_JSON).exists()


def test_prepare_splits_is_stratified_for_classification(tmp_path: Path):
    _write_basic_competition(tmp_path, n=60)
    contract = infer_contract(tmp_path, n_folds=3, holdout_fraction=0.2, seed=42)
    artifact = prepare_splits(tmp_path, contract)
    assert artifact is not None
    splits = pd.read_csv(tmp_path / SPLIT_CSV)
    train = pd.read_csv(tmp_path / "train.csv")
    merged = splits.merge(train, left_on="row_index", right_index=True)
    dev = merged[merged["split"] == "dev"]
    counts = dev.groupby("fold")["target"].value_counts().unstack(fill_value=0)
    # Each fold should have both classes represented (stratification).
    assert (counts > 0).all().all()
