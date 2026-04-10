import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solver.config import SolverConfig  # noqa: E402
from solver.interpreter import ExecResult  # noqa: E402
from solver.nodes import Journal, SearchNode  # noqa: E402
from solver.parsing import ParsedScores  # noqa: E402
from solver.runner import _detect_fake_success, _execute_node  # noqa: E402
from solver.strategies import build_branch_history, pick_required_strategy  # noqa: E402
from solver.task_classify import TaskProfile  # noqa: E402


class DummyInterpreter:
    def __init__(self, data_dir: Path, result: ExecResult):
        self.data_dir = data_dir
        self._result = result

    def run(self, code: str, node_id: str) -> ExecResult:
        return self._result


def test_detect_fake_success_flags_sample_submission_clone(tmp_path: Path):
    node_dir = tmp_path / "nodes" / "d001"
    input_dir = node_dir / "input"
    input_dir.mkdir(parents=True)

    sample = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
    sample_path = input_dir / "sample_submission.csv"
    sub_path = node_dir / "submission.csv"
    sample.to_csv(sample_path, index=False)
    sample.to_csv(sub_path, index=False)

    node = SearchNode(id="d001", stage="draft", code="print('hi')")
    node.result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.5\nFINAL HOLDOUT SCORE: 0.5",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )

    reason = _detect_fake_success(node, ParsedScores(val_score=0.8, holdout_score=0.7, maximize=True))
    assert reason is not None
    assert "constant prediction" in reason or "byte-identical" in reason


def test_execute_node_uses_metric_hint_and_cfg_gap_threshold(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n001"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.8, 0.2]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.90\nFINAL HOLDOUT SCORE: 0.70\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(id="n001", stage="draft", code="print('ok')")

    cfg = SolverConfig()
    cfg.search.holdout_gap_rel_threshold = 0.05
    profile = TaskProfile(category="tabular", objective="regression", maximize=False)

    _execute_node(interpreter, journal, node, cfg=cfg, task_profile=profile)

    assert node.scores.maximize is False
    assert node.is_suspicious
    assert not node.is_buggy


def test_execute_node_flags_metric_direction_mismatch(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n002"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.8, 0.2]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.90\nMETRIC DIRECTION: MAXIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(id="n002", stage="draft", code="print('ok')")

    cfg = SolverConfig()
    profile = TaskProfile(category="tabular", objective="regression", maximize=False)

    _execute_node(interpreter, journal, node, cfg=cfg, task_profile=profile)

    assert node.scores.maximize is True
    assert node.is_suspicious
    assert not node.is_buggy


def test_build_branch_history_reports_positive_delta_for_minimize_metric():
    journal = Journal()

    draft = SearchNode(id="d001", stage="draft", code="print('a')")
    draft.branch_root_id = "d001"
    draft.scores = ParsedScores(val_score=0.50, maximize=False)

    improve = SearchNode(id="i002", stage="improve", code="print('b')", parent_id="d001")
    improve.branch_root_id = "d001"
    improve.scores = ParsedScores(val_score=0.40, maximize=False)

    journal.add(draft)
    journal.add(improve)

    rows = build_branch_history(journal, "d001")
    assert rows[0].delta is None
    assert rows[1].delta is not None
    assert rows[1].delta > 0


def test_pick_required_strategy_uses_timeseries_plan():
    choice = pick_required_strategy(
        fraction_used=0.1,
        used=set(),
        task_type="timeseries",
        objective="forecasting",
        metric_name="error_metric",
    )
    assert choice in {"cv:timeseries_split", "fe:lag_features", "fe:datetime_expansion", "fe:aggregation_groupby"}


def test_execute_node_flags_identical_scores_as_suspicious(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n003"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.812345\nFINAL HOLDOUT SCORE: 0.812345\nMETRIC DIRECTION: MAXIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(id="n003", stage="draft", code="print('ok')")

    _execute_node(interpreter, journal, node, cfg=SolverConfig(), task_profile=TaskProfile(maximize=True))

    assert node.is_suspicious
    assert any("numerically identical" in reason for reason in node.suspicion_reasons)


def test_execute_node_flags_perfect_score_as_suspicious(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n004"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 1.0\nFINAL HOLDOUT SCORE: 0.94\nMETRIC DIRECTION: MAXIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(id="n004", stage="draft", code="print('ok')")

    _execute_node(interpreter, journal, node, cfg=SolverConfig(), task_profile=TaskProfile(maximize=True))

    assert node.is_suspicious
    assert any("nearly perfect" in reason for reason in node.suspicion_reasons)


def test_execute_node_flags_multiclass_probabilities_that_do_not_sum_to_one(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "cat": [0.0, 0.0], "dog": [0.0, 0.0], "fox": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n005"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "cat": [0.7, 0.6],
            "dog": [0.7, 0.2],
            "fox": [0.1, 0.3],
        }
    ).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.62\nFINAL HOLDOUT SCORE: 0.64\nMETRIC DIRECTION: MINIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(id="n005", stage="draft", code="print('ok')")
    profile = TaskProfile(
        category="tabular",
        objective="multiclass_classification",
        metric_name="logloss",
        maximize=False,
        prediction_mode="multiclass_probabilities",
        sample_target_cols=("cat", "dog", "fox"),
    )

    _execute_node(interpreter, journal, node, cfg=SolverConfig(), task_profile=profile)

    assert node.is_suspicious
    assert any("rows do not sum to 1" in reason for reason in node.suspicion_reasons)


def test_execute_node_flags_timeseries_random_split_code(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"timestamp": ["2024-01-01"], "target": [0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n006"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"timestamp": ["2024-01-01"], "target": [0.1]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.20\nFINAL HOLDOUT SCORE: 0.21\nMETRIC DIRECTION: MINIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(
        id="n006",
        stage="draft",
        code="from sklearn.model_selection import KFold\ncv = KFold(n_splits=5, shuffle=True)",
    )
    profile = TaskProfile(
        category="timeseries",
        objective="forecasting",
        maximize=False,
        validation_mode="time_ordered",
        likely_time_cols=("timestamp",),
    )

    _execute_node(interpreter, journal, node, cfg=SolverConfig(), task_profile=profile)

    assert node.is_suspicious
    assert any("time-ordered task" in reason for reason in node.suspicion_reasons)


def test_execute_node_flags_group_leakage_split(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"patient_id": [1, 2], "target": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n007"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"patient_id": [1, 2], "target": [0, 1]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.77\nFINAL HOLDOUT SCORE: 0.75\nMETRIC DIRECTION: MAXIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(
        id="n007",
        stage="draft",
        code="from sklearn.model_selection import StratifiedKFold\ncv = StratifiedKFold(n_splits=5)",
    )
    profile = TaskProfile(
        category="tabular",
        objective="binary_classification",
        maximize=True,
        validation_mode="stratified_group",
        likely_group_cols=("patient_id",),
    )

    _execute_node(interpreter, journal, node, cfg=SolverConfig(), task_profile=profile)

    assert node.is_suspicious
    assert any("group-leakage task" in reason for reason in node.suspicion_reasons)


def test_execute_node_flags_overfit_trend_from_branch_history(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    prior = SearchNode(id="d001", stage="draft", code="print('old')")
    prior.branch_root_id = "d001"
    prior.result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.80\nFINAL HOLDOUT SCORE: 0.78\nMETRIC DIRECTION: MAXIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=tmp_path / "prior.csv",
    )
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.9]}).to_csv(prior.result.submission_path, index=False)
    prior.scores = ParsedScores(val_score=0.80, holdout_score=0.78, maximize=True)

    journal = Journal()
    journal.add(prior)

    node_dir = tmp_path / "nodes" / "n008"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout="FINAL VAL SCORE: 0.84\nFINAL HOLDOUT SCORE: 0.77\nMETRIC DIRECTION: MAXIMIZE\n",
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    node = SearchNode(id="n008", stage="improve", code="print('new')", parent_id="d001")
    node.branch_root_id = "d001"

    _execute_node(interpreter, journal, node, cfg=SolverConfig(), task_profile=TaskProfile(maximize=True))

    assert node.is_suspicious
    assert any("validation improved" in reason for reason in node.suspicion_reasons)


def test_execute_node_flags_high_fold_variance(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )

    node_dir = tmp_path / "nodes" / "n009"
    node_dir.mkdir(parents=True)
    sub_path = node_dir / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(sub_path, index=False)

    result = ExecResult(
        return_code=0,
        stdout=(
            "FINAL VAL SCORE: 0.81\n"
            "FINAL HOLDOUT SCORE: 0.79\n"
            "METRIC DIRECTION: MAXIMIZE\n"
            "CV FOLD SCORES: 0.40, 0.92, 0.85, 0.93, 0.48\n"
        ),
        stderr="",
        duration_seconds=1.0,
        submission_path=sub_path,
    )
    interpreter = DummyInterpreter(data_dir=data_dir, result=result)
    journal = Journal()
    node = SearchNode(id="n009", stage="draft", code="print('ok')")

    _execute_node(interpreter, journal, node, cfg=SolverConfig(), task_profile=TaskProfile(maximize=True))

    assert node.is_suspicious
    assert any("CV fold scores are unstable" in reason for reason in node.suspicion_reasons)
