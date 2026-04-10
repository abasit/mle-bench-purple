import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solver.nodes import SearchNode  # noqa: E402
from solver.prompts import _classify_error, build_debug_prompt, build_draft_prompt, build_improve_prompt, build_repair_prompt  # noqa: E402
from solver.task_classify import TaskProfile  # noqa: E402


def test_build_draft_prompt_includes_stepwise_multiclass_route():
    profile = TaskProfile(
        category="tabular",
        objective="multiclass_classification",
        metric_name="logloss",
        maximize=False,
        prediction_mode="multiclass_probabilities",
        validation_mode="stratified",
        route_id="tabular_multiclass_prob",
        sample_target_cols=("cat", "dog", "fox"),
        primary_target="species",
        n_classes=3,
    )

    messages = build_draft_prompt(
        task_desc="Predict the correct animal class.",
        task_type="tabular",
        data_files=["train.csv", "test.csv", "sample_submission.csv"],
        sample_sub_preview="# sample_submission.csv\nid,cat,dog,fox",
        kb_card="",
        data_preview="train.csv has columns id,species,feature",
        task_profile_summary="Task profile:\n- route_id: tabular_multiclass_prob",
        task_profile=profile,
        env_summary="- Python: 3.11",
        time_remaining=3600,
        variant=0,
        total_variants=3,
    )

    prompt = messages[1]["content"]
    assert "TASK ROUTE" in prompt
    assert "TABULAR PLAYBOOK" in prompt
    assert "multiclass_probabilities" in prompt
    assert "stratified CV" in prompt
    assert "loss_function='MultiClass'" in prompt
    assert "Do not ensemble" in messages[0]["content"]
    assert "numeric-only model" in messages[0]["content"]


def test_build_debug_prompt_includes_data_preview_and_session_reuse():
    parent = SearchNode(id="d001", stage="draft", code="print('old')")

    messages = build_debug_prompt(
        parent=parent,
        error_summary="NameError: model is not defined",
        cleaned_log="Traceback...",
        time_remaining=1200,
        data_preview="DATA EXPLORATION: train.csv columns are id,target,x1",
        task_profile_summary="Task profile: binary classification",
        task_profile=TaskProfile(category="tabular", objective="binary_classification", maximize=True),
        env_summary="- Python: 3.11",
        prior_attempts=[],
    )

    prompt = messages[1]["content"]
    assert "SESSION RUNTIME" in prompt
    assert "TABULAR PLAYBOOK" in prompt
    assert "Data / exploration context" in prompt
    assert "train.csv columns are id,target,x1" in prompt


def test_classify_error_upgrades_boolean_unknown_fill_to_categorical_mismatch():
    error_class, advice = _classify_error(
        "TypeError: Invalid value 'Unknown' for dtype 'boolean'",
        "",
    )

    assert error_class == "categorical_model_mismatch"
    assert "OneHotEncoder/LabelEncoder/pd.factorize" in advice


def test_classify_error_upgrades_mixed_bool_string_encoder_failure():
    error_class, advice = _classify_error(
        "TypeError: Encoders require their input argument must be uniformly strings or numbers. Got ['bool', 'str']",
        "",
    )

    assert error_class == "categorical_model_mismatch"
    assert "single dtype" in advice


def test_build_improve_prompt_includes_data_preview_and_session_reuse():
    parent = SearchNode(id="d001", stage="draft", code="print('old')")
    parent.scores.maximize = True
    parent.scores.val_score = 0.75
    parent.scores.holdout_score = 0.73

    messages = build_improve_prompt(
        parent=parent,
        journal_summary="i002(improve val=0.7600) [model:catboost]",
        time_remaining=1200,
        fraction_used=0.5,
        data_preview="DATA EXPLORATION: train.csv columns are id,target,x1",
        task_profile_summary="Task profile: binary classification",
        task_profile=TaskProfile(category="tabular", objective="binary_classification", maximize=True),
        env_summary="- Python: 3.11",
        branch_history_rows=[],
        used_strategies={"model:catboost"},
        untried_strategies=["fe:missing_indicators"],
        required_strategy="fe:missing_indicators",
        task_type="tabular",
    )

    prompt = messages[1]["content"]
    assert "SESSION RUNTIME" in prompt
    assert "TABULAR PLAYBOOK" in prompt
    assert "Data / exploration context" in prompt
    assert "train.csv columns are id,target,x1" in prompt


def test_build_repair_prompt_biases_toward_working_baseline():
    messages = build_repair_prompt(
        code="print('old')",
        error_summary="TypeError: bad dtype",
        cleaned_log="Traceback...",
        time_remaining=900,
        repair_attempt=1,
        total_repairs=1,
        data_preview="DATA EXPLORATION: train.csv columns are id,target,x1",
        task_profile_summary="Task profile: binary classification",
        task_profile=TaskProfile(category="tabular", objective="binary_classification", maximize=True),
        env_summary="- Python: 3.11",
    )

    prompt = messages[1]["content"]
    assert "WORKING BASELINE FIRST" in prompt
    assert "GROUNDING RULES" in prompt
    assert "simpler baseline" in prompt
