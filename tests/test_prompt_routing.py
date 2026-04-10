import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solver.prompts import build_draft_prompt  # noqa: E402
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
    assert "STEPWISE ROUTE" in prompt
    assert "Each row must sum to 1" in prompt
    assert "VALIDATION ROUTE" in prompt
    assert "loss_function='MultiClass'" in prompt
