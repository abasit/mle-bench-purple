"""Sanity tests for the prompt builders."""

from purple_next.prompts import (
    DISPOSITIONS,
    SYSTEM_PROMPT,
    build_debug_prompt,
    build_draft_prompt,
    build_improve_prompt,
    classify_error,
    disposition_for_run,
)


def test_draft_prompt_contains_disposition_and_protocol():
    msgs = build_draft_prompt(
        task_desc="predict something",
        data_files=["train.csv", "test.csv"],
        data_preview="# train.csv\n...",
        contract_summary="- metric: auc (maximize)",
        env_summary="- Python 3.11",
        time_remaining_s=3600.0,
        disposition=disposition_for_run(1),
        variant=0,
    )
    assert len(msgs) == 2
    user = msgs[1]["content"]
    assert "DATA FIRST" in user
    assert "- metric: auc" in user
    assert "_splits.csv" in msgs[0]["content"]


def test_improve_prompt_includes_parent_stdout_and_hint():
    msgs = build_improve_prompt(
        parent_code="print('parent')",
        parent_cv=0.80,
        parent_holdout=0.78,
        parent_stdout_tail="class balance: 60/40\nfold 1: 0.80",
        direction="higher is better",
        hint_index=0,
        contract_summary="- metric: auc (maximize)",
        time_remaining_s=1200.0,
        fraction_used=0.5,
    )
    user = msgs[1]["content"]
    assert "class balance: 60/40" in user
    assert "feature engineering" in user
    assert "cv=0.80000" in user


def test_debug_prompt_classifies_oom():
    msgs = build_debug_prompt(
        parent_code="print('x')",
        error_summary="CUDA out of memory",
        log_tail="torch.cuda.OutOfMemoryError: ...",
        contract_summary="",
        time_remaining_s=500.0,
    )
    user = msgs[1]["content"]
    assert "Error class: oom" in user
    assert "Reduce memory" in user


def test_classify_error_buckets_work():
    assert classify_error("ValueError: cannot reshape array", "")[0] == "shape_mismatch"
    assert classify_error("FileNotFoundError: no such file", "")[0] == "data_loading"
    assert classify_error("ModuleNotFoundError: no module named 'foo'", "")[0] == "import_error"
    assert classify_error("subprocess timed out", "")[0] == "timeout"


def test_dispositions_nonempty_and_indexable():
    assert len(DISPOSITIONS) >= 3
    assert "SPEED" in disposition_for_run(0)
    assert "GO BIG" in disposition_for_run(2)
    # Wraps around for large indices.
    assert disposition_for_run(5) == disposition_for_run(5 % len(DISPOSITIONS))


def test_system_prompt_mentions_outcome_json():
    assert "OUTCOME_JSON" in SYSTEM_PROMPT
    assert "_splits.csv" in SYSTEM_PROMPT
