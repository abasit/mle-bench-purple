"""Tests for purple_next.config — yaml parity + validation."""

from pathlib import Path

from purple_next.config import Config


def test_yaml_parses_and_validates():
    yaml_path = Path(__file__).resolve().parents[1] / "purple_next.yaml"
    cfg = Config.from_yaml(yaml_path)
    # API key may be blank at rest; ignore that for validation.
    errs = [e for e in cfg.validate() if "api_key" not in e]
    assert errs == []
    assert cfg.search.pass_k >= 1
    assert cfg.search.num_drafts >= 1
    assert cfg.search.n_folds >= 2


def test_config_defaults_are_sane():
    cfg = Config()
    assert cfg.search.holdout_fraction == 0.20
    assert cfg.search.ucb_explore_c == 1.0
    assert cfg.exec.timeout > 0
