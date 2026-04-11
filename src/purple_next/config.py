"""Flat dataclass config loaded from ``purple_next.yaml``.

Intentionally tiny: the new architecture moves most validation machinery
into the runner-owned protocol, so there's far less to tune.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    model: str = "Qwen/Qwen3.5-397B-A17B-fast"
    base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    api_key: str = ""
    temperature: float = 0.6
    max_tokens: int = 16000
    timeout: float = 600.0
    max_retries: int = 3


@dataclass
class SearchConfig:
    num_drafts: int = 2
    max_steps: int = 20
    max_parallel: int = 2
    pass_k: int = 3
    dispositions: list[str] = field(default_factory=list)  # filled from yaml or defaults
    seat_temperatures: list[float] = field(default_factory=list)  # per-seat LLM temp; empty → use llm.temperature
    ucb_explore_c: float = 1.0
    max_debug_attempts_per_node: int = 2
    final_top_k: int = 3
    grace_seconds: float = 180.0
    holdout_fraction: float = 0.20
    n_folds: int = 5


@dataclass
class ExecConfig:
    timeout: float = 1800.0


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    time_limit: float = 30600.0
    seed: int = 42
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        cfg = cls()
        if not path.exists():
            return cfg
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg.merge(raw)
        return cfg

    def merge(self, raw: dict[str, Any]) -> None:
        if not isinstance(raw, dict):
            return
        for key in ("time_limit", "seed", "log_level"):
            if key in raw:
                setattr(self, key, type(getattr(self, key))(raw[key]))
        llm = raw.get("llm") or {}
        for key in ("model", "base_url", "api_key", "temperature", "max_tokens", "timeout", "max_retries"):
            if key in llm:
                setattr(self.llm, key, type(getattr(self.llm, key))(llm[key]))
        search = raw.get("search") or {}
        for key in (
            "num_drafts", "max_steps", "max_parallel", "pass_k",
            "ucb_explore_c", "max_debug_attempts_per_node", "final_top_k",
            "grace_seconds", "holdout_fraction", "n_folds",
        ):
            if key in search:
                setattr(self.search, key, type(getattr(self.search, key))(search[key]))
        if "dispositions" in search and isinstance(search["dispositions"], list):
            self.search.dispositions = [str(d) for d in search["dispositions"]]
        if "seat_temperatures" in search and isinstance(search["seat_temperatures"], list):
            self.search.seat_temperatures = [float(t) for t in search["seat_temperatures"]]
        exec_block = raw.get("exec") or {}
        if "timeout" in exec_block:
            self.exec.timeout = float(exec_block["timeout"])

    def resolve_env(self) -> None:
        if self.llm.api_key:
            return
        # Match env var to base_url so the right provider key is picked when
        # the user has multiple provider keys exported at once.
        host = self.llm.base_url.lower()
        if "nebius" in host:
            preferred = ("NEBIUS_API_KEY",)
        elif "anthropic" in host:
            preferred = ("ANTHROPIC_API_KEY",)
        elif "openai" in host:
            preferred = ("OPENAI_API_KEY",)
        else:
            preferred = ()
        for var in (*preferred, "NEBIUS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            val = os.environ.get(var, "").strip()
            if val:
                self.llm.api_key = val
                return

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.search.pass_k < 1:
            errs.append("search.pass_k must be >= 1")
        if self.search.num_drafts < 1:
            errs.append("search.num_drafts must be >= 1")
        if self.search.max_steps < self.search.num_drafts:
            errs.append("search.max_steps must be >= num_drafts")
        if self.search.holdout_fraction <= 0 or self.search.holdout_fraction >= 0.5:
            errs.append("search.holdout_fraction must be in (0, 0.5)")
        if self.search.n_folds < 2:
            errs.append("search.n_folds must be >= 2")
        if self.exec.timeout <= 0:
            errs.append("exec.timeout must be > 0")
        return errs
