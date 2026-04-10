"""YAML config loader.

Schema is intentionally flat and small. Any unrecognised key is silently
ignored. The on-disk yaml uses an ``agent.code`` block for the LLM section
to keep deployment configs stable; an alternative top-level ``llm`` block is
also accepted.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("solver")


# ── Sub-configs ───────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 12000
    # Per-HTTP-call timeout. Keep this short or silent hangs will burn the
    # entire wall-clock budget. Heartbeat logging during a call is enabled in
    # the LLM client itself.
    timeout: float = 240.0
    max_retries: int = 3


@dataclass
class ExecConfig:
    timeout: float = 1800.0   # per-node code execution timeout (seconds)


@dataclass
class SearchConfig:
    num_drafts: int = 2                  # initial parallel drafts (variants 0..N-1)
    max_steps: int = 60                  # total node generations across the run
    max_parallel: int = 2                # worker pool size for phase 2
    max_debug_attempts_per_node: int = 2 # per-node cap (lineage cap also applies)
    improve_top_k: int = 2               # rotate improves over top-K validated nodes
    final_candidate_top_k: int = 2       # keep top-K single-model submissions at the end
    inline_repair_attempts: int = 1      # bounded generate-run-fix loop before marking a node buggy
    grace_seconds: float = 180           # don't spawn new steps under this remaining
    # Relative gap threshold for the val/holdout honesty check.
    # gap = |val - holdout| / max(|val|, |holdout|, 0.01)
    # gap > threshold → node is flagged ``suspicious``.
    holdout_gap_rel_threshold: float = 0.20
    metric_improve_eps: float = 1e-4
    fold_variance_rel_threshold: float = 0.20
    identical_score_abs_tol: float = 1e-6
    perfect_score_abs_tol: float = 1e-6


@dataclass
class SolverConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    time_limit: float = 30600.0
    seed: int = 42
    workspace_dir: Path = Path("./runs")
    data_dir: Path = Path("./data")
    desc_file: Path | None = None
    use_kb: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "SolverConfig":
        cfg = cls()
        if not path.exists():
            logger.warning(f"[config] {path} not found, using defaults")
            return cfg
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning(f"[config] failed to parse {path}: {e}; using defaults")
            return cfg
        cfg.merge_dict(raw)
        return cfg

    def merge_dict(self, raw: dict[str, Any]) -> None:
        """Merge a YAML dict into this config.

        Accepted top-level shapes:
            time_limit, seed, log_level, use_kb
            agent.{time_limit, seed, steps, initial_drafts}
            agent.code.{model, base_url, api_key, temp, max_tokens, timeout, max_retries}
            agent.feedback.{...}             # secondary LLM block, fills gaps in code.*
            agent.llm.{...}                  # alternative single LLM block
            agent.search.{num_drafts, max_steps, final_candidate_top_k, ...}
            search.{...}                     # alternative top-level search block
            llm.{...}                        # alternative top-level llm block
            exec.timeout
        """
        if not isinstance(raw, dict):
            return

        if "time_limit" in raw:
            self.time_limit = float(raw["time_limit"])
        if "seed" in raw:
            self.seed = int(raw["seed"])
        if "log_level" in raw:
            self.log_level = str(raw["log_level"])
        if "use_kb" in raw:
            self.use_kb = bool(raw["use_kb"])

        agent = raw.get("agent") or {}
        if isinstance(agent, dict):
            self._merge_agent(agent)

        exec_block = raw.get("exec") or {}
        if isinstance(exec_block, dict):
            if "timeout" in exec_block:
                self.exec.timeout = float(exec_block["timeout"])

        llm_block = raw.get("llm") or {}
        if isinstance(llm_block, dict):
            self._merge_llm(llm_block)

        search_block = raw.get("search") or {}
        if isinstance(search_block, dict):
            self._merge_search(search_block)

    def _merge_agent(self, agent: dict[str, Any]) -> None:
        if "time_limit" in agent:
            self.time_limit = float(agent["time_limit"])
        if "seed" in agent:
            self.seed = int(agent["seed"])
        if "steps" in agent:
            self.search.max_steps = int(agent["steps"])
        if "initial_drafts" in agent:
            self.search.num_drafts = int(agent["initial_drafts"])

        # We honour ``agent.code`` first (canonical) then let other blocks fill gaps.
        primary = agent.get("code") if isinstance(agent.get("code"), dict) else None
        secondary = agent.get("feedback") if isinstance(agent.get("feedback"), dict) else None
        explicit_llm = agent.get("llm") if isinstance(agent.get("llm"), dict) else None
        if secondary:
            self._merge_llm(secondary)
        if primary:
            self._merge_llm(primary)
        if explicit_llm:
            self._merge_llm(explicit_llm)

        search_block = agent.get("search")
        if isinstance(search_block, dict):
            self._merge_search(search_block)

    def _merge_llm(self, block: dict[str, Any]) -> None:
        if "model" in block:
            self.llm.model = str(block["model"])
        if "base_url" in block:
            self.llm.base_url = str(block["base_url"])
        if "api_key" in block:
            api_key = str(block["api_key"]).strip()
            if api_key:
                self.llm.api_key = api_key
        if "temperature" in block:
            self.llm.temperature = float(block["temperature"])
        elif "temp" in block:
            self.llm.temperature = float(block["temp"])
        if "max_tokens" in block:
            self.llm.max_tokens = int(block["max_tokens"])
        if "timeout" in block:
            self.llm.timeout = float(block["timeout"])
        if "max_retries" in block:
            self.llm.max_retries = int(block["max_retries"])

    def _merge_search(self, block: dict[str, Any]) -> None:
        for key in (
            "num_drafts",
            "max_steps",
            "max_parallel",
            "max_debug_attempts_per_node",
            "improve_top_k",
            "final_candidate_top_k",
            "inline_repair_attempts",
            "grace_seconds",
            "holdout_gap_rel_threshold",
            "metric_improve_eps",
            "fold_variance_rel_threshold",
            "identical_score_abs_tol",
            "perfect_score_abs_tol",
        ):
            if key in block:
                cur = getattr(self.search, key)
                setattr(self.search, key, type(cur)(block[key]))
        if "ensemble_top_k" in block and "final_candidate_top_k" not in block:
            self.search.final_candidate_top_k = int(block["ensemble_top_k"])
        # Backwards-compat for the old absolute-gap key.
        if "holdout_gap_threshold" in block and "holdout_gap_rel_threshold" not in block:
            self.search.holdout_gap_rel_threshold = float(block["holdout_gap_threshold"])

    def resolve_env(self) -> None:
        """Pull the API key from common env-vars if it isn't set in the yaml."""
        if not self.llm.api_key:
            for var in (
                "NEBIUS_API_KEY",
                "OPENAI_API_KEY",
                "TOKENFACTORY_API_KEY",
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
            ):
                val = os.environ.get(var, "").strip()
                if val:
                    self.llm.api_key = val
                    logger.info(f"[config] LLM api_key loaded from ${var}")
                    break

    def validate(self) -> list[str]:
        """Return a list of human-readable validation errors. Empty = OK."""
        errors: list[str] = []
        if not self.llm.api_key:
            errors.append(
                "LLM api_key is empty — set agent.code.api_key in solver.yaml or "
                "export NEBIUS_API_KEY / OPENAI_API_KEY in the environment."
            )
        if not self.llm.model:
            errors.append("LLM model is empty.")
        if self.time_limit <= 0:
            errors.append(f"time_limit must be > 0 (got {self.time_limit})")
        if self.search.num_drafts < 1:
            errors.append(f"search.num_drafts must be >= 1 (got {self.search.num_drafts})")
        if self.search.max_steps < self.search.num_drafts:
            errors.append(
                f"search.max_steps ({self.search.max_steps}) < num_drafts ({self.search.num_drafts})"
            )
        if self.search.final_candidate_top_k < 1:
            errors.append(
                "search.final_candidate_top_k must be >= 1 "
                f"(got {self.search.final_candidate_top_k})"
            )
        if self.search.inline_repair_attempts < 0:
            errors.append(
                "search.inline_repair_attempts must be >= 0 "
                f"(got {self.search.inline_repair_attempts})"
            )
        if self.exec.timeout <= 0:
            errors.append(f"exec.timeout must be > 0 (got {self.exec.timeout})")
        return errors

    def __str__(self) -> str:
        return (
            f"SolverConfig(model={self.llm.model}, time_limit={self.time_limit:.0f}s, "
            f"num_drafts={self.search.num_drafts}, max_steps={self.search.max_steps}, "
            f"max_parallel={self.search.max_parallel}, "
            f"final_candidate_top_k={self.search.final_candidate_top_k}, "
            f"inline_repair_attempts={self.search.inline_repair_attempts})"
        )
