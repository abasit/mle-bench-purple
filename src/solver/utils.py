"""Output hygiene, time tracking, path helpers.

These helpers exist to keep the LLM context clean: tracebacks come through
filtered, warnings are stripped, output is tail-truncated, and the time
budget is exposed via a single object so we never sprinkle ``time.time()``
calls throughout the loop.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("solver")


# ── Output hygiene ────────────────────────────────────────────────────────
# Sklearn / pandas / torch warnings can blow up context length and crowd out
# the actual error message. We strip multi-line warning blocks before any
# output is shown to the LLM.

_WARNING_BLOCK_RE = re.compile(
    r"^/.+?:\d+:.*?Warning:.*?$\n(?:^\s+.*?$\n)*",
    re.MULTILINE,
)

# Optional secondary patterns for warnings that don't start with a path.
_LOOSE_WARNING_RE = re.compile(
    r"^[A-Za-z]+Warning:.*?$\n(?:^\s+.*?$\n)*",
    re.MULTILINE,
)


def strip_warnings(text: str) -> str:
    """Remove multi-line warning blocks emitted by sklearn/pandas/torch."""
    if not text:
        return text
    cleaned = _WARNING_BLOCK_RE.sub("", text)
    cleaned = _LOOSE_WARNING_RE.sub("", cleaned)
    return cleaned.strip("\n")


def truncate_tail(text: str, max_chars: int) -> str:
    """Truncate output but keep the *tail* — tracebacks live there."""
    if not text or len(text) <= max_chars:
        return text
    keep = max(max_chars - 80, 200)
    truncated = len(text) - keep
    return f"[...{truncated} chars truncated...]\n{text[-keep:]}"


def filter_traceback_frames(text: str) -> str:
    """Strip interpreter / importlib frames so only user-code frames remain."""
    if not text:
        return text
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if "interpreter.py" not in line
        and "importlib" not in line
        and "/solver/" not in line
    )


def clean_exec_output(stdout: str, stderr: str, max_chars: int = 6000) -> str:
    """Standard pipeline: combine, strip warnings, filter frames, tail-truncate."""
    parts: list[str] = []
    if stdout:
        parts.append(strip_warnings(stdout))
    if stderr:
        cleaned_err = filter_traceback_frames(strip_warnings(stderr))
        if cleaned_err:
            parts.append("---STDERR---")
            parts.append(cleaned_err)
    combined = "\n".join(p for p in parts if p)
    return truncate_tail(combined, max_chars)


# ── Time tracking ─────────────────────────────────────────────────────────


class TimeBudget:
    """Wallclock budget tracker.

    All time logic in the runner goes through this object so we never sprinkle
    raw ``time.time()`` calls throughout the codebase.
    """

    def __init__(self, total_seconds: float, grace_seconds: float = 120):
        self.total = float(total_seconds)
        self.grace = float(grace_seconds)
        self.started_at = time.time()

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed())

    def fraction_used(self) -> float:
        if self.total <= 0:
            return 1.0
        return min(1.0, self.elapsed() / self.total)

    def can_spawn_step(self) -> bool:
        """True if there's enough headroom to start a fresh step."""
        return self.remaining() > self.grace

    def can_spawn_with_reserve(self, reserve: float) -> bool:
        """True if there's enough budget to start a step *and* keep ``reserve`` left."""
        return self.remaining() > (self.grace + reserve)

    def __str__(self) -> str:
        return (
            f"TimeBudget(elapsed={self.elapsed():.0f}s, "
            f"remaining={self.remaining():.0f}s, "
            f"total={self.total:.0f}s)"
        )


# ── Path helpers ──────────────────────────────────────────────────────────


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_sample_submission(data_dir: Path) -> Path | None:
    """Find a sample_submission CSV file regardless of exact filename."""
    if not data_dir.exists():
        return None
    candidates: list[Path] = []
    for p in data_dir.rglob("*.csv"):
        name = p.name.lower()
        if "sample" in name and "submission" in name:
            candidates.append(p)
    if not candidates:
        return None
    # Prefer the shortest path (most likely the canonical one).
    return sorted(candidates, key=lambda p: (len(p.parts), len(p.name)))[0]


def list_data_files(data_dir: Path, max_files: int = 30) -> list[str]:
    if not data_dir.exists():
        return []
    files: list[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file():
            files.append(p)
            if len(files) >= max_files:
                break
    return [str(p.relative_to(data_dir)) for p in files]


# ── Misc ──────────────────────────────────────────────────────────────────


def fmt_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def first_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[:n]) + f"\n[... {len(lines) - n} more lines ...]"


# ── Environment introspection ─────────────────────────────────────────────


def detect_environment() -> dict[str, str]:
    """Best-effort snapshot of the runtime environment.

    Used by the runner to (a) log diagnostics and (b) tell the LLM whether a
    GPU is available so it doesn't waste a step trying ``torch.cuda``.
    """
    info: dict[str, str] = {}
    import platform
    import sys
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform(terse=True)

    # CPU count
    try:
        import os as _os
        info["cpu_count"] = str(_os.cpu_count() or "?")
    except Exception:
        info["cpu_count"] = "?"

    # Memory
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info["ram_gb"] = f"{vm.total / (1024**3):.1f}"
    except Exception:
        info["ram_gb"] = "?"

    # GPU
    info["gpu"] = "none"
    info["gpu_count"] = "0"
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            info["gpu_count"] = str(torch.cuda.device_count())
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return info


def env_summary_for_prompt(env: dict[str, str]) -> str:
    gpu_line = (
        f"GPU: {env['gpu']} (count={env['gpu_count']}) — torch.cuda available"
        if env.get("gpu_count", "0") != "0"
        else "GPU: NONE — DO NOT use .cuda(); set device='cpu' everywhere"
    )
    return (
        f"- Python: {env.get('python','?')}\n"
        f"- OS: {env.get('platform','?')}\n"
        f"- CPUs: {env.get('cpu_count','?')}\n"
        f"- RAM: {env.get('ram_gb','?')} GB\n"
        f"- {gpu_line}"
    )


def safe_get(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk a chain of attribute / dict accesses safely."""
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def join_truthy(items: Iterable[Any], sep: str = "\n") -> str:
    return sep.join(str(i) for i in items if i)
