"""Per-node subprocess executor.

Runs each candidate script in an isolated Python subprocess with:
- a copy of ./input (read-only symlink or mirror to the shared data dir)
- a fresh workspace subdirectory per node
- a wall-clock timeout

Session state reuse is deliberately NOT reimplemented here — the new design
accepts the cold-start cost in exchange for simplicity and deterministic
iteration. (If it turns out to matter on a specific task we can add it back
behind a flag.)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("purple_next")


@dataclass
class ExecResult:
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    submission_path: Path | None = None
    error_summary: str = ""
    log_tail_chars: int = 6000

    @property
    def is_success(self) -> bool:
        return self.return_code == 0 and not self.timed_out

    @property
    def has_submission(self) -> bool:
        return self.submission_path is not None and self.submission_path.exists()

    def tail(self, max_chars: int | None = None) -> str:
        limit = max_chars if max_chars is not None else self.log_tail_chars
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append("---STDERR---")
            parts.append(self.stderr)
        combined = "\n".join(p for p in parts if p)
        if len(combined) <= limit:
            return combined
        return combined[-limit:]


class Interpreter:
    def __init__(self, *, workspace_dir: Path, data_dir: Path, timeout: float):
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.timeout = float(timeout)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def run(self, code: str, node_id: str) -> ExecResult:
        node_dir = self.workspace_dir / f"node_{node_id}"
        node_dir.mkdir(parents=True, exist_ok=True)

        # Mirror input dir (symlink when possible, copytree as fallback).
        input_link = node_dir / "input"
        if input_link.exists() or input_link.is_symlink():
            try:
                if input_link.is_symlink() or input_link.is_file():
                    input_link.unlink()
                else:
                    shutil.rmtree(input_link)
            except Exception:
                pass
        try:
            os.symlink(self.data_dir, input_link, target_is_directory=True)
        except (OSError, NotImplementedError):
            shutil.copytree(self.data_dir, input_link)

        script_path = node_dir / "solution.py"
        script_path.write_text(code, encoding="utf-8")

        started = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, "solution.py"],
                cwd=str(node_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
            rc = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            rc = -1
            timed_out = True
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, (bytes, bytearray)) else (exc.stdout or "")
            stderr = (exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, (bytes, bytearray)) else (exc.stderr or "")) + f"\nTimeoutExpired after {self.timeout:.0f}s"
        except Exception as exc:
            rc = -1
            stdout = ""
            stderr = f"InterpreterError: {type(exc).__name__}: {exc}"

        duration = time.time() - started
        sub_path = node_dir / "submission.csv"

        error_summary = ""
        if rc != 0 or timed_out:
            error_summary = _summarize_error(stderr)

        # Save stdout/stderr to disk for post-run debugging.
        try:
            (node_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (node_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        except Exception:
            pass

        return ExecResult(
            return_code=rc,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=timed_out,
            submission_path=sub_path if sub_path.exists() else None,
            error_summary=error_summary,
        )


def _summarize_error(stderr: str) -> str:
    if not stderr:
        return ""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if ":" in line and not line.startswith("File "):
            return line[:240]
    return lines[-1][:240]
