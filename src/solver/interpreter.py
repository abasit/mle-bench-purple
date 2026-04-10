"""Subprocess-based Python code executor.

Each node in the search tree runs in its OWN fresh subprocess so:
    - imports cannot leak across nodes,
    - one node's runaway memory doesn't poison the next,
    - each node produces its own ``submission.csv`` (required for ensembling).

Design notes
------------
- The generated code is written to ``solution.py`` inside a per-node workdir.
- We invoke ``python solution.py`` with ``cwd=node_workdir``. The node sees:
    ./input/         → directory symlink (or copy) of the competition data
    ./submission.csv → expected output path
- We capture stdout and stderr, enforce a wallclock timeout via Popen.kill,
  and rebrand timeouts as ``TimeoutError`` in the result so the LLM gets a
  clean signal instead of a confusing SIGTERM.
- We use the *real* Python from the parent process (``sys.executable``) so
  the subprocess inherits the same package set without re-creating an env.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .utils import clean_exec_output

logger = logging.getLogger("solver")


@dataclass
class ExecResult:
    """Outcome of running a node's solution.py."""

    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    submission_path: Path | None = None
    error_summary: str = ""

    @property
    def is_success(self) -> bool:
        return self.return_code == 0 and not self.timed_out

    @property
    def has_submission(self) -> bool:
        return self.submission_path is not None and self.submission_path.exists()

    def cleaned_log(self, max_chars: int = 6000) -> str:
        return clean_exec_output(self.stdout, self.stderr, max_chars=max_chars)


class Interpreter:
    """Per-node Python subprocess executor."""

    def __init__(
        self,
        workspace_dir: Path,
        data_dir: Path,
        timeout: float = 1800,
        python_executable: str | None = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.timeout = float(timeout)
        self.python = python_executable or sys.executable
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────

    def run(self, code: str, node_id: str) -> ExecResult:
        """Execute ``code`` in a fresh subprocess for the given node id."""
        node_dir = self.node_dir(node_id)
        node_dir.mkdir(parents=True, exist_ok=True)

        self._link_input(node_dir)
        solution_path = node_dir / "solution.py"
        solution_path.write_text(code, encoding="utf-8")

        submission_path = node_dir / "submission.csv"
        if submission_path.exists():
            submission_path.unlink()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("MPLBACKEND", "Agg")

        start = time.time()
        timed_out = False
        return_code = -1
        stdout = ""
        stderr = ""

        try:
            proc = subprocess.Popen(
                [self.python, "solution.py"],
                cwd=str(node_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                # On POSIX we put the child in its own process group so we can
                # signal the whole tree on timeout. On Windows there's no
                # equivalent — we just call .kill() which terminates the proc.
                start_new_session=(os.name != "nt"),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                    if os.name == "nt"
                    else 0
                ),
            )

            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill(proc)
                # Drain whatever output exists.
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except Exception:
                    stdout, stderr = "", ""
                return_code = -signal.SIGTERM if hasattr(signal, "SIGTERM") else -1
                stderr = (stderr or "") + (
                    f"\nTimeoutError: execution exceeded {self.timeout:.0f}s time limit"
                )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[interpreter] subprocess launch failed for {node_id}: {e}")
            stderr = f"InterpreterError: {type(e).__name__}: {e}"
            return_code = -1

        duration = time.time() - start

        sub_path = submission_path if submission_path.exists() else None

        error_summary = ""
        if not (return_code == 0 and not timed_out):
            error_summary = self._extract_error_summary(stderr)

        result = ExecResult(
            return_code=return_code,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_seconds=duration,
            timed_out=timed_out,
            submission_path=sub_path,
            error_summary=error_summary,
        )

        logger.info(
            f"[interpreter] node {node_id}: rc={return_code} "
            f"duration={duration:.0f}s timed_out={timed_out} "
            f"submission={'YES' if sub_path else 'no'}"
        )
        return result

    # ── helpers ───────────────────────────────────────────────────────────

    def node_dir(self, node_id: str) -> Path:
        return self.workspace_dir / "nodes" / node_id

    def _link_input(self, node_dir: Path) -> None:
        """Make the competition data available at ``./input/`` inside node_dir.

        Strategy:
            1. Try a directory-symlink (cheap).
            2. On Windows, try a junction (works without symlink privileges).
            3. If linking fails, copy the directory.
        """
        target = node_dir / "input"
        if target.exists() or target.is_symlink():
            return
        try:
            target.symlink_to(self.data_dir.resolve(), target_is_directory=True)
            return
        except (OSError, NotImplementedError) as e:
            logger.debug(f"[interpreter] symlink failed ({e}), falling back to copy")
        if os.name == "nt":
            try:
                self._create_windows_junction(target, self.data_dir.resolve())
                return
            except Exception as e:
                logger.debug(f"[interpreter] junction failed ({e}), falling back to copy")
        try:
            shutil.copytree(self.data_dir, target)
        except Exception:
            logger.exception(
                f"[interpreter] failed to materialise input dir at {target}"
            )

    def _kill(self, proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if proc.poll() is None:
                    proc.kill()
            else:
                # Kill the whole process group.
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(2)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _create_windows_junction(target: Path, source: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not target.exists():
            raise OSError(
                f"mklink /J failed rc={completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()}"
            )

    @staticmethod
    def _extract_error_summary(stderr: str) -> str:
        """Pull the last (and most informative) Traceback / Error line out of stderr."""
        if not stderr:
            return ""
        # Walk lines from the bottom looking for the canonical "ErrorName: message" form.
        for line in reversed(stderr.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            if any(
                stripped.startswith(prefix + ":")
                or " " + prefix + ":" in stripped
                for prefix in (
                    "Error",
                    "Exception",
                    "TypeError",
                    "ValueError",
                    "KeyError",
                    "FileNotFoundError",
                    "RuntimeError",
                    "ImportError",
                    "ModuleNotFoundError",
                    "AttributeError",
                    "IndexError",
                    "AssertionError",
                    "MemoryError",
                    "TimeoutError",
                    "NameError",
                    "ZeroDivisionError",
                    "OSError",
                )
            ):
                return stripped[:300]
        return stderr.splitlines()[-1].strip()[:300]
