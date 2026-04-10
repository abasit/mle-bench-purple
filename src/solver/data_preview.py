"""Data preview — gives the LLM a snapshot of what's in ./input/.

Works across competition types:
    - Tabular: previews CSVs (columns, dtypes, sample values, missing rates, target dist)
    - Vision: counts image files by extension, previews any metadata CSVs
    - NLP: same as tabular but also notes text column lengths
    - Audio: counts audio files, previews metadata
    - Parquet: reads with pd.read_parquet

No competition-specific hardcoding. Target column is inferred by comparing
train columns vs sample_submission columns (the column in train that appears
in sample_submission but not in test is likely the target).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("solver")

_MAX_PREVIEW_CHARS = 5000
_PREVIEW_ROW_LIMIT = 5
_PREVIEW_VALUES_PER_COL = 6
_TABULAR_EXTS = {".csv", ".tsv", ".parquet", ".jsonl"}


def build_data_preview(data_dir: Path) -> str:
    if not data_dir.exists():
        return ""

    parts: list[str] = []

    # 1. Directory structure summary — always useful.
    parts.append(_dir_summary(data_dir))

    # 2. Preview tabular files (CSV / parquet / TSV).
    try:
        import pandas as pd
        tabular_files = _select_tabular_files(data_dir)
        sample_sub_cols = _get_sample_submission_columns(data_dir, pd)
        for path in tabular_files[:4]:
            try:
                parts.append(_preview_tabular(path, data_dir, pd, sample_sub_cols))
            except Exception as e:
                rel = path.relative_to(data_dir).as_posix()
                parts.append(f"{rel}: could not read ({e})")
    except ImportError:
        pass

    text = "\n\n".join(p for p in parts if p).strip()
    if len(text) > _MAX_PREVIEW_CHARS:
        text = text[:_MAX_PREVIEW_CHARS] + "\n[... truncated ...]"
    return text


# ---------------------------------------------------------------------------
# Directory summary
# ---------------------------------------------------------------------------

def _dir_summary(data_dir: Path) -> str:
    """Count files by type and list subdirectories."""
    ext_counts: dict[str, int] = {}
    subdirs: list[str] = []
    total = 0
    for p in data_dir.rglob("*"):
        if p.is_dir():
            try:
                rel = p.relative_to(data_dir).as_posix()
                if rel != "." and "/" not in rel:  # top-level subdirs only
                    subdirs.append(rel)
            except Exception:
                pass
            continue
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        total += 1
        if total > 10000:
            break

    lines = [f"Directory: {total} files"]
    if ext_counts:
        top = sorted(ext_counts.items(), key=lambda x: -x[1])[:8]
        lines.append("  " + ", ".join(f"{ext}: {cnt}" for ext, cnt in top))
    if subdirs:
        lines.append("  Subdirs: " + ", ".join(sorted(subdirs)[:10]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tabular file selection + preview
# ---------------------------------------------------------------------------

def _select_tabular_files(data_dir: Path) -> list[Path]:
    """Pick tabular files in priority order: train, test, sample_submission, labels, other."""
    found: list[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _TABULAR_EXTS:
            found.append(p)
    if not found:
        return []

    def rank(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        if "train" in name and "label" not in name:
            return (0, len(name))
        if "test" in name:
            return (1, len(name))
        if "sample" in name and "submission" in name:
            return (2, len(name))
        if "label" in name or "target" in name:
            return (3, len(name))
        return (4, len(name))

    found.sort(key=rank)
    return found


def _read_tabular(path: Path, pd, nrows: int | None = None):
    """Read a tabular file regardless of format."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.head(nrows)
        return df
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def _get_sample_submission_columns(data_dir: Path, pd) -> set[str]:
    """Read the sample submission to help identify the target column."""
    for p in data_dir.rglob("*"):
        name = p.name.lower()
        if "sample" in name and "submission" in name and p.suffix.lower() in _TABULAR_EXTS:
            try:
                df = _read_tabular(p, pd, nrows=1)
                return set(df.columns)
            except Exception:
                pass
    return set()


def _preview_tabular(path: Path, data_dir: Path, pd, sample_sub_cols: set[str]) -> str:
    """Build a compact preview for one tabular file."""
    rel = path.relative_to(data_dir).as_posix()
    head = _read_tabular(path, pd, nrows=_PREVIEW_ROW_LIMIT)

    # Row count (cheap line count).
    row_count = -1
    if path.suffix.lower() != ".parquet":
        try:
            with path.open("rb") as fh:
                row_count = sum(1 for _ in fh) - 1
        except Exception:
            pass
    else:
        try:
            row_count = len(pd.read_parquet(path, columns=[head.columns[0]]))
        except Exception:
            pass

    rows_str = str(row_count) if row_count >= 0 else "?"
    n_cols = len(head.columns)

    # Read a sample for stats (only for files that look like training data).
    name_lower = path.name.lower()
    is_trainlike = "train" in name_lower or "label" in name_lower
    sample_df = None
    if is_trainlike and row_count != 0:
        try:
            sample_df = _read_tabular(path, pd, nrows=5000)
        except Exception:
            pass

    lines: list[str] = []
    lines.append(f"{rel}  ({rows_str} rows, {n_cols} cols)")

    # Columns with dtypes, missing rates, sample values.
    for col in head.columns:
        dtype = str(head[col].dtype)
        sample_vals = _short_sample(head[col].dropna().tolist())
        miss_str = ""
        if sample_df is not None and col in sample_df.columns:
            miss = float(sample_df[col].isna().mean())
            if miss > 0:
                miss_str = f" miss={miss:.0%}"
        lines.append(f"  {col}: {dtype}{miss_str}  {sample_vals}")

    # Head rows.
    try:
        head_str = head.to_string(index=False, max_colwidth=20)
        for line in head_str.splitlines():
            lines.append(f"  {line}")
    except Exception:
        pass

    # Infer target column by diffing train columns vs sample_submission columns.
    if sample_df is not None and sample_sub_cols:
        target_candidates = [
            c for c in sample_df.columns
            if c in sample_sub_cols and c not in _likely_id_cols(sample_df)
        ]
        if not target_candidates:
            ranked = sorted(
                [c for c in sample_df.columns if c not in _likely_id_cols(sample_df)],
                key=_target_name_score,
                reverse=True,
            )
            if ranked:
                target_candidates = ranked[:1]
        if target_candidates:
            target_col = target_candidates[0]
            lines.append(_target_stats(sample_df, target_col, pd))

    return "\n".join(lines)


def _likely_id_cols(df) -> set[str]:
    """Heuristic: columns that look like IDs (first col, or name contains 'id')."""
    ids = {df.columns[0]}
    for c in df.columns:
        if c.lower() in {"id", "index"} or c.lower().endswith("id") or c.lower().endswith("_id"):
            ids.add(c)
    return ids


def _target_stats(df, col: str, pd) -> str:
    """One-line target summary: dtype, nunique, distribution or range."""
    s = df[col]
    try:
        nunique = int(s.nunique(dropna=True))
    except Exception:
        return f"  Target: {col}"

    parts = [f"  Target: {col} ({s.dtype}, {nunique} unique)"]

    if nunique <= 20 and not pd.api.types.is_float_dtype(s):
        try:
            vc = s.value_counts(dropna=True).head(6)
            dist = ", ".join(f"{v!r}:{c}" for v, c in vc.items())
            parts.append(f" [{dist}]")
        except Exception:
            pass
    elif pd.api.types.is_numeric_dtype(s):
        try:
            parts.append(f" range=[{s.min():.4g}, {s.max():.4g}] mean={s.mean():.4g}")
        except Exception:
            pass

    return "".join(parts)


def _target_name_score(name: str) -> int:
    lower = str(name).lower()
    score = 0
    if lower in {"target", "label", "class", "y"}:
        score += 10
    if "target" in lower:
        score += 6
    if "label" in lower or "class" in lower or "category" in lower:
        score += 5
    if "score" in lower or "rating" in lower or "sentiment" in lower:
        score += 3
    return score


def _short_sample(values: Iterable, n: int = _PREVIEW_VALUES_PER_COL) -> str:
    out: list[str] = []
    for v in list(values)[:n]:
        s = repr(v)
        if len(s) > 24:
            s = s[:21] + "..."
        out.append(s)
    return "[" + ", ".join(out) + "]"
