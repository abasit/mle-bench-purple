"""Interactive data exploration — Phase 0.

Before generating drafts, run a few lightweight Python snippets against the
actual competition data and capture the output. This gives the LLM *real*
data statistics (not just the static preview built by data_preview.py) and
lets it discover things the heuristic preview misses: unusual delimiters,
nested directories, binary file formats, multi-table relationships, etc.

The exploration runs in the same subprocess sandbox as nodes but with a
short timeout (60s) and no submission expected. The output is injected
into the draft prompts as an "exploration report".
"""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent

from .interpreter import Interpreter

logger = logging.getLogger("solver")

# Each snippet is (label, code). They run sequentially in separate subprocesses.
_EXPLORE_SNIPPETS: list[tuple[str, str]] = [
    (
        "file_listing",
        dedent("""\
            import os
            for root, dirs, files in os.walk('./input'):
                depth = root.replace('./input', '').count(os.sep)
                indent = '  ' * depth
                print(f'{indent}{os.path.basename(root)}/')
                if depth < 2:
                    sub_indent = '  ' * (depth + 1)
                    for f in sorted(files)[:30]:
                        size = os.path.getsize(os.path.join(root, f))
                        if size > 1024*1024:
                            size_str = f'{size/1024/1024:.1f}MB'
                        elif size > 1024:
                            size_str = f'{size/1024:.0f}KB'
                        else:
                            size_str = f'{size}B'
                        print(f'{sub_indent}{f}  ({size_str})')
                    if len(files) > 30:
                        print(f'{sub_indent}... and {len(files)-30} more files')
        """),
    ),
    (
        "tabular_stats",
        dedent("""\
            import pandas as pd
            import os, glob
            csvs = sorted(glob.glob('./input/**/*.csv', recursive=True))
            csvs += sorted(glob.glob('./input/**/*.parquet', recursive=True))
            csvs += sorted(glob.glob('./input/**/*.tsv', recursive=True))
            for path in csvs[:5]:
                name = os.path.relpath(path, './input')
                try:
                    if path.endswith('.parquet'):
                        df = pd.read_parquet(path)
                    elif path.endswith('.tsv'):
                        df = pd.read_csv(path, sep='\\t', nrows=5000)
                    else:
                        df = pd.read_csv(path, nrows=5000)
                    print(f'\\n=== {name} ===')
                    print(f'Shape: {df.shape}')
                    print(f'Columns: {list(df.columns)}')
                    print(f'Dtypes:\\n{df.dtypes.to_string()}')
                    print(f'Missing:\\n{df.isnull().sum().to_string()}')
                    print(f'\\nHead:')
                    print(df.head(3).to_string())
                    # Numeric stats
                    num_cols = df.select_dtypes('number').columns.tolist()
                    if num_cols:
                        print(f'\\nNumeric describe:')
                        print(df[num_cols[:8]].describe().to_string())
                    # Categorical stats
                    cat_cols = df.select_dtypes(['object', 'bool']).columns.tolist()
                    for c in cat_cols[:4]:
                        vc = df[c].value_counts(dropna=False).head(8)
                        print(f'\\n{c} value_counts: {dict(vc)}')
                except Exception as e:
                    print(f'{name}: ERROR {e}')
        """),
    ),
    (
        "sample_submission_check",
        dedent("""\
            import pandas as pd, glob, os
            subs = [f for f in glob.glob('./input/**/*sample*submission*', recursive=True)
                    if f.endswith('.csv')]
            if not subs:
                subs = [f for f in glob.glob('./input/**/*submission*', recursive=True)
                        if f.endswith('.csv')]
            for path in subs[:1]:
                name = os.path.relpath(path, './input')
                df = pd.read_csv(path)
                print(f'Sample submission: {name}')
                print(f'Shape: {df.shape}')
                print(f'Columns: {list(df.columns)}')
                print(f'Dtypes: {dict(df.dtypes)}')
                print(f'Head:\\n{df.head(5).to_string()}')
                for c in df.columns:
                    if c.lower() not in ('id', 'index') and not c.lower().endswith('id'):
                        print(f'\\nTarget column "{c}": nunique={df[c].nunique()}, '
                              f'dtype={df[c].dtype}, sample={df[c].head(5).tolist()}')
        """),
    ),
]


def run_exploration(
    interpreter: Interpreter,
    *,
    timeout: float = 60.0,
) -> str:
    """Run exploration snippets and return a combined report.

    Each snippet runs in its own subprocess (via the standard Interpreter)
    with a short timeout. Failures are logged but don't block the pipeline.

    Returns a formatted string suitable for injection into draft prompts.
    """
    original_timeout = interpreter.timeout
    interpreter.timeout = timeout

    sections: list[str] = []
    for label, code in _EXPLORE_SNIPPETS:
        try:
            result = interpreter.run(code, f"explore_{label}")
            output = result.stdout.strip()
            if result.stderr.strip() and not result.is_success:
                output += f"\n[stderr]: {result.stderr.strip()[:500]}"
            if output:
                sections.append(f"## {label}\n{_truncate(output, 2000)}")
            else:
                logger.debug(f"[explore] {label}: no output")
        except Exception as e:
            logger.warning(f"[explore] {label} failed: {e}")

    interpreter.timeout = original_timeout

    if not sections:
        return ""

    report = "DATA EXPLORATION (live output from running code on the actual data):\n\n"
    report += "\n\n".join(sections)

    # Cap total length
    if len(report) > 6000:
        report = report[:6000] + "\n[... exploration truncated ...]"

    logger.info(f"[explore] exploration report: {len(report)} chars, {len(sections)} sections")
    return report


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Keep tail (more informative) when truncating.
    return "...\n" + text[-(max_chars - 4):]
