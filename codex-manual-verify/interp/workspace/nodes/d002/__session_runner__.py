from __future__ import annotations

import os
import sys
import types
from pathlib import Path

try:
    import dill as _serializer
except Exception:  # pragma: no cover - runtime fallback
    import cloudpickle as _serializer  # type: ignore[no-redef]


def _log(message: str) -> None:
    print(f"[session] {message}", file=sys.stderr)


def _load_parent_state(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        obj = _serializer.load(fh)
    return obj if isinstance(obj, dict) else {}


def _collect_persistable(namespace: dict[str, object]) -> dict[str, object]:
    keep: dict[str, object] = {}
    skipped: list[str] = []
    for key, value in namespace.items():
        if key.startswith("__") or key.startswith("SESSION_"):
            continue
        if isinstance(value, types.ModuleType):
            continue
        try:
            _serializer.dumps(value)
        except Exception:
            skipped.append(key)
            continue
        keep[key] = value
    if skipped:
        _log(f"skipped {len(skipped)} non-persistable globals: {', '.join(skipped[:8])}")
    return keep


def _persist_state(path: Path, namespace: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = _collect_persistable(namespace)
    try:
        with tmp.open("wb") as fh:
            _serializer.dump(payload, fh)
        os.replace(tmp, path)
        _log(f"persisted {len(payload)} globals to {path.name}")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def main() -> int:
    solution_path = Path("solution.py").resolve()
    state_path_raw = os.environ.get("MLE_SESSION_STATE_PATH", "").strip()
    parent_state_raw = os.environ.get("MLE_PARENT_SESSION_STATE_PATH", "").strip()
    parent_node_id = os.environ.get("MLE_SESSION_PARENT_NODE_ID", "").strip()

    state_path = Path(state_path_raw).resolve() if state_path_raw else None
    parent_state_path = Path(parent_state_raw).resolve() if parent_state_raw else None

    restored: dict[str, object] = {}
    if parent_state_path and parent_state_path.exists():
        try:
            restored = _load_parent_state(parent_state_path)
            _log(
                f"restored {len(restored)} globals from "
                f"{parent_node_id or parent_state_path.name}"
            )
        except Exception as exc:
            _log(f"failed to restore parent state {parent_state_path}: {type(exc).__name__}: {exc}")

    main_mod = types.ModuleType("__main__")
    namespace = main_mod.__dict__
    namespace.update(restored)
    namespace["__name__"] = "__main__"
    namespace["__file__"] = str(solution_path)
    namespace["__package__"] = None
    namespace["__builtins__"] = __builtins__
    namespace["SESSION_RESTORED"] = bool(restored)
    namespace["SESSION_PARENT_NODE_ID"] = parent_node_id
    namespace["SESSION_RESTORED_KEYS"] = tuple(sorted(k for k in restored if not k.startswith("__"))[:200])

    old_main = sys.modules.get("__main__")
    old_argv = list(sys.argv)
    sys.modules["__main__"] = main_mod
    sys.argv = [str(solution_path)]

    try:
        source = solution_path.read_text(encoding="utf-8")
        code = compile(source, str(solution_path), "exec")
        exec(code, namespace, namespace)
        return 0
    finally:
        try:
            if state_path is not None:
                _persist_state(state_path, namespace)
        finally:
            sys.argv = old_argv
            if old_main is not None:
                sys.modules["__main__"] = old_main
            else:
                sys.modules.pop("__main__", None)


if __name__ == "__main__":
    raise SystemExit(main())