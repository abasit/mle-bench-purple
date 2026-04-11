"""Append-only node journal with cheap queries."""

from __future__ import annotations

import logging
import threading
from typing import Iterator

from .node import SearchNode

logger = logging.getLogger("purple_next")


class Journal:
    def __init__(self) -> None:
        self._nodes: list[SearchNode] = []
        self._by_id: dict[str, SearchNode] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[SearchNode]:
        return iter(list(self._nodes))

    def next_id(self, stage: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{stage[:1]}{self._counter:03d}"

    def add(self, node: SearchNode) -> None:
        with self._lock:
            self._nodes.append(node)
            self._by_id[node.id] = node
            logger.info(f"[journal] +{node.short()} total={len(self._nodes)}")

    def get(self, node_id: str) -> SearchNode | None:
        return self._by_id.get(node_id)

    def parent_of(self, node: SearchNode) -> SearchNode | None:
        if node.parent_id is None:
            return None
        return self._by_id.get(node.parent_id)

    def all_valid(self) -> list[SearchNode]:
        return [n for n in self._nodes if n.is_valid]

    def all_buggy(self) -> list[SearchNode]:
        return [n for n in self._nodes if n.is_buggy]

    def branches(self) -> dict[str, list[SearchNode]]:
        out: dict[str, list[SearchNode]] = {}
        for n in self._nodes:
            root = n.branch_root_id or n.id
            out.setdefault(root, []).append(n)
        return out

    def best(self, *, maximize: bool = True) -> SearchNode | None:
        valid = self.all_valid()
        if not valid:
            return None

        def key(n: SearchNode) -> float:
            score = n.cv_score if n.cv_score is not None else float("-inf")
            return score if maximize else -score

        non_suspicious = [n for n in valid if not n.is_suspicious and n.review_verdict not in {"leaky"}]
        pool = non_suspicious or valid
        return max(pool, key=key)

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self._nodes),
            "valid": sum(1 for n in self._nodes if n.is_valid),
            "buggy": sum(1 for n in self._nodes if n.is_buggy),
            "drafts": sum(1 for n in self._nodes if n.stage == "draft"),
            "improves": sum(1 for n in self._nodes if n.stage == "improve"),
            "debugs": sum(1 for n in self._nodes if n.stage == "debug"),
        }

    def snapshot(self) -> list[dict]:
        rows: list[dict] = []
        for n in self._nodes:
            rows.append({
                "id": n.id,
                "stage": n.stage,
                "parent_id": n.parent_id,
                "branch_root_id": n.branch_root_id,
                "cv_score": n.cv_score,
                "holdout_score": n.holdout_score,
                "maximize": n.maximize,
                "is_buggy": n.is_buggy,
                "is_suspicious": n.is_suspicious,
                "review_verdict": n.review_verdict,
                "review_reasons": list(n.review_reasons),
                "suspicion_reasons": list(n.suspicion_reasons),
                "improve_hint_index": n.improve_hint_index,
                "notes": n.notes,
                "created_at": n.created_at,
            })
        return rows
