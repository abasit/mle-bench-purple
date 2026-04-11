"""UCB-over-branches selector with debug-first logic.

A "branch" is the lineage rooted at a single draft node. At each step we:

1. Pick a debug target if any buggy node still has spare budget (debug-first).
2. Otherwise, pick the branch with the highest UCB score and improve its best
   current valid descendant.

UCB score per branch:
    mean_reward + c * sqrt( ln(total_plays) / branch_plays )

mean_reward is the best valid cv_score in the branch, normalized so the best
branch is 1 and the worst is 0 (ties = 1). branch_plays is the number of
improve nodes rooted in that branch so far. Early in the search every branch
has zero plays, so exploration dominates; late in the search the bonus
collapses and the selector exploits the best branch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .journal import Journal
from .node import SearchNode

logger = logging.getLogger("purple_next")


@dataclass
class NextAction:
    kind: str              # "debug" | "improve"
    parent: SearchNode


class Selector:
    def __init__(self, *, max_debug_attempts_per_node: int, explore_c: float = 1.0):
        self.max_debug_attempts = int(max_debug_attempts_per_node)
        self.c = float(explore_c)

    def pick(self, journal: Journal, *, excluded: set[str] | None = None, maximize: bool = True) -> NextAction | None:
        excluded = excluded or set()

        debug_target = self._pick_debug(journal, excluded)
        if debug_target is not None:
            debug_target.debug_attempts += 1
            logger.info(
                f"[selector] debug pick: {debug_target.id} "
                f"(attempts={debug_target.debug_attempts}/{self.max_debug_attempts})"
            )
            return NextAction(kind="debug", parent=debug_target)

        improve_target = self._pick_improve_ucb(journal, excluded, maximize)
        if improve_target is not None:
            return NextAction(kind="improve", parent=improve_target)
        return None

    # ── debug ────────────────────────────────────────────────────────────

    def _pick_debug(self, journal: Journal, excluded: set[str]) -> SearchNode | None:
        for node in reversed(list(journal)):
            if not node.is_buggy:
                continue
            if node.id in excluded:
                continue
            if node.debug_attempts >= self.max_debug_attempts:
                continue
            root = node.branch_root_id or node.id
            if self._branch_has_valid(journal, root):
                # Once a branch has a valid candidate, chase improvement not repair.
                continue
            return node
        return None

    @staticmethod
    def _branch_has_valid(journal: Journal, root: str) -> bool:
        for n in journal:
            if (n.branch_root_id or n.id) == root and n.is_valid:
                return True
        return False

    # ── improve (UCB) ────────────────────────────────────────────────────

    def _pick_improve_ucb(
        self,
        journal: Journal,
        excluded: set[str],
        maximize: bool,
    ) -> SearchNode | None:
        branches = journal.branches()
        # Only branches that contain at least one valid node can be improved.
        candidates: list[tuple[str, SearchNode, int, float]] = []
        for root_id, nodes in branches.items():
            valid = [n for n in nodes if n.is_valid and n.id not in excluded]
            if not valid:
                continue
            best = max(
                valid,
                key=lambda n: (n.cv_score if maximize else -n.cv_score) if n.cv_score is not None else float("-inf"),
            )
            plays = sum(1 for n in nodes if n.stage == "improve")
            cv = best.cv_score if best.cv_score is not None else 0.0
            candidates.append((root_id, best, plays, cv if maximize else -cv))

        if not candidates:
            return None

        # Normalize cv scores to [0, 1] across branches for the exploitation term.
        scores = [c[3] for c in candidates]
        lo, hi = min(scores), max(scores)
        spread = hi - lo if hi > lo else 1.0
        normed = [(s - lo) / spread for s in scores]

        total_plays = sum(c[2] for c in candidates) or 1
        best_idx = -1
        best_ucb = float("-inf")
        for i, (root_id, node, plays, _raw) in enumerate(candidates):
            mean_reward = normed[i]
            explore = self.c * math.sqrt(math.log(total_plays + 1) / (plays + 1))
            ucb = mean_reward + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        root_id, chosen, plays, _ = candidates[best_idx]
        logger.info(
            f"[selector] improve pick (UCB): node={chosen.id} branch={root_id} "
            f"plays={plays} ucb={best_ucb:.3f}"
        )
        return chosen
