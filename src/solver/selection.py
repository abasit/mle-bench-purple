"""Next-action policy.

Priority order:
    1. Debug a buggy node that still has spare debug budget AND whose lineage
       hasn't accumulated too many failed debug descendants.
    2. Improve a node from the validated set, with **branch-fair rotation
       and novelty pressure**: every distinct branch (lineage from a draft
       root) gets attention in proportion to its inverse improve count, so
       a strong branch can't monopolise the search budget. Within that, the
       branch with the most untried strategies (novelty headroom) wins ties.
    3. Return None — nothing useful left to do.

Backtracking: when a branch has plateaued (consecutive improves with no
score gain), the selector may pick an earlier, simpler ancestor as the
parent instead of the current best. This allows exploring a different
strategy direction from a less overengineered starting point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import SolverConfig
from .nodes import Journal, SearchNode

logger = logging.getLogger("solver")


# A buggy lineage that has produced this many failed debug attempts (across
# all descendants of the original draft) is considered hopeless and skipped.
_MAX_FAILED_DEBUGS_PER_LINEAGE = 4

# After this many consecutive improves with no score gain, backtrack to an
# earlier ancestor instead of continuing from the current best.
_PLATEAU_THRESHOLD = 3


@dataclass
class NextAction:
    kind: str        # "debug" | "improve"
    parent: SearchNode


class Selector:
    """Stateful next-action picker.

    Tracks per-branch improve counts so the rotation across branches is fair
    even when one branch dominates the val score. The branch with the most
    untried strategies wins ties, then highest val score breaks remaining ties.
    """

    def __init__(
        self,
        cfg: SolverConfig,
        task_type: str = "tabular",
        objective: str = "",
        metric_name: str = "",
    ):
        self.cfg = cfg
        self.task_type = task_type
        self.objective = objective
        self.metric_name = metric_name
        self._improve_cursor = 0
        # branch_root_id -> number of improves spawned from this branch.
        self._branch_improve_counts: dict[str, int] = {}

    def pick(
        self,
        journal: Journal,
        excluded_ids: set[str] | None = None,
    ) -> NextAction | None:
        """Pick the next action.

        Args:
            journal: the full journal of nodes so far.
            excluded_ids: parent ids that are currently being acted on by other
                workers. The selector will not return any of these.

        IMPORTANT: this method has the side effect of incrementing
        ``target.debug_attempts`` for the chosen debug target. Callers MUST
        actually use the returned action; if they discard it, the increment
        cannot be cleanly reversed without race conditions, so we just don't
        offer that option.
        """
        excluded = excluded_ids or set()

        # 1. Debug-first.
        debug_target = self._pick_debug_target(journal, excluded)
        if debug_target is not None:
            debug_target.debug_attempts += 1
            logger.info(
                f"[selector] debug pick: {debug_target.id} "
                f"(now attempts={debug_target.debug_attempts}/{self.cfg.search.max_debug_attempts_per_node})"
            )
            return NextAction(kind="debug", parent=debug_target)

        # 2. Improve a top-K validated node.
        improve_target = self._pick_improve_target(journal, excluded)
        if improve_target is not None:
            return NextAction(kind="improve", parent=improve_target)

        return None

    # ── debug ─────────────────────────────────────────────────────────────

    def _pick_debug_target(
        self,
        journal: Journal,
        excluded: set[str],
    ) -> SearchNode | None:
        """Pick the most recent buggy node worth fixing.

        Filters applied (in order):
            (a) node must be buggy
            (b) node id must not be in ``excluded`` (already in flight)
            (c) node's own debug_attempts < max_attempts
            (d) lineage's failed-debug count < lineage cap
            (e) lineage hasn't already produced a valid descendant
        """
        max_attempts = self.cfg.search.max_debug_attempts_per_node
        lineage_failed: dict[str, int] = self._failed_debug_counts_by_lineage(journal)

        for node in reversed(list(journal)):
            if not node.is_buggy:
                continue
            if node.id in excluded:
                continue
            if node.debug_attempts >= max_attempts:
                continue

            root = self._root_of(journal, node)
            if root is None:
                continue
            if lineage_failed.get(root.id, 0) >= _MAX_FAILED_DEBUGS_PER_LINEAGE:
                logger.debug(
                    f"[selector] skipping {node.id}: lineage {root.id} has "
                    f"{lineage_failed.get(root.id, 0)} failed debugs (cap "
                    f"{_MAX_FAILED_DEBUGS_PER_LINEAGE})"
                )
                continue
            if self._has_successful_descendant(journal, node):
                continue

            return node

        return None

    @staticmethod
    def _failed_debug_counts_by_lineage(journal: Journal) -> dict[str, int]:
        """For each root draft, count buggy debug-stage descendants."""
        all_nodes = list(journal)
        by_id = {n.id: n for n in all_nodes}

        # Resolve each node's root by walking up parent_ids. Same fall-back
        # logic as ``_root_of``: if the parent isn't in the journal, treat the
        # current node as the root rather than emitting None.
        root_of: dict[str, str] = {}
        for n in all_nodes:
            cur = n
            seen: set[str] = set()
            while cur.parent_id is not None:
                if cur.id in seen:
                    break
                seen.add(cur.id)
                parent = by_id.get(cur.parent_id)
                if parent is None:
                    break
                cur = parent
            root_of[n.id] = cur.id

        counts: dict[str, int] = {}
        for n in all_nodes:
            if n.is_buggy and n.stage == "debug":
                root = root_of.get(n.id, n.id)
                counts[root] = counts.get(root, 0) + 1
        return counts

    @staticmethod
    def _root_of(journal: Journal, node: SearchNode) -> SearchNode | None:
        """Walk up parent_id pointers to find the original draft.

        Returns the last non-None ancestor — even if a parent_id points to
        a node that's missing from the journal (which can happen during
        partial recovery), we treat the deepest known ancestor as the root
        rather than collapsing to None.
        """
        seen: set[str] = set()
        cur: SearchNode = node
        while cur.parent_id is not None:
            if cur.id in seen:
                return cur  # cycle defence — return the last seen
            seen.add(cur.id)
            parent = journal.parent_of(cur)
            if parent is None:
                # Parent isn't in the journal — treat current as the root.
                return cur
            cur = parent
        return cur

    @staticmethod
    def _has_successful_descendant(journal: Journal, ancestor: SearchNode) -> bool:
        """Walk forward through children to see if any descendant is valid."""
        targets = {ancestor.id}
        for n in journal:
            if n.parent_id in targets:
                if n.is_valid:
                    return True
                targets.add(n.id)
        return False

    # ── improve ───────────────────────────────────────────────────────────

    def _pick_improve_target(
        self,
        journal: Journal,
        excluded: set[str],
    ) -> SearchNode | None:
        """Branch-fair improve selection with novelty pressure and backtracking.

        Algorithm:
            1. Group all valid (non-suspicious if possible) nodes by their
               ``branch_root_id``.
            2. For each branch, pick its best validated node as the candidate.
               If the branch has *plateaued* (several consecutive improves with
               no score gain), backtrack to an earlier ancestor instead.
            3. Filter out branches whose candidate is in ``excluded``
               (already being acted on by another worker).
            4. Sort branches by:
                 (a) ascending improve count  → fairness across branches
                 (b) descending novelty headroom → prefer branches with
                     unexplored strategies
                 (c) descending val score      → break ties by quality
            5. Pick the best branch's candidate, increment its improve count.
        """
        # Late import — strategies imports nodes for typing only.
        from .strategies import branch_distinct_strategies, untried_strategies

        valid = journal.all_valid()
        if not valid:
            return None

        # Group by branch.
        by_branch: dict[str, list[SearchNode]] = {}
        for n in valid:
            root = getattr(n, "branch_root_id", None) or n.id
            by_branch.setdefault(root, []).append(n)

        if not by_branch:
            return None

        # Determine maximize direction once for all branches.
        maximize = journal.maximize

        # Build (root, best_node, improve_count, novelty_headroom, val) tuples.
        candidates: list[tuple[str, SearchNode, int, int, float]] = []
        for root, nodes in by_branch.items():
            # Check if this branch has plateaued; if so, backtrack.
            backtrack_target = self._pick_backtrack_target(
                journal, root, nodes, maximize, excluded,
            )
            if backtrack_target is not None:
                logger.info(
                    f"[selector] backtracking branch {root}: "
                    f"picking earlier node {backtrack_target.id} "
                    f"(val={backtrack_target.val_score}) instead of branch best"
                )
                candidate = backtrack_target
            else:
                candidate = self._best_in_branch(
                    nodes,
                    maximize,
                    eps=self.cfg.search.metric_improve_eps,
                )
            if candidate is None or candidate.id in excluded:
                continue
            improve_count = self._branch_improve_counts.get(root, 0)
            tried = branch_distinct_strategies(journal, root)
            headroom = len(
                untried_strategies(
                    tried,
                    task_type=self.task_type,
                    objective=self.objective,
                    metric_name=self.metric_name,
                )
            )
            val = candidate.val_score if candidate.val_score is not None else float("-inf")
            candidates.append((root, candidate, improve_count, headroom, val))

        if not candidates:
            return None

        # Sort: fewest improves first, then highest novelty headroom, then best val.
        # Negate val for ascending sort when maximize=False.
        val_sign = -1.0 if maximize else 1.0
        candidates.sort(key=lambda c: (c[2], -c[3], val_sign * c[4]))

        chosen_root, chosen_node, _, _, _ = candidates[0]
        self._branch_improve_counts[chosen_root] = (
            self._branch_improve_counts.get(chosen_root, 0) + 1
        )
        logger.info(
            f"[selector] improve pick: branch={chosen_root} node={chosen_node.id} "
            f"(branch improves now={self._branch_improve_counts[chosen_root]})"
        )
        return chosen_node

    def _pick_backtrack_target(
        self,
        journal: Journal,
        branch_root: str,
        valid_nodes: list[SearchNode],
        maximize: bool,
        excluded: set[str],
    ) -> SearchNode | None:
        """If a branch has plateaued, pick an earlier simpler ancestor.

        A branch has plateaued when the last ``_PLATEAU_THRESHOLD`` consecutive
        improve-stage nodes produced no score improvement over the best score
        seen *before* them in the branch. When this happens, we walk backward
        through the branch's valid nodes and pick the earliest one whose score
        is within 5% of the best — a simpler, less overengineered starting
        point for a different strategy direction.

        Returns None if the branch hasn't plateaued or no suitable ancestor
        exists.
        """
        from .strategies import build_branch_history

        history = build_branch_history(journal, branch_root)
        # Need enough improve-stage entries to judge a plateau.
        improve_rows = [r for r in history if r.stage == "improve" and not r.is_buggy]
        if len(improve_rows) < _PLATEAU_THRESHOLD:
            return None

        # Check if the last N improves all had non-positive delta.
        recent = improve_rows[-_PLATEAU_THRESHOLD:]
        if any(r.delta is not None and r.delta > 1e-6 for r in recent):
            return None  # at least one recent improve made progress

        # Branch has plateaued. Find an earlier valid node to backtrack to.
        # Sort valid nodes by creation time.
        sorted_valid = sorted(valid_nodes, key=lambda n: n.created_at)
        if len(sorted_valid) <= 1:
            return None

        best = self._best_in_branch(valid_nodes, maximize)
        if best is None or best.val_score is None:
            return None

        # Walk from earliest to latest, pick the earliest node that:
        #   - is not the current best
        #   - is not excluded
        #   - has a score within 5% relative of the best
        best_val = best.val_score
        for node in sorted_valid:
            if node.id == best.id or node.id in excluded:
                continue
            if node.val_score is None:
                continue
            denom = max(abs(best_val), 0.01)
            gap = abs(best_val - node.val_score) / denom
            if gap <= 0.05:
                return node

        # No suitable ancestor found — fallback: pick the original draft
        # if it's valid and not excluded.
        for node in sorted_valid:
            if node.stage == "draft" and node.id not in excluded:
                return node

        return None

    @staticmethod
    def _best_in_branch(
        nodes: list[SearchNode],
        maximize: bool,
        eps: float = 0.0,
    ) -> SearchNode | None:
        scored = [n for n in nodes if n.val_score is not None and n.is_valid]
        if not scored:
            return None
        # Prefer non-suspicious if any are present.
        honest = [n for n in scored if not n.is_suspicious]
        pool = honest or scored
        ordered = sorted(
            pool,
            key=lambda n: (n.val_score if maximize else -n.val_score),
            reverse=True,
        )
        best = ordered[0]
        best_val = best.val_score
        if best_val is None or eps <= 0:
            return best
        tied = [
            n
            for n in ordered
            if n.val_score is not None and abs(n.val_score - best_val) <= eps
        ]
        if not tied:
            return best
        return max(tied, key=lambda n: n.created_at)
