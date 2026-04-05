"""Node evaluation: backpropagate, check_improvement, get_node_reward."""

import logging
import math
import time
import random

from .search_node import SearchNode

logger = logging.getLogger("MLEvolve")


def backpropagate(node: SearchNode, value: float, add_to_tree=True):
    """Propagate reward up the tree; update debug_success, continue_improve, lock."""
    logger.info(f"[backprop] node {node.id}, reward={value}")
    while node is not None:
        if node.parent and node.is_buggy is False and node.parent.is_buggy is True:
            node.parent.is_debug_success = True
        elif node.parent and node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
            node.parent.is_debug_success = True
        if node.parent and node.parent.stage != "root":
            node.parent.continue_improve = node.continue_improve
        if node.stage in ["draft", "fusion_draft"] and node.lock:
            node.lock = False
        if node.improve_failure_depth > 0:
            node.improve_failure_depth = 0
        node.update(value, add_to_tree)
        node = node.parent


def _propagate_reward_up(node: SearchNode, value: float):
    """Propagate a reward value up a node's parent chain (visits + total_reward only)."""
    current = node
    while current is not None:
        current.update(value, add=True)
        current = current.parent


def graph_backpropagate(agent, node: SearchNode, value: float, add_to_tree=True):
    """GBOP-inspired backpropagation: tree backprop + asymmetric cross-branch penalty sharing.

    Improvement over vanilla GBOP: only share *negative* rewards cross-branch,
    weighted by Jaccard similarity. This prevents reward contamination (a strong
    node in branch A inflating a weak equivalent in branch B) while still teaching
    branches to avoid approaches that failed elsewhere.
    """
    # Step 1: Normal tree backpropagation
    backpropagate(node, value, add_to_tree)

    # Step 2: Asymmetric, similarity-weighted penalty sharing
    # Positive rewards stay local — each branch earns its own wins.
    # Penalties are shared so branches learn what doesn't work cross-branch.
    if value >= 0 or not hasattr(agent, 'similarity_registry'):
        return

    targets = agent.similarity_registry.get_weighted_penalty_targets(node.id, agent.journal)
    if not targets:
        return

    for eq_node, sim in targets:
        weighted_penalty = value * sim  # negative * [0,1] → still negative, scaled by similarity
        logger.info(
            f"[graph-backprop] penalty {node.id[:8]} → {eq_node.id[:8]} "
            f"(branch {node.branch_id}→{eq_node.branch_id}, "
            f"sim={sim:.2f}, penalty={weighted_penalty:.3f})"
        )
        _propagate_reward_up(eq_node, weighted_penalty)


def _update_beta(agent, node: SearchNode):
    """Wire the dormant Bayesian fields: update Beta(alpha, beta) after evaluation.

    success = metric improved vs parent (or first node in branch).
    Also propagates the outcome to sufficiently similar nodes in other branches
    so Thompson Sampling at the draft level benefits from cross-branch signal.
    """
    if node.is_buggy is True or node.is_buggy is None:
        success = False
    elif node.metric is None or node.metric.value is None:
        success = False
    else:
        parent = node.parent
        if (parent and parent.metric and parent.metric.value is not None
                and not parent.is_buggy):
            improvement = (
                node.metric.value - parent.metric.value
                if agent.metric_maximize
                else parent.metric.value - node.metric.value
            )
            success = improvement > agent.scfg.metric_improvement_threshold
        else:
            success = True  # first evaluated node in branch, treat as success

    node.update_beta(success)
    logger.debug(f"[beta] node {node.id[:8]} success={success} → α={node.alpha} β={node.beta}")

    # Propagate outcome to similar nodes in other branches (threshold: sim >= 0.5)
    if hasattr(agent, 'similarity_registry'):
        for eq_node, sim in agent.similarity_registry.get_weighted_penalty_targets(
            node.id, agent.journal
        ):
            if sim >= 0.5:
                eq_node.update_beta(success)
                logger.debug(
                    f"[beta] cross-branch {node.id[:8]}→{eq_node.id[:8]} "
                    f"success={success} sim={sim:.2f}"
                )


def get_node_reward(agent, node: SearchNode):
    """Compute MCTS reward for a node.

    Scale: [-1, 2.0]
    ─────────────────────────────────────────────
    Bug / no metric          → -1.0  (failure)
    Clean run, no improvement→  1.0  (baseline success)
    Debug success (was buggy)→ +0.3  (small bonus – debugging is less
                                       valuable than a genuine improvement)
    Beat global best         → +0.0 … +0.7 magnitude bonus
                                (proportional to relative improvement,
                                 saturates at ~20% relative gain → +0.7)
    ─────────────────────────────────────────────
    Motivation for changes vs. original {-1, +1, +1.5}:
    • Old code gave debug-success the SAME reward (+1.5) as beating the
      global best.  This over-rewards debugging and biases UCT toward
      continuing broken branches rather than exploring new ones.
    • Magnitude-aware bonus lets UCT distinguish a tiny improvement
      (+0.001) from a large one (+10%), making backpropagation more
      informative.
    """
    if node.is_buggy is True or node.is_buggy is None:
        return -1.0
    if node.metric.value is None:
        return -1.0

    # Base reward for a clean, metric-producing run
    reward = 1.0

    # ── Global-improvement bonus (magnitude-aware) ──────────────────────
    if node.metric.value is not None and agent.best_metric is not None:
        improvement = (
            node.metric.value - agent.best_metric
            if node.metric.maximize
            else agent.best_metric - node.metric.value
        )
        if improvement > 0:
            logger.info(f"Node {node.id} is better than the best node {agent.best_node.id} now!")
            # Relative improvement capped at 20% → full +0.7 bonus.
            # Uses log1p so even tiny gains get a positive signal.
            scale = max(abs(agent.best_metric), 1e-6)
            relative = improvement / scale
            magnitude_bonus = min(0.7, 0.7 * (1 - math.exp(-relative / 0.2)))
            reward += magnitude_bonus

    # ── Debug-success bonus (was buggy, now fixed) ───────────────────────
    # Smaller than the improvement bonus: fixing a bug is necessary but
    # much less valuable than actually beating the best metric.
    if node.parent and node.parent.stage != "root":
        if node.parent.is_buggy is True:
            reward += 0.3
        # (no additional bonus for non-buggy parent — base 1.0 covers it)

    return min(reward, 2.0)  # hard cap to keep UCT numerics stable


def check_improvement(agent, cur_node: SearchNode, parent_node: SearchNode):

    improvement = 0
    should_backpropagate = False

    if (agent.search_start_time and
        cur_node.stage != "root" and
        cur_node.branch_id is not None):

        time_elapsed = time.time() - agent.search_start_time
        time_progress = time_elapsed / agent.acfg.time_limit

        if not hasattr(agent, 'branch_node_count'):
            agent.branch_node_count = {}

        branch_id = cur_node.branch_id
        agent.branch_node_count[branch_id] = agent.branch_node_count.get(branch_id, 0) + 1
        current_count = agent.branch_node_count[branch_id]

        force_backprop = False

        scfg = agent.scfg

        if time_progress >= scfg.force_backprop_late_threshold:
            if random.random() < scfg.force_backprop_late_prob:
                force_backprop = True
                logger.info(f"[Force Backprop] Late stage ({time_progress:.1%}), "
                        f"node {cur_node.id} (stage={cur_node.stage}, branch={branch_id}, #{current_count})")

        elif time_progress >= scfg.force_backprop_mid_threshold and current_count % scfg.force_backprop_mid_modulo == 0:
            force_backprop = True
            logger.info(f"[Force Backprop] Mid stage ({time_progress:.1%}), "
                       f"branch {branch_id} node #{current_count}, "
                       f"node {cur_node.id} (stage={cur_node.stage})")

        if force_backprop:
            skip_force_backprop = False

            if (not cur_node.is_buggy and
                cur_node.metric is not None and
                cur_node.metric.value is not None):

                recent_window = scfg.recent_best_window
                recent_nodes = [n for n in agent.journal[-recent_window:]
                               if (not n.is_buggy and n.metric and n.metric.value is not None)]

                if recent_nodes:
                    if cur_node.metric.maximize:
                        recent_best = max(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value >= recent_best.metric.value
                    else:
                        recent_best = min(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value <= recent_best.metric.value

                    if is_recent_best:
                        logger.info(f"[Smart Backprop] Node {cur_node.id} is recent best "
                                  f"(metric={cur_node.metric.value:.4f}), skip force backprop to continue improvement chain")
                        skip_force_backprop = True

            if not skip_force_backprop:
                if (not cur_node.is_buggy and
                    cur_node.metric is not None and
                    cur_node.metric.value is not None):

                    local_best = cur_node.local_best_node
                    if local_best and local_best.metric and local_best.metric.value is not None:
                        if agent.metric_maximize:
                            is_better = cur_node.metric.value > local_best.metric.value
                        else:
                            is_better = cur_node.metric.value < local_best.metric.value

                        if is_better:
                            cur_node.local_best_node = cur_node
                            logger.info(f"  └─ Updated local_best: {cur_node.metric.value:.4f} "
                                      f"(prev: {local_best.metric.value:.4f})")
                    else:
                        cur_node.local_best_node = cur_node
                        logger.info(f"  └─ Set as local_best: {cur_node.metric.value:.4f}")

                _update_beta(agent, cur_node)
                reward = get_node_reward(agent, cur_node)
                graph_backpropagate(agent, cur_node, reward)
                return True

    local_best_node = cur_node.local_best_node
    local_best_metric = local_best_node.metric.value

    if cur_node.is_buggy is False:
        new_metric = cur_node.metric.value
        if parent_node.is_buggy:
            logger.info(f"[eval] debug success for {parent_node.id}")
            if new_metric:
                if local_best_metric:
                    debug_improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
                    if debug_improvement > 0:
                        cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
                else:
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
            else:
                should_backpropagate = True

        if new_metric is not None and local_best_metric is not None:
            improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
            if improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < agent.scfg.max_improve_failure:
                local_best_node.improve_failure_depth += 1
                action = "continue"
                cur_node.continue_improve = True
            elif improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth >= agent.scfg.max_improve_failure:
                action = "terminal"
                cur_node.continue_improve = False
                should_backpropagate = True
                cur_node.is_terminal = True
            else:
                action = "continue"
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement={improvement:.6f}, action={action}")
        elif new_metric is not None:
            cur_node.local_best_node = cur_node
            cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=continue")
        else:
            should_backpropagate = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
    elif cur_node.is_buggy is None:
        logger.warning(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
        should_backpropagate = True
    else:
        if cur_node.debug_depth >= agent.scfg.back_debug_depth:
            should_backpropagate = True
            if cur_node.debug_depth >= agent.scfg.max_debug_depth:
                cur_node.is_terminal = True

    _update_beta(agent, cur_node)
    if should_backpropagate:
        reward = get_node_reward(agent, cur_node)
        graph_backpropagate(agent, cur_node, reward)
    else:
        agent.current_node_list.append(cur_node)
    return should_backpropagate
