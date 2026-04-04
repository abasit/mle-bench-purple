"""Node selection: UCT select, get_exploration_weight, get_top_k_nodes_global, select_from_top_k_weighted, select_with_soft_switch."""

import logging
import math
import random
import time
from typing import List

from .search_node import SearchNode
from .conditions import should_trigger_branch_fusion
logger = logging.getLogger("MLEvolve")


def _piecewise_decay(t, initial_C=1.414, T1=100, T2=200, alpha=0.01, lower_bound=0.7):
    """Piecewise decay: initial_C until T1, linear to lower_bound by T2, then lower_bound."""
    if t < T1:
        return initial_C
    elif T1 <= t <= T2:
        return max(initial_C - alpha * (t - T1), lower_bound)
    else:
        return lower_bound


def _compute_exploration_constant(agent):
    """Compute exploration constant C from search progress (piecewise decay)."""
    dcfg = agent.cfg.agent.decay
    n1 = agent.scfg.num_drafts * (agent.scfg.num_improves ** 2)
    n2 = round(agent.acfg.steps * dcfg.phase_ratios[0])
    t1 = min(n1, n2)
    t2 = round(agent.acfg.steps * dcfg.phase_ratios[1])
    return _piecewise_decay(
        t=agent.current_step,
        initial_C=dcfg.exploration_constant,
        T1=t1,
        T2=t2,
        alpha=dcfg.alpha,
        lower_bound=dcfg.lower_bound,
    )


def _class_uct_value(child: SearchNode, C: float, agent) -> float:
    """UCT using aggregated visits/rewards across equivalence class.

    Per GBOP (Leurent & Maillard 2020), merging equivalent states gives
    tighter value bounds by pooling observations.
    """
    if not hasattr(agent, 'similarity_registry'):
        return child.uct_value(exploration_constant=C)

    class_visits, class_reward = agent.similarity_registry.get_class_stats(
        child.id, agent.journal
    )
    if class_visits == 0:
        return float('inf')
    exploitation = class_reward / class_visits
    parent_visits = child.parent.visits if child.parent else 1
    if parent_visits <= 0:
        parent_visits = 1
    exploration = C * math.sqrt(math.log(parent_visits) / class_visits)
    return exploitation + exploration


def _is_equivalent_expanded(agent, node: SearchNode) -> bool:
    """Check if any equivalent node in another branch already has children."""
    if not hasattr(agent, 'similarity_registry'):
        return False
    eq_ids = agent.similarity_registry.get_equivalence_class(node.id)
    if len(eq_ids) <= 1:
        return False
    id2node = {n.id: n for n in agent.journal.nodes}
    for eq_id in eq_ids:
        if eq_id == node.id:
            continue
        eq_node = id2node.get(eq_id)
        if eq_node and eq_node.branch_id != node.branch_id and len(eq_node.children) > 0:
            return True
    return False


def select(agent, node: SearchNode):
    """UCT selection: recurse from node, return node to expand (root lock for drafts)."""
    def _best_child(n: SearchNode) -> SearchNode:
        C = _compute_exploration_constant(agent)
        if agent.is_root(n):
            filtered_children = [child for child in n.children if not child.lock]
            selected_node = n
            if len(filtered_children) > 0:
                selected_node = max(filtered_children,
                                    key=lambda child: _class_uct_value(child, C, agent))
            if selected_node.stage in ["draft", "fusion_draft"]:
                selected_node.lock = True
            return selected_node
        else:
            return max(n.children, key=lambda child: _class_uct_value(child, C, agent))

    while node and not node.is_terminal:
        if not node.reached_child_limit(scfg=agent.scfg):
            if node.is_buggy and node.is_debug_success is True:
                node = _best_child(node)
            elif node.continue_improve and len(node.children) > 0:
                node = _best_child(node)
            else:
                # Skip if an equivalent node in another branch is already expanded
                if _is_equivalent_expanded(agent, node):
                    logger.info(
                        f"[select] Skipping {node.id[:8]}: equivalent node "
                        f"already expanded in another branch"
                    )
                    node.is_terminal = True
                    return select(agent, agent.virtual_root)
                logger.info(f"[select] → node {node.id} (method=expand)")
                return node
        else:
            if agent.is_root(node) and should_trigger_branch_fusion(agent) and random.random() < agent.acfg.branch_fusion_trigger_prob:
                logger.info(f"Root node {node.id} is fully expanded for regular drafts, aggregation conditions met (including probability), returning root")
                return node
            node = _best_child(node)
    logger.info(f"[select] → node {node.id} (method=uct)")
    return node


def get_exploration_weight(time_elapsed: float, total_time: float,
                           switch_start: float = 0.5,
                           switch_end: float = 0.7,
                           min_weight: float = 0.2) -> float:
    """Exploration weight: 1.0 until switch_start, linear decay to min_weight by switch_end."""
    time_progress = time_elapsed / total_time

    if time_progress < switch_start:
        return 1.0
    elif time_progress < switch_end:
        decay_progress = (time_progress - switch_start) / (switch_end - switch_start)
        return 1.0 - (1.0 - min_weight) * decay_progress
    else:
        return min_weight


def get_top_k_nodes_global(agent, k: int, max_from_same_branch: int) -> List[dict]:
    """Select top-k nodes globally with branch diversity (recomputed each call). Returns list of {node, branch_id, metric, rank}."""
    all_nodes = []
    for branch_id in agent.branch_all_nodes:
        for node in agent.branch_all_nodes[branch_id]:
            if not node.is_buggy and node.metric is not None and node.metric.value is not None:
                all_nodes.append(node)

    if not all_nodes:
        logger.warning("No valid nodes found for Top-K selection")
        return []

    maximize = agent.metric_maximize
    all_nodes.sort(
        key=lambda n: n.metric.value,
        reverse=maximize
    )

    logger.info(f"Total valid nodes: {len(all_nodes)}, requesting Top-{k}")

    selected = []
    branch_count = {}

    for node in all_nodes:
        if len(selected) >= k:
            break

        branch_id = node.branch_id
        current_count = branch_count.get(branch_id, 0)

        if current_count >= max_from_same_branch:
            logger.debug(f"Branch {branch_id} reached limit ({max_from_same_branch}), skipping node with metric={node.metric.value:.4f}")
            continue

        selected.append({
            'node': node,
            'branch_id': branch_id,
            'metric': node.metric.value,
            'rank': len(selected) + 1
        })
        branch_count[branch_id] = current_count + 1

    if selected:
        branch_distribution = {}
        for item in selected:
            bid = item['branch_id']
            branch_distribution[bid] = branch_distribution.get(bid, 0) + 1

        metrics_str = ", ".join([f"Rank{item['rank']}={item['metric']:.4f}(B{item['branch_id']})" for item in selected])
        logger.info(f"📊 Top-{len(selected)} selected: {metrics_str}")
        logger.info(f"📊 Branch distribution: {branch_distribution}")

    return selected


def select_from_top_k_weighted(agent, top_k_nodes: List[dict]) -> SearchNode:
    """Weighted random choice from top-k nodes (weight = 1/rank)."""
    if not top_k_nodes:
        return select(agent, agent.virtual_root)

    weights = [1.0 / item['rank'] for item in top_k_nodes]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    selected = random.choices(top_k_nodes, weights=probabilities)[0]

    logger.info(f"🎯 Selected: Rank{selected['rank']} (Branch {selected['branch_id']}, "
                f"metric={selected['metric']:.4f}, prob={probabilities[top_k_nodes.index(selected)]:.1%})")

    return selected['node']


def select_with_soft_switch(agent) -> SearchNode:
    """Soft switch: exploration (UCT) vs exploitation (Top-K) by time progress."""
    if agent.search_start_time is None:
        logger.info("📊 Search not started yet, using standard UCT")
        return select(agent, agent.virtual_root)

    time_elapsed = time.time() - agent.search_start_time
    total_time = agent.acfg.time_limit
    time_progress = time_elapsed / total_time

    scfg = agent.scfg

    exploration_weight = get_exploration_weight(
        time_elapsed, total_time,
        switch_start=scfg.explore_switch_start,
        switch_end=scfg.explore_switch_end,
        min_weight=scfg.min_exploration_weight,
    )

    if random.random() < exploration_weight:
        logger.info(f"📊 Exploration mode (weight={exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")
        return select(agent, agent.virtual_root)

    else:
        # Top-K exploitation
        logger.info(f"🎯 Exploitation mode (weight={1-exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")

        if time_progress < scfg.explore_switch_end:
            k = scfg.topk_early_k
            max_from_same_branch = scfg.topk_early_max_per_branch
            phase = f"early-mid (<{scfg.explore_switch_end:.0%})"
        else:
            k = scfg.topk_late_k
            max_from_same_branch = scfg.topk_late_max_per_branch
            phase = f"late (>={scfg.explore_switch_end:.0%})"

        logger.info(f"📊 Phase: {phase}, requesting Top-{k} (max {max_from_same_branch} per branch)")

        top_k_nodes = get_top_k_nodes_global(
            agent,
            k=k,
            max_from_same_branch=max_from_same_branch
        )

        if not top_k_nodes:
            logger.warning("No valid Top-K nodes found, fallback to standard UCT")
            return select(agent, agent.virtual_root)

        available_nodes = [
            item for item in top_k_nodes
            if not item['node'].reached_child_limit(agent.scfg, for_topk=True)
        ]

        if available_nodes:
            selected_node = select_from_top_k_weighted(agent, available_nodes)
            logger.info(f"✅ Selected unexpanded Top-K node {selected_node.id} (from {len(available_nodes)}/{len(top_k_nodes)} available)")
            selected_node._topk_triggered = True
            return selected_node
        else:
            logger.info(f"⚠️ All Top-{len(top_k_nodes)} nodes fully expanded, will apply UCT from selected node")
            selected_node = select_from_top_k_weighted(agent, top_k_nodes)
            logger.info(f"Selected fully expanded node {selected_node.id}, applying UCT from it")
            uct_node = select(agent, selected_node)
            uct_node._topk_triggered = True
            return uct_node
