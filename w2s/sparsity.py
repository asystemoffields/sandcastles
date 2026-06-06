"""
sparsity.py -- Sparsity analysis, pruning, and structured sparsity detection.

Pruned and quantized models often have significant weight sparsity.
Every zero weight is a multiply-accumulate path eliminated from the
netlist.  This module provides tools to analyze, exploit, and report
sparsity throughout the compilation pipeline.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from w2s.core import ComputeGraph, Operation, OpType


# ---------------------------------------------------------------------------
#  Sparsity report
# ---------------------------------------------------------------------------

@dataclass
class LayerSparsity:
    """Sparsity statistics for a single layer."""
    name: str
    op_type: OpType
    total_weights: int
    zero_weights: int
    sparsity: float                # 0.0 to 1.0
    structured_2_4: bool           # has 2:4 structured sparsity
    structured_n_m: Optional[Tuple[int, int]] = None  # detected N:M pattern


@dataclass
class SparsityReport:
    """Sparsity analysis for an entire compute graph."""
    layers: List[LayerSparsity] = field(default_factory=list)
    total_weights: int = 0
    total_zeros: int = 0
    overall_sparsity: float = 0.0
    eliminated_multipliers: int = 0
    original_multipliers: int = 0

    def __str__(self) -> str:
        lines = []
        lines.append("=== Sparsity Report ===")
        lines.append(f"Overall: {self.overall_sparsity:.1%} sparse "
                     f"({self.total_zeros:,}/{self.total_weights:,} weights are zero)")
        if self.original_multipliers > 0:
            savings = self.eliminated_multipliers / self.original_multipliers
            lines.append(f"Multipliers eliminated: {self.eliminated_multipliers:,} "
                         f"of {self.original_multipliers:,} ({savings:.1%} area savings)")
        lines.append("")

        lines.append(f"  {'Layer':<30} {'Type':<10} {'Params':>8} "
                     f"{'Zeros':>8} {'Sparsity':>9}  Structured")
        lines.append(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*9}  {'-'*12}")

        for ls in self.layers:
            struct = ""
            if ls.structured_2_4:
                struct = "2:4"
            elif ls.structured_n_m:
                struct = f"{ls.structured_n_m[0]}:{ls.structured_n_m[1]}"
            lines.append(
                f"  {ls.name:<30} {ls.op_type.value:<10} {ls.total_weights:>8,} "
                f"{ls.zero_weights:>8,} {ls.sparsity:>8.1%}  {struct}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Analysis
# ---------------------------------------------------------------------------

def analyze_sparsity(graph: ComputeGraph) -> SparsityReport:
    """
    Analyze weight sparsity across all layers in the graph.

    Uses q_weights if available (post-quantization), otherwise float weights.
    """
    report = SparsityReport()

    for op in graph.topological_order():
        weights = op.q_weights if op.q_weights else op.weights
        if not weights:
            continue

        # Only analyze weight matrices (not bias, running stats, etc.)
        weight_keys = _get_weight_keys(op)
        if not weight_keys:
            continue

        total = 0
        zeros = 0
        for key in weight_keys:
            if key not in weights:
                continue
            w = weights[key]
            total += w.size
            zeros += int(np.count_nonzero(w == 0))

        if total == 0:
            continue

        sparsity = zeros / total
        has_2_4 = False
        nm_pattern = None

        # Check for structured sparsity on primary weight matrix
        primary_key = weight_keys[0]
        if primary_key in weights:
            w = weights[primary_key]
            has_2_4 = detect_structured_2_4(w)
            if not has_2_4:
                nm_pattern = detect_structured_nm(w)

        layer_info = LayerSparsity(
            name=op.name,
            op_type=op.op_type,
            total_weights=total,
            zero_weights=zeros,
            sparsity=sparsity,
            structured_2_4=has_2_4,
            structured_n_m=nm_pattern,
        )
        report.layers.append(layer_info)
        report.total_weights += total
        report.total_zeros += zeros

        # Count multiplier savings
        n_muls = _count_multipliers(op, weights)
        n_muls_after = _count_multipliers_sparse(op, weights)
        report.original_multipliers += n_muls
        report.eliminated_multipliers += (n_muls - n_muls_after)

    if report.total_weights > 0:
        report.overall_sparsity = report.total_zeros / report.total_weights

    return report


def _get_weight_keys(op: Operation) -> List[str]:
    """Return the weight tensor keys that contribute to multiplier count."""
    if op.op_type in (OpType.DENSE, OpType.CONV1D, OpType.CONV2D):
        return ['weight']
    if op.op_type == OpType.MULTI_HEAD_ATTENTION:
        return ['q_weight', 'k_weight', 'v_weight', 'out_weight']
    if op.op_type == OpType.GROUPED_QUERY_ATTENTION:
        return ['q_weight', 'k_weight', 'v_weight', 'out_weight']
    if op.op_type == OpType.SWIGLU:
        return ['gate_weight', 'up_weight', 'down_weight']
    if op.op_type == OpType.EMBEDDING:
        return ['weight']
    if op.op_type in (OpType.LAYERNORM, OpType.RMSNORM, OpType.BATCHNORM):
        return ['scale']
    return []


def _count_multipliers(op: Operation, weights: Dict) -> int:
    """Count total multiplications for an op (ignoring sparsity)."""
    if op.op_type == OpType.DENSE:
        w = weights.get('weight')
        return w.size if w is not None else 0
    if op.op_type in (OpType.CONV1D, OpType.CONV2D):
        w = weights.get('weight')
        return w.size if w is not None else 0
    if op.op_type in (OpType.MULTI_HEAD_ATTENTION, OpType.GROUPED_QUERY_ATTENTION):
        total = 0
        for k in ('q_weight', 'k_weight', 'v_weight', 'out_weight'):
            w = weights.get(k)
            if w is not None:
                total += w.size
        return total
    if op.op_type == OpType.SWIGLU:
        total = 0
        for k in ('gate_weight', 'up_weight', 'down_weight'):
            w = weights.get(k)
            if w is not None:
                total += w.size
        return total
    return 0


def _count_multipliers_sparse(op: Operation, weights: Dict) -> int:
    """Count non-zero multiplications (actual hardware after sparsity)."""
    keys = _get_weight_keys(op)
    total = 0
    for k in keys:
        w = weights.get(k)
        if w is not None:
            total += int(np.count_nonzero(w))
    return total


# ---------------------------------------------------------------------------
#  Structured sparsity detection
# ---------------------------------------------------------------------------

def detect_structured_2_4(weights: np.ndarray, threshold: float = 0.99) -> bool:
    """
    Check if a weight matrix actually has 2:4 structured sparsity.

    In 2:4 sparsity, every group of 4 consecutive elements along the last
    axis keeps *exactly* the top-2 (so at most 2 are non-zero) and the
    realized density sits at ~50%.  A loose "at least 90% of groups have
    <=2 non-zeros" test wrongly flags plain *unstructured* high sparsity
    (e.g. a 70% sparse random matrix trivially has <=2 non-zeros in almost
    every group of 4, yet is not 2:4 structured at all).

    To avoid that mislabeling we require BOTH:
      1. (nearly) ALL groups comply with the <=2 budget
         (fraction >= ``threshold``, default 0.99), and
      2. a majority of groups actually *use* the budget, i.e. have exactly
         2 non-zeros — this is the fingerprint of top-2 pruning and is what
         separates true 2:4 from a matrix that is merely very sparse.
    """
    if weights.ndim < 2:
        return False

    flat = weights.reshape(-1, weights.shape[-1])
    cols = flat.shape[1]

    if cols < 4:
        return False

    # Check groups of 4 along columns
    n_groups = cols // 4
    compliant = 0   # groups with <= 2 non-zeros
    exactly_2 = 0   # groups with exactly 2 non-zeros
    total = 0

    for row in flat:
        for g in range(n_groups):
            group = row[g * 4:(g + 1) * 4]
            nnz = np.count_nonzero(group)
            total += 1
            if nnz <= 2:
                compliant += 1
            if nnz == 2:
                exactly_2 += 1

    if total == 0:
        return False

    return (compliant / total) >= threshold and (exactly_2 / total) >= 0.5


def detect_structured_nm(
    weights: np.ndarray,
    candidates: List[Tuple[int, int]] = None,
    threshold: float = 0.9,
) -> Optional[Tuple[int, int]]:
    """
    Detect N:M structured sparsity patterns.

    Checks if the weight matrix has at most N non-zero values in every
    group of M consecutive elements.  Returns the tightest (smallest M)
    pattern found, or None.
    """
    if weights.ndim < 2:
        return None

    if candidates is None:
        candidates = [(1, 4), (2, 8), (4, 8), (2, 4)]

    flat = weights.reshape(-1, weights.shape[-1])
    cols = flat.shape[1]

    best = None
    for n, m in candidates:
        if cols < m:
            continue
        if n == 2 and m == 4:
            continue  # already handled by detect_structured_2_4

        n_groups = cols // m
        compliant = 0
        total = 0

        for row in flat:
            for g in range(n_groups):
                group = row[g * m:(g + 1) * m]
                nnz = np.count_nonzero(group)
                total += 1
                if nnz <= n:
                    compliant += 1

        if total > 0 and (compliant / total) >= threshold:
            if best is None or m < best[1]:
                best = (n, m)

    return best


# ---------------------------------------------------------------------------
#  Pruning
# ---------------------------------------------------------------------------

def prune_weights(
    graph: ComputeGraph,
    threshold: float = 0.0,
    target_sparsity: Optional[float] = None,
) -> ComputeGraph:
    """
    Prune small weights to zero.

    Args:
        graph:           ComputeGraph with q_weights populated.
        threshold:       Zero out weights with abs value <= threshold.
                         Ignored if target_sparsity is set.
        target_sparsity: If set (0.0-1.0), prune the smallest weights
                         globally to reach this sparsity level.

    Returns the same graph (mutated in place).
    """
    if target_sparsity is not None:
        return _prune_to_target(graph, target_sparsity)

    for op in graph.operations:
        if not op.q_weights:
            continue
        for key in _get_weight_keys(op):
            if key not in op.q_weights:
                continue
            w = op.q_weights[key]
            mask = np.abs(w) <= threshold
            op.q_weights[key] = np.where(mask, 0, w)

    return graph


def _prune_to_target(graph: ComputeGraph, target: float) -> ComputeGraph:
    """
    Prune globally to reach a target sparsity level.

    Selects EXACTLY ``n_to_prune`` weights — the n smallest by absolute
    magnitude — and zeros them.  A naive ``|w| <= threshold`` mask massively
    overshoots the target on quantized integer weights, where many values
    tie at the threshold magnitude.  Here we rank all weights and zero the
    first n, breaking ties deterministically by global (stable) order.
    """
    # Collect all (already non-zero) weight magnitudes with their locations.
    abs_vals: List[int] = []
    # locations: (weight_array, flat_index_within_array)
    locations: List[Tuple[np.ndarray, int]] = []

    for op in graph.operations:
        if not op.q_weights:
            continue
        for key in _get_weight_keys(op):
            if key not in op.q_weights:
                continue
            w = op.q_weights[key]
            flat = w.reshape(-1)
            for j in range(flat.size):
                abs_vals.append(abs(int(flat[j])))
                locations.append((w, j))

    if not abs_vals:
        return graph

    abs_arr = np.asarray(abs_vals)
    n_total = abs_arr.size
    n_to_prune = int(n_total * target)

    if n_to_prune <= 0:
        return graph
    if n_to_prune > n_total:
        n_to_prune = n_total

    # Stable argsort over magnitudes: ties are broken by ascending global
    # index, so the choice of which tied weights to prune is deterministic.
    order = np.argsort(abs_arr, kind="stable")
    to_prune = order[:n_to_prune]

    for idx in to_prune:
        w, j = locations[idx]
        # .flat assignment writes back to the original array regardless of
        # memory layout (reshape/ravel may return a copy).
        w.flat[j] = 0

    return graph


def enforce_structured_2_4(graph: ComputeGraph) -> ComputeGraph:
    """
    Enforce 2:4 structured sparsity on all weight matrices.

    For every group of 4 consecutive elements along the last axis,
    keep the 2 largest (by absolute value) and zero the rest.
    This is compatible with NVIDIA's Ampere sparse tensor cores
    and maps cleanly to hardware mux-select circuits.
    """
    for op in graph.operations:
        if not op.q_weights:
            continue
        for key in _get_weight_keys(op):
            if key not in op.q_weights:
                continue
            w = op.q_weights[key]
            if w.ndim < 2 or w.shape[-1] < 4:
                continue
            op.q_weights[key] = _apply_2_4(w)

    return graph


def _apply_2_4(w: np.ndarray) -> np.ndarray:
    """Apply 2:4 structured pruning along the last axis."""
    shape = w.shape
    flat = w.reshape(-1, shape[-1]).copy()
    cols = flat.shape[1]
    n_groups = cols // 4

    for row_idx in range(flat.shape[0]):
        for g in range(n_groups):
            start = g * 4
            group = flat[row_idx, start:start + 4]
            abs_vals = np.abs(group)
            # Keep top 2 by absolute value
            if np.count_nonzero(group) > 2:
                indices = np.argsort(abs_vals)
                # Zero the 2 smallest
                flat[row_idx, start + indices[0]] = 0
                flat[row_idx, start + indices[1]] = 0

    return flat.reshape(shape)
