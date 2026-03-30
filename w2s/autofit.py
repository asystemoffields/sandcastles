"""
autofit.py — Automatically fit a model to a target device.

Given a model and a target device, find the best combination of
quantization bit width, mixed precision, sparsity, and compilation mode
to make the design fit while preserving as much accuracy as possible.

The approach:
  1. Estimate area at uniform int8 — if it fits, done.
  2. Run per-layer sensitivity analysis (measure output error at int4 vs int8).
  3. Greedily downgrade the least-sensitive layers to int4 until the design fits.
  4. If still too big, apply sparsity (prune the least-important weights).
  5. If still too big, try sequential mode.
  6. Report the final configuration or declare the model too large.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from w2s.core import ComputeGraph, Operation, OpType, QuantConfig, QuantScheme
from w2s.quantize import quantize_graph, calibrate, forward_op_float
from w2s.estimate import estimate as estimate_asic


# ---------------------------------------------------------------------------
#  Sensitivity analysis
# ---------------------------------------------------------------------------

@dataclass
class LayerSensitivity:
    """Quantization sensitivity for a single layer."""
    name: str
    op_type: OpType
    n_params: int
    error_int8: float     # output MSE when this layer is at int8
    error_int4: float     # output MSE when this layer is at int4
    sensitivity: float    # error_int4 - error_int8 (higher = more sensitive)


@dataclass
class SensitivityReport:
    """Per-layer sensitivity analysis results."""
    layers: List[LayerSensitivity]
    baseline_error: float  # error with all layers at int8

    def ranked(self) -> List[LayerSensitivity]:
        """Return layers ranked by sensitivity (least sensitive first)."""
        return sorted(self.layers, key=lambda ls: ls.sensitivity)

    def __str__(self) -> str:
        lines = ["=== Sensitivity Analysis ===", ""]
        lines.append(f"  {'Layer':<30} {'Type':<10} {'Params':>8} "
                     f"{'int8 err':>10} {'int4 err':>10} {'Sensitivity':>12}")
        lines.append(f"  {'-'*30} {'-'*10} {'-'*8} "
                     f"{'-'*10} {'-'*10} {'-'*12}")
        for ls in self.ranked():
            lines.append(
                f"  {ls.name:<30} {ls.op_type.value:<10} {ls.n_params:>8,} "
                f"{ls.error_int8:>10.4f} {ls.error_int4:>10.4f} "
                f"{ls.sensitivity:>12.4f}"
            )
        return "\n".join(lines)


# Ops that have weights and can be quantized at different bit widths
_QUANTIZABLE_OPS = {
    OpType.DENSE, OpType.CONV1D, OpType.CONV2D,
    OpType.MULTI_HEAD_ATTENTION, OpType.GROUPED_QUERY_ATTENTION,
    OpType.SWIGLU, OpType.EMBEDDING,
}


def analyze_sensitivity(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
) -> SensitivityReport:
    """
    Run per-layer sensitivity analysis.

    For each weighted layer, quantize it at int4 while keeping everything
    else at int8, and measure the output error (MSE) vs the float baseline.

    This is an O(n_layers) process — one forward pass per layer.
    """
    # Get float baseline outputs
    float_outputs = _float_forward(graph, calibration_data)

    # Get int8 baseline error
    config_8 = QuantConfig(bits=8, scheme=QuantScheme.SYMMETRIC)
    graph_8 = _quantize_copy(graph, calibration_data, config_8)
    int8_outputs = _float_forward_quantized(graph_8, calibration_data)
    baseline_error = _compute_mse(float_outputs, int8_outputs)

    # Per-layer sensitivity
    weighted_ops = [
        op for op in graph.topological_order()
        if op.op_type in _QUANTIZABLE_OPS and op.weights
    ]

    layers = []
    for op in weighted_ops:
        n_params = sum(int(np.prod(w.shape)) for w in op.weights.values())

        # Error with this layer at int4, everything else at int8
        bits_map_4 = {op.name: 4}
        graph_4 = _quantize_copy(graph, calibration_data, config_8, bits_map_4)
        outputs_4 = _float_forward_quantized(graph_4, calibration_data)
        error_4 = _compute_mse(float_outputs, outputs_4)

        layers.append(LayerSensitivity(
            name=op.name,
            op_type=op.op_type,
            n_params=n_params,
            error_int8=baseline_error,
            error_int4=error_4,
            sensitivity=error_4 - baseline_error,
        ))

    return SensitivityReport(layers=layers, baseline_error=baseline_error)


# ---------------------------------------------------------------------------
#  Auto-fit engine
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Result of auto-fit optimization."""
    fits: bool
    mode: str                          # "combinational" or "sequential"
    bits: int                          # default bit width
    bits_map: Optional[Dict[str, int]] # per-layer overrides (None = uniform)
    sparsity: float                    # applied sparsity level (0.0 = none)
    estimated_luts: int
    device_luts: int
    n_layers_downgraded: int
    config_summary: str

    def __str__(self) -> str:
        lines = ["=== Auto-Fit Result ==="]
        if self.fits:
            lines.append(f"Status:     FITS")
        else:
            lines.append(f"Status:     DOES NOT FIT")
        lines.append(f"Mode:       {self.mode}")
        lines.append(f"Default:    int{self.bits}")
        if self.bits_map:
            int4_layers = [k for k, v in self.bits_map.items() if v == 4]
            if int4_layers:
                lines.append(f"Downgraded: {len(int4_layers)} layers to int4")
                for name in int4_layers[:10]:
                    lines.append(f"  - {name}")
                if len(int4_layers) > 10:
                    lines.append(f"  ... and {len(int4_layers) - 10} more")
        if self.sparsity > 0:
            lines.append(f"Sparsity:   {self.sparsity:.0%}")
        lines.append(f"Area:       ~{self.estimated_luts:,} LUTs "
                     f"({self.estimated_luts / self.device_luts:.0%} of device)")
        lines.append("")
        lines.append(f"Config:     {self.config_summary}")
        return "\n".join(lines)


def autofit(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    device_luts: int,
    prefer_combinational: bool = False,
    max_sparsity: float = 0.75,
    sparsity_steps: List[float] = None,
) -> FitResult:
    """
    Automatically find the best configuration to fit a model on a device.

    Args:
        graph:                 ComputeGraph with float weights.
        calibration_data:      {input_name: array} for calibration.
        device_luts:           Target device LUT capacity.
        prefer_combinational:  Try combinational mode before sequential.
        max_sparsity:          Maximum sparsity level to try (0.0-1.0).
        sparsity_steps:        Sparsity levels to try (default: [0.25, 0.5, 0.75]).

    Returns:
        A FitResult describing the best configuration found.
    """
    if sparsity_steps is None:
        sparsity_steps = [0.25, 0.5, 0.75]
    sparsity_steps = [s for s in sparsity_steps if s <= max_sparsity]

    modes = (["combinational", "sequential"] if prefer_combinational
             else ["sequential", "combinational"])

    # Strategy 1: Try uniform int8
    for mode in modes:
        result = _try_config(graph, calibration_data, 8, None, 0.0, mode, device_luts)
        if result.fits:
            return result

    # Strategy 2: Run sensitivity analysis, greedily downgrade to int4
    sensitivity = analyze_sensitivity(graph, calibration_data)
    ranked = sensitivity.ranked()  # least sensitive first

    for mode in modes:
        bits_map = {}
        for layer in ranked:
            bits_map[layer.name] = 4
            result = _try_config(graph, calibration_data, 8, dict(bits_map),
                                 0.0, mode, device_luts)
            if result.fits:
                return result

    # Strategy 3: All int4
    for mode in modes:
        result = _try_config(graph, calibration_data, 4, None, 0.0, mode, device_luts)
        if result.fits:
            return result

    # Strategy 4: Add sparsity
    for sparsity in sparsity_steps:
        for mode in modes:
            # int4 + sparsity
            result = _try_config(graph, calibration_data, 4, None,
                                 sparsity, mode, device_luts)
            if result.fits:
                return result

            # int8 + sparsity (better accuracy than int4)
            result = _try_config(graph, calibration_data, 8, None,
                                 sparsity, mode, device_luts)
            if result.fits:
                return result

    # Nothing fits — return the best attempt
    return FitResult(
        fits=False,
        mode="sequential",
        bits=4,
        bits_map=None,
        sparsity=max_sparsity,
        estimated_luts=result.estimated_luts if result else 0,
        device_luts=device_luts,
        n_layers_downgraded=0,
        config_summary="Model too large for target device",
    )


def autofit_fpga(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    device=None,
    prefer_combinational: bool = False,
    max_sparsity: float = 0.75,
) -> FitResult:
    """
    Auto-fit for a specific FPGA device.

    Convenience wrapper that extracts device_luts from an FPGADevice.
    """
    if device is None:
        from w2s.fpga import ICE40_UP5K
        device = ICE40_UP5K
    return autofit(
        graph, calibration_data, device.lut4s,
        prefer_combinational=prefer_combinational,
        max_sparsity=max_sparsity,
    )


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _try_config(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    bits: int,
    bits_map: Optional[Dict[str, int]],
    sparsity: float,
    mode: str,
    device_luts: int,
) -> FitResult:
    """Try a specific configuration and estimate area."""
    config = QuantConfig(bits=bits, scheme=QuantScheme.SYMMETRIC)

    # Deep copy to avoid mutating the original
    g = _deep_copy_graph(graph)
    g.quant_config = config
    quantize_graph(g, calibration_data, config, bits_map=bits_map)

    # Apply sparsity
    if sparsity > 0:
        from w2s.sparsity import prune_weights
        prune_weights(g, target_sparsity=sparsity)

    # Estimate
    report = estimate_asic(g, mode=mode)

    fits = report.estimated_luts <= device_luts

    # Build config summary
    parts = [f"int{bits}"]
    n_downgraded = 0
    if bits_map:
        int4_count = sum(1 for v in bits_map.values() if v == 4)
        n_downgraded = int4_count
        if int4_count > 0:
            parts.append(f"{int4_count} layers@int4")
    if sparsity > 0:
        parts.append(f"{sparsity:.0%} sparse")
    parts.append(mode)
    config_summary = ", ".join(parts)

    return FitResult(
        fits=fits,
        mode=mode,
        bits=bits,
        bits_map=bits_map if bits_map else None,
        sparsity=sparsity,
        estimated_luts=report.estimated_luts,
        device_luts=device_luts,
        n_layers_downgraded=n_downgraded,
        config_summary=config_summary,
    )


def _deep_copy_graph(graph: ComputeGraph) -> ComputeGraph:
    """Deep-copy a graph so quantization doesn't mutate the original."""
    g = ComputeGraph(
        name=graph.name,
        input_names=list(graph.input_names),
        input_shapes=dict(graph.input_shapes),
        output_names=list(graph.output_names),
        quant_config=QuantConfig(
            bits=graph.quant_config.bits,
            scheme=graph.quant_config.scheme,
            granularity=graph.quant_config.granularity,
        ),
    )
    for op in graph.operations:
        new_op = Operation(
            op_type=op.op_type,
            name=op.name,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            attrs=dict(op.attrs),
            weights={k: v.copy() for k, v in op.weights.items()},
        )
        g.add(new_op)
    return g


def _quantize_copy(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    config: QuantConfig,
    bits_map: Optional[Dict[str, int]] = None,
) -> ComputeGraph:
    """Quantize a copy of the graph without mutating the original."""
    g = _deep_copy_graph(graph)
    g.quant_config = config
    quantize_graph(g, calibration_data, config, bits_map=bits_map)
    return g


def _float_forward(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run the float forward pass and return all tensor values."""
    tensor_values = {}
    for name, data in calibration_data.items():
        tensor_values[name] = data.astype(np.float64)

    for op in graph.topological_order():
        outputs = forward_op_float(op, tensor_values)
        tensor_values.update(outputs)

    # Return only the graph outputs
    return {name: tensor_values[name]
            for name in graph.output_names
            if name in tensor_values}


def _float_forward_quantized(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Run a float forward pass using quantized weights.

    This simulates the effect of quantization on the output: use the
    dequantized weights (q_weights / weight_scale) instead of float weights,
    to measure quantization-induced error.
    """
    tensor_values = {}
    for name, data in calibration_data.items():
        tensor_values[name] = data.astype(np.float64)

    for op in graph.topological_order():
        # Temporarily swap in dequantized weights for the forward pass
        orig_weights = op.weights
        if op.q_weights:
            deq_weights = {}
            for key, q_arr in op.q_weights.items():
                if key in ('running_mean', 'running_var'):
                    deq_weights[key] = orig_weights.get(key, q_arr.astype(np.float64))
                    continue
                # Dequantize: divide by weight scale
                ws = op.q_params.get('weight_scale')
                if ws is not None and key in orig_weights:
                    if isinstance(ws, dict):
                        # Per-projection scales (MHA, SwiGLU)
                        proj_scale = ws.get(key, 1.0)
                        if isinstance(proj_scale, np.ndarray):
                            proj_scale = float(proj_scale[0]) if proj_scale.size == 1 else proj_scale
                        if isinstance(proj_scale, np.ndarray):
                            # Per-channel: expand scale for broadcasting
                            s = np.ones_like(q_arr, dtype=np.float64)
                            for c in range(min(len(proj_scale), q_arr.shape[0])):
                                s[c] = proj_scale[c]
                            deq_weights[key] = q_arr.astype(np.float64) / s
                        else:
                            deq_weights[key] = q_arr.astype(np.float64) / float(proj_scale)
                    elif isinstance(ws, np.ndarray) and ws.size > 1:
                        # Per-channel scale
                        s = np.ones_like(q_arr, dtype=np.float64)
                        for c in range(min(len(ws), q_arr.shape[0])):
                            s[c] = ws[c]
                        deq_weights[key] = q_arr.astype(np.float64) / s
                    else:
                        scale = float(ws) if not isinstance(ws, np.ndarray) else float(ws[0])
                        if scale > 0:
                            deq_weights[key] = q_arr.astype(np.float64) / scale
                        else:
                            deq_weights[key] = q_arr.astype(np.float64)
                else:
                    deq_weights[key] = q_arr.astype(np.float64)
            op.weights = deq_weights

        outputs = forward_op_float(op, tensor_values)
        tensor_values.update(outputs)

        # Restore original weights
        if op.q_weights:
            op.weights = orig_weights

    return {name: tensor_values[name]
            for name in graph.output_names
            if name in tensor_values}


def _compute_mse(
    a: Dict[str, np.ndarray],
    b: Dict[str, np.ndarray],
) -> float:
    """Compute mean squared error between two output dicts."""
    total = 0.0
    count = 0
    for key in a:
        if key in b:
            diff = a[key].astype(np.float64) - b[key].astype(np.float64)
            total += float(np.mean(diff ** 2))
            count += 1
    return total / max(count, 1)
