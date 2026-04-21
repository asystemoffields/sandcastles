"""
quantize.py -- Quantization engine for weights2silicon.

Takes a ComputeGraph with float weights and calibration data, quantizes
all weights, computes requantization parameters, and populates
op.q_weights and op.q_params on every Operation that carries weights.

Activation ranges are determined by running a float forward pass over
the calibration data so that requantization multipliers map the
accumulator scale back to the correct output-tensor scale.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from w2s.core import (
    ComputeGraph,
    OpType,
    Operation,
    QuantConfig,
    QuantGranularity,
    QuantScheme,
)
from w2s.emit import acc_bits_for


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def quantize_graph(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    config: QuantConfig = None,
    bits_map: Optional[Dict[str, int]] = None,
) -> ComputeGraph:
    """
    Quantize all operations in the graph.

    1. Run a float forward pass with calibration_data to determine
       activation ranges.
    2. For each op, quantize weights and compute requantization
       parameters.
    3. Populate op.q_weights and op.q_params.

    Args:
        graph:            ComputeGraph with float weights.
        calibration_data: {input_name: array} for activation range calibration.
        config:           Default quantization config (bit width, scheme, etc.).
        bits_map:         Optional per-layer bit width override.
                          Maps op name -> bit width (4, 8, or 16).
                          Ops not in the map use config.bits.

    Returns the same graph (mutated in place).
    """
    if config is None:
        config = graph.quant_config

    bits = config.bits
    scheme = config.scheme
    granularity = config.granularity

    # Step 1 -- calibrate: run float forward pass, collect tensor ranges
    tensor_ranges = calibrate(graph, calibration_data)
    graph.tensor_ranges = tensor_ranges

    # For mixed precision, we need per-tensor scales at potentially different
    # bit widths.  We compute scales at the default bit width, then override
    # for ops with custom widths.  The key insight: tensor scales depend on
    # the bit width of the consumer op, not the producer.
    tensor_scales: Dict[str, float] = {}
    for tname, (lo, hi) in tensor_ranges.items():
        tensor_scales[tname] = _scale_from_range(lo, hi, bits, scheme)

    # If mixed precision, recompute scales for tensors consumed/produced
    # by ops with custom bit widths
    if bits_map:
        ordered_ops = graph.topological_order()
        for op in ordered_ops:
            op_bits = bits_map.get(op.name, bits)
            if op_bits != bits:
                # Store per-op bits for use during Verilog generation
                op.attrs['bits'] = op_bits
                # Recompute output tensor scales at this op's bit width
                for out_name in op.outputs:
                    if out_name in tensor_ranges:
                        lo, hi = tensor_ranges[out_name]
                        tensor_scales[out_name] = _scale_from_range(
                            lo, hi, op_bits, scheme)

    graph.tensor_scales = tensor_scales

    # Step 2 -- quantize in topological order
    ordered_ops = graph.topological_order()

    for op in ordered_ops:
        op_bits = bits if bits_map is None else bits_map.get(op.name, bits)
        _quantize_op(op, tensor_scales, op_bits, scheme, granularity)

    return graph


# ---------------------------------------------------------------------------
#  Calibration -- float forward pass
# ---------------------------------------------------------------------------

def calibrate(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
) -> Dict[str, Tuple[float, float]]:
    """
    Run a float forward pass over *calibration_data* to determine the
    min/max range of every intermediate tensor.

    Returns {tensor_name: (min_val, max_val)}.
    """
    tensor_values: Dict[str, np.ndarray] = {}

    # Seed with calibration inputs
    for name, data in calibration_data.items():
        tensor_values[name] = data.astype(np.float64)

    ordered_ops = graph.topological_order()

    for op in ordered_ops:
        outputs = forward_op_float(op, tensor_values)
        tensor_values.update(outputs)

    # Collect ranges
    ranges: Dict[str, Tuple[float, float]] = {}
    for name, val in tensor_values.items():
        lo = float(np.min(val))
        hi = float(np.max(val))
        # Guard against perfectly-zero tensors -- give them a small range
        # so scale is not infinite.
        if lo == hi:
            lo = lo - 1e-6
            hi = hi + 1e-6
        ranges[name] = (lo, hi)

    return ranges


# ---------------------------------------------------------------------------
#  Float forward pass for a single operation
# ---------------------------------------------------------------------------

def forward_op_float(
    op: Operation,
    tensor_values: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Execute one operation in float precision.

    Reads its inputs from *tensor_values* and returns a dict mapping
    each output tensor name to its computed value.
    """
    otype = op.op_type
    out_name = op.outputs[0]

    # ---- linear / dense ---------------------------------------------------
    if otype == OpType.DENSE:
        x = tensor_values[op.inputs[0]]
        w = op.weights['weight']                  # (n_out, n_in)
        b = op.weights.get('bias')                # (n_out,) or None
        y = x @ w.T
        if b is not None:
            y = y + b
        act = op.attrs.get('activation', 'none')
        if act == 'relu':
            y = np.maximum(y, 0.0)
        return {out_name: y}

    # ---- convolutions -----------------------------------------------------
    if otype == OpType.CONV2D:
        return {out_name: _conv2d_float(op, tensor_values)}

    if otype == OpType.CONV1D:
        return {out_name: _conv1d_float(op, tensor_values)}

    # ---- element-wise activations -----------------------------------------
    if otype == OpType.RELU:
        x = tensor_values[op.inputs[0]]
        return {out_name: np.maximum(x, 0.0)}

    if otype == OpType.GELU:
        x = tensor_values[op.inputs[0]]
        # tanh approximation (matches PyTorch default)
        c = math.sqrt(2.0 / math.pi)
        y = 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
        return {out_name: y}

    if otype == OpType.SIGMOID:
        x = tensor_values[op.inputs[0]]
        return {out_name: 1.0 / (1.0 + np.exp(-x))}

    if otype == OpType.TANH:
        x = tensor_values[op.inputs[0]]
        return {out_name: np.tanh(x)}

    if otype == OpType.SILU:
        x = tensor_values[op.inputs[0]]
        sig = 1.0 / (1.0 + np.exp(-x))
        return {out_name: x * sig}

    if otype == OpType.SOFTMAX:
        x = tensor_values[op.inputs[0]]
        axis = op.attrs.get('axis', -1)
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return {out_name: e / np.sum(e, axis=axis, keepdims=True)}

    # ---- normalisation ----------------------------------------------------
    if otype == OpType.LAYERNORM:
        x = tensor_values[op.inputs[0]]
        eps = op.attrs.get('eps', 1e-5)
        # Normalise over the last N dims corresponding to normalized_shape
        norm_shape = op.attrs.get('normalized_shape', (x.shape[-1],))
        n_dims = len(norm_shape)
        axes = tuple(range(-n_dims, 0))
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        scale = op.weights.get('scale')
        bias = op.weights.get('bias')
        if scale is not None:
            x_norm = x_norm * scale
        if bias is not None:
            x_norm = x_norm + bias
        return {out_name: x_norm}

    if otype == OpType.RMSNORM:
        x = tensor_values[op.inputs[0]]
        eps = op.attrs.get('eps', 1e-5)
        ms = np.mean(x ** 2, axis=-1, keepdims=True)
        x_norm = x / np.sqrt(ms + eps)
        scale = op.weights.get('scale')
        if scale is not None:
            x_norm = x_norm * scale
        return {out_name: x_norm}

    if otype == OpType.BATCHNORM:
        x = tensor_values[op.inputs[0]]
        eps = op.attrs.get('eps', 1e-5)
        rm = op.weights['running_mean']
        rv = op.weights['running_var']
        scale = op.weights.get('scale')
        bias = op.weights.get('bias')
        C = len(rm)

        # Channel axis is set explicitly at import time (ONNX spec: axis 1
        # for (N, C, ...) inputs; use axis 0 for bare (C, ...) inputs).  The
        # previous heuristic silently misfired when batch happened to equal
        # channel count.
        c_axis = int(op.attrs.get('c_axis', 1))
        if x.ndim == 1:
            c_axis = 0                      # (C,) has no other axis to use
        if c_axis < 0:
            c_axis += x.ndim
        if not (0 <= c_axis < x.ndim):
            raise ValueError(
                f"BatchNorm {op.name!r}: c_axis={op.attrs.get('c_axis')} "
                f"is out of range for input ndim={x.ndim} (shape {x.shape})"
            )
        if x.shape[c_axis] != C:
            raise ValueError(
                f"BatchNorm {op.name!r}: input shape {x.shape} has "
                f"{x.shape[c_axis]} channels at axis {c_axis} but "
                f"running_mean has length {C}.  Check c_axis attr."
            )

        bc_shape = [1] * x.ndim
        bc_shape[c_axis] = C
        bc_shape = tuple(bc_shape)

        rm_b = rm.reshape(bc_shape)
        rv_b = rv.reshape(bc_shape)
        x_norm = (x - rm_b) / np.sqrt(rv_b + eps)
        if scale is not None:
            x_norm = x_norm * scale.reshape(bc_shape)
        if bias is not None:
            x_norm = x_norm + bias.reshape(bc_shape)
        return {out_name: x_norm}

    # ---- element-wise binary ----------------------------------------------
    if otype == OpType.ADD:
        a = tensor_values[op.inputs[0]]
        b = tensor_values[op.inputs[1]]
        return {out_name: a + b}

    if otype == OpType.MULTIPLY:
        a = tensor_values[op.inputs[0]]
        b = tensor_values[op.inputs[1]]
        return {out_name: a * b}

    # ---- embedding --------------------------------------------------------
    if otype == OpType.EMBEDDING:
        indices = tensor_values[op.inputs[0]]
        weight = op.weights['weight']   # (V, D)
        return {out_name: weight[indices.astype(np.intp)]}

    # ---- pooling ----------------------------------------------------------
    if otype == OpType.MAXPOOL2D:
        return {out_name: _pool2d_float(op, tensor_values, mode='max')}

    if otype == OpType.AVGPOOL2D:
        return {out_name: _pool2d_float(op, tensor_values, mode='avg')}

    if otype == OpType.GLOBAL_AVGPOOL:
        x = tensor_values[op.inputs[0]]
        # Average over all spatial dims (everything after channel dim)
        if x.ndim >= 3:
            axes = tuple(range(2, x.ndim))
        elif x.ndim == 2:
            axes = (1,)
        else:
            axes = ()
        return {out_name: np.mean(x, axis=axes, keepdims=False) if axes else x}

    # ---- structural -------------------------------------------------------
    if otype == OpType.RESHAPE:
        x = tensor_values[op.inputs[0]]
        target = op.attrs.get('target_shape', (-1,))
        return {out_name: x.reshape(target)}

    if otype == OpType.FLATTEN:
        x = tensor_values[op.inputs[0]]
        if x.ndim <= 1:
            return {out_name: x}
        # Flatten all dims except the first (batch dim for calibration).
        # Verilog generate_flatten does a full 1D rewire on single samples;
        # here we preserve dim-0 so batched calibration data flows through
        # downstream Dense layers correctly.
        return {out_name: x.reshape(x.shape[0], -1)}

    if otype == OpType.CONCAT:
        arrays = [tensor_values[n] for n in op.inputs]
        axis = op.attrs.get('axis', 0)
        return {out_name: np.concatenate(arrays, axis=axis)}

    # ---- multi-head attention ---------------------------------------------
    if otype == OpType.MULTI_HEAD_ATTENTION:
        return {out_name: _mha_float(op, tensor_values)}

    # ---- swiglu -------------------------------------------------------------
    if otype == OpType.SWIGLU:
        return {out_name: _swiglu_float(op, tensor_values)}

    # ---- rope ---------------------------------------------------------------
    if otype == OpType.ROPE:
        return {out_name: _rope_float(op, tensor_values)}

    # ---- grouped query attention --------------------------------------------
    if otype == OpType.GROUPED_QUERY_ATTENTION:
        return {out_name: _gqa_float(op, tensor_values)}

    # ---- kv cache -----------------------------------------------------------
    if otype == OpType.KV_CACHE:
        # Pass-through for calibration: output = input
        x = tensor_values[op.inputs[0]]
        return {out_name: x.copy()}

    raise ValueError(f"forward_op_float: unsupported op type {otype}")


# ---------------------------------------------------------------------------
#  Conv helpers (float)
# ---------------------------------------------------------------------------

def _conv2d_float(op: Operation, tv: Dict[str, np.ndarray]) -> np.ndarray:
    """Naive Conv2D forward: supports batched or unbatched input."""
    x = tv[op.inputs[0]]
    w = op.weights['weight']                       # (Co, Ci, kH, kW)
    b = op.weights.get('bias')                     # (Co,) or None
    Co, Ci, kH, kW = w.shape
    sH, sW = op.attrs.get('stride', (1, 1))
    pH, pW = op.attrs.get('padding', (0, 0))

    # Handle batched vs unbatched input
    batched = (x.ndim == 4)
    if not batched:
        x = x[np.newaxis]                          # (1, Ci, H, W)

    N, _Ci, H, W = x.shape

    # Zero-pad
    if pH > 0 or pW > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)))

    Hp = x.shape[2]
    Wp = x.shape[3]
    Ho = (Hp - kH) // sH + 1
    Wo = (Wp - kW) // sW + 1

    # im2col
    col = np.zeros((N, Ci * kH * kW, Ho * Wo), dtype=x.dtype)
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i * sH: i * sH + kH, j * sW: j * sW + kW]
            col[:, :, idx] = patch.reshape(N, -1)
            idx += 1

    w_mat = w.reshape(Co, -1)                      # (Co, Ci*kH*kW)
    out = np.einsum('ij,njk->nik', w_mat, col)     # (N, Co, Ho*Wo)
    out = out.reshape(N, Co, Ho, Wo)

    if b is not None:
        out = out + b[np.newaxis, :, np.newaxis, np.newaxis]

    # Fused activation
    act = op.attrs.get('activation', 'none')
    if act == 'relu':
        out = np.maximum(out, 0.0)

    if not batched:
        out = out[0]

    return out


def _conv1d_float(op: Operation, tv: Dict[str, np.ndarray]) -> np.ndarray:
    """Naive Conv1D forward."""
    x = tv[op.inputs[0]]
    w = op.weights['weight']                       # (Co, Ci, kW)
    b = op.weights.get('bias')
    Co, Ci, kW = w.shape
    sW = op.attrs.get('stride', (1,))
    if isinstance(sW, (tuple, list)):
        sW = sW[0]
    pW = op.attrs.get('padding', (0,))
    if isinstance(pW, (tuple, list)):
        pW = pW[0]

    batched = (x.ndim == 3)
    if not batched:
        x = x[np.newaxis]                          # (1, Ci, W)

    N, _Ci, W = x.shape
    if pW > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pW, pW)))

    Wp = x.shape[2]
    Wo = (Wp - kW) // sW + 1

    col = np.zeros((N, Ci * kW, Wo), dtype=x.dtype)
    for j in range(Wo):
        patch = x[:, :, j * sW: j * sW + kW]
        col[:, :, j] = patch.reshape(N, -1)

    w_mat = w.reshape(Co, -1)
    out = np.einsum('ij,njk->nik', w_mat, col)
    out = out.reshape(N, Co, Wo)

    if b is not None:
        out = out + b[np.newaxis, :, np.newaxis]

    if not batched:
        out = out[0]

    return out


# ---------------------------------------------------------------------------
#  Pool helper (float)
# ---------------------------------------------------------------------------

def _pool2d_float(
    op: Operation,
    tv: Dict[str, np.ndarray],
    mode: str = 'max',
) -> np.ndarray:
    """Naive maxpool2d / avgpool2d forward."""
    x = tv[op.inputs[0]]
    kH, kW = op.attrs.get('kernel_size', (2, 2))
    sH, sW = op.attrs.get('stride', (kH, kW))
    pH, pW = op.attrs.get('padding', (0, 0))

    batched = (x.ndim == 4)
    if not batched:
        x = x[np.newaxis]

    N, C, H, W = x.shape
    if pH > 0 or pW > 0:
        pad_val = -1e30 if mode == 'max' else 0.0
        x_pad = np.full((N, C, H + 2 * pH, W + 2 * pW), pad_val, dtype=x.dtype)
        x_pad[:, :, pH:pH + H, pW:pW + W] = x
        x = x_pad

    Hp = x.shape[2]
    Wp = x.shape[3]
    Ho = (Hp - kH) // sH + 1
    Wo = (Wp - kW) // sW + 1

    out = np.zeros((N, C, Ho, Wo), dtype=x.dtype)
    for i in range(Ho):
        for j in range(Wo):
            window = x[:, :, i * sH: i * sH + kH, j * sW: j * sW + kW]
            if mode == 'max':
                out[:, :, i, j] = window.reshape(N, C, -1).max(axis=-1)
            else:
                out[:, :, i, j] = window.reshape(N, C, -1).mean(axis=-1)

    if not batched:
        out = out[0]
    return out


# ---------------------------------------------------------------------------
#  Multi-head attention (float)
# ---------------------------------------------------------------------------

def _mha_float(op: Operation, tv: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Full multi-head attention forward pass.

    Expects op.weights to contain q_weight, q_bias, k_weight, k_bias,
    v_weight, v_bias, out_weight, out_bias.

    Input is assumed to be a single tensor of shape (..., seq_len, embed_dim).
    For simplicity, self-attention is assumed (Q, K, V all derived from the
    same input).
    """
    x = tv[op.inputs[0]]

    num_heads = op.attrs['num_heads']
    head_dim = op.attrs['head_dim']
    embed_dim = op.attrs.get('embed_dim', num_heads * head_dim)

    Wq = op.weights['q_weight']   # (E, E)
    bq = op.weights.get('q_bias')
    Wk = op.weights['k_weight']
    bk = op.weights.get('k_bias')
    Wv = op.weights['v_weight']
    bv = op.weights.get('v_bias')
    Wo = op.weights['out_weight']
    bo = op.weights.get('out_bias')

    # Linear projections
    Q = x @ Wq.T
    K = x @ Wk.T
    V = x @ Wv.T
    if bq is not None:
        Q = Q + bq
    if bk is not None:
        K = K + bk
    if bv is not None:
        V = V + bv

    # Determine batch/sequence dims.  x can be (seq, E), (B, seq, E), etc.
    orig_shape = Q.shape
    if Q.ndim == 2:
        # (seq, E)  -- treat as single batch
        seq_len = orig_shape[0]
        Q = Q.reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    elif Q.ndim == 3:
        # (B, seq, E)
        B, seq_len, _ = orig_shape
        Q = Q.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"MHA: unexpected input ndim {Q.ndim}")

    # Scaled dot-product attention
    scale = math.sqrt(head_dim)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale  # (B, H, S, S)
    # Softmax over last axis
    scores_max = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - scores_max)
    attn = e / np.sum(e, axis=-1, keepdims=True)

    # Weighted sum
    ctx = np.matmul(attn, V)   # (B, H, S, head_dim)

    # Concat heads
    if orig_shape[-1] == embed_dim and len(orig_shape) == 2:
        ctx = ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, embed_dim)
        ctx = ctx[0]  # back to (seq, E)
    else:
        B = ctx.shape[0]
        seq_len = ctx.shape[2]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, seq_len, embed_dim)

    # Output projection
    out = ctx @ Wo.T
    if bo is not None:
        out = out + bo

    return out


# ---------------------------------------------------------------------------
#  SwiGLU (float)
# ---------------------------------------------------------------------------

def _swiglu_float(op: Operation, tv: Dict[str, np.ndarray]) -> np.ndarray:
    """
    SwiGLU forward pass:
        gate = x @ gate_W.T + gate_b
        up   = x @ up_W.T + up_b
        hidden = silu(gate) * up
        out  = hidden @ down_W.T + down_b
    """
    x = tv[op.inputs[0]]

    gate_w = op.weights['gate_weight']    # (ffn_dim, dim)
    up_w   = op.weights['up_weight']      # (ffn_dim, dim)
    down_w = op.weights['down_weight']    # (dim, ffn_dim)

    gate_b = op.weights.get('gate_bias')
    up_b   = op.weights.get('up_bias')
    down_b = op.weights.get('down_bias')

    gate = x @ gate_w.T
    if gate_b is not None:
        gate = gate + gate_b

    up = x @ up_w.T
    if up_b is not None:
        up = up + up_b

    # SiLU activation on gate: silu(x) = x * sigmoid(x)
    sig = 1.0 / (1.0 + np.exp(-gate))
    hidden = (gate * sig) * up

    out = hidden @ down_w.T
    if down_b is not None:
        out = out + down_b

    return out


# ---------------------------------------------------------------------------
#  RoPE (float)
# ---------------------------------------------------------------------------

def _rope_float(op: Operation, tv: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Rotary position embeddings forward pass.

    Input: (seq_len, embed_dim) or (embed_dim,).
    cos_table: (max_seq_len, half_dim)
    sin_table: (max_seq_len, half_dim)

    For each position pos, for each pair (d, d+half):
        out[d]      = x[d]*cos[pos,d] - x[d+half]*sin[pos,d]
        out[d+half] = x[d]*sin[pos,d] + x[d+half]*cos[pos,d]
    """
    x = tv[op.inputs[0]]

    cos_table = op.weights['cos_table']   # (max_seq_len, half)
    sin_table = op.weights['sin_table']   # (max_seq_len, half)

    dim = op.attrs['dim']
    half = dim // 2

    if x.ndim == 1:
        # Single position — treat as position 0
        out = np.zeros_like(x)
        cos_vals = cos_table[0, :half]
        sin_vals = sin_table[0, :half]
        out[:half]    = x[:half] * cos_vals - x[half:] * sin_vals
        out[half:]    = x[:half] * sin_vals + x[half:] * cos_vals
    else:
        # (seq_len, dim) — apply per-position
        seq_len = x.shape[0]
        out = np.zeros_like(x)
        for pos in range(seq_len):
            cos_vals = cos_table[pos, :half]
            sin_vals = sin_table[pos, :half]
            out[pos, :half]    = x[pos, :half] * cos_vals - x[pos, half:] * sin_vals
            out[pos, half:]    = x[pos, :half] * sin_vals + x[pos, half:] * cos_vals

    return out


# ---------------------------------------------------------------------------
#  GQA (float)
# ---------------------------------------------------------------------------

def _gqa_float(op: Operation, tv: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Grouped-query attention forward pass.

    Same as MHA but K/V have fewer heads (num_kv_heads < num_heads).
    Each KV head is shared by (num_heads // num_kv_heads) query heads.
    """
    x = tv[op.inputs[0]]

    num_heads    = op.attrs['num_heads']
    num_kv_heads = op.attrs['num_kv_heads']
    head_dim     = op.attrs['head_dim']
    embed_dim    = op.attrs.get('embed_dim', num_heads * head_dim)

    Wq = op.weights['q_weight']       # (embed_dim, embed_dim)
    bq = op.weights.get('q_bias')
    Wk = op.weights['k_weight']       # (kv_dim, embed_dim)
    bk = op.weights.get('k_bias')
    Wv = op.weights['v_weight']       # (kv_dim, embed_dim)
    bv = op.weights.get('v_bias')
    Wo = op.weights['out_weight']     # (embed_dim, embed_dim)
    bo = op.weights.get('out_bias')

    kv_dim = num_kv_heads * head_dim
    heads_per_group = num_heads // num_kv_heads

    # Linear projections
    Q = x @ Wq.T
    K = x @ Wk.T
    V = x @ Wv.T
    if bq is not None:
        Q = Q + bq
    if bk is not None:
        K = K + bk
    if bv is not None:
        V = V + bv

    # Determine batch/sequence dims
    orig_shape = Q.shape
    if Q.ndim == 2:
        seq_len = orig_shape[0]
        Q = Q.reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(1, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(1, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    elif Q.ndim == 3:
        B, seq_len, _ = orig_shape
        Q = Q.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"GQA: unexpected input ndim {Q.ndim}")

    # Expand K/V heads to match Q heads via repeat
    # K shape: (B, num_kv_heads, S, head_dim)
    # -> repeat each KV head heads_per_group times along head axis
    K = np.repeat(K, heads_per_group, axis=1)  # (B, num_heads, S, head_dim)
    V = np.repeat(V, heads_per_group, axis=1)

    # Scaled dot-product attention
    scale = math.sqrt(head_dim)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - scores_max)
    attn = e / np.sum(e, axis=-1, keepdims=True)

    ctx = np.matmul(attn, V)  # (B, num_heads, S, head_dim)

    # Concat heads
    if orig_shape[-1] == embed_dim and len(orig_shape) == 2:
        ctx = ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, embed_dim)
        ctx = ctx[0]
    else:
        B = ctx.shape[0]
        seq_len = ctx.shape[2]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, seq_len, embed_dim)

    # Output projection
    out = ctx @ Wo.T
    if bo is not None:
        out = out + bo

    return out


# ---------------------------------------------------------------------------
#  Quantize tensor
# ---------------------------------------------------------------------------

def quantize_tensor(
    arr: np.ndarray,
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize a float tensor.

    Returns (quantized_int_array, scales).

    For PER_TENSOR: scales is a 1-element array.
    For PER_CHANNEL: scales has shape (arr.shape[axis],).
    """
    if granularity == QuantGranularity.PER_CHANNEL and arr.ndim >= 2:
        n_channels = arr.shape[axis]
        scales = np.zeros(n_channels, dtype=np.float64)
        q_arr = np.zeros_like(arr, dtype=np.int64)

        for c in range(n_channels):
            # Slice along the channel axis
            slc = [slice(None)] * arr.ndim
            slc[axis] = c
            channel_data = arr[tuple(slc)]

            q_channel, s = _quantize_1d(channel_data, bits, scheme)
            q_arr[tuple(slc)] = q_channel
            scales[c] = s

        return q_arr, scales
    else:
        q_arr, s = _quantize_1d(arr, bits, scheme)
        return q_arr, np.array([s], dtype=np.float64)


def _quantize_1d(
    arr: np.ndarray,
    bits: int,
    scheme: QuantScheme,
) -> Tuple[np.ndarray, float]:
    """
    Quantize a tensor with a single (per-tensor) scale.

    Returns (int_array, scale_float).
    """
    arr = arr.astype(np.float64)

    if scheme == QuantScheme.SYMMETRIC:
        qmax = (1 << (bits - 1)) - 1                       # 127 for 8-bit
        abs_max = float(np.max(np.abs(arr)))
        if abs_max < 1e-30:
            abs_max = 1e-30
        scale = qmax / abs_max
        q = np.clip(np.round(arr * scale), -qmax, qmax).astype(np.int64)
        return q, scale

    else:  # ASYMMETRIC
        qmin_int = -(1 << (bits - 1))                      # -128 for 8-bit
        qmax_int = (1 << (bits - 1)) - 1                   # 127

        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi - lo < 1e-30:
            hi = lo + 1e-30

        scale = (hi - lo) / (qmax_int - qmin_int)
        zero_point = round(-lo / scale) + qmin_int
        zero_point = int(np.clip(zero_point, qmin_int, qmax_int))

        q = np.clip(np.round(arr / scale + zero_point), qmin_int, qmax_int)
        q = q.astype(np.int64)
        # For asymmetric we return the inverse scale (values-per-unit) so
        # that the requantization logic stays uniform.  The Verilog backend
        # only uses symmetric mode, but the calibration path may request
        # asymmetric.
        return q, 1.0 / scale


# ---------------------------------------------------------------------------
#  Requantization parameter computation
# ---------------------------------------------------------------------------

def compute_requant(
    input_scale: float,
    weight_scale: float,
    output_scale: float,
    shift: int = 16,
) -> Tuple[int, int]:
    """
    Compute integer requantization multiplier and shift.

    After MAC:  acc = sum(w_q * x_q) + b_q
    acc is in scale  (input_scale * weight_scale).

    To convert to output scale:
        out_q = (acc * M) >> S

    where:
        M = round(output_scale / (input_scale * weight_scale) * 2^S)

    If M overflows a 32-bit signed integer we bump S until it fits.
    """
    acc_scale = input_scale * weight_scale
    if acc_scale < 1e-30:
        acc_scale = 1e-30
    if output_scale < 1e-30:
        output_scale = 1e-30

    ratio = output_scale / acc_scale

    # Try the requested shift; if M overflows int32, try smaller shifts
    # to reduce the multiplier magnitude.  Fall back to clamping if needed.
    for s in range(shift, max(shift - 16, 0), -1):
        M = round(ratio * (1 << s))
        if abs(M) <= 0x7FFFFFFF:
            return int(M), int(s)

    # Fallback: use the requested shift with clamped multiplier
    M = round(ratio * (1 << shift))
    M = max(-0x7FFFFFFF, min(0x7FFFFFFF, M))
    return int(M), int(shift)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _scale_from_range(
    lo: float, hi: float, bits: int, scheme: QuantScheme
) -> float:
    """Derive a quantization scale from an observed min/max range."""
    if scheme == QuantScheme.SYMMETRIC:
        qmax = (1 << (bits - 1)) - 1
        abs_max = max(abs(lo), abs(hi))
        if abs_max < 1e-30:
            abs_max = 1e-30
        return qmax / abs_max
    else:
        qmin_int = -(1 << (bits - 1))
        qmax_int = (1 << (bits - 1)) - 1
        span = hi - lo
        if span < 1e-30:
            span = 1e-30
        scale = (qmax_int - qmin_int) / span
        return scale


# Ops that carry weight tensors needing quantization
_WEIGHTED_OPS = {
    OpType.DENSE,
    OpType.CONV1D,
    OpType.CONV2D,
    OpType.LAYERNORM,
    OpType.RMSNORM,
    OpType.BATCHNORM,
    OpType.MULTI_HEAD_ATTENTION,
    OpType.GROUPED_QUERY_ATTENTION,
    OpType.SWIGLU,
    OpType.ROPE,
    OpType.KV_CACHE,
    OpType.EMBEDDING,
}


def _quantize_op(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Quantize one operation in-place."""
    if op.op_type not in _WEIGHTED_OPS:
        return
    if not op.weights:
        return

    # ---- quantize each weight tensor ------------------------------------
    for wname, warr in op.weights.items():
        # Running stats (batchnorm) are not quantized -- they are folded
        # into scale/bias during inference.
        if wname in ('running_mean', 'running_var'):
            continue

        q_arr, w_scales = quantize_tensor(
            warr, bits, scheme, granularity, axis=0,
        )
        op.q_weights[wname] = q_arr

    # ---- requantization parameters (only for MAC-style ops) -------------
    if op.op_type in (OpType.DENSE, OpType.CONV1D, OpType.CONV2D):
        _compute_mac_requant(op, tensor_scales, bits, scheme, granularity)

    elif op.op_type == OpType.MULTI_HEAD_ATTENTION:
        _compute_mha_requant(op, tensor_scales, bits, scheme, granularity)

    elif op.op_type == OpType.GROUPED_QUERY_ATTENTION:
        _compute_gqa_requant(op, tensor_scales, bits, scheme, granularity)

    elif op.op_type == OpType.SWIGLU:
        _compute_swiglu_requant(op, tensor_scales, bits, scheme, granularity)

    elif op.op_type == OpType.ROPE:
        _compute_rope_requant(op, tensor_scales, bits, scheme, granularity)

    elif op.op_type == OpType.KV_CACHE:
        # No weights to quantize — pass through
        pass

    elif op.op_type in (OpType.LAYERNORM, OpType.RMSNORM, OpType.BATCHNORM):
        _compute_norm_requant(op, tensor_scales, bits, scheme, granularity)

    elif op.op_type == OpType.EMBEDDING:
        _compute_embedding_requant(op, tensor_scales, bits, scheme, granularity)


def _compute_mac_requant(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Compute requantization params for DENSE / CONV1D / CONV2D."""
    input_name = op.inputs[0]
    output_name = op.outputs[0]
    input_scale = tensor_scales.get(input_name, 1.0)
    output_scale = tensor_scales.get(output_name, 1.0)

    weight = op.weights['weight']

    # Quantize the weight tensor and get its scales
    q_w, w_scales = quantize_tensor(
        weight, bits, scheme, granularity, axis=0,
    )
    op.q_weights['weight'] = q_w

    # Quantize bias in the accumulator scale (input_scale * weight_scale)
    bias = op.weights.get('bias')
    if bias is not None:
        if granularity == QuantGranularity.PER_CHANNEL and weight.ndim >= 2:
            # Each output channel has its own bias scale
            bias_q = np.zeros_like(bias, dtype=np.int64)
            for c in range(len(bias)):
                bias_scale = input_scale * w_scales[c]
                if bias_scale < 1e-30:
                    bias_scale = 1e-30
                bias_q[c] = int(round(bias[c] * bias_scale))
            op.q_weights['bias'] = bias_q
        else:
            # Per-tensor path (or weight.ndim < 2): collapse w_scales to one
            # scalar.  If somehow we still received a multi-element array
            # here (e.g. a 1-D per-channel weight under PER_CHANNEL), it
            # means the caller handed us inconsistent config; raise rather
            # than silently averaging.
            ws_arr = np.asarray(w_scales).reshape(-1)
            if ws_arr.size == 1:
                ws_scalar = float(ws_arr[0])
            else:
                raise ValueError(
                    f"Bias quantization for {op.name!r}: expected scalar "
                    f"weight scale (per-tensor or 1-D weight) but got "
                    f"{ws_arr.size} scales.  Check quantization granularity "
                    f"vs. weight shape {weight.shape}."
                )
            bias_scale = input_scale * ws_scalar
            if bias_scale < 1e-30:
                bias_scale = 1e-30
            op.q_weights['bias'] = np.round(bias * bias_scale).astype(np.int64)

    # Determine accumulator bit-width
    if weight.ndim == 2:
        n_in = weight.shape[1]
    elif weight.ndim == 3:
        n_in = weight.shape[1] * weight.shape[2]       # Conv1D
    elif weight.ndim == 4:
        n_in = weight.shape[1] * weight.shape[2] * weight.shape[3]  # Conv2D
    else:
        n_in = int(np.prod(weight.shape[1:]))
    a_bits = acc_bits_for(n_in, bits)

    # Compute requantization multiplier(s)
    if granularity == QuantGranularity.PER_CHANNEL and weight.ndim >= 2:
        n_out = weight.shape[0]
        requant_mults = np.zeros(n_out, dtype=np.int64)
        # Two-pass approach: first find the maximum shift needed across
        # all channels, then recompute all multipliers with that shift.
        shifts = np.zeros(n_out, dtype=np.int64)
        for c in range(n_out):
            m, s = compute_requant(input_scale, w_scales[c], output_scale, 16)
            requant_mults[c] = m
            shifts[c] = s
        shift = int(np.max(shifts))
        for c in range(n_out):
            m, s = compute_requant(input_scale, w_scales[c], output_scale, shift)
            requant_mults[c] = m
        op.q_params = {
            'requant_mult': requant_mults,
            'requant_shift': shift,
            'input_scale': float(input_scale),
            'weight_scale': w_scales,
            'output_scale': float(output_scale),
            'acc_bits': a_bits,
        }
    else:
        ws = float(w_scales[0])
        m, s = compute_requant(input_scale, ws, output_scale)
        op.q_params = {
            'requant_mult': m,
            'requant_shift': s,
            'input_scale': float(input_scale),
            'weight_scale': ws,
            'output_scale': float(output_scale),
            'acc_bits': a_bits,
        }


def _compute_mha_requant(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Compute per-projection requantization params for multi-head attention.

    The Verilog generator (attention.py) expects per-projection keys:
        q_requant_mult, q_requant_shift,
        k_requant_mult, k_requant_shift,
        v_requant_mult, v_requant_shift,
        out_requant_mult, out_requant_shift,
        acc_bits.
    """
    input_name = op.inputs[0]
    output_name = op.outputs[0]
    input_scale = tensor_scales.get(input_name, 1.0)
    output_scale = tensor_scales.get(output_name, 1.0)

    embed_dim = op.attrs.get(
        'embed_dim',
        op.attrs['num_heads'] * op.attrs['head_dim'],
    )
    a_bits = acc_bits_for(embed_dim, bits)

    # Quantize all weight matrices
    proj_names = [
        'q_weight', 'k_weight', 'v_weight', 'out_weight',
        'q_bias', 'k_bias', 'v_bias', 'out_bias',
    ]
    w_scale_map = {}
    for pname in proj_names:
        if pname in op.weights:
            warr = op.weights[pname]
            q_arr, ws = quantize_tensor(warr, bits, scheme, granularity, axis=0)
            op.q_weights[pname] = q_arr
            w_scale_map[pname] = float(ws[0]) if ws.size == 1 else ws

    # Per-projection requantization: each projection (q, k, v, out) gets
    # its own requant_mult/shift, computed from its weight scale.
    op.q_params = {
        'input_scale': float(input_scale),
        'weight_scale': w_scale_map,
        'output_scale': float(output_scale),
        'acc_bits': a_bits,
    }

    for proj in ('q', 'k', 'v', 'out'):
        wkey = f'{proj}_weight'
        ws = w_scale_map.get(wkey, 1.0)
        if isinstance(ws, np.ndarray):
            ws = float(ws[0])
        m, s = compute_requant(input_scale, float(ws), output_scale)
        op.q_params[f'{proj}_requant_mult'] = m
        op.q_params[f'{proj}_requant_shift'] = s


def _compute_norm_requant(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Compute requantization params for normalization layers."""
    input_name = op.inputs[0]
    output_name = op.outputs[0]
    input_scale = tensor_scales.get(input_name, 1.0)
    output_scale = tensor_scales.get(output_name, 1.0)

    # Norm layers have a scale (gamma) tensor -- use its mean magnitude
    # as a proxy for the weight scale in the requantization math.
    scale_w = op.weights.get('scale')
    if scale_w is not None:
        q_arr, ws = quantize_tensor(scale_w, bits, scheme, granularity, axis=0)
        op.q_weights['scale'] = q_arr
        ws_float = float(ws[0]) if ws.size == 1 else float(np.mean(ws))
    else:
        ws_float = 1.0

    bias_w = op.weights.get('bias')
    if bias_w is not None:
        q_arr, _ = quantize_tensor(bias_w, bits, scheme, granularity, axis=0)
        op.q_weights['bias'] = q_arr

    m, s = compute_requant(input_scale, ws_float, output_scale)
    a_bits = acc_bits_for(
        int(scale_w.shape[0]) if scale_w is not None else 1,
        bits,
    )

    op.q_params = {
        'requant_mult': m,
        'requant_shift': s,
        'input_scale': float(input_scale),
        'weight_scale': ws_float,
        'output_scale': float(output_scale),
        'acc_bits': a_bits,
    }


def _compute_embedding_requant(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Compute requantization params for embedding lookup."""
    output_name = op.outputs[0]
    output_scale = tensor_scales.get(output_name, 1.0)

    weight = op.weights['weight']                   # (V, D)
    q_arr, ws = quantize_tensor(weight, bits, scheme, granularity, axis=0)
    op.q_weights['weight'] = q_arr
    ws_float = float(ws[0]) if ws.size == 1 else ws

    op.q_params = {
        'requant_mult': 1,
        'requant_shift': 0,
        'input_scale': 1.0,                         # no input scale for embeddings
        'weight_scale': ws_float,
        'output_scale': float(output_scale),
        'acc_bits': bits,
    }


def _compute_swiglu_requant(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Compute per-projection requantization params for SwiGLU.

    The Verilog generator (transformer.py) expects:
        gate_requant_mult, gate_requant_shift,
        up_requant_mult,   up_requant_shift,
        down_requant_mult, down_requant_shift,
        acc_bits.
    """
    input_name = op.inputs[0]
    output_name = op.outputs[0]
    input_scale = tensor_scales.get(input_name, 1.0)
    output_scale = tensor_scales.get(output_name, 1.0)

    # Quantize all weight tensors
    weight_names = [
        'gate_weight', 'up_weight', 'down_weight',
        'gate_bias', 'up_bias', 'down_bias',
    ]
    w_scale_map = {}
    for wname in weight_names:
        if wname in op.weights:
            warr = op.weights[wname]
            q_arr, ws = quantize_tensor(warr, bits, scheme, granularity, axis=0)
            op.q_weights[wname] = q_arr
            w_scale_map[wname] = float(ws[0]) if ws.size == 1 else ws

    # Accumulator bits — use the largest weight dimension
    gate_w = op.weights['gate_weight']         # (ffn_dim, dim)
    down_w = op.weights['down_weight']         # (dim, ffn_dim)
    n_in = max(gate_w.shape[1], down_w.shape[1])
    a_bits = acc_bits_for(n_in, bits)

    op.q_params = {
        'input_scale': float(input_scale),
        'weight_scale': w_scale_map,
        'output_scale': float(output_scale),
        'acc_bits': a_bits,
    }

    # Per-projection requant: gate, up, down
    for proj in ('gate', 'up', 'down'):
        wkey = f'{proj}_weight'
        ws = w_scale_map.get(wkey, 1.0)
        if isinstance(ws, np.ndarray):
            ws = float(ws[0])
        m, s = compute_requant(input_scale, float(ws), output_scale)
        op.q_params[f'{proj}_requant_mult'] = m
        op.q_params[f'{proj}_requant_shift'] = s


def _compute_gqa_requant(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Compute per-projection requantization params for grouped-query attention.

    The Verilog generator (transformer.py) expects:
        q_requant_mult, q_requant_shift,
        k_requant_mult, k_requant_shift,
        v_requant_mult, v_requant_shift,
        out_requant_mult, out_requant_shift,
        acc_bits.
    """
    input_name = op.inputs[0]
    output_name = op.outputs[0]
    input_scale = tensor_scales.get(input_name, 1.0)
    output_scale = tensor_scales.get(output_name, 1.0)

    embed_dim = op.attrs.get(
        'embed_dim',
        op.attrs['num_heads'] * op.attrs['head_dim'],
    )
    a_bits = acc_bits_for(embed_dim, bits)

    # Quantize all weight matrices
    proj_names = [
        'q_weight', 'k_weight', 'v_weight', 'out_weight',
        'q_bias', 'k_bias', 'v_bias', 'out_bias',
    ]
    w_scale_map = {}
    for pname in proj_names:
        if pname in op.weights:
            warr = op.weights[pname]
            q_arr, ws = quantize_tensor(warr, bits, scheme, granularity, axis=0)
            op.q_weights[pname] = q_arr
            w_scale_map[pname] = float(ws[0]) if ws.size == 1 else ws

    op.q_params = {
        'input_scale': float(input_scale),
        'weight_scale': w_scale_map,
        'output_scale': float(output_scale),
        'acc_bits': a_bits,
    }

    # Per-projection requant: q, k, v, out
    for proj in ('q', 'k', 'v', 'out'):
        wkey = f'{proj}_weight'
        ws = w_scale_map.get(wkey, 1.0)
        if isinstance(ws, np.ndarray):
            ws = float(ws[0])
        m, s = compute_requant(input_scale, float(ws), output_scale)
        op.q_params[f'{proj}_requant_mult'] = m
        op.q_params[f'{proj}_requant_shift'] = s


def _compute_rope_requant(
    op: Operation,
    tensor_scales: Dict[str, float],
    bits: int,
    scheme: QuantScheme,
    granularity: QuantGranularity,
) -> None:
    """Compute requantization params for rotary position embeddings.

    The cos/sin tables are lookup tables that get quantized.  The
    rotation itself is a 2-term MAC per output element, so we
    provide a generic requant_mult/shift pair plus acc_bits.
    """
    input_name = op.inputs[0]
    output_name = op.outputs[0]
    input_scale = tensor_scales.get(input_name, 1.0)
    output_scale = tensor_scales.get(output_name, 1.0)

    # Quantize cos and sin tables (lookup tables, per-tensor is fine)
    for tname in ('cos_table', 'sin_table'):
        if tname in op.weights:
            warr = op.weights[tname]
            q_arr, ws = quantize_tensor(warr, bits, scheme, granularity, axis=0)
            op.q_weights[tname] = q_arr

    # The rotation MAC has 2 multiply-add terms per output element.
    # Input is multiplied by a cos/sin table value (both quantised to
    # `bits`), so the effective "weight scale" is the table's scale.
    cos_table = op.weights['cos_table']
    _, cos_ws = quantize_tensor(cos_table, bits, scheme, granularity, axis=0)
    table_scale = float(cos_ws[0]) if cos_ws.size == 1 else float(np.mean(cos_ws))

    # acc_bits: 2 terms (the rotation sum has exactly 2 products)
    a_bits = acc_bits_for(2, bits)

    m, s = compute_requant(input_scale, table_scale, output_scale)

    op.q_params = {
        'requant_mult': m,
        'requant_shift': s,
        'input_scale': float(input_scale),
        'weight_scale': table_scale,
        'output_scale': float(output_scale),
        'acc_bits': a_bits,
    }
