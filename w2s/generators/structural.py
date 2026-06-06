"""
structural.py -- Verilog generators for structural / element-wise operations.

ADD and MULTIPLY produce combinational arithmetic with saturation.
RESHAPE and FLATTEN are zero-cost rewirings (no logic emitted).
CONCAT joins wire-name lists along the specified axis.
"""

from typing import Dict, List, Optional, Tuple

from w2s.core import Operation, TensorWires
from w2s.emit import (
    section_comment,
    wire_signed,
    sign_extend_wire,
    sign_extend_expr,
    slit,
    saturate_linear,
    requantize_lines,
    requant_prod_bits,
    compute_rescale,
)


# ---------------------------------------------------------------------------
#  Per-operand scale alignment
# ---------------------------------------------------------------------------
#
#  Add / Multiply / Concat combine *raw* quantized integers.  That is only
#  correct if every operand shares one fixed-point scale.  When operands carry
#  different scales (the common case — two unrelated layers feeding a residual
#  add, or a concat of heterogeneous branches) the integers must first be
#  rescaled to a common (output) scale.
#
#  The per-operand scales are threaded to the generator through op.q_params:
#       op.q_params['input_scales'] = [s0, s1, ...]   # one per op.inputs
#       op.q_params['output_scale'] = s_out
#  When those keys are absent we fall back to the legacy "shared scale"
#  behaviour (operands are assumed already aligned).  When they are present but
#  the scales genuinely differ, we emit an explicit per-operand requant so the
#  hardware is correct rather than silently wrong.

def _structural_scales(op: Operation, n_inputs: int):
    """Return (input_scales, output_scale) or (None, None) if not plumbed."""
    in_scales = op.q_params.get('input_scales') if op.q_params else None
    out_scale = op.q_params.get('output_scale') if op.q_params else None
    if in_scales is None or out_scale is None:
        return None, None
    in_scales = [float(s) for s in in_scales]
    if len(in_scales) != n_inputs:
        raise ValueError(
            f"{op.name!r}: input_scales has {len(in_scales)} entries but op has "
            f"{n_inputs} inputs"
        )
    return in_scales, float(out_scale)


def _scales_aligned(in_scales, out_scale) -> bool:
    """True if every operand already sits at the output scale (no rescale)."""
    return all(abs(s - out_scale) <= 1e-12 * max(abs(s), abs(out_scale), 1.0)
               for s in in_scales)


# ---------------------------------------------------------------------------
#  ADD (residual connection)
# ---------------------------------------------------------------------------

def generate_add(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for element-wise addition (residual connection).

    Both inputs must have the same shape.  Each pair of elements is
    sign-extended to (bits+1), added, and saturated back to *bits*.

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    in_a = wire_map[op.inputs[0]]
    in_b = wire_map[op.inputs[1]]
    n = in_a.numel
    out_name = op.outputs[0]
    out_wire_names = [f"{op.name}_out_{i}" for i in range(n)]

    in_scales, out_scale = _structural_scales(op, 2)
    rescale = in_scales is not None and not _scales_aligned(in_scales, out_scale)

    # One extra bit prevents overflow before saturation
    wide = bits + 1

    lines: List[str] = []

    lines += section_comment(
        f"{op.name}: Add  shape={in_a.shape}  ({n} elements)"
        + ("  [per-operand requant to output scale]" if rescale else "")
    )
    lines.append("")

    if rescale:
        # Map each operand from its own scale to the shared output scale, then
        # add the aligned integers.
        m_a, s_a = compute_rescale(in_scales[0], out_scale)
        m_b, s_b = compute_rescale(in_scales[1], out_scale)
        wa = requant_prod_bits(bits, m_a)
        wb = requant_prod_bits(bits, m_b)
        wsum = max(wa, wb) + 1
        for i in range(n):
            prefix = f"{op.name}_e{i}"
            rq_a, sh_a = requantize_lines(in_a.wire_names[i], bits, m_a, s_a,
                                          f"{prefix}_a")
            rq_b, sh_b = requantize_lines(in_b.wire_names[i], bits, m_b, s_b,
                                          f"{prefix}_b")
            lines += rq_a
            lines += rq_b
            sum_name = f"{prefix}_sum"
            lines.append(
                f"    wire signed [{wsum - 1}:0] {sum_name} = "
                f"{sign_extend_expr(sh_a, wa, wsum)} + "
                f"{sign_extend_expr(sh_b, wb, wsum)};"
            )
            out_wire = out_wire_names[i]
            lines += saturate_linear(sum_name, wsum, out_wire, bits)
            lines.append("")
    else:
        for i in range(n):
            prefix = f"{op.name}_e{i}"
            a_wire = in_a.wire_names[i]
            b_wire = in_b.wire_names[i]

            # Sign-extend both operands
            a_ext = f"{prefix}_a"
            b_ext = f"{prefix}_b"
            lines.append(sign_extend_wire(a_wire, bits, a_ext, wide))
            lines.append(sign_extend_wire(b_wire, bits, b_ext, wide))

            # Sum
            sum_name = f"{prefix}_sum"
            lines.append(
                f"    wire signed [{wide - 1}:0] {sum_name} = {a_ext} + {b_ext};"
            )

            # Saturate back to target width
            out_wire = out_wire_names[i]
            lines += saturate_linear(sum_name, wide, out_wire, bits)
            lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=in_a.shape,
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}


# ---------------------------------------------------------------------------
#  MULTIPLY (element-wise)
# ---------------------------------------------------------------------------

def generate_multiply(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for element-wise multiplication.

    Both inputs must have the same shape.  Each pair is sign-extended
    to (bits) and multiplied, yielding a (2*bits)-wide product which
    is then saturated back to *bits*.

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    in_a = wire_map[op.inputs[0]]
    in_b = wire_map[op.inputs[1]]
    n = in_a.numel
    out_name = op.outputs[0]
    out_wire_names = [f"{op.name}_out_{i}" for i in range(n)]

    prod_bits = 2 * bits

    in_scales, out_scale = _structural_scales(op, 2)
    # The product of a (scale s_a) and b (scale s_b) lands at scale s_a*s_b; to
    # express it at the output scale we requantize by out/(s_a*s_b).
    rescale = in_scales is not None

    lines: List[str] = []

    lines += section_comment(
        f"{op.name}: Multiply  shape={in_a.shape}  ({n} elements)"
        + ("  [product requant to output scale]" if rescale else "")
    )
    lines.append("")

    if rescale:
        m_p, s_p = compute_rescale(in_scales[0] * in_scales[1], out_scale)
        for i in range(n):
            prefix = f"{op.name}_e{i}"
            a_ext = f"{prefix}_a"
            b_ext = f"{prefix}_b"
            lines.append(sign_extend_wire(in_a.wire_names[i], bits, a_ext, prod_bits))
            lines.append(sign_extend_wire(in_b.wire_names[i], bits, b_ext, prod_bits))
            prod_name = f"{prefix}_prod"
            lines.append(
                f"    wire signed [{prod_bits - 1}:0] {prod_name} = "
                f"{a_ext} * {b_ext};"
            )
            rq, shifted = requantize_lines(prod_name, prod_bits, m_p, s_p,
                                           f"{prefix}_rq")
            lines += rq
            out_wire = out_wire_names[i]
            lines += saturate_linear(shifted, requant_prod_bits(prod_bits, m_p),
                                     out_wire, bits)
            lines.append("")
    else:
        for i in range(n):
            prefix = f"{op.name}_e{i}"
            a_wire = in_a.wire_names[i]
            b_wire = in_b.wire_names[i]

            # Sign-extend both operands to product width for clean multiply
            a_ext = f"{prefix}_a"
            b_ext = f"{prefix}_b"
            lines.append(sign_extend_wire(a_wire, bits, a_ext, prod_bits))
            lines.append(sign_extend_wire(b_wire, bits, b_ext, prod_bits))

            # Multiply
            prod_name = f"{prefix}_prod"
            lines.append(
                f"    wire signed [{prod_bits - 1}:0] {prod_name} = "
                f"{a_ext} * {b_ext};"
            )

            # Right-shift by (bits-1) to rescale (matches integer forward pass)
            shift_name = f"{prefix}_shift"
            lines.append(
                f"    wire signed [{prod_bits - 1}:0] {shift_name} = "
                f"{prod_name} >>> {bits - 1};"
            )

            # Saturate back to target width
            out_wire = out_wire_names[i]
            lines += saturate_linear(shift_name, prod_bits, out_wire, bits)
            lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=in_a.shape,
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}


# ---------------------------------------------------------------------------
#  RESHAPE
# ---------------------------------------------------------------------------

def generate_reshape(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for a reshape -- pure rewiring, no logic emitted.

    The wire_names list stays in the same flat order; only the logical
    shape tuple changes to op.attrs['target_shape'].

    Returns (verilog_lines, new_wires).
    """
    in_tw = wire_map[op.inputs[0]]
    target_shape = tuple(op.attrs['target_shape'])
    out_name = op.outputs[0]

    lines: List[str] = []
    lines += section_comment(
        f"{op.name}: Reshape  {in_tw.shape} -> {target_shape}  (rewire only)"
    )
    lines.append("")

    out_tw = TensorWires(
        wire_names=list(in_tw.wire_names),   # same wires, new shape
        shape=target_shape,
        bits=in_tw.bits,
        signed=in_tw.signed,
    )

    return lines, {out_name: out_tw}


# ---------------------------------------------------------------------------
#  FLATTEN
# ---------------------------------------------------------------------------

def generate_flatten(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for flatten -- pure rewiring, no logic emitted.

    All dimensions are collapsed into a single flat dimension.

    Returns (verilog_lines, new_wires).
    """
    in_tw = wire_map[op.inputs[0]]
    n = in_tw.numel
    out_name = op.outputs[0]

    lines: List[str] = []
    lines += section_comment(
        f"{op.name}: Flatten  {in_tw.shape} -> ({n},)  (rewire only)"
    )
    lines.append("")

    out_tw = TensorWires(
        wire_names=list(in_tw.wire_names),   # same wires, flat shape
        shape=(n,),
        bits=in_tw.bits,
        signed=in_tw.signed,
    )

    return lines, {out_name: out_tw}


# ---------------------------------------------------------------------------
#  CONCAT
# ---------------------------------------------------------------------------

def generate_concat(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for tensor concatenation along *axis*.

    All input tensors must have identical shapes on every axis except
    the concatenation axis.  The output wires are the input wire lists
    interleaved in the correct order for C-contiguous (row-major) layout.

    Returns (verilog_lines, new_wires).
    """
    axis = op.attrs.get('axis', 0)
    out_name = op.outputs[0]

    # Gather all input TensorWires
    inputs = [wire_map[name] for name in op.inputs]
    n_inputs = len(inputs)
    ndim = len(inputs[0].shape)

    # Normalize negative axis
    if axis < 0:
        axis = ndim + axis

    in_scales, out_scale = _structural_scales(op, n_inputs)
    rescale = in_scales is not None and not _scales_aligned(in_scales, out_scale)
    # Per-input (mult, shift) mapping that input's scale to the output scale.
    rescale_ms = (
        [compute_rescale(s, out_scale) for s in in_scales] if rescale else None
    )

    lines: List[str] = []

    # Compute output shape
    out_shape = list(inputs[0].shape)
    for tw in inputs[1:]:
        out_shape[axis] += tw.shape[axis]
    out_shape = tuple(out_shape)

    lines += section_comment(
        f"{op.name}: Concat  {n_inputs} inputs along axis={axis} "
        f"-> {out_shape}  "
        + ("(per-operand requant to output scale)" if rescale else "(rewire only)")
    )
    lines.append("")

    # --- Interleave wire names in correct row-major order ---------------
    #
    # Strategy: iterate over every index in the output shape.  For each
    # output index, determine which input tensor and which position in
    # that tensor it maps to, then copy the corresponding wire name.

    out_wire_names: List[str] = []

    # Pre-compute axis offsets for each input
    axis_offsets: List[int] = []
    running = 0
    for tw in inputs:
        axis_offsets.append(running)
        running += tw.shape[axis]

    # Total output elements
    total = 1
    for s in out_shape:
        total *= s

    # Strides for output shape (row-major)
    out_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1]

    for flat in range(total):
        # Decompose flat index into multi-dim index
        md = []
        rem = flat
        for d in range(ndim):
            md.append(rem // out_strides[d])
            rem %= out_strides[d]

        # Determine which input owns this position along the concat axis
        pos_on_axis = md[axis]
        src_idx = 0
        for k in range(n_inputs - 1, -1, -1):
            if pos_on_axis >= axis_offsets[k]:
                src_idx = k
                break

        src_tw = inputs[src_idx]
        # Map to source coordinates: only the concat-axis coordinate changes
        src_md = list(md)
        src_md[axis] = md[axis] - axis_offsets[src_idx]

        # Convert source multi-dim index to flat (row-major)
        src_flat = 0
        src_stride = 1
        for d in range(ndim - 1, -1, -1):
            src_flat += src_md[d] * src_stride
            src_stride *= src_tw.shape[d]

        src_wire = src_tw.wire_names[src_flat]
        if rescale:
            # Requant this element from its input's scale to the output scale,
            # producing a fresh output wire.
            m, s = rescale_ms[src_idx]
            new_wire = f"{op.name}_out_{flat}"
            rq, shifted = requantize_lines(src_wire, src_tw.bits, m, s,
                                           f"{op.name}_rq{flat}")
            lines += rq
            lines += saturate_linear(shifted,
                                     requant_prod_bits(src_tw.bits, m),
                                     new_wire, bits)
            out_wire_names.append(new_wire)
        else:
            out_wire_names.append(src_wire)

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=out_shape,
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}
