"""
conv.py -- Verilog generators for Conv1D and Conv2D.

Produces combinational Verilog where every convolution weight is a numeric
constant in the logic fabric.  For each output pixel, a MAC tree is emitted
over the kernel window, followed by bias addition, requantization, and
saturation.  Zero-padded input positions are skipped entirely (the
synthesis tool never sees a multiply-by-zero for padding).
"""

import numpy as np
from typing import Dict, List, Tuple

from w2s.core import Operation, TensorWires
from w2s.emit import (
    sign_extend_wire,
    mac_term,
    requantize_lines,
    saturate,
    section_comment,
    wire_signed,
    slit,
)


# ---------------------------------------------------------------------------
#  Conv2D
# ---------------------------------------------------------------------------

def generate_conv2d(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate combinational Verilog for a 2-D convolution.

    Weight shape : (C_out, C_in, kH, kW) -- quantized integers
    Bias shape   : (C_out,) or None
    Input shape  : (C_in, H_in, W_in)
    Output shape : (C_out, H_out, W_out)

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    weight = op.q_weights['weight']                         # (Co, Ci, kH, kW)
    bias = op.q_weights.get('bias', None)                   # (Co,) or None
    float_weight = op.weights.get('weight', None)
    float_bias = op.weights.get('bias', None)

    C_out, C_in, kH, kW = weight.shape
    kernel_size = op.attrs.get('kernel_size', (kH, kW))
    sH, sW = op.attrs.get('stride', (1, 1))
    pH, pW = op.attrs.get('padding', (0, 0))

    requant_mult = op.q_params['requant_mult']              # ndarray (Co,) or scalar
    requant_shift = op.q_params['requant_shift']             # int (shared across all channels)
    acc_bits = int(op.q_params['acc_bits'])
    per_channel = isinstance(requant_mult, np.ndarray)

    activation = op.attrs.get('activation', 'none')

    # ---- input wires ---------------------------------------------------
    inp_name = op.inputs[0]
    inp_tw = wire_map[inp_name]
    # inp_tw.shape == (C_in, H_in, W_in)
    H_in = inp_tw.shape[1]
    W_in = inp_tw.shape[2]

    # ---- output geometry -----------------------------------------------
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1

    out_name = op.outputs[0]
    n_out = C_out * H_out * W_out
    out_wire_names = [f"{op.name}_out_{i}" for i in range(n_out)]

    lines: List[str] = []

    # header comment
    n_weights = C_out * C_in * kH * kW
    n_biases = C_out if bias is not None else 0
    requant_shift_val = int(requant_shift)
    lines += section_comment(
        f"{op.name}: Conv2D  in({C_in},{H_in},{W_in}) "
        f"k({kH},{kW}) s({sH},{sW}) p({pH},{pW}) -> "
        f"out({C_out},{H_out},{W_out})"
    )
    lines.append(
        f"    // {n_weights} weights + {n_biases} biases hardwired"
    )
    lines.append(
        f"    // Requantize: {'per-channel' if per_channel else 'per-tensor'} mult, shift={requant_shift_val}"
    )
    lines.append("")

    # ---- per-output-pixel generation -----------------------------------
    for co in range(C_out):
        mult_co = int(requant_mult[co]) if per_channel else int(requant_mult)
        shift_co = int(requant_shift)

        for oh in range(H_out):
            for ow in range(W_out):
                flat_idx = co * H_out * W_out + oh * W_out + ow
                prefix = f"{op.name}_n{flat_idx}"

                lines.append(
                    f"    // --- output[{co},{oh},{ow}] (flat {flat_idx}) ---"
                )

                # 1) sign-extend every needed input wire to acc_bits
                ext_map: Dict[Tuple[int, int, int], str] = {}
                for ci in range(C_in):
                    for r in range(kH):
                        for s in range(kW):
                            ih = oh * sH - pH + r
                            iw = ow * sW - pW + s
                            # skip out-of-bounds (zero-padded)
                            if ih < 0 or ih >= H_in or iw < 0 or iw >= W_in:
                                continue
                            wval = int(weight[co, ci, r, s])
                            if wval == 0:
                                continue
                            key = (ci, ih, iw)
                            if key not in ext_map:
                                inp_flat = ci * H_in * W_in + ih * W_in + iw
                                src_wire = inp_tw.wire_names[inp_flat]
                                ext_name = f"{prefix}_ext_c{ci}_h{ih}_w{iw}"
                                lines.append(
                                    sign_extend_wire(
                                        src_wire, bits, ext_name, acc_bits
                                    )
                                )
                                ext_map[key] = ext_name

                # 2) accumulator = sum of weight*input + bias
                acc_name = f"{prefix}_acc"
                terms: List[str] = []
                for ci in range(C_in):
                    for r in range(kH):
                        for s in range(kW):
                            ih = oh * sH - pH + r
                            iw = ow * sW - pW + s
                            if ih < 0 or ih >= H_in or iw < 0 or iw >= W_in:
                                continue
                            wval = int(weight[co, ci, r, s])
                            if wval == 0:
                                continue
                            key = (ci, ih, iw)
                            fval = (
                                float(float_weight[co, ci, r, s])
                                if float_weight is not None
                                else None
                            )
                            comment = (
                                f"W[{co},{ci},{r},{s}]={fval:+.4f}"
                                if fval is not None
                                else f"W[{co},{ci},{r},{s}]"
                            )
                            terms.append(
                                mac_term(wval, ext_map[key], acc_bits, comment)
                            )

                # bias
                if bias is not None:
                    bval = int(bias[co])
                    fb = (
                        float(float_bias[co])
                        if float_bias is not None
                        else None
                    )
                    bcmt = (
                        f"  // bias[{co}]={fb:+.4f}" if fb is not None
                        else f"  // bias[{co}]"
                    )
                    terms.append(f"{slit(acc_bits, bval)};{bcmt}")

                # emit accumulator
                if not terms:
                    # degenerate: all weights zero, no bias
                    lines.append(wire_signed(acc_name, acc_bits))
                    lines.append(
                        f"    assign {acc_name} = {slit(acc_bits, 0)};"
                    )
                else:
                    lines.append(
                        f"    wire signed [{acc_bits - 1}:0] {acc_name} ="
                    )
                    for ti, t in enumerate(terms):
                        pfx = "  " if ti == 0 else "+ "
                        sfx = ";" if ti == len(terms) - 1 else ""
                        lines.append(f"    {pfx}{t}{sfx}")
                lines.append("")

                # 3) requantize (per-channel)
                rq_lines, shifted = requantize_lines(
                    acc_name, acc_bits, mult_co, shift_co, prefix
                )
                lines += rq_lines

                # 4) saturate + optional activation
                out_wire = out_wire_names[flat_idx]
                prod_bits = min(acc_bits + 18, 64)
                lines += saturate(shifted, prod_bits, out_wire, bits, activation)
                lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=(C_out, H_out, W_out),
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}


# ---------------------------------------------------------------------------
#  Conv1D
# ---------------------------------------------------------------------------

def generate_conv1d(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate combinational Verilog for a 1-D convolution.

    Weight shape : (C_out, C_in, kW) -- quantized integers
    Bias shape   : (C_out,) or None
    Input shape  : (C_in, W_in)
    Output shape : (C_out, W_out)

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    weight = op.q_weights['weight']                         # (Co, Ci, kW)
    bias = op.q_weights.get('bias', None)                   # (Co,) or None
    float_weight = op.weights.get('weight', None)
    float_bias = op.weights.get('bias', None)

    C_out, C_in, kW = weight.shape
    sW = op.attrs.get('stride', (1,))
    if isinstance(sW, (tuple, list)):
        sW = sW[0]
    pW = op.attrs.get('padding', (0,))
    if isinstance(pW, (tuple, list)):
        pW = pW[0]

    requant_mult = op.q_params['requant_mult']              # ndarray (Co,) or scalar
    requant_shift = op.q_params['requant_shift']             # int (shared across all channels)
    acc_bits = int(op.q_params['acc_bits'])
    per_channel = isinstance(requant_mult, np.ndarray)

    activation = op.attrs.get('activation', 'none')

    # ---- input wires ---------------------------------------------------
    inp_name = op.inputs[0]
    inp_tw = wire_map[inp_name]
    # inp_tw.shape == (C_in, W_in)
    W_in = inp_tw.shape[1]

    # ---- output geometry -----------------------------------------------
    W_out = (W_in + 2 * pW - kW) // sW + 1

    out_name = op.outputs[0]
    n_out = C_out * W_out
    out_wire_names = [f"{op.name}_out_{i}" for i in range(n_out)]

    lines: List[str] = []

    # header comment
    n_weights = C_out * C_in * kW
    n_biases = C_out if bias is not None else 0
    requant_shift_val = int(requant_shift)
    lines += section_comment(
        f"{op.name}: Conv1D  in({C_in},{W_in}) "
        f"k({kW}) s({sW}) p({pW}) -> out({C_out},{W_out})"
    )
    lines.append(
        f"    // {n_weights} weights + {n_biases} biases hardwired"
    )
    lines.append(
        f"    // Requantize: {'per-channel' if per_channel else 'per-tensor'} mult, shift={requant_shift_val}"
    )
    lines.append("")

    # ---- per-output-element generation ---------------------------------
    for co in range(C_out):
        mult_co = int(requant_mult[co]) if per_channel else int(requant_mult)
        shift_co = int(requant_shift)

        for ow in range(W_out):
            flat_idx = co * W_out + ow
            prefix = f"{op.name}_n{flat_idx}"

            lines.append(
                f"    // --- output[{co},{ow}] (flat {flat_idx}) ---"
            )

            # 1) sign-extend needed input wires to acc_bits
            ext_map: Dict[Tuple[int, int], str] = {}
            for ci in range(C_in):
                for s in range(kW):
                    iw = ow * sW - pW + s
                    if iw < 0 or iw >= W_in:
                        continue
                    wval = int(weight[co, ci, s])
                    if wval == 0:
                        continue
                    key = (ci, iw)
                    if key not in ext_map:
                        inp_flat = ci * W_in + iw
                        src_wire = inp_tw.wire_names[inp_flat]
                        ext_name = f"{prefix}_ext_c{ci}_w{iw}"
                        lines.append(
                            sign_extend_wire(
                                src_wire, bits, ext_name, acc_bits
                            )
                        )
                        ext_map[key] = ext_name

            # 2) accumulator = sum of weight*input + bias
            acc_name = f"{prefix}_acc"
            terms: List[str] = []
            for ci in range(C_in):
                for s in range(kW):
                    iw = ow * sW - pW + s
                    if iw < 0 or iw >= W_in:
                        continue
                    wval = int(weight[co, ci, s])
                    if wval == 0:
                        continue
                    key = (ci, iw)
                    fval = (
                        float(float_weight[co, ci, s])
                        if float_weight is not None
                        else None
                    )
                    comment = (
                        f"W[{co},{ci},{s}]={fval:+.4f}"
                        if fval is not None
                        else f"W[{co},{ci},{s}]"
                    )
                    terms.append(
                        mac_term(wval, ext_map[key], acc_bits, comment)
                    )

            # bias
            if bias is not None:
                bval = int(bias[co])
                fb = (
                    float(float_bias[co])
                    if float_bias is not None
                    else None
                )
                bcmt = (
                    f"  // bias[{co}]={fb:+.4f}" if fb is not None
                    else f"  // bias[{co}]"
                )
                terms.append(f"{slit(acc_bits, bval)};{bcmt}")

            # emit accumulator
            if not terms:
                lines.append(wire_signed(acc_name, acc_bits))
                lines.append(
                    f"    assign {acc_name} = {slit(acc_bits, 0)};"
                )
            else:
                lines.append(
                    f"    wire signed [{acc_bits - 1}:0] {acc_name} ="
                )
                for ti, t in enumerate(terms):
                    pfx = "  " if ti == 0 else "+ "
                    sfx = ";" if ti == len(terms) - 1 else ""
                    lines.append(f"    {pfx}{t}{sfx}")
            lines.append("")

            # 3) requantize (per-channel)
            rq_lines, shifted = requantize_lines(
                acc_name, acc_bits, mult_co, shift_co, prefix
            )
            lines += rq_lines

            # 4) saturate + optional activation
            out_wire = out_wire_names[flat_idx]
            prod_bits = min(acc_bits + 18, 64)
            lines += saturate(shifted, prod_bits, out_wire, bits, activation)
            lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=(C_out, W_out),
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}
