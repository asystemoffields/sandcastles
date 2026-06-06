"""
dense.py — Verilog generator for fully-connected (dense) layers.

Produces purely combinational logic:  output = weight * input + bias
with requantization and optional fused ReLU.
"""

import numpy as np
from typing import Dict, List, Tuple

from w2s.core import Operation, TensorWires
from w2s import emit


def generate_dense(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for a dense (fully-connected) layer.

    For each output neuron j:
        acc_j = sum_i(weight[j][i] * input[i]) + bias[j]
        requant_j = (acc_j * requant_mult) >>> requant_shift
        output_j = saturate(requant_j)

    Returns:
        verilog_lines  -- list of Verilog source lines
        new_wires      -- dict mapping output tensor name to TensorWires
    """
    # --- Unpack weights and parameters ---
    weight = op.q_weights['weight']                 # (n_out, n_in)
    n_out, n_in = weight.shape
    bias = op.q_weights.get('bias', np.zeros(n_out, dtype=np.int64))  # (n_out,)
    weight_f = op.weights['weight']                 # float, for comments
    bias_f = op.weights.get('bias', np.zeros(n_out, dtype=np.float64))  # float, for comments

    # Default to identity requant (mult=1, shift=0) when unspecified, matching
    # forward_int and the sequential compiler, so the three execution paths
    # never silently disagree on a hand-built / partially-quantized graph.
    requant_mult = op.q_params.get('requant_mult', 1)   # int or ndarray
    requant_shift = op.q_params.get('requant_shift', 0)  # int
    acc_bits = op.q_params.get('acc_bits', emit.acc_bits_for(n_in, bits))
    activation = op.attrs.get('activation', 'none')

    per_channel = isinstance(requant_mult, np.ndarray)

    # --- Input tensor ---
    in_tensor = wire_map[op.inputs[0]]
    out_tensor_name = op.outputs[0]

    lines: List[str] = []

    # Section header
    lines += emit.section_comment(
        f"Dense: {op.name}  ({n_in} -> {n_out})"
        + (f"  activation={activation}" if activation != "none" else "")
    )
    lines.append("")

    # Sign-extend every input wire to acc_bits
    se_names: List[str] = []
    for i in range(n_in):
        src = in_tensor.wire_names[i]
        dst = f"{op.name}_se_{i}"
        se_names.append(dst)
        lines.append(emit.sign_extend_wire(src, bits, dst, acc_bits))
    lines.append("")

    # --- Per-neuron MAC + requantize + saturate ---
    out_wire_names: List[str] = []

    for j in range(n_out):
        out_wire = f"{op.name}_out_{j}"
        out_wire_names.append(out_wire)

        lines.append(f"    // --- neuron {j} ---")

        # Build MAC terms, skipping zero weights
        terms: List[str] = []
        for i in range(n_in):
            w = int(weight[j, i])
            if w == 0:
                continue
            comment = f"w={weight_f[j, i]:.4f}"
            terms.append(emit.mac_term(w, se_names[i], acc_bits, comment=comment))

        # Bias term
        bias_val = int(bias[j])

        # Accumulator wire
        acc_name = f"{op.name}_acc_{j}"

        if not terms and bias_val == 0:
            # Entire neuron is zero
            lines.append(emit.wire_signed(acc_name, acc_bits))
            lines.append(f"    assign {acc_name} = {emit.slit(acc_bits, 0)};")
        elif not terms:
            # Only bias, no weight terms
            lines.append(
                f"    wire signed [{acc_bits - 1}:0] {acc_name} = "
                f"{emit.slit(acc_bits, bias_val)};  // bias only"
            )
        else:
            # Full MAC expression
            lines.append(f"    wire signed [{acc_bits - 1}:0] {acc_name} =")
            all_terms = list(terms)
            # Add bias as last term
            bias_comment = f"bias={bias_f[j]:.4f}" if bias_val != 0 else ""
            if bias_val != 0:
                all_terms.append(emit.slit(acc_bits, bias_val))

            for idx, t in enumerate(all_terms):
                sep = "+" if idx > 0 else " "
                is_last = (idx == len(all_terms) - 1)
                # Put semicolon BEFORE any comment (// starts comment to EOL)
                if "//" in t:
                    code, cmt = t.split("//", 1)
                    line = f"        {sep} {code.rstrip()}{';' if is_last else ''}  //{cmt}"
                else:
                    cmt = f"  // {bias_comment}" if (is_last and bias_comment and bias_val != 0 and t == all_terms[-1]) else ""
                    line = f"        {sep} {t}{';' if is_last else ''}{cmt}"
                lines.append(line)

        # Requantize
        mult_j = int(requant_mult[j]) if per_channel else int(requant_mult)
        rq_lines, shifted_wire = emit.requantize_lines(
            acc_name, acc_bits, mult_j, requant_shift,
            prefix=f"{op.name}_n{j}",
        )
        lines += rq_lines

        # Saturate (with optional fused ReLU)
        prod_bits = min(acc_bits + 18, 64)
        lines += emit.saturate(shifted_wire, prod_bits, out_wire, bits, activation)
        lines.append("")

    # --- Build output TensorWires ---
    new_wires = {
        out_tensor_name: TensorWires(
            wire_names=out_wire_names,
            shape=(n_out,),
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires
