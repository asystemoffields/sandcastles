"""
attention.py -- Verilog generator for multi-head self-attention.

Produces purely combinational logic for fixed sequence length with
hardwired projection weights.  Every weight constant is burned into
the fabric as a literal -- no RAM, no clocks.

Pipeline:
  1. Q/K/V linear projections  (dense layers with hardwired weights)
  2. Reshape into heads         (wire remapping, zero logic)
  3. Attention scores            (Q * K^T per head, then rescale)
  4. ReLU-attention softmax      (ReLU + normalise-by-sum approximation)
  5. Weighted sum                (attn_weights * V per head)
  6. Concat heads                (wire remapping, zero logic)
  7. Output projection           (dense layer with hardwired weights)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Tuple

from w2s.core import Operation, TensorWires
from w2s import emit


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def generate_mha(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for multi-head self-attention.

    op.attrs must contain:
        num_heads, head_dim, embed_dim, seq_len

    Returns:
        verilog_lines  -- list of Verilog source lines
        new_wires      -- dict mapping output tensor name to TensorWires
    """
    # -- Unpack attrs --
    num_heads = op.attrs['num_heads']
    head_dim = op.attrs['head_dim']
    embed_dim = op.attrs['embed_dim']
    seq_len = op.attrs['seq_len']
    # Causal (autoregressive) attention: position i may only attend to j <= i.
    # Decoder/LM importers must set attrs['causal']=True; otherwise the model
    # would attend to future tokens.  Defaults to False (bidirectional).
    causal = bool(op.attrs.get('causal', False))
    assert embed_dim == num_heads * head_dim

    # -- Quantised weights / params --
    q_weight = op.q_weights['q_weight']        # (embed_dim, embed_dim)
    q_bias = op.q_weights['q_bias']            # (embed_dim,)
    k_weight = op.q_weights['k_weight']
    k_bias = op.q_weights['k_bias']
    v_weight = op.q_weights['v_weight']
    v_bias = op.q_weights['v_bias']
    out_weight = op.q_weights['out_weight']
    out_bias = op.q_weights['out_bias']

    qp = op.q_params
    acc_bits = int(qp['acc_bits'])

    # -- Input tensor: shape (seq_len, embed_dim), row-major flat --
    in_tw = wire_map[op.inputs[0]]
    out_name = op.outputs[0]
    pfx = op.name                              # unique prefix for wires

    lines: List[str] = []

    # ==================================================================
    #  1.  Q / K / V projections
    # ==================================================================
    q_wires = _projection(
        lines, pfx, "q", in_tw,
        q_weight, q_bias,
        int(qp['q_requant_mult']), int(qp['q_requant_shift']),
        seq_len, embed_dim, acc_bits, bits,
    )
    k_wires = _projection(
        lines, pfx, "k", in_tw,
        k_weight, k_bias,
        int(qp['k_requant_mult']), int(qp['k_requant_shift']),
        seq_len, embed_dim, acc_bits, bits,
    )
    v_wires = _projection(
        lines, pfx, "v", in_tw,
        v_weight, v_bias,
        int(qp['v_requant_mult']), int(qp['v_requant_shift']),
        seq_len, embed_dim, acc_bits, bits,
    )

    # ==================================================================
    #  2.  Reshape into heads -- zero-cost wire remapping
    # ==================================================================
    #  q_wires[s * embed_dim + d] -> Q_heads[h][s][hd]
    #  where h = d // head_dim, hd = d % head_dim
    lines += emit.section_comment(
        f"Reshape Q/K/V into {num_heads} heads x {seq_len} positions "
        f"x {head_dim} dims (wire remapping only)"
    )
    lines.append("")

    # Build lookup tables:  heads[h][s][hd] = wire_name
    def _reshape_heads(flat_wires: List[str]) -> List[List[List[str]]]:
        heads: List[List[List[str]]] = []
        for h in range(num_heads):
            head: List[List[str]] = []
            for s in range(seq_len):
                row: List[str] = []
                for hd in range(head_dim):
                    d = h * head_dim + hd
                    row.append(flat_wires[s * embed_dim + d])
                head.append(row)
            heads.append(head)
        return heads

    q_heads = _reshape_heads(q_wires)
    k_heads = _reshape_heads(k_wires)
    v_heads = _reshape_heads(v_wires)

    # ==================================================================
    #  3.  Attention scores:  score[h][i][j] = Q[h,i,:] . K[h,j,:]
    # ==================================================================
    lines += emit.section_comment(
        f"Attention scores: {num_heads} heads, "
        f"{seq_len}x{seq_len} score matrices"
    )
    lines.append("")

    # Precompute 1/sqrt(head_dim) rescale as mult + shift.
    # attn_scale_mult and attn_scale_shift are derived from q_params if
    # present; otherwise we compute our own fixed-point approximation.
    attn_mult = int(qp.get('attn_requant_mult', 0))
    attn_shift = int(qp.get('attn_requant_shift', 0))
    if attn_mult == 0:
        # Approximate 1/sqrt(head_dim) as M >> S with M in ~16-bit range
        inv_sqrt = 1.0 / math.sqrt(head_dim)
        attn_shift = 15
        attn_mult = max(1, int(round(inv_sqrt * (1 << attn_shift))))

    score_acc = emit.acc_bits_for(head_dim, bits)

    # score[h][i][j]  wire names
    score_wires: List[List[List[str]]] = []

    for h in range(num_heads):
        head_scores: List[List[str]] = []
        lines.append(f"    // --- head {h} scores ---")

        for i in range(seq_len):
            row_scores: List[str] = []
            for j in range(seq_len):
                sc_pfx = f"{pfx}_sc_h{h}_i{i}_j{j}"
                acc_w = f"{sc_pfx}_acc"

                # Sign-extend Q and K to score_acc bits, build dot product
                terms: List[str] = []
                se_lines: List[str] = []
                for hd in range(head_dim):
                    q_se = f"{sc_pfx}_qse{hd}"
                    k_se = f"{sc_pfx}_kse{hd}"
                    se_lines.append(
                        emit.sign_extend_wire(
                            q_heads[h][i][hd], bits, q_se, score_acc
                        )
                    )
                    se_lines.append(
                        emit.sign_extend_wire(
                            k_heads[h][j][hd], bits, k_se, score_acc
                        )
                    )
                    terms.append(f"({q_se} * {k_se})")
                lines += se_lines

                # Accumulator
                lines.append(f"    wire signed [{score_acc - 1}:0] {acc_w} =")
                for idx, t in enumerate(terms):
                    sep = " " if idx == 0 else "+"
                    lines.append(f"        {sep} {t}")
                lines[-1] += ";"

                # Rescale by 1/sqrt(head_dim) via requantize
                rq_lines, shifted = emit.requantize_lines(
                    acc_w, score_acc, attn_mult, attn_shift,
                    prefix=sc_pfx,
                )
                lines += rq_lines

                # Saturate back to `bits`
                sat_wire = f"{sc_pfx}_sat"
                lines += emit.saturate(shifted, 64, sat_wire, bits)
                lines.append("")

                row_scores.append(sat_wire)
            head_scores.append(row_scores)
        score_wires.append(head_scores)

    # ==================================================================
    #  4.  Softmax approximation -- ReLU attention
    # ==================================================================
    #
    #  For each (h, i) row of scores:
    #    a) ReLU: clamp negatives to zero
    #    b) Sum the ReLU'd values
    #    c) Compute approximate reciprocal of sum (fixed-point)
    #    d) Multiply each ReLU'd score by the reciprocal
    #
    #  This is "ReLU-attention" -- a well-known linear-attention
    #  approximation that avoids exponentiation entirely.
    # ==================================================================

    lines += emit.section_comment("ReLU-attention softmax approximation")
    lines.append("")

    # Internal precision for the normalised attention weights.
    # We use `bits` for the final attn weights (matching the rest of the
    # datapath) but need wider intermediates during division.
    norm_bits = 2 * bits + 4
    # Maximum positive value the normalised weight can take (full-precision
    # "1.0" in our fixed-point scheme).
    frac_shift = bits - 1   # fractional bits in the output weights
    one_fp = (1 << frac_shift)  # fixed-point 1.0

    attn_wires: List[List[List[str]]] = []  # [h][i][j]

    for h in range(num_heads):
        head_attn: List[List[str]] = []
        lines.append(f"    // --- head {h} ReLU-attention ---")

        for i in range(seq_len):
            row_attn: List[str] = []
            rpfx = f"{pfx}_relu_h{h}_i{i}"

            # (a) ReLU each score
            relu_names: List[str] = []
            for j in range(seq_len):
                rw = f"{rpfx}_r{j}"
                relu_names.append(rw)
                if causal and j > i:
                    # Causal mask: future positions contribute nothing (and are
                    # excluded from the normalising sum).
                    lines.append(
                        f"    wire signed [{bits - 1}:0] {rw} = {bits}'sd0;"
                        f"  // causal mask (j>i)"
                    )
                    continue
                src = score_wires[h][i][j]
                lines.append(
                    f"    wire signed [{bits - 1}:0] {rw} = "
                    f"({src} > {emit.slit(bits, 0)}) ? {src} : {bits}'sd0;"
                )
            lines.append("")

            # (b) Sum the ReLU'd values -- tree of additions.
            #     We sign-extend to norm_bits first to avoid overflow.
            se_relu: List[str] = []
            for j in range(seq_len):
                se_w = f"{rpfx}_se{j}"
                se_relu.append(se_w)
                lines.append(
                    emit.sign_extend_wire(relu_names[j], bits, se_w, norm_bits)
                )

            sum_wire = f"{rpfx}_sum"
            lines.append(f"    wire signed [{norm_bits - 1}:0] {sum_wire} =")
            for j, sw in enumerate(se_relu):
                sep = " " if j == 0 else "+"
                lines.append(f"        {sep} {sw}")
            lines[-1] += ";"
            lines.append("")

            # (c) Approximate reciprocal of sum via LUT.
            #     After ReLU the scores are non-negative, so the sum
            #     ranges from 0 to seq_len * max_positive (where
            #     max_positive = (1 << (bits-1)) - 1).
            #     For each possible sum value s we precompute
            #         recip = round(scale_factor / s)
            #     where scale_factor = one_fp << frac_shift, then
            #     multiply and shift instead of dividing at runtime.
            recip_bits = norm_bits + frac_shift
            recip_wire = f"{rpfx}_recip"
            numer = (one_fp << frac_shift)

            max_relu_val = (1 << (bits - 1)) - 1
            max_sum_val = seq_len * max_relu_val
            # sum_wire is signed norm_bits wide; after ReLU the actual
            # values are non-negative so we use the unsigned magnitude.
            sum_ubits = max(1, max_sum_val.bit_length())

            # Cap reciprocal LUT at 4096 entries to keep synthesis
            # tractable.  For sums beyond the LUT range we reuse the
            # last entry -- the reciprocal is monotonically decreasing
            # so this is a safe (slightly over-weighted) approximation.
            lut_max = min(max_sum_val, 4096)

            lines.append(f"    reg signed [{recip_bits - 1}:0] {recip_wire};")
            lines.append(f"    always @(*) begin")
            lines.append(f"        case ({sum_wire}[{sum_ubits - 1}:0])")
            for s in range(1, lut_max + 1):
                rv = int(round(numer / s))
                rv = min(rv, (1 << (recip_bits - 1)) - 1)  # stay in signed range
                lines.append(
                    f"            {sum_ubits}'d{s}: {recip_wire} = "
                    f"{emit.slit(recip_bits, rv)};"
                )
            # For sums > lut_max, use the last LUT entry's value as default
            if max_sum_val > 4096:
                default_rv = int(round(numer / lut_max))
                default_rv = min(default_rv, (1 << (recip_bits - 1)) - 1)
            else:
                default_rv = 0
            lines.append(
                f"            default: {recip_wire} = {emit.slit(recip_bits, default_rv)};"
            )
            lines.append(f"        endcase")
            lines.append(f"    end")
            lines.append("")

            # (d) Multiply each ReLU'd score by reciprocal, shift back.
            for j in range(seq_len):
                aw = f"{rpfx}_aw{j}"
                prod_w = f"{rpfx}_prod{j}"
                se_relu_j = se_relu[j]
                prod_bits = recip_bits + norm_bits
                lines.append(
                    f"    wire signed [{prod_bits - 1}:0] {prod_w} = "
                    f"{{{{{prod_bits - norm_bits}{{{se_relu_j}[{norm_bits - 1}]}}}}, {se_relu_j}}} "
                    f"* {{{{{prod_bits - recip_bits}{{{recip_wire}[{recip_bits - 1}]}}}}, {recip_wire}}};"
                )
                # Shift right by frac_shift to get back to bits-wide value
                shift_w = f"{rpfx}_sh{j}"
                lines.append(
                    f"    wire signed [{prod_bits - 1}:0] {shift_w} = "
                    f"{prod_w} >>> {frac_shift};"
                )
                # Saturate to bits
                lines += emit.saturate(shift_w, prod_bits, aw, bits)
                row_attn.append(aw)
            lines.append("")

            head_attn.append(row_attn)
        attn_wires.append(head_attn)

    # ==================================================================
    #  5.  Weighted sum:  context[h,i,d] = sum_j(attn[h,i,j] * V[h,j,d])
    # ==================================================================

    lines += emit.section_comment(
        f"Weighted sum: context vectors ({num_heads} heads)"
    )
    lines.append("")

    ctx_acc_bits = emit.acc_bits_for(seq_len, bits)
    # context[h][i][hd]  wire names
    ctx_wires: List[List[List[str]]] = []

    for h in range(num_heads):
        head_ctx: List[List[str]] = []
        lines.append(f"    // --- head {h} context ---")
        for i in range(seq_len):
            pos_ctx: List[str] = []
            for hd in range(head_dim):
                cpfx = f"{pfx}_ctx_h{h}_i{i}_d{hd}"
                acc_w = f"{cpfx}_acc"

                # Sign extend attn and V to ctx_acc_bits, dot-product over j
                se_lines_a: List[str] = []
                terms: List[str] = []
                for j in range(seq_len):
                    a_se = f"{cpfx}_ase{j}"
                    v_se = f"{cpfx}_vse{j}"
                    se_lines_a.append(
                        emit.sign_extend_wire(
                            attn_wires[h][i][j], bits, a_se, ctx_acc_bits
                        )
                    )
                    se_lines_a.append(
                        emit.sign_extend_wire(
                            v_heads[h][j][hd], bits, v_se, ctx_acc_bits
                        )
                    )
                    terms.append(f"({a_se} * {v_se})")
                lines += se_lines_a

                lines.append(f"    wire signed [{ctx_acc_bits - 1}:0] {acc_w} =")
                for idx, t in enumerate(terms):
                    sep = " " if idx == 0 else "+"
                    lines.append(f"        {sep} {t}")
                lines[-1] += ";"

                # Rescale before saturating: the normalised attn weights sum to
                # 2^(bits-1) ("1.0" in Q(bits-1)), so the weighted sum Sum_j
                # attn*V is 2^(bits-1) too large.  Shifting it back down is
                # mandatory -- without it every context element overflows int{bits}
                # and saturates to +/-max regardless of the actual values.
                ctx_sh = f"{cpfx}_sh"
                lines.append(
                    f"    wire signed [{ctx_acc_bits - 1}:0] {ctx_sh} = "
                    f"{acc_w} >>> {frac_shift};  // / 2^(bits-1) attn-weight scale"
                )
                sat_w = f"{cpfx}_sat"
                lines += emit.saturate(ctx_sh, ctx_acc_bits, sat_w, bits)
                pos_ctx.append(sat_w)
                lines.append("")
            head_ctx.append(pos_ctx)
        ctx_wires.append(head_ctx)

    # ==================================================================
    #  6.  Concat heads -- wire remapping (zero logic)
    # ==================================================================
    #  ctx_wires[h][s][hd] -> concat[s][h * head_dim + hd]
    #  Flat order: s * embed_dim + d  where d = h * head_dim + hd

    lines += emit.section_comment(
        "Concat heads (wire remapping only)"
    )
    lines.append("")

    concat_flat: List[str] = []  # length = seq_len * embed_dim
    for s in range(seq_len):
        for h in range(num_heads):
            for hd in range(head_dim):
                concat_flat.append(ctx_wires[h][s][hd])

    # ==================================================================
    #  7.  Output projection
    # ==================================================================

    out_wires_flat = _projection(
        lines, pfx, "out", None,
        out_weight, out_bias,
        int(qp['out_requant_mult']), int(qp['out_requant_shift']),
        seq_len, embed_dim, acc_bits, bits,
        input_flat=concat_flat,
    )

    # --- Build output TensorWires ---
    new_wires = {
        out_name: TensorWires(
            wire_names=out_wires_flat,
            shape=(seq_len, embed_dim),
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _projection(
    lines: List[str],
    pfx: str,
    proj_tag: str,
    in_tw,
    weight: np.ndarray,
    bias: np.ndarray,
    requant_mult: int,
    requant_shift: int,
    seq_len: int,
    embed_dim: int,
    acc_bits: int,
    bits: int,
    input_flat: List[str] | None = None,
) -> List[str]:
    """
    Emit a dense projection applied independently to each sequence
    position.

    weight: (embed_dim, embed_dim)
    bias:   (embed_dim,)
    Input:  (seq_len, embed_dim)  in row-major flat order.

    If *input_flat* is supplied it is used instead of in_tw.wire_names.

    Returns a flat list of output wire names (seq_len * embed_dim).
    """
    tag = f"{pfx}_{proj_tag}"

    lines += emit.section_comment(
        f"{proj_tag.upper()} projection: {seq_len} positions x "
        f"{embed_dim}->{embed_dim}"
    )
    lines.append("")

    flat_in = input_flat if input_flat is not None else in_tw.wire_names
    out_flat: List[str] = []

    for s in range(seq_len):
        # Sign-extend input wires for this position
        se_names: List[str] = []
        for i in range(embed_dim):
            src = flat_in[s * embed_dim + i]
            dst = f"{tag}_s{s}_se{i}"
            se_names.append(dst)
            lines.append(emit.sign_extend_wire(src, bits, dst, acc_bits))
        lines.append("")

        for d in range(embed_dim):
            out_wire = f"{tag}_s{s}_d{d}"
            out_flat.append(out_wire)
            acc_name = f"{tag}_s{s}_acc{d}"

            # MAC terms, skipping zeros
            terms: List[str] = []
            for i in range(embed_dim):
                w = int(weight[d, i])
                if w == 0:
                    continue
                terms.append(emit.mac_term(w, se_names[i], acc_bits))

            bias_val = int(bias[d])

            if not terms and bias_val == 0:
                lines.append(
                    f"    wire signed [{acc_bits - 1}:0] {acc_name} = "
                    f"{emit.slit(acc_bits, 0)};"
                )
            elif not terms:
                lines.append(
                    f"    wire signed [{acc_bits - 1}:0] {acc_name} = "
                    f"{emit.slit(acc_bits, bias_val)};  // bias only"
                )
            else:
                lines.append(f"    wire signed [{acc_bits - 1}:0] {acc_name} =")
                for idx, t in enumerate(terms):
                    sep = " " if idx == 0 else "+"
                    lines.append(f"        {sep} {t}")
                if bias_val != 0:
                    lines.append(f"        + {emit.slit(acc_bits, bias_val)}")
                lines[-1] += ";"

            # Requantize
            rq_lines, shifted = emit.requantize_lines(
                acc_name, acc_bits, requant_mult, requant_shift,
                prefix=f"{tag}_s{s}_n{d}",
            )
            lines += rq_lines

            # Saturate to output bits (linear, no activation)
            lines += emit.saturate(shifted, 64, out_wire, bits)
            lines.append("")

    return out_flat
