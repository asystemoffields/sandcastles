"""
transformer.py -- Verilog generators for modern LLM building blocks.

Covers the four operations needed beyond basic MHA to compile
DeepSeek, Llama, Qwen, Mistral, Gemma, etc.:

  - SwiGLU   (gated FFN with SiLU activation)
  - RoPE     (rotary position embeddings)
  - GQA      (grouped-query attention)
  - KV cache (register-file buffer for sequential inference)

All generators follow the standard protocol: accept (op, wire_map, bits),
return (verilog_lines, new_wires).  Wire names are prefixed with op.name
for uniqueness.  No cross-imports from other generator modules.
"""

import math
import numpy as np
from typing import Dict, List, Tuple

from w2s.core import Operation, TensorWires
from w2s import emit


# ===================================================================
#  SwiGLU
# ===================================================================

def generate_swiglu(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for a SwiGLU feed-forward block.

    SwiGLU(x) = SiLU(x @ gate_weight.T + gate_bias)
              * (x @ up_weight.T + up_bias)
    output    = swiglu_hidden @ down_weight.T + down_bias

    All three projections are hardwired dense layers with requantization.
    SiLU is approximated with an 8-segment piecewise-linear LUT.
    """
    pfx = op.name

    # -- Weights --
    gate_weight = op.q_weights['gate_weight']      # (ffn_dim, embed_dim)
    up_weight   = op.q_weights['up_weight']         # (ffn_dim, embed_dim)
    down_weight = op.q_weights['down_weight']       # (embed_dim, ffn_dim)

    gate_bias = op.q_weights.get('gate_bias')       # (ffn_dim,) or None
    up_bias   = op.q_weights.get('up_bias')
    down_bias = op.q_weights.get('down_bias')

    ffn_dim, embed_dim = gate_weight.shape
    out_dim = down_weight.shape[0]

    # -- Quant params --
    qp = op.q_params
    acc_bits = int(qp['acc_bits'])

    # -- Input / output --
    in_tw = wire_map[op.inputs[0]]
    out_name = op.outputs[0]

    lines: List[str] = []

    # ==================================================================
    #  1. Gate projection:  gate = x @ gate_weight.T + gate_bias
    # ==================================================================
    lines += emit.section_comment(
        f"SwiGLU: {pfx}  ({embed_dim} -> {ffn_dim} -> {out_dim})"
    )
    lines.append("")

    lines += emit.section_comment(
        f"Gate projection ({embed_dim} -> {ffn_dim})"
    )
    lines.append("")

    # Sign-extend input to acc width
    in_se: List[str] = []
    for i in range(embed_dim):
        src = in_tw.wire_names[i]
        dst = f"{pfx}_in_se_{i}"
        in_se.append(dst)
        lines.append(emit.sign_extend_wire(src, bits, dst, acc_bits))
    lines.append("")

    gate_wires = _dense_proj(
        lines, pfx, "gate", in_se,
        gate_weight, gate_bias,
        int(qp['gate_requant_mult']), int(qp['gate_requant_shift']),
        ffn_dim, embed_dim, acc_bits, bits,
    )

    # ==================================================================
    #  2. Up projection:  up = x @ up_weight.T + up_bias
    # ==================================================================
    lines += emit.section_comment(
        f"Up projection ({embed_dim} -> {ffn_dim})"
    )
    lines.append("")

    up_wires = _dense_proj(
        lines, pfx, "up", in_se,
        up_weight, up_bias,
        int(qp['up_requant_mult']), int(qp['up_requant_shift']),
        ffn_dim, embed_dim, acc_bits, bits,
    )

    # ==================================================================
    #  3. SiLU on gate:  silu(x) = x * sigmoid(x)
    # ==================================================================
    #
    #  Piecewise-linear approximation of SiLU with 8 segments.
    #  SiLU(x) = x * sigma(x).  For 8-bit quantised inputs in [-128,127]:
    #
    #  The breakpoints and coefficients are pre-computed for the
    #  quantised integer range.  Slopes/offsets are Q8 fixed-point
    #  (i.e. the true slope is slope_int / 256).
    #
    #  Segments (covering signed 8-bit range, easily scaled for other widths):
    #    x < -96 : silu ~= 0                 (slope=0,   offset=0)
    #    -96...-64: slowly rising from 0      (slope=2,   offset=192)
    #    -64...-32: transition                (slope=16,  offset=1088)
    #    -32...0  : approaching linear        (slope=64,  offset=2624)
    #    0...32   : near-linear               (slope=192, offset=2624)
    #    32...64  : approaching identity      (slope=240, offset=1088)
    #    64...96  : near identity             (slope=254, offset=192)
    #    x >= 96  : silu ~= x                 (slope=256, offset=0)
    # ==================================================================

    lines += emit.section_comment("SiLU activation on gate (8-segment PWL)")
    lines.append("")

    # Scale breakpoints for the actual bit width
    qmax = 2 ** (bits - 1) - 1
    # Breakpoints at 75%, 50%, 25% of negative range and mirrored
    bp_fracs = [-0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75]
    breakpoints = [int(round(f * qmax)) for f in bp_fracs]

    # SiLU slopes in Q8 fixed-point for each of the 8 segments
    silu_slopes  = [0, 2, 16, 64, 192, 240, 254, 256]
    silu_offsets = [0, 192, 1088, 2624, 2624, 1088, 192, 0]

    silu_wires: List[str] = []
    for j in range(ffn_dim):
        silu_out = f"{pfx}_silu_{j}"
        silu_wires.append(silu_out)
        lines += emit.pwl_lut_lines(
            input_wire=gate_wires[j],
            output_wire=silu_out,
            breakpoints=breakpoints,
            slopes=silu_slopes,
            offsets=silu_offsets,
            input_bits=bits,
            output_bits=bits,
            lut_prefix=f"{pfx}_silu_lut{j}",
        )
        lines.append("")

    # ==================================================================
    #  4. Element-wise multiply:  hidden = silu(gate) * up
    # ==================================================================
    #
    #  Sign-extend both operands to 2*bits, multiply, then shift back
    #  by (bits-1) to keep the result in the original range, and
    #  saturate to bits width.
    # ==================================================================

    lines += emit.section_comment("Element-wise multiply: hidden = silu(gate) * up")
    lines.append("")

    mul_bits = 2 * bits
    hidden_wires: List[str] = []

    for j in range(ffn_dim):
        mpfx = f"{pfx}_mul_{j}"
        a_se = f"{mpfx}_a"
        b_se = f"{mpfx}_b"
        prod = f"{mpfx}_prod"
        shifted = f"{mpfx}_sh"
        out_w = f"{pfx}_hidden_{j}"
        hidden_wires.append(out_w)

        lines.append(emit.sign_extend_wire(silu_wires[j], bits, a_se, mul_bits))
        lines.append(emit.sign_extend_wire(up_wires[j], bits, b_se, mul_bits))
        lines.append(
            f"    wire signed [{mul_bits - 1}:0] {prod} = {a_se} * {b_se};"
        )
        # Shift right by (bits-1) to rescale the product
        lines.append(
            f"    wire signed [{mul_bits - 1}:0] {shifted} = "
            f"{prod} >>> {bits - 1};"
        )
        lines += emit.saturate(shifted, mul_bits, out_w, bits)
        lines.append("")

    # ==================================================================
    #  5. Down projection:  output = hidden @ down_weight.T + down_bias
    # ==================================================================

    lines += emit.section_comment(
        f"Down projection ({ffn_dim} -> {out_dim})"
    )
    lines.append("")

    # Sign-extend hidden to acc width
    hidden_se: List[str] = []
    for j in range(ffn_dim):
        src = hidden_wires[j]
        dst = f"{pfx}_hid_se_{j}"
        hidden_se.append(dst)
        lines.append(emit.sign_extend_wire(src, bits, dst, acc_bits))
    lines.append("")

    out_wire_names = _dense_proj(
        lines, pfx, "down", hidden_se,
        down_weight, down_bias,
        int(qp['down_requant_mult']), int(qp['down_requant_shift']),
        out_dim, ffn_dim, acc_bits, bits,
    )

    # -- Output TensorWires --
    new_wires = {
        out_name: TensorWires(
            wire_names=out_wire_names,
            shape=(out_dim,),
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires


# ===================================================================
#  RoPE (Rotary Position Embeddings)
# ===================================================================

def generate_rope(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for rotary position embeddings.

    For each dimension pair (i, i + dim//2):
        x_rot[i]          = x[i]*cos[pos][i] - x[i+dim//2]*sin[pos][i]
        x_rot[i+dim//2]   = x[i]*sin[pos][i] + x[i+dim//2]*cos[pos][i]

    The cos/sin tables are hardwired as ROM indexed by position.
    The position index arrives as a second input wire.
    """
    pfx = op.name

    # -- Attrs and params --
    dim = op.attrs['dim']
    max_seq_len = op.attrs['max_seq_len']
    half = dim // 2

    qp = op.q_params
    requant_mult = int(qp['requant_mult'])
    requant_shift = int(qp['requant_shift'])
    acc_bits = int(qp['acc_bits'])

    # -- Tables --
    cos_table = op.q_weights['cos_table']    # (max_seq_len, half)
    sin_table = op.q_weights['sin_table']    # (max_seq_len, half)

    # -- Inputs --
    in_tw = wire_map[op.inputs[0]]           # (dim,) vector
    pos_tw = wire_map[op.inputs[1]]          # (1,) position index
    pos_wire = pos_tw.wire_names[0]
    out_name = op.outputs[0]

    # Position index bit width -- enough to address max_seq_len
    pos_bits = max(1, math.ceil(math.log2(max(max_seq_len, 2))))

    lines: List[str] = []

    lines += emit.section_comment(
        f"RoPE: {pfx}  dim={dim}, max_seq_len={max_seq_len}"
    )
    lines.append("")

    # ==================================================================
    #  ROM tables for cos and sin, indexed by position
    # ==================================================================
    #
    #  For each half-dimension d (0..half-1), generate a case statement
    #  that maps position -> cos[pos][d] and sin[pos][d].
    #
    #  To keep wire counts manageable, we generate one cos and one sin
    #  wire per half-dimension, selected by the position index.
    # ==================================================================

    lines += emit.section_comment("cos/sin ROM tables (case on position)")
    lines.append("")

    cos_wires: List[str] = []
    sin_wires: List[str] = []

    for d in range(half):
        cos_w = f"{pfx}_cos_{d}"
        sin_w = f"{pfx}_sin_{d}"
        cos_wires.append(cos_w)
        sin_wires.append(sin_w)

        # cos ROM
        lines.append(emit.reg_signed(cos_w, bits))
        lines.append(f"    always @(*) begin")
        lines.append(f"        case ({pos_wire}[{pos_bits - 1}:0])")
        for p in range(max_seq_len):
            lines.append(
                f"            {emit.ulit(pos_bits, p)}: "
                f"{cos_w} = {emit.slit(bits, int(cos_table[p, d]))};"
            )
        lines.append(
            f"            default: {cos_w} = {emit.slit(bits, 0)};"
        )
        lines.append(f"        endcase")
        lines.append(f"    end")

        # sin ROM
        lines.append(emit.reg_signed(sin_w, bits))
        lines.append(f"    always @(*) begin")
        lines.append(f"        case ({pos_wire}[{pos_bits - 1}:0])")
        for p in range(max_seq_len):
            lines.append(
                f"            {emit.ulit(pos_bits, p)}: "
                f"{sin_w} = {emit.slit(bits, int(sin_table[p, d]))};"
            )
        lines.append(
            f"            default: {sin_w} = {emit.slit(bits, 0)};"
        )
        lines.append(f"        endcase")
        lines.append(f"    end")
        lines.append("")

    # ==================================================================
    #  Rotation: MAC with two terms per output element
    # ==================================================================
    #
    #  For pair (d, d+half):
    #    out[d]      = x[d]*cos[d] - x[d+half]*sin[d]
    #    out[d+half] = x[d]*sin[d] + x[d+half]*cos[d]
    # ==================================================================

    lines += emit.section_comment("Rotation: multiply-and-add per pair")
    lines.append("")

    # Products of two acc_bits-wide values need 2*acc_bits bits
    rope_prod_bits = 2 * acc_bits

    out_wire_names: List[str] = [""] * dim

    for d in range(half):
        rpfx = f"{pfx}_rot_{d}"

        x_lo = in_tw.wire_names[d]
        x_hi = in_tw.wire_names[d + half]
        cos_w = cos_wires[d]
        sin_w = sin_wires[d]

        # Sign-extend inputs and table values to acc_bits
        x_lo_se = f"{rpfx}_xlo_se"
        x_hi_se = f"{rpfx}_xhi_se"
        cos_se  = f"{rpfx}_cos_se"
        sin_se  = f"{rpfx}_sin_se"
        lines.append(emit.sign_extend_wire(x_lo, bits, x_lo_se, acc_bits))
        lines.append(emit.sign_extend_wire(x_hi, bits, x_hi_se, acc_bits))
        lines.append(emit.sign_extend_wire(cos_w, bits, cos_se, acc_bits))
        lines.append(emit.sign_extend_wire(sin_w, bits, sin_se, acc_bits))

        # out[d] = x[d]*cos - x[d+half]*sin
        # Use wider intermediate to avoid product overflow
        acc_lo = f"{rpfx}_acc_lo"
        lines.append(
            f"    wire signed [{rope_prod_bits - 1}:0] {acc_lo} = "
            f"({x_lo_se} * {cos_se}) - ({x_hi_se} * {sin_se});"
        )

        rq_lines, shifted_lo = emit.requantize_lines(
            acc_lo, rope_prod_bits, requant_mult, requant_shift,
            prefix=f"{rpfx}_lo",
        )
        lines += rq_lines
        out_lo = f"{pfx}_rout_{d}"
        lines += emit.saturate(shifted_lo, 64, out_lo, bits)
        out_wire_names[d] = out_lo

        # out[d+half] = x[d]*sin + x[d+half]*cos
        # Use wider intermediate to avoid product overflow
        acc_hi = f"{rpfx}_acc_hi"
        lines.append(
            f"    wire signed [{rope_prod_bits - 1}:0] {acc_hi} = "
            f"({x_lo_se} * {sin_se}) + ({x_hi_se} * {cos_se});"
        )

        rq_lines, shifted_hi = emit.requantize_lines(
            acc_hi, rope_prod_bits, requant_mult, requant_shift,
            prefix=f"{rpfx}_hi",
        )
        lines += rq_lines
        out_hi = f"{pfx}_rout_{d + half}"
        lines += emit.saturate(shifted_hi, 64, out_hi, bits)
        out_wire_names[d + half] = out_hi

        lines.append("")

    # -- Output TensorWires --
    new_wires = {
        out_name: TensorWires(
            wire_names=out_wire_names,
            shape=(dim,),
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires


# ===================================================================
#  GQA (Grouped Query Attention)
# ===================================================================

def generate_gqa(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for grouped-query attention.

    Like standard MHA, but K and V heads are shared across groups of
    Q heads.  When computing attention for Q head h, use K/V head
    (h * num_kv_heads // num_heads).  This is pure index remapping --
    no extra logic beyond the smaller K/V projections.
    """
    pfx = op.name

    # -- Attrs --
    num_heads    = op.attrs['num_heads']
    num_kv_heads = op.attrs['num_kv_heads']
    head_dim     = op.attrs['head_dim']
    embed_dim    = op.attrs['embed_dim']
    seq_len      = op.attrs['seq_len']
    causal       = bool(op.attrs.get('causal', False))   # mask j>i for decoders
    heads_per_group = num_heads // num_kv_heads

    assert embed_dim == num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # -- Weights --
    q_weight   = op.q_weights['q_weight']        # (embed_dim, embed_dim)
    q_bias     = op.q_weights['q_bias']           # (embed_dim,)
    k_weight   = op.q_weights['k_weight']         # (kv_dim, embed_dim)
    k_bias     = op.q_weights['k_bias']           # (kv_dim,)
    v_weight   = op.q_weights['v_weight']         # (kv_dim, embed_dim)
    v_bias     = op.q_weights['v_bias']           # (kv_dim,)
    out_weight = op.q_weights['out_weight']       # (embed_dim, embed_dim)
    out_bias   = op.q_weights['out_bias']         # (embed_dim,)

    qp = op.q_params
    acc_bits = int(qp['acc_bits'])

    # -- Input --
    in_tw = wire_map[op.inputs[0]]
    out_name = op.outputs[0]

    lines: List[str] = []

    lines += emit.section_comment(
        f"GQA: {pfx}  heads={num_heads}, kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}, seq_len={seq_len}"
    )
    lines.append("")

    # ==================================================================
    #  1. Q projection: (seq_len, embed_dim) -> (seq_len, embed_dim)
    # ==================================================================
    q_wires = _seq_projection(
        lines, pfx, "q", in_tw.wire_names,
        q_weight, q_bias,
        int(qp['q_requant_mult']), int(qp['q_requant_shift']),
        seq_len, embed_dim, embed_dim, acc_bits, bits,
    )

    # ==================================================================
    #  2. K projection: (seq_len, embed_dim) -> (seq_len, kv_dim)
    # ==================================================================
    k_wires = _seq_projection(
        lines, pfx, "k", in_tw.wire_names,
        k_weight, k_bias,
        int(qp['k_requant_mult']), int(qp['k_requant_shift']),
        seq_len, embed_dim, kv_dim, acc_bits, bits,
    )

    # ==================================================================
    #  3. V projection: (seq_len, embed_dim) -> (seq_len, kv_dim)
    # ==================================================================
    v_wires = _seq_projection(
        lines, pfx, "v", in_tw.wire_names,
        v_weight, v_bias,
        int(qp['v_requant_mult']), int(qp['v_requant_shift']),
        seq_len, embed_dim, kv_dim, acc_bits, bits,
    )

    # ==================================================================
    #  4. Reshape into heads (wire remapping)
    # ==================================================================
    #
    #  Q heads: q_wires[s * embed_dim + h*head_dim + hd]
    #  K/V heads: kv_wires[s * kv_dim + kh*head_dim + hd]
    #
    #  For Q head h, the corresponding K/V head is:
    #    kh = h * num_kv_heads // num_heads   (= h // heads_per_group)
    # ==================================================================

    lines += emit.section_comment(
        f"Reshape: {num_heads} Q heads, {num_kv_heads} K/V heads "
        f"(wire remapping only)"
    )
    lines.append("")

    # q_heads[h][s][hd], k_heads[kh][s][hd], v_heads[kh][s][hd]
    def _reshape(flat: List[str], n_heads: int, dim_per_head: int,
                 total_dim: int) -> List[List[List[str]]]:
        heads: List[List[List[str]]] = []
        for h in range(n_heads):
            head: List[List[str]] = []
            for s in range(seq_len):
                row: List[str] = []
                for hd in range(dim_per_head):
                    d = h * dim_per_head + hd
                    row.append(flat[s * total_dim + d])
                head.append(row)
            heads.append(head)
        return heads

    q_heads = _reshape(q_wires, num_heads, head_dim, embed_dim)
    k_heads = _reshape(k_wires, num_kv_heads, head_dim, kv_dim)
    v_heads = _reshape(v_wires, num_kv_heads, head_dim, kv_dim)

    # ==================================================================
    #  5. Attention scores + ReLU-attention + weighted sum (per Q head)
    # ==================================================================
    #
    #  For Q head h, use K/V head kh = h // heads_per_group.
    #  This is the only difference from standard MHA.
    # ==================================================================

    lines += emit.section_comment(
        f"Attention scores ({num_heads} Q heads, grouped K/V)"
    )
    lines.append("")

    score_acc = emit.acc_bits_for(head_dim, bits)

    # Attention scale: 1/sqrt(head_dim)
    attn_mult = int(qp.get('attn_requant_mult', 0))
    attn_shift = int(qp.get('attn_requant_shift', 0))
    if attn_mult == 0:
        inv_sqrt = 1.0 / math.sqrt(head_dim)
        attn_shift = 15
        attn_mult = max(1, int(round(inv_sqrt * (1 << attn_shift))))

    # ReLU-attention parameters (same as MHA)
    norm_bits = 2 * bits + 4
    frac_shift = bits - 1
    one_fp = 1 << frac_shift

    # Context accumulator width
    ctx_acc_bits = emit.acc_bits_for(seq_len, bits)

    # ctx_wires[h][s][hd]
    ctx_wires: List[List[List[str]]] = []

    for h in range(num_heads):
        kh = h // heads_per_group  # K/V head index for this Q head
        head_ctx: List[List[str]] = []

        lines.append(f"    // --- Q head {h} (K/V head {kh}) ---")

        for i in range(seq_len):
            # --- Compute scores for this (head, position) row ---
            score_row: List[str] = []
            for j in range(seq_len):
                sc_pfx = f"{pfx}_sc_h{h}_i{i}_j{j}"
                acc_w = f"{sc_pfx}_acc"

                se_lines: List[str] = []
                terms: List[str] = []
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
                            k_heads[kh][j][hd], bits, k_se, score_acc
                        )
                    )
                    terms.append(f"({q_se} * {k_se})")
                lines += se_lines

                lines.append(f"    wire signed [{score_acc - 1}:0] {acc_w} =")
                for idx, t in enumerate(terms):
                    sep = " " if idx == 0 else "+"
                    lines.append(f"        {sep} {t}")
                lines[-1] += ";"

                rq_lines, shifted = emit.requantize_lines(
                    acc_w, score_acc, attn_mult, attn_shift,
                    prefix=sc_pfx,
                )
                lines += rq_lines

                sat_wire = f"{sc_pfx}_sat"
                lines += emit.saturate(shifted, 64, sat_wire, bits)
                lines.append("")
                score_row.append(sat_wire)

            # --- ReLU-attention for this row ---
            rpfx = f"{pfx}_relu_h{h}_i{i}"

            # (a) ReLU
            relu_names: List[str] = []
            for j in range(seq_len):
                rw = f"{rpfx}_r{j}"
                relu_names.append(rw)
                if causal and j > i:
                    # Causal mask: future positions contribute nothing.
                    lines.append(
                        f"    wire signed [{bits - 1}:0] {rw} = {bits}'sd0;"
                        f"  // causal mask (j>i)"
                    )
                    continue
                src = score_row[j]
                lines.append(
                    f"    wire signed [{bits - 1}:0] {rw} = "
                    f"({src} > {emit.slit(bits, 0)}) ? {src} : {bits}'sd0;"
                )
            lines.append("")

            # (b) Sum
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

            # (c) Reciprocal via LUT (no variable-divisor division).
            #     After ReLU the scores are non-negative, so the sum
            #     ranges from 0 to seq_len * max_positive.  For each
            #     possible sum value s we precompute
            #         recip = round(scale_factor / s)
            #     then multiply-and-shift instead of dividing.
            recip_bits = norm_bits + frac_shift
            recip_wire = f"{rpfx}_recip"
            numer = one_fp << frac_shift

            max_relu_val = (1 << (bits - 1)) - 1
            max_sum_val = seq_len * max_relu_val
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
                rv = min(rv, (1 << (recip_bits - 1)) - 1)
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

            # (d) Normalised attention weights
            attn_row: List[str] = []
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
                shift_w = f"{rpfx}_sh{j}"
                lines.append(
                    f"    wire signed [{prod_bits - 1}:0] {shift_w} = "
                    f"{prod_w} >>> {frac_shift};"
                )
                lines += emit.saturate(shift_w, prod_bits, aw, bits)
                attn_row.append(aw)
            lines.append("")

            # --- Weighted sum: context[h,i,hd] = sum_j(attn * V) ---
            pos_ctx: List[str] = []
            for hd in range(head_dim):
                cpfx = f"{pfx}_ctx_h{h}_i{i}_d{hd}"
                acc_w = f"{cpfx}_acc"

                se_lines_c: List[str] = []
                c_terms: List[str] = []
                for j in range(seq_len):
                    a_se = f"{cpfx}_ase{j}"
                    v_se = f"{cpfx}_vse{j}"
                    se_lines_c.append(
                        emit.sign_extend_wire(
                            attn_row[j], bits, a_se, ctx_acc_bits
                        )
                    )
                    se_lines_c.append(
                        emit.sign_extend_wire(
                            v_heads[kh][j][hd], bits, v_se, ctx_acc_bits
                        )
                    )
                    c_terms.append(f"({a_se} * {v_se})")
                lines += se_lines_c

                lines.append(
                    f"    wire signed [{ctx_acc_bits - 1}:0] {acc_w} ="
                )
                for idx, t in enumerate(c_terms):
                    sep = " " if idx == 0 else "+"
                    lines.append(f"        {sep} {t}")
                lines[-1] += ";"

                # Rescale by the attn-weight fixed-point scale (2^(bits-1)) before
                # saturating -- without it the weighted sum overflows and every
                # context element clamps to +/-max.  (See generate_mha.)
                ctx_sh = f"{cpfx}_sh"
                lines.append(
                    f"    wire signed [{ctx_acc_bits - 1}:0] {ctx_sh} = "
                    f"{acc_w} >>> {frac_shift};"
                )
                sat_w = f"{cpfx}_sat"
                lines += emit.saturate(ctx_sh, ctx_acc_bits, sat_w, bits)
                pos_ctx.append(sat_w)
                lines.append("")

            head_ctx.append(pos_ctx)
        ctx_wires.append(head_ctx)

    # ==================================================================
    #  6. Concat heads (wire remapping)
    # ==================================================================

    lines += emit.section_comment("Concat heads (wire remapping only)")
    lines.append("")

    concat_flat: List[str] = []
    for s in range(seq_len):
        for h in range(num_heads):
            for hd in range(head_dim):
                concat_flat.append(ctx_wires[h][s][hd])

    # ==================================================================
    #  7. Output projection
    # ==================================================================

    out_wires_flat = _seq_projection(
        lines, pfx, "out", concat_flat,
        out_weight, out_bias,
        int(qp['out_requant_mult']), int(qp['out_requant_shift']),
        seq_len, embed_dim, embed_dim, acc_bits, bits,
    )

    # -- Output TensorWires --
    new_wires = {
        out_name: TensorWires(
            wire_names=out_wires_flat,
            shape=(seq_len, embed_dim),
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires


# ===================================================================
#  KV Cache
# ===================================================================

def generate_kv_cache(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for a KV cache register file.

    Stores K or V vectors token-by-token into a register array,
    and exposes all stored entries as combinational output wires.

    Inputs:
        op.inputs[0] — new K/V vector, shape (num_heads * head_dim,)
        op.inputs[1] — position index, shape (1,)
    Output:
        All cache entries, shape (max_seq_len, num_heads * head_dim)
    """
    pfx = op.name

    num_heads   = op.attrs['num_heads']
    head_dim    = op.attrs['head_dim']
    max_seq_len = op.attrs['max_seq_len']
    vec_dim     = num_heads * head_dim
    total_elems = max_seq_len * vec_dim

    # Position index bit width
    pos_bits = max(1, math.ceil(math.log2(max(max_seq_len, 2))))

    # -- Inputs --
    in_tw  = wire_map[op.inputs[0]]    # (vec_dim,) new vector
    pos_tw = wire_map[op.inputs[1]]    # (1,) position index
    pos_wire = pos_tw.wire_names[0]
    out_name = op.outputs[0]

    # -- Clock and write-enable inputs --
    clk_wire = f"{pfx}_clk"
    wr_en_wire = f"{pfx}_wr_en"

    lines: List[str] = []

    lines += emit.section_comment(
        f"KV Cache: {pfx}  heads={num_heads}, head_dim={head_dim}, "
        f"max_seq_len={max_seq_len}"
    )
    lines.append("")

    # Declare clk and wr_en as input wires
    lines.append(f"    input {clk_wire};")
    lines.append(f"    input {wr_en_wire};")
    lines.append("")

    # ==================================================================
    #  Register file declaration
    # ==================================================================
    #
    #  reg signed [bits-1:0] cache_mem [0:total_elems-1];
    #
    #  Writes are clocked (posedge clk) and gated by wr_en.
    #  Reads are combinational -- all entries exposed as output wires.
    # ==================================================================

    lines.append(
        f"    reg signed [{bits - 1}:0] {pfx}_mem "
        f"[0:{total_elems - 1}];"
    )
    lines.append("")

    # -- Write logic: clocked store of new vector at position --
    lines += emit.section_comment(
        "Write: store new K/V vector at position (clocked)"
    )
    lines.append("")

    lines.append(f"    always @(posedge {clk_wire}) begin")
    lines.append(f"        if ({wr_en_wire}) begin")
    for d in range(vec_dim):
        addr_expr = f"{pos_wire}[{pos_bits - 1}:0] * {vec_dim} + {d}"
        lines.append(
            f"            {pfx}_mem[{addr_expr}] <= "
            f"{in_tw.wire_names[d]};"
        )
    lines.append(f"        end")
    lines.append(f"    end")
    lines.append("")

    # -- Read logic: expose all cache entries as output wires --
    lines += emit.section_comment(
        "Read: expose all cache positions as output wires"
    )
    lines.append("")

    out_wire_names: List[str] = []
    for s in range(max_seq_len):
        for d in range(vec_dim):
            flat_idx = s * vec_dim + d
            out_w = f"{pfx}_out_{s}_{d}"
            out_wire_names.append(out_w)
            lines.append(
                f"    wire signed [{bits - 1}:0] {out_w} = "
                f"{pfx}_mem[{flat_idx}];"
            )
        lines.append("")

    # -- Output TensorWires --
    new_wires = {
        out_name: TensorWires(
            wire_names=out_wire_names,
            shape=(max_seq_len, vec_dim),
            bits=bits,
            signed=True,
        ),
        f"{pfx}_clk": TensorWires(
            wire_names=[clk_wire],
            shape=(1,),
            bits=1,
            signed=False,
        ),
        f"{pfx}_wr_en": TensorWires(
            wire_names=[wr_en_wire],
            shape=(1,),
            bits=1,
            signed=False,
        ),
    }

    return lines, new_wires


# ===================================================================
#  Internal helpers (not exported)
# ===================================================================

def _dense_proj(
    lines: List[str],
    pfx: str,
    proj_tag: str,
    input_se: List[str],
    weight: np.ndarray,
    bias,
    requant_mult: int,
    requant_shift: int,
    n_out: int,
    n_in: int,
    acc_bits: int,
    bits: int,
) -> List[str]:
    """
    Emit a single-position dense projection (used by SwiGLU).

    input_se: list of sign-extended input wire names (already acc_bits wide).
    weight:   (n_out, n_in) quantised weight matrix.
    bias:     (n_out,) quantised bias, or None.

    Returns list of output wire names of length n_out.
    """
    tag = f"{pfx}_{proj_tag}"
    out_wires: List[str] = []

    for j in range(n_out):
        out_w = f"{tag}_out_{j}"
        out_wires.append(out_w)
        acc_name = f"{tag}_acc_{j}"

        # MAC terms, skipping zero weights
        terms: List[str] = []
        for i in range(n_in):
            w = int(weight[j, i])
            if w == 0:
                continue
            terms.append(emit.mac_term(w, input_se[i], acc_bits))

        bias_val = int(bias[j]) if bias is not None else 0

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
            prefix=f"{tag}_n{j}",
        )
        lines += rq_lines

        # Saturate to output bits (linear, no activation)
        lines += emit.saturate(shifted, 64, out_w, bits)
        lines.append("")

    return out_wires


def _seq_projection(
    lines: List[str],
    pfx: str,
    proj_tag: str,
    input_flat: List[str],
    weight: np.ndarray,
    bias: np.ndarray,
    requant_mult: int,
    requant_shift: int,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    acc_bits: int,
    bits: int,
) -> List[str]:
    """
    Emit a dense projection applied independently to each sequence position.

    weight: (out_dim, in_dim)
    bias:   (out_dim,)
    Input layout:  input_flat[s * in_dim + d]
    Output layout: out_flat[s * out_dim + d]

    Returns flat list of output wire names (seq_len * out_dim).
    """
    tag = f"{pfx}_{proj_tag}"

    lines += emit.section_comment(
        f"{proj_tag.upper()} projection: {seq_len} pos x "
        f"{in_dim}->{out_dim}"
    )
    lines.append("")

    out_flat: List[str] = []

    for s in range(seq_len):
        # Sign-extend input wires for this position
        se_names: List[str] = []
        for i in range(in_dim):
            src = input_flat[s * in_dim + i]
            dst = f"{tag}_s{s}_se{i}"
            se_names.append(dst)
            lines.append(emit.sign_extend_wire(src, bits, dst, acc_bits))
        lines.append("")

        for d in range(out_dim):
            out_wire = f"{tag}_s{s}_d{d}"
            out_flat.append(out_wire)
            acc_name = f"{tag}_s{s}_acc{d}"

            # MAC terms, skipping zeros
            terms: List[str] = []
            for i in range(in_dim):
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
                lines.append(
                    f"    wire signed [{acc_bits - 1}:0] {acc_name} ="
                )
                for idx, t in enumerate(terms):
                    sep = " " if idx == 0 else "+"
                    lines.append(f"        {sep} {t}")
                if bias_val != 0:
                    lines.append(
                        f"        + {emit.slit(acc_bits, bias_val)}"
                    )
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
