"""
norm.py -- Verilog generators for normalization layers.

LayerNorm:  scale * (x - mean) / sqrt(var + eps) + bias
RMSNorm:   scale * x / sqrt(mean(x^2) + eps)
BatchNorm:  (inference) folded into per-channel affine: y = w*x + b
"""

import math
import numpy as np
from typing import Dict, List, Tuple

from w2s.core import Operation, TensorWires
from w2s import emit


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

ACC_BITS = 32          # width of intermediate accumulators
RSQRT_LUT_BITS = 4    # (legacy, unused) 16-entry LUT for reciprocal-sqrt
RSQRT_FRAC_BITS = 24  # fractional bits in rsqrt result. Must be large enough
                      # that 1/sqrt(var) stays well-resolved as an integer even
                      # for the largest variance (~2^32 at int16): rsqrt_min ~=
                      # 2^(F-16), so F>=24 keeps >=8 bits of resolution there.
INTERP_FRAC_BITS = 8  # fractional bits for linear interpolation delta
NORM_FRAC_BITS = 7    # fractional bits carried in the normalised value (x-mean)/sqrt(var)
NORM_VAR_BITS = 48    # width of the (full-precision) integer variance fed to rsqrt


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2_int(n: int) -> int:
    return int(math.log2(n))


def _precompute_reciprocal(d: int, frac_bits: int = 16) -> int:
    """Fixed-point reciprocal:  round(2**frac_bits / d)."""
    return round((1 << frac_bits) / d)


def _adder_tree_lines(
    inputs: List[str],
    result_name: str,
    acc_bits: int,
    prefix: str,
) -> List[str]:
    """
    Emit a balanced binary adder tree reducing *inputs* into *result_name*.

    Each level halves the number of wires; the final wire is assigned to
    *result_name*.  If the list has a single element the result is just an
    alias.
    """
    if not inputs:
        return [f"    wire signed [{acc_bits - 1}:0] {result_name} = {emit.slit(acc_bits, 0)};"]

    if len(inputs) == 1:
        return [f"    wire signed [{acc_bits - 1}:0] {result_name} = {inputs[0]};"]

    lines: List[str] = []
    cur = list(inputs)
    level = 0

    while len(cur) > 1:
        nxt: List[str] = []
        for pair_idx in range(0, len(cur) - 1, 2):
            w = f"{prefix}_t{level}_{pair_idx // 2}"
            lines.append(
                f"    wire signed [{acc_bits - 1}:0] {w} = {cur[pair_idx]} + {cur[pair_idx + 1]};"
            )
            nxt.append(w)
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
        level += 1

    # Final alias
    lines.append(f"    wire signed [{acc_bits - 1}:0] {result_name} = {cur[0]};")
    return lines


RSQRT_MANT_BITS = 5    # mantissa LUT address bits (32 entries)


def _rsqrt_lut_lines(
    var_wire: str,
    var_bits: int,
    result_wire: str,
    result_bits: int,
    prefix: str,
) -> List[str]:
    """
    Reciprocal-sqrt of a non-negative integer variance, returned in
    Q{RSQRT_FRAC_BITS} fixed-point.

    Correctness across the *full* dynamic range of the variance is obtained by
    range reduction, not a flat LUT.  The variance is written as
    ``var = mantissa * 2**e`` (``mantissa`` in ``[1, 2)``) by finding the
    most-significant-bit position ``e``; ``1/sqrt(mantissa)`` is read from a
    small LUT and the ``2**(-e/2)`` factor is applied by a shift (with an extra
    1/sqrt(2) multiply for odd ``e``).

    The previous implementation binned the variance linearly over
    ``[0, 2**var_bits)``; real variances (tens to thousands) all fell in bin 0,
    so it returned ~1.0 (≈6e-5 in Q14) for every input and the normalised
    output collapsed to zero.  This version is accurate to <1 LSB on average.
    """
    F = RSQRT_FRAC_BITS                                   # result fractional bits
    MB = RSQRT_MANT_BITS                                  # mantissa LUT bits
    SQRT1_2 = int(round((1 << F) * 0.70710678118654752))  # 1/sqrt(2) in QF
    W = var_bits
    ebits = max((W - 1).bit_length(), 1)
    p = prefix
    lines: List[str] = []

    # Clamp to >= 1.  Variance is non-negative; this supplies the eps guard and
    # prevents 1/sqrt(0).
    vc = f"{p}_vc"
    lines.append(
        f"    wire [{W - 1}:0] {vc} = "
        f"({var_wire} <= 0) ? {W}'d1 : {var_wire}[{W - 1}:0];"
    )

    # Leading-one position e:  2**e <= vc < 2**(e+1).
    e = f"{p}_e"
    terms = " : ".join(f"{vc}[{i}] ? {ebits}'d{i}" for i in range(W - 1, 0, -1))
    lines.append(f"    wire [{ebits - 1}:0] {e} = {terms} : {ebits}'d0;")

    # Normalise so the leading one sits at bit W-1; the MB bits just below it
    # form the mantissa fraction (implicit leading 1 dropped).
    nrm = f"{p}_nrm"
    lines.append(f"    wire [{W - 1}:0] {nrm} = {vc} << ({W - 1} - {e});")
    midx = f"{p}_midx"
    lines.append(f"    wire [{MB - 1}:0] {midx} = {nrm}[{W - 2}:{W - 1 - MB}];")

    # LUT:  1/sqrt(1 + midx/2**MB)  in QF.
    mrs = f"{p}_mrs"
    lines.append(f"    reg [{F}:0] {mrs};")
    lines.append(f"    always @(*) begin")
    lines.append(f"        case ({midx})")
    for idx in range(1 << MB):
        val = int(round((1 << F) / math.sqrt(1.0 + idx / (1 << MB))))
        lines.append(f"            {MB}'d{idx}: {mrs} = {F + 1}'d{val};")
    lines.append(f"            default: {mrs} = {F + 1}'d{1 << F};")
    lines.append(f"        endcase")
    lines.append(f"    end")

    # mrs * (1/sqrt(2)) for the odd-exponent case.
    rprod = f"{p}_rprod"
    lines.append(f"    wire [{2 * F}:0] {rprod} = {mrs} * {F + 1}'d{SQRT1_2};")

    # Apply 2**(-e/2):  even e -> mrs >> e/2 ; odd e -> (mrs/sqrt2) >> (e-1)/2.
    rr = f"{p}_rr"
    lines.append(f"    reg [{result_bits - 1}:0] {rr};")
    lines.append(f"    always @(*) begin")
    lines.append(f"        case ({e})")
    for ev in range(W):
        if ev % 2 == 0:
            lines.append(f"            {ebits}'d{ev}: {rr} = {mrs} >> {ev // 2};")
        else:
            lines.append(f"            {ebits}'d{ev}: {rr} = {rprod} >> {F + (ev - 1) // 2};")
    lines.append(f"            default: {rr} = {result_bits}'d0;")
    lines.append(f"        endcase")
    lines.append(f"    end")

    lines.append(f"    wire signed [{result_bits - 1}:0] {result_wire} = {rr};")
    return lines


# ---------------------------------------------------------------------------
#  LayerNorm
# ---------------------------------------------------------------------------

def generate_layernorm(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for LayerNorm.

    LayerNorm(x) = scale * (x - mean) / sqrt(var + eps) + bias

    Steps emitted:
      1. Sign-extend inputs to ACC_BITS.
      2. Compute mean via adder tree + divide by D.
      3. Subtract mean from each element.
      4. Compute variance via adder tree of squared differences / D.
      5. Reciprocal sqrt via 16-entry LUT + interpolation.
      6. For each element i: out[i] = saturate(scale[i] * (x[i] - mean) * rsqrt + bias[i])
    """
    scale = op.q_weights['scale']       # (D,)  int
    bias_q = op.q_weights['bias']       # (D,)  int
    scale_f = op.weights['scale']       # float, for comments
    bias_f = op.weights['bias']         # float, for comments
    D = len(scale)

    in_tensor = wire_map[op.inputs[0]]
    out_tensor_name = op.outputs[0]

    lines: List[str] = []
    p = op.name   # short prefix

    lines += emit.section_comment(f"LayerNorm: {op.name}  (D={D})")
    lines.append("")

    # -- 1. Sign-extend inputs to ACC_BITS --
    se_names: List[str] = []
    for i in range(D):
        src = in_tensor.wire_names[i]
        dst = f"{p}_se_{i}"
        se_names.append(dst)
        lines.append(emit.sign_extend_wire(src, bits, dst, ACC_BITS))
    lines.append("")

    # -- 2. Mean: adder tree then divide by D --
    lines += emit.section_comment("Mean computation")
    sum_wire = f"{p}_sum"
    lines += _adder_tree_lines(se_names, sum_wire, ACC_BITS, prefix=f"{p}_ms")
    lines.append("")

    mean_wire = f"{p}_mean"
    if _is_power_of_2(D):
        shift = _log2_int(D)
        lines.append(
            f"    wire signed [{ACC_BITS - 1}:0] {mean_wire} = {sum_wire} >>> {shift};"
            f"  // /D (D={D}, shift={shift})"
        )
    else:
        recip = _precompute_reciprocal(D, frac_bits=16)
        prod_wire = f"{p}_mean_prod"
        lines.append(
            f"    wire signed [{2 * ACC_BITS - 1}:0] {prod_wire} = "
            f"{sum_wire} * {emit.slit(ACC_BITS, recip)};"
            f"  // * (1/D) in Q16"
        )
        lines.append(
            f"    wire signed [{ACC_BITS - 1}:0] {mean_wire} = "
            f"{prod_wire}[{2 * ACC_BITS - 1}:16];"
        )
    lines.append("")

    # -- 3. Subtract mean from each element --
    lines += emit.section_comment("x - mean")
    diff_names: List[str] = []
    for i in range(D):
        d = f"{p}_diff_{i}"
        diff_names.append(d)
        lines.append(f"    wire signed [{ACC_BITS - 1}:0] {d} = {se_names[i]} - {mean_wire};")
    lines.append("")

    # -- 4. Variance: mean of squared differences (full precision) --
    #    Sum diff^2 at 64-bit width and divide by D with NO early per-term
    #    right-shift.  The old code did `diff^2 >>> bits` per term, which drove
    #    realistic (small) variances to zero and destroyed the normalisation.
    #    var_wire is the true integer variance of x_int.
    lines += emit.section_comment("Variance computation")
    sq_names: List[str] = []
    for i in range(D):
        sq = f"{p}_sq_{i}"
        sq_names.append(sq)
        lines.append(
            f"    wire signed [63:0] {sq} = {diff_names[i]} * {diff_names[i]};"
        )
    lines.append("")

    var_sum_wire = f"{p}_var_sum"
    lines += _adder_tree_lines(sq_names, var_sum_wire, 64, prefix=f"{p}_vs")
    lines.append("")

    var_wire = f"{p}_var"
    if _is_power_of_2(D):
        shift = _log2_int(D)
        lines.append(
            f"    wire [{NORM_VAR_BITS - 1}:0] {var_wire} = {var_sum_wire} >>> {shift};"
            f"  // /D (D={D})"
        )
    else:
        recip = _precompute_reciprocal(D, frac_bits=16)
        vp = f"{p}_var_prod"
        lines.append(
            f"    wire signed [127:0] {vp} = {var_sum_wire} * {emit.slit(64, recip)};"
        )
        lines.append(
            f"    wire [{NORM_VAR_BITS - 1}:0] {var_wire} = "
            f"{vp}[{NORM_VAR_BITS - 1 + 16}:16];"
        )
    lines.append("")

    # -- 5. Reciprocal sqrt of the variance --
    lines += emit.section_comment("Reciprocal sqrt (1/sqrt(var + eps))")
    rsqrt_wire = f"{p}_rsqrt"
    lines += _rsqrt_lut_lines(
        var_wire, NORM_VAR_BITS, rsqrt_wire, ACC_BITS,
        prefix=f"{p}_rlut",
    )
    lines.append("")

    # -- 6. Output:  out[i] = gamma_q[i] * normed[i] + beta_q[i] --
    #    normed[i] = (x[i]-mean)/sqrt(var) in Q{NORM_FRAC_BITS} (dimensionless).
    #    gamma_q/beta_q are quantised at the OUTPUT scale (see quantize.py), so
    #    after the gamma multiply a >>> NORM_FRAC_BITS lands the result directly
    #    in the output integer domain.
    lines += emit.section_comment("Scale, multiply by rsqrt, add bias")
    out_wire_names: List[str] = []

    for i in range(D):
        out_wire = f"{p}_out_{i}"
        out_wire_names.append(out_wire)

        s_val = int(scale[i])
        b_val = int(bias_q[i])
        s_flt = float(scale_f[i])
        b_flt = float(bias_f[i])

        # normed = (x-mean)*rsqrt, keeping NORM_FRAC_BITS fractional bits
        sd = f"{p}_sd_{i}"
        lines.append(
            f"    wire signed [63:0] {sd} = {diff_names[i]} * {rsqrt_wire};"
            f"  // (x-mean)/sqrt(var) << {RSQRT_FRAC_BITS}"
        )
        normed = f"{p}_normed_{i}"
        lines.append(
            f"    wire signed [{ACC_BITS - 1}:0] {normed} = "
            f"{sd} >>> {RSQRT_FRAC_BITS - NORM_FRAC_BITS};"
            f"  // Q{NORM_FRAC_BITS}"
        )

        # out = (gamma_q * normed) >> NORM_FRAC_BITS + beta_q
        pre_sat = f"{p}_pre_{i}"
        lines.append(
            f"    wire signed [{ACC_BITS - 1}:0] {pre_sat} = "
            f"(({emit.slit(ACC_BITS, s_val)} * {normed}) >>> {NORM_FRAC_BITS})"
            f" + {emit.slit(ACC_BITS, b_val)};"
            f"  // s={s_flt:.4f} b={b_flt:.4f}"
        )

        lines += emit.saturate_linear(pre_sat, ACC_BITS, out_wire, bits)
        lines.append("")

    # -- Output wires --
    new_wires = {
        out_tensor_name: TensorWires(
            wire_names=out_wire_names,
            shape=(D,),
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires


# ---------------------------------------------------------------------------
#  RMSNorm
# ---------------------------------------------------------------------------

def generate_rmsnorm(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for RMSNorm.

    RMSNorm(x) = scale * x / sqrt(mean(x^2) + eps)

    Same as LayerNorm but without mean subtraction and without bias.
    """
    scale = op.q_weights['scale']       # (D,)
    scale_f = op.weights['scale']
    D = len(scale)

    in_tensor = wire_map[op.inputs[0]]
    out_tensor_name = op.outputs[0]

    lines: List[str] = []
    p = op.name

    lines += emit.section_comment(f"RMSNorm: {op.name}  (D={D})")
    lines.append("")

    # -- 1. Sign-extend inputs --
    se_names: List[str] = []
    for i in range(D):
        src = in_tensor.wire_names[i]
        dst = f"{p}_se_{i}"
        se_names.append(dst)
        lines.append(emit.sign_extend_wire(src, bits, dst, ACC_BITS))
    lines.append("")

    # -- 2. Mean of squares: sum(x^2) / D, full precision (no early shift) --
    lines += emit.section_comment("Mean of squares")
    sq_names: List[str] = []
    for i in range(D):
        sq = f"{p}_sq_{i}"
        sq_names.append(sq)
        lines.append(
            f"    wire signed [63:0] {sq} = {se_names[i]} * {se_names[i]};"
        )
    lines.append("")

    sq_sum_wire = f"{p}_sqsum"
    lines += _adder_tree_lines(sq_names, sq_sum_wire, 64, prefix=f"{p}_ss")
    lines.append("")

    ms_wire = f"{p}_ms"
    if _is_power_of_2(D):
        shift = _log2_int(D)
        lines.append(
            f"    wire [{NORM_VAR_BITS - 1}:0] {ms_wire} = {sq_sum_wire} >>> {shift};"
            f"  // /D (D={D})"
        )
    else:
        recip = _precompute_reciprocal(D, frac_bits=16)
        prod = f"{p}_ms_prod"
        lines.append(
            f"    wire signed [127:0] {prod} = {sq_sum_wire} * {emit.slit(64, recip)};"
        )
        lines.append(
            f"    wire [{NORM_VAR_BITS - 1}:0] {ms_wire} = "
            f"{prod}[{NORM_VAR_BITS - 1 + 16}:16];"
        )
    lines.append("")

    # -- 3. Reciprocal sqrt of the mean-of-squares --
    lines += emit.section_comment("Reciprocal sqrt (1/sqrt(mean(x^2) + eps))")
    rsqrt_wire = f"{p}_rsqrt"
    lines += _rsqrt_lut_lines(
        ms_wire, NORM_VAR_BITS, rsqrt_wire, ACC_BITS,
        prefix=f"{p}_rlut",
    )
    lines.append("")

    # -- 4. Output: scale[i] * x[i] * rsqrt, then saturate --
    lines += emit.section_comment("Scale and multiply by rsqrt")
    out_wire_names: List[str] = []

    for i in range(D):
        out_wire = f"{p}_out_{i}"
        out_wire_names.append(out_wire)

        s_val = int(scale[i])
        s_flt = float(scale_f[i])

        # normed = x * rsqrt, keeping NORM_FRAC_BITS fractional bits (= x/rms in QN)
        xr = f"{p}_xr_{i}"
        lines.append(
            f"    wire signed [63:0] {xr} = {se_names[i]} * {rsqrt_wire};"
        )
        xr_trunc = f"{p}_xrt_{i}"
        lines.append(
            f"    wire signed [{ACC_BITS - 1}:0] {xr_trunc} = "
            f"{xr} >>> {RSQRT_FRAC_BITS - NORM_FRAC_BITS};"
            f"  // Q{NORM_FRAC_BITS}"
        )

        # out = (gamma_q * normed) >> NORM_FRAC_BITS  (gamma_q quantised at output scale)
        pre_sat = f"{p}_pre_{i}"
        lines.append(
            f"    wire signed [{ACC_BITS - 1}:0] {pre_sat} = "
            f"({emit.slit(ACC_BITS, s_val)} * {xr_trunc}) >>> {NORM_FRAC_BITS};"
            f"  // s={s_flt:.4f}"
        )

        # saturate
        lines += emit.saturate_linear(pre_sat, ACC_BITS, out_wire, bits)
        lines.append("")

    new_wires = {
        out_tensor_name: TensorWires(
            wire_names=out_wire_names,
            shape=(D,),
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires


# ---------------------------------------------------------------------------
#  BatchNorm (inference — folded into affine transform)
# ---------------------------------------------------------------------------

def generate_batchnorm(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for BatchNorm in inference mode.

    At inference time, BN is folded into a per-channel affine transform:
        y[c] = bn_weight[c] * x[c] + bn_bias[c]
    where (computed at compile time in float, then quantized):
        bn_weight[c] = scale[c] / sqrt(running_var[c] + eps)
        bn_bias[c]   = bias[c] - scale[c] * running_mean[c] / sqrt(running_var[c] + eps)

    This yields simple per-element multiply-add with hardwired constants.
    """
    # -- Use pre-quantized weights from the main quantization pipeline --
    # The quantizer stores quantized scale (gamma) and bias (beta) in
    # op.q_weights, and skips running_mean / running_var.  We fold the
    # BN into a per-channel affine using the quantized scale/bias and
    # the float running stats (which are exact constants, not learned).
    scale_q = op.q_weights['scale']                # (C,) int — quantized gamma
    bias_q = op.q_weights['bias']                  # (C,) int — quantized beta

    # Float originals for folding factors and comments
    scale_f = op.weights['scale']                  # (C,)
    bias_f = op.weights['bias']                    # (C,)
    running_mean_f = op.weights['running_mean']    # (C,)
    running_var_f = op.weights['running_var']      # (C,)
    eps = op.attrs.get('eps', 1e-5)
    C = len(scale_f)

    in_tensor = wire_map[op.inputs[0]]
    out_tensor_name = op.outputs[0]

    # -- Resolve channel axis from the explicit op attr ------------------
    # ``c_axis`` is pinned at import time (1 for ONNX NCHW, 0 for (C,...)
    # inputs), removing the heuristic that silently mis-identified the
    # channel dim when batch happened to equal channel count.
    in_shape = tuple(in_tensor.shape)
    c_axis = int(op.attrs.get('c_axis', 1))
    if len(in_shape) == 1:
        c_axis = 0
    if c_axis < 0:
        c_axis += len(in_shape)
    if not (0 <= c_axis < len(in_shape)):
        raise ValueError(
            f"BatchNorm {op.name!r}: c_axis={op.attrs.get('c_axis')} out "
            f"of range for input shape {in_shape}"
        )
    if in_shape[c_axis] != C:
        raise ValueError(
            f"BatchNorm {op.name!r}: input shape {in_shape} has "
            f"{in_shape[c_axis]} channels at axis {c_axis} but BN has "
            f"C={C}.  Check c_axis attr."
        )

    # -- Fold BN into per-channel affine using quantized scale/bias --
    # At inference: y[c] = (scale/sqrt(var+eps)) * x[c]
    #                    + (bias - scale*mean/sqrt(var+eps))
    # The quantizer already mapped scale -> scale_q and bias -> bias_q
    # using the global quantization scale.  We apply the folding factors
    # (1/sqrt(var+eps) and -mean/sqrt(var+eps)) in float and round, so
    # the result stays on the same quantization grid.
    inv_std = 1.0 / np.sqrt(running_var_f + eps)

    # bn_weight_q[c] = round(scale_q[c] * inv_std[c])
    bn_weight_q = np.clip(
        np.round(scale_q.astype(np.float64) * inv_std),
        -(1 << (bits - 1)), (1 << (bits - 1)) - 1,
    ).astype(np.int64)

    # bn_bias_q[c] = bias_q[c] - round(scale_q[c] * mean[c] * inv_std[c])
    bn_bias_q = np.clip(
        np.round(bias_q.astype(np.float64)
                 - scale_q.astype(np.float64) * running_mean_f * inv_std),
        -(1 << (bits - 1)), (1 << (bits - 1)) - 1,
    ).astype(np.int64)

    # Float folded weights for Verilog comments
    bn_weight_f = scale_f * inv_std
    bn_bias_f = bias_f - scale_f * running_mean_f * inv_std

    lines: List[str] = []
    p = op.name

    lines += emit.section_comment(
        f"BatchNorm (folded affine): {op.name}  shape={in_shape} C={C} "
        f"c_axis={c_axis}"
    )
    lines.append(f"    // Folded at compile time: y = bn_w[c]*x + bn_b[c]")
    lines.append(f"    // bn_w[c] = scale[c] / sqrt(running_var[c] + eps)")
    lines.append(f"    // bn_b[c] = bias[c] - scale[c] * running_mean[c] / sqrt(running_var[c] + eps)")
    lines.append("")

    # For each flat output position, figure out which channel it belongs
    # to by decomposing the flat index according to in_shape.  This works
    # for any ndim: (C,), (C, H, W), (1, C, H, W), (N, C, L), etc.
    numel = 1
    for d in in_shape:
        numel *= d
    # Strides for C-order (row-major) flat indexing.
    strides = [1] * len(in_shape)
    for i in range(len(in_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * in_shape[i + 1]

    out_wire_names: List[str] = []
    se_names: List[str] = []

    # Sign-extend every input wire once.
    for i in range(numel):
        src = in_tensor.wire_names[i]
        dst = f"{p}_se_{i}"
        se_names.append(dst)
        lines.append(emit.sign_extend_wire(src, bits, dst, ACC_BITS))
    lines.append("")

    for flat in range(numel):
        c = (flat // strides[c_axis]) % in_shape[c_axis]
        out_wire = f"{p}_out_{flat}"
        out_wire_names.append(out_wire)

        w_val = int(bn_weight_q[c])
        b_val = int(bn_bias_q[c])
        w_flt = float(bn_weight_f[c])
        b_flt = float(bn_bias_f[c])

        acc = f"{p}_acc_{flat}"

        if w_val == 0 and b_val == 0:
            lines.append(
                f"    wire signed [{ACC_BITS - 1}:0] {acc} = {emit.slit(ACC_BITS, 0)};"
                f"  // pos {flat} c={c}"
            )
        elif w_val == 0:
            lines.append(
                f"    wire signed [{ACC_BITS - 1}:0] {acc} = "
                f"{emit.slit(ACC_BITS, b_val)};"
                f"  // pos {flat} c={c} bias only"
            )
        else:
            lines.append(
                f"    wire signed [{ACC_BITS - 1}:0] {acc} = "
                f"(({emit.slit(ACC_BITS, w_val)} * {se_names[flat]}) >>> {bits})"
                f" + {emit.slit(ACC_BITS, b_val)};"
                f"  // pos {flat} c={c} w={w_flt:.4f} b={b_flt:.4f}"
            )

        lines += emit.saturate_linear(acc, ACC_BITS, out_wire, bits)

    lines.append("")

    new_wires = {
        out_tensor_name: TensorWires(
            wire_names=out_wire_names,
            shape=in_shape,
            bits=bits,
            signed=True,
        )
    }

    return lines, new_wires
