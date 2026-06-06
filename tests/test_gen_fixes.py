"""
Tests for confirmed bugs fixed in the Verilog generators.

Each test is RTL-grounded: the emitted Verilog is compiled with Icarus and the
real hardware outputs are compared against an independent integer reference.
Every test is written so that it FAILS against the pre-fix generators and
PASSES against the fixed ones.

Fixes covered:
  1. conv.py     — Conv2D/Conv1D now honour `dilation`.
  2. structural.py — Add/Multiply/Concat now align per-operand fixed-point
                     scales before combining raw integers.
  3. emit.py     — requantize_lines rounds to nearest (was truncating).
  4. emit.py     — requantize_lines sizes the product wire to the real
                   multiplier width (was assuming <=16-bit, truncating MSBs).
"""

import numpy as np
import pytest

from w2s.core import QuantConfig, QuantGranularity
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph

from tests.rtl_harness import simulate, have_iverilog

pytestmark = pytest.mark.skipif(
    not have_iverilog(), reason="iverilog/vvp not available"
)


# ---------------------------------------------------------------------------
#  Integer reference helpers (independent of forward_int, which ignores
#  dilation and truncates — exactly the bugs under test).
# ---------------------------------------------------------------------------

def _round_shift(prod, shift):
    """Arithmetic round-to-nearest right shift, matching fixed requantize_lines."""
    if shift > 0:
        return (prod + (1 << (shift - 1))) >> shift
    return prod >> shift


def _quant_input(x, scale, qmax):
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * scale),
                   -qmax, qmax).astype(np.int64)


def _int_dilated_conv2d(x_int, w, b, stride, pad, dil, mult, shift, qmax,
                        activation="none"):
    Co, Ci, kH, kW = w.shape
    sH, sW = stride
    pH, pW = pad
    dH, dW = dil
    C, H, W = x_int.shape
    kH_eff = (kH - 1) * dH + 1
    kW_eff = (kW - 1) * dW + 1
    H_out = (H + 2 * pH - kH_eff) // sH + 1
    W_out = (W + 2 * pW - kW_eff) // sW + 1
    out = []
    for co in range(Co):
        m = int(mult[co]) if isinstance(mult, np.ndarray) else int(mult)
        for oh in range(H_out):
            for ow in range(W_out):
                acc = 0
                for ci in range(Ci):
                    for r in range(kH):
                        for s in range(kW):
                            ih = oh * sH - pH + r * dH
                            iw = ow * sW - pW + s * dW
                            if ih < 0 or ih >= H or iw < 0 or iw >= W:
                                continue
                            acc += int(w[co, ci, r, s]) * int(x_int[ci, ih, iw])
                acc += int(b[co])
                val = _round_shift(acc * m, int(shift))
                if activation == "relu":
                    val = max(0, min(qmax, val))
                else:
                    val = max(-qmax, min(qmax, val))
                out.append(val)
    return np.array(out, dtype=np.int64)


def _int_dilated_conv1d(x_int, w, b, stride, pad, dil, mult, shift, qmax,
                        activation="none"):
    Co, Ci, kW = w.shape
    sW = stride
    pW = pad
    dW = dil
    C, W = x_int.shape
    kW_eff = (kW - 1) * dW + 1
    W_out = (W + 2 * pW - kW_eff) // sW + 1
    out = []
    for co in range(Co):
        m = int(mult[co]) if isinstance(mult, np.ndarray) else int(mult)
        for ow in range(W_out):
            acc = 0
            for ci in range(Ci):
                for s in range(kW):
                    iw = ow * sW - pW + s * dW
                    if iw < 0 or iw >= W:
                        continue
                    acc += int(w[co, ci, s]) * int(x_int[ci, iw])
            acc += int(b[co])
            val = _round_shift(acc * m, int(shift))
            if activation == "relu":
                val = max(0, min(qmax, val))
            else:
                val = max(-qmax, min(qmax, val))
            out.append(val)
    return np.array(out, dtype=np.int64)


# ===========================================================================
#  Fix 1 — Conv dilation
# ===========================================================================

class TestConvDilation:
    def test_conv2d_dilation(self):
        bits = 8
        qmax = 2 ** (bits - 1) - 1
        np.random.seed(7)
        C_in, C_out, kH, kW = 2, 3, 2, 2
        w = np.random.randn(C_out, C_in, kH, kW) * 0.4
        b = np.random.randn(C_out) * 0.05

        gb = GraphBuilder("conv2d_dil")
        inp = gb.input("img", shape=(C_in, 5, 5))
        co = gb.conv2d(inp, weight=w, bias=b, stride=(1, 1), padding=(0, 0),
                       name="cv")
        gb.output(co)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)

        # Inject dilation (the builder has no dilation kwarg).
        graph.get_op("cv").attrs["dilation"] = (2, 2)

        calib = {"img": np.random.randn(8, C_in, 5, 5) * 0.5}
        quantize_graph(graph, calib)

        op = graph.get_op("cv")
        x = np.random.randn(C_in, 5, 5) * 0.5
        scale_x = graph.tensor_scales["img"]
        x_int = _quant_input(x, scale_x, qmax)

        ref_dil = _int_dilated_conv2d(
            x_int, op.q_weights["weight"], op.q_weights["bias"],
            (1, 1), (0, 0), (2, 2),
            op.q_params["requant_mult"], op.q_params["requant_shift"], qmax,
        )
        ref_nodil = _int_dilated_conv2d(
            x_int, op.q_weights["weight"], op.q_weights["bias"],
            (1, 1), (0, 0), (1, 1),
            op.q_params["requant_mult"], op.q_params["requant_shift"], qmax,
        )
        # The test must be non-trivial: dilation must actually change the result
        # (else it could pass against the buggy generator by coincidence).
        # Output geometry differs (3x3 vs 4x4), so compare against the buggy
        # generator's would-be output: it ignores dilation entirely.
        assert ref_dil.size == C_out * 3 * 3

        rtl = simulate(graph, {"img": x})
        out = rtl[op.outputs[0]]
        assert out.shape == ref_dil.shape, (out.shape, ref_dil.shape)
        assert np.array_equal(out, ref_dil), (
            f"RTL conv2d (dilation=2) != dilated reference\n"
            f"  rtl={out.tolist()}\n  ref={ref_dil.tolist()}"
        )
        # And confirm dilation genuinely matters for these weights/inputs.
        assert not np.array_equal(ref_dil, ref_nodil[: ref_dil.size])

    def test_conv1d_dilation(self):
        bits = 8
        qmax = 2 ** (bits - 1) - 1
        np.random.seed(11)
        C_in, C_out, kW = 2, 2, 3
        w = np.random.randn(C_out, C_in, kW) * 0.4
        b = np.random.randn(C_out) * 0.05

        # Build a Conv1D op directly (builder lacks a conv1d helper).
        from w2s.core import OpType, Operation
        gb = GraphBuilder("conv1d_dil")
        inp = gb.input("seq", shape=(C_in, 9))
        op = Operation(
            op_type=OpType.CONV1D, name="cv1",
            inputs=[inp], outputs=["cv1_out"],
            attrs={"kernel_size": (kW,), "stride": (1,), "padding": (0,),
                   "dilation": (2,)},
            weights={"weight": w, "bias": b},
        )
        gb.graph.add(op)
        gb.output("cv1_out")
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)

        calib = {"seq": np.random.randn(8, C_in, 9) * 0.5}
        quantize_graph(graph, calib)

        op = graph.get_op("cv1")
        x = np.random.randn(C_in, 9) * 0.5
        scale_x = graph.tensor_scales["seq"]
        x_int = _quant_input(x, scale_x, qmax)

        # effective kernel = (3-1)*2+1 = 5 -> W_out = 9-5+1 = 5
        ref_dil = _int_dilated_conv1d(
            x_int, op.q_weights["weight"], op.q_weights["bias"],
            1, 0, 2,
            op.q_params["requant_mult"], op.q_params["requant_shift"], qmax,
        )
        ref_nodil = _int_dilated_conv1d(
            x_int, op.q_weights["weight"], op.q_weights["bias"],
            1, 0, 1,
            op.q_params["requant_mult"], op.q_params["requant_shift"], qmax,
        )
        assert ref_dil.size == C_out * 5

        rtl = simulate(graph, {"seq": x})
        out = rtl["cv1_out"]
        assert out.shape == ref_dil.shape
        assert np.array_equal(out, ref_dil), (
            f"RTL conv1d (dilation=2) != dilated reference\n"
            f"  rtl={out.tolist()}\n  ref={ref_dil.tolist()}"
        )
        assert not np.array_equal(ref_dil, ref_nodil[: ref_dil.size])


# ===========================================================================
#  Fix 1 regression — plain conv (dilation=1) still exact
# ===========================================================================

class TestConvNoDilationRegression:
    def test_conv2d_default_matches_reference(self):
        """Default (dilation=1) conv path must stay exact after the dilation
        and requant-rounding changes."""
        bits = 8
        qmax = 2 ** (bits - 1) - 1
        np.random.seed(3)
        C_in, C_out, kH, kW = 1, 4, 3, 3
        w = np.random.randn(C_out, C_in, kH, kW) * 0.5
        b = np.random.randn(C_out) * 0.1

        gb = GraphBuilder("conv2d_plain")
        inp = gb.input("image", shape=(C_in, 8, 8))
        co = gb.conv2d(inp, weight=w, bias=b, stride=(1, 1), padding=(0, 0),
                       activation="relu", name="conv1")
        gb.output(co)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)
        quantize_graph(graph, {"image": np.random.randn(10, C_in, 8, 8) * 0.5})

        op = graph.get_op("conv1")
        x = np.random.randn(C_in, 8, 8) * 0.5
        scale_x = graph.tensor_scales["image"]
        x_int = _quant_input(x, scale_x, qmax)
        ref = _int_dilated_conv2d(
            x_int, op.q_weights["weight"], op.q_weights["bias"],
            op.attrs["stride"], op.attrs["padding"], (1, 1),
            op.q_params["requant_mult"], op.q_params["requant_shift"], qmax,
            activation=op.attrs.get("activation", "none"),
        )
        rtl = simulate(graph, {"image": x})
        out = rtl[op.outputs[0]]
        assert np.array_equal(out, ref), (
            f"default conv2d RTL != reference\n  rtl={out.tolist()}\n  ref={ref.tolist()}"
        )


# ===========================================================================
#  Fix 2 — structural per-operand scale alignment
# ===========================================================================

def _build_two_dense(scale_a_w, scale_b_w, seed, combine, name):
    """Two dense layers (different weight magnitudes -> different output
    scales) feeding a structural op.  Returns (graph, op_names...)."""
    np.random.seed(seed)
    Wa = np.random.randn(4, 4) * scale_a_w
    Wb = np.random.randn(4, 4) * scale_b_w
    gb = GraphBuilder(name)
    x = gb.input("x", shape=(4,))
    a = gb.dense(x, Wa, name="da")
    b = gb.dense(x, Wb, name="db")
    out = combine(gb, a, b)
    gb.output(a)
    gb.output(b)
    gb.output(out)
    graph = gb.build()
    graph.quant_config = QuantConfig(bits=8,
                                     granularity=QuantGranularity.PER_TENSOR)
    calib = {"x": np.random.randn(16, 4) * 1.0}
    quantize_graph(graph, calib)
    return graph, "da", "db", out


def _thread_scales(graph, op_name, in_tensor_names, out_tensor_name):
    op = graph.get_op(op_name)
    op.q_params["input_scales"] = [graph.tensor_scales[t] for t in in_tensor_names]
    op.q_params["output_scale"] = graph.tensor_scales[out_tensor_name]


class TestStructuralScaleAlignment:
    def test_add_different_scales(self):
        bits = 8
        qmax = 2 ** (bits - 1) - 1
        graph, da, db, out = _build_two_dense(
            scale_a_w=2.0, scale_b_w=0.2, seed=5,
            combine=lambda gb, a, b: gb.add(a, b, name="res"), name="addscale")
        a_t, b_t = "da_out", "db_out"
        s_a = graph.tensor_scales[a_t]
        s_b = graph.tensor_scales[b_t]
        s_out = graph.tensor_scales[out]
        # Scales must genuinely differ for this to exercise the fix.
        assert max(s_a, s_b) / min(s_a, s_b) > 2.0

        _thread_scales(graph, "res", [a_t, b_t], out)

        x = np.random.randn(4) * 1.0
        rtl = simulate(graph, {"x": x})
        a_int = rtl[a_t].astype(np.float64)
        b_int = rtl[b_t].astype(np.float64)
        expected = np.clip(
            np.round((a_int / s_a + b_int / s_b) * s_out), -qmax, qmax
        ).astype(np.int64)
        got = rtl[out]
        assert np.max(np.abs(got - expected)) <= 1, (
            f"Add RTL != rescaled reference\n  rtl={got.tolist()}\n"
            f"  expected={expected.tolist()}"
        )
        # The buggy generator does raw a+b; show that differs from correct.
        raw = np.clip(rtl[a_t] + rtl[b_t], -qmax, qmax)
        assert not np.array_equal(raw, expected)

    def test_concat_different_scales(self):
        bits = 8
        qmax = 2 ** (bits - 1) - 1
        graph, da, db, out = _build_two_dense(
            scale_a_w=2.0, scale_b_w=0.2, seed=8,
            combine=lambda gb, a, b: gb.concat([a, b], axis=0, name="cat"),
            name="catscale")
        a_t, b_t = "da_out", "db_out"
        s_a = graph.tensor_scales[a_t]
        s_b = graph.tensor_scales[b_t]
        s_out = graph.tensor_scales[out]
        assert max(s_a, s_b) / min(s_a, s_b) > 2.0

        _thread_scales(graph, "cat", [a_t, b_t], out)

        x = np.random.randn(4) * 1.0
        rtl = simulate(graph, {"x": x})
        a_int = rtl[a_t].astype(np.float64)
        b_int = rtl[b_t].astype(np.float64)
        exp_a = np.clip(np.round(a_int / s_a * s_out), -qmax, qmax)
        exp_b = np.clip(np.round(b_int / s_b * s_out), -qmax, qmax)
        expected = np.concatenate([exp_a, exp_b]).astype(np.int64)
        got = rtl[out]
        assert np.max(np.abs(got - expected)) <= 1, (
            f"Concat RTL != rescaled reference\n  rtl={got.tolist()}\n"
            f"  expected={expected.tolist()}"
        )
        # Buggy generator just rewires (raw concat); show that differs.
        raw = np.concatenate([rtl[a_t], rtl[b_t]])
        assert not np.array_equal(raw, expected)

    def test_multiply_different_scales(self):
        bits = 8
        qmax = 2 ** (bits - 1) - 1
        graph, da, db, out = _build_two_dense(
            scale_a_w=1.5, scale_b_w=0.3, seed=13,
            combine=lambda gb, a, b: gb.multiply(a, b, name="mul"),
            name="mulscale")
        a_t, b_t = "da_out", "db_out"
        s_a = graph.tensor_scales[a_t]
        s_b = graph.tensor_scales[b_t]
        s_out = graph.tensor_scales[out]

        _thread_scales(graph, "mul", [a_t, b_t], out)

        x = np.random.randn(4) * 1.0
        rtl = simulate(graph, {"x": x})
        a_int = rtl[a_t].astype(np.float64)
        b_int = rtl[b_t].astype(np.float64)
        expected = np.clip(
            np.round((a_int / s_a) * (b_int / s_b) * s_out), -qmax, qmax
        ).astype(np.int64)
        got = rtl[out]
        assert np.max(np.abs(got - expected)) <= 1, (
            f"Multiply RTL != rescaled reference\n  rtl={got.tolist()}\n"
            f"  expected={expected.tolist()}"
        )


# ===========================================================================
#  Fix 3 — round-to-nearest requantization
# ===========================================================================

class TestRequantRounding:
    def test_dense_rounds_to_nearest(self):
        bits = 8
        gb = GraphBuilder("rnd")
        x = gb.input("x", shape=(1,))
        out = gb.dense(x, np.array([[0.5]]), name="drnd")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)
        quantize_graph(graph, {"x": np.array([[1.0]])})

        # Override to a controlled requant: acc = 3, mult = 3, shift = 1.
        #   truncating  : (3*3) >> 1            = 4
        #   round-nearest: (3*3 + 1) >> 1       = 5
        op = graph.get_op("drnd")
        graph.tensor_scales["x"] = 1.0
        op.q_weights["weight"] = np.array([[1]], dtype=np.int64)
        op.q_params["requant_mult"] = 3
        op.q_params["requant_shift"] = 1
        op.q_params["acc_bits"] = 16

        rtl = simulate(graph, {"x": np.array([3.0])})
        got = int(rtl["drnd_out"][0])
        assert got == 5, f"expected round-to-nearest 5, got {got} (truncation gives 4)"


# ===========================================================================
#  Fix 4 — product-wire width fits the real multiplier
# ===========================================================================

class TestRequantWideMultiplier:
    def test_wide_multiplier_no_msb_truncation(self):
        bits = 8
        qmax = 2 ** (bits - 1) - 1
        gb = GraphBuilder("wide")
        x = gb.input("x", shape=(1,))
        out = gb.dense(x, np.array([[0.5]]), name="dwide")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)
        quantize_graph(graph, {"x": np.array([[1.0]])})

        # acc = bias = 2**22, multiplier ~2**21 (22-bit, > the old 16-bit
        # assumption), shift = 20.  acc_bits = 24.
        #   true product = 2**22 * (2**21+1) = 2**43 + 2**22  (needs ~44 bits)
        #   old prod_bits = acc_bits + 18 = 42  -> product wraps mod 2**42,
        #                   collapses to a tiny value -> output 4
        #   fixed         -> 2**43>>20 = 2**23 -> saturates to qmax (127)
        op = graph.get_op("dwide")
        graph.tensor_scales["x"] = 1.0
        op.q_weights["weight"] = np.array([[0]], dtype=np.int64)
        op.q_weights["bias"] = np.array([2 ** 22], dtype=np.int64)
        op.q_params["requant_mult"] = 2 ** 21 + 1
        op.q_params["requant_shift"] = 20
        op.q_params["acc_bits"] = 24

        # independent reference (full-precision python ints)
        acc = 2 ** 22
        prod = acc * (2 ** 21 + 1)
        ref = _round_shift(prod, 20)
        ref = max(-qmax, min(qmax, ref))
        assert ref == qmax  # sanity: this saturates

        rtl = simulate(graph, {"x": np.array([1.0])})
        got = int(rtl["dwide_out"][0])
        assert got == ref, (
            f"wide-multiplier requant truncated MSBs: got {got}, expected {ref}"
        )
