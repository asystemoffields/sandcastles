"""
End-to-end RTL regression tests for fixes that span the quantizer + generators:

  * structural Add aligns operands quantized at different scales (quantize.py now
    plumbs q_params['input_scales']/['output_scale'] into the Add generator).
  * the builder/importer mark decoder attention causal.
  * combinational, sequential and forward_int all round-to-nearest consistently.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import forward_int
from tests.rtl_harness import simulate, have_iverilog

needs_rtl = pytest.mark.skipif(not have_iverilog(), reason="iverilog/vvp not installed")


@needs_rtl
def test_residual_add_aligns_different_scales():
    """Two dense branches with very different output magnitudes (=> very
    different quant scales) feeding an Add.  The integer sum is only correct if
    each operand is rescaled to the common output scale first."""
    rng = np.random.default_rng(2)
    E = 6
    Wa = rng.standard_normal((E, E)) * 0.1   # small -> fine scale
    Wb = rng.standard_normal((E, E)) * 2.0   # large -> coarse scale
    gb = GraphBuilder("addnet")
    x = gb.input("x", shape=(E,))
    a = gb.dense(x, Wa, np.zeros(E), name="da")
    b = gb.dense(x, Wb, np.zeros(E), name="db")
    s = gb.add(a, b, name="sum")
    gb.output(s)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"x": rng.standard_normal((50, E))})

    add_op = [o for o in g.topological_order() if o.op_type.name == "ADD"][0]
    sa, sb = add_op.q_params["input_scales"]
    assert abs(sa - sb) / max(sa, sb) > 0.5, "test needs genuinely different scales"

    osc = add_op.q_params["output_scale"]
    worst = 0
    for _ in range(10):
        xin = rng.standard_normal(E)
        ideal = np.clip(np.round((xin @ Wa.T + xin @ Wb.T) * osc), -127, 127)
        rtl = simulate(g, {"x": xin})[g.output_names[0]]
        worst = max(worst, int(np.abs(rtl - ideal).max()))
    # A few LSB of requant noise is expected; the broken (unaligned) path would
    # be off by the ~12x scale ratio (full saturation), so this margin still
    # decisively distinguishes aligned from unaligned.
    assert worst <= 6, f"residual add mismatch {worst} (scale alignment broken?)"


def test_builder_marks_causal_attention():
    """gb.mha/gb.gqa accept causal and record it so decoder importers can set
    it; the generator reads attrs['causal']."""
    gb = GraphBuilder("c")
    x = gb.input("x", shape=(4,))
    E = 4
    z = np.zeros(E)
    W = np.eye(E)
    gb.mha(x, q_weight=W, q_bias=z, k_weight=W, k_bias=z, v_weight=W, v_bias=z,
           out_weight=W, out_bias=z, num_heads=2, causal=True, name="a")
    g = gb.build()
    op = [o for o in g.topological_order() if o.op_type.name == "MULTI_HEAD_ATTENTION"][0]
    assert op.attrs.get("causal") is True


@needs_rtl
def test_combinational_matches_forward_int_rounding():
    """After the rounding fix, combinational RTL must match forward_int exactly
    (both round-to-nearest) on a multi-layer dense net."""
    rng = np.random.default_rng(4)
    gb = GraphBuilder("rnd")
    x = gb.input("x", shape=(5,))
    h = gb.dense(x, rng.standard_normal((6, 5)) * 0.4, rng.standard_normal(6) * 0.1,
                 activation="relu", name="h")
    o = gb.dense(h, rng.standard_normal((3, 6)) * 0.4, rng.standard_normal(3) * 0.1,
                 name="o")
    gb.output(o)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"x": rng.standard_normal((50, 5))})
    for _ in range(12):
        xin = rng.standard_normal(5)
        fi = forward_int(g, {"x": xin})[g.output_names[0]].ravel()
        rtl = simulate(g, {"x": xin})[g.output_names[0]]
        assert np.array_equal(rtl, fi), f"RTL {rtl} != golden {fi}"
