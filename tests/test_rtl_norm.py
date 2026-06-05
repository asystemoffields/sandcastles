"""
RTL regression tests for LayerNorm / RMSNorm.

These simulate the *emitted Verilog* (via iverilog) and compare it to the true
floating-point normalisation.  Before the audit fixes the emitted hardware
collapsed every normalised output to ~0 (the rsqrt LUT binned variance over
[0, 2**32) and returned ~1.0 for all realistic variances), so these tests
would have produced ~100% error.  They now hold to a few percent.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from tests.rtl_harness import simulate, quantize_inputs, have_iverilog

pytestmark = pytest.mark.skipif(not have_iverilog(),
                                reason="iverilog/vvp not installed")


def _run(kind, D, bits, seed):
    np.random.seed(seed)
    gamma = np.random.randn(D) * 0.5 + 1.0
    beta = np.random.randn(D) * 0.2
    gb = GraphBuilder(f"{kind}_{D}_{bits}")
    x = gb.input("x", shape=(D,))
    y = gb.layernorm(x, gamma, beta, name="n") if kind == "ln" \
        else gb.rmsnorm(x, gamma, name="n")
    gb.output(y)
    g = gb.build()
    g.quant_config = QuantConfig(bits=bits)
    quantize_graph(g, {"x": np.random.randn(30, D) * 2.0})
    osc = g.tensor_scales[g.output_names[0]]
    qmax = 2 ** (bits - 1) - 1

    worst = 0.0
    nonzero_seen = False
    for _ in range(8):
        xin = np.random.randn(D) * np.random.uniform(0.5, 3.0)
        qin = quantize_inputs(g, {"x": xin})["x"].astype(np.float64)
        if kind == "ln":
            n = (qin - qin.mean()) / np.sqrt(qin.var() + 1e-5)
            ideal = np.clip(np.round((gamma * n + beta) * osc), -qmax, qmax)
        else:
            n = qin / np.sqrt((qin ** 2).mean() + 1e-5)
            ideal = np.clip(np.round(gamma * n * osc), -qmax, qmax)
        rtl = simulate(g, {"x": xin})[g.output_names[0]]
        nonzero_seen = nonzero_seen or np.any(np.abs(rtl) > qmax // 8)
        worst = max(worst, np.abs(rtl - ideal).max() / qmax)
    # The output must actually track the input (regression guard against the
    # old "everything collapses to ~0 / beta" failure mode)...
    assert nonzero_seen, f"{kind} D={D} bits={bits}: RTL output never left zero"
    # ...and match the true float normalisation to a few percent.
    assert worst < 0.05, f"{kind} D={D} bits={bits}: rel-err {worst:.3f} too high"


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("D", [8, 16, 32])
def test_layernorm_rtl_matches_float(D, bits):
    _run("ln", D, bits, seed=D * bits + 1)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("D", [8, 16, 32])
def test_rmsnorm_rtl_matches_float(D, bits):
    _run("rms", D, bits, seed=D * bits + 7)
