"""
Regression tests for two CONFIRMED bugs in the SEQUENTIAL compiler
(w2s/sequential/compile.py).

Bug 1 (topology):  the sequential compiler hard-wires a linear-chain dataflow
    buf0(input) -> buf1(layer0) -> ... -> bufN(output) and only validated op
    *types* (Dense/Reshape/Flatten), never that the graph is actually a single
    linear chain.  A branched / residual / multi-output all-Dense graph was
    therefore silently miscompiled.  It must now raise NotImplementedError
    naming the offending op.

Bug 2 (accumulator width):  the MAC accumulator and bias ROM were hardcoded to
    32 bits, ignoring the quantizer's computed acc_bits.  For int16 a single
    product is ~2^30 and the running sum overflows signed-32 for fan-in >= 2,
    wrapping silently.  The accumulator must now be sized from the per-layer
    required width (> 32 for int16) and produce overflow-free results.

Each test below FAILS on the pre-fix code and PASSES on the fixed code.
"""

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.sequential.compile import compile_sequential
from w2s.graph import generate_sequential_testbench, forward_int


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _linear_chain_graph(name, sizes, bits, seed=0, activation="none"):
    """Build + quantize a simple Dense MLP (single linear chain)."""
    rng = np.random.default_rng(seed)
    gb = GraphBuilder(name)
    t = gb.input("x", shape=(sizes[0],))
    for i in range(len(sizes) - 1):
        n_out, n_in = sizes[i + 1], sizes[i]
        W = rng.uniform(0.5, 1.0, (n_out, n_in))
        b = np.zeros(n_out)
        act = activation if i < len(sizes) - 2 else "none"
        t = gb.dense(t, W, b, activation=act, name=f"d{i}")
    gb.output(t)
    g = gb.build()
    g.quant_config = QuantConfig(bits=bits)
    Xcal = rng.uniform(0.5, 1.0, (8, sizes[0]))
    quantize_graph(g, {"x": Xcal})
    return g


def _mac_acc_width(verilog_text):
    m = re.search(r"reg signed \[(\d+):0\] mac_acc;", verilog_text)
    assert m, "could not find mac_acc declaration"
    return int(m.group(1)) + 1


def _run_sequential_rtl(graph, X):
    """
    Compile `graph` to sequential Verilog, drive it through iverilog/vvp with
    the project's testbench, and return the simulator stdout.  Golden outputs
    come from the integer reference forward_int.  Inputs are pre-quantized
    exactly as forward_int does internally.
    """
    bits = graph.quant_config.bits
    qmax = 2 ** (bits - 1) - 1
    scale = graph.tensor_scales["x"]
    qin = np.clip(np.round(X * scale), -qmax, qmax).astype(np.int64)
    exp = forward_int(graph, {"x": X})
    out_name = graph.output_names[0]

    d = tempfile.mkdtemp()
    vpath = compile_sequential(graph, d)
    tbpath = generate_sequential_testbench(
        graph, {"x": qin}, {out_name: exp[out_name]}, output_dir=d
    )
    sim = os.path.join(d, "sim.vvp")
    r = subprocess.run(
        ["iverilog", "-o", sim, vpath, tbpath],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"iverilog failed:\n{r.stderr}"
    r2 = subprocess.run(["vvp", sim], capture_output=True, text=True)
    return r2.stdout


_HAVE_IVERILOG = shutil.which("iverilog") is not None and shutil.which("vvp") is not None


# ---------------------------------------------------------------------------
#  Bug 1 — topology guard
# ---------------------------------------------------------------------------

class TestLinearChainGuard:

    def test_two_parallel_dense_branch_raises(self):
        """
        Two Dense layers both consuming the graph input is a branch, not a
        linear chain.  The pre-fix compiler silently emitted a chain;
        it must now raise NotImplementedError naming the offending op.
        """
        rng = np.random.default_rng(1)
        gb = GraphBuilder("branch_net")
        x = gb.input("x", shape=(4,))
        a = gb.dense(x, rng.standard_normal((3, 4)), np.zeros(3), name="branch_a")
        b = gb.dense(x, rng.standard_normal((3, 4)), np.zeros(3), name="branch_b")
        gb.output(a)  # single output; branch_b is a parallel consumer of x
        g = gb.build()
        g.quant_config = QuantConfig(bits=8)
        quantize_graph(g, {"x": rng.standard_normal((6, 4))})

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(NotImplementedError) as ei:
                compile_sequential(g, d)
        # The offending op (the second parallel consumer of x) must be named.
        assert "branch_b" in str(ei.value)

    def test_residual_skip_connection_raises(self):
        """
        A residual add re-consuming an earlier activation is not a linear
        chain; sequential mode must refuse it.
        """
        rng = np.random.default_rng(2)
        gb = GraphBuilder("resid_net")
        x = gb.input("x", shape=(4,))
        h = gb.dense(x, rng.standard_normal((4, 4)), np.zeros(4), name="d0")
        h2 = gb.dense(h, rng.standard_normal((4, 4)), np.zeros(4), name="d1")
        s = gb.add(h, h2, name="resid")  # skip connection: consumes h again
        gb.output(s)
        g = gb.build()
        g.quant_config = QuantConfig(bits=8)
        quantize_graph(g, {"x": rng.standard_normal((6, 4))})

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(NotImplementedError) as ei:
                compile_sequential(g, d)
        assert "resid" in str(ei.value)

    def test_multiple_graph_outputs_raises(self):
        """A graph with two outputs cannot be streamed as one buffer."""
        rng = np.random.default_rng(3)
        gb = GraphBuilder("multiout_net")
        x = gb.input("x", shape=(4,))
        h = gb.dense(x, rng.standard_normal((4, 4)), np.zeros(4), name="d0")
        a = gb.dense(h, rng.standard_normal((2, 4)), np.zeros(2), name="head_a")
        b = gb.dense(h, rng.standard_normal((2, 4)), np.zeros(2), name="head_b")
        gb.output(a)
        gb.output(b)
        g = gb.build()
        g.quant_config = QuantConfig(bits=8)
        quantize_graph(g, {"x": rng.standard_normal((6, 4))})

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(NotImplementedError):
                compile_sequential(g, d)

    def test_genuine_linear_chain_still_compiles(self):
        """A real linear-chain MLP must still compile (no false positive)."""
        g = _linear_chain_graph("good_chain", [4, 6, 2], bits=8,
                                seed=4, activation="relu")
        with tempfile.TemporaryDirectory() as d:
            vpath = compile_sequential(g, d)
            assert os.path.exists(vpath)


# ---------------------------------------------------------------------------
#  Bug 2 — accumulator / bias ROM width
# ---------------------------------------------------------------------------

class TestAccumulatorWidth:

    def test_int16_accumulator_wider_than_32(self):
        """
        For int16 a single product is ~2^30, so summing fan-in >= 2 needs more
        than 32 accumulator bits.  The pre-fix compiler hardcoded 32 bits; the
        accumulator (and the bias ROM that feeds it) must now exceed 32 bits.
        """
        g = _linear_chain_graph("acc16", [8, 5, 2], bits=16, seed=5)
        with tempfile.TemporaryDirectory() as d:
            vpath = compile_sequential(g, d)
            text = open(vpath).read()
        width = _mac_acc_width(text)
        assert width > 32, f"int16 accumulator only {width} bits (overflows)"
        # The bias ROM feeds the accumulator and must match its width.
        bias_w = int(re.search(r"function signed \[(\d+):0\] l0_b;", text).group(1)) + 1
        assert bias_w == width, (
            f"bias ROM width {bias_w} != accumulator width {width}"
        )

    def test_int8_accumulator_unchanged(self):
        """int8 designs (which never needed > 32 bits) keep the 32-bit acc."""
        g = _linear_chain_graph("acc8", [8, 5, 2], bits=8, seed=6)
        with tempfile.TemporaryDirectory() as d:
            vpath = compile_sequential(g, d)
            text = open(vpath).read()
        assert _mac_acc_width(text) == 32

    @pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
    def test_int16_no_overflow_rtl(self):
        """
        End-to-end RTL check: an int16 Dense whose accumulator exceeds 2^31
        must match the integer golden reference.  On the pre-fix (32-bit)
        accumulator the running sum wraps and the simulation reports FAIL;
        with the widened accumulator it PASSES.
        """
        # n_in = 4 with near-1.0 positive weights/inputs pushes the layer-0
        # accumulator past signed-32 range (each product ~1.07e9).
        np.random.seed(7)
        n_in = 4
        W1 = np.random.uniform(0.5, 1.0, (3, n_in))
        W2 = np.random.uniform(-1, 1, (2, 3)) * 0.05
        gb = GraphBuilder("ovf16")
        x = gb.input("x", shape=(n_in,))
        h = gb.dense(x, W1, np.zeros(3), name="h")
        o = gb.dense(h, W2, np.zeros(2), name="o")
        gb.output(o)
        g = gb.build()
        g.quant_config = QuantConfig(bits=16)
        quantize_graph(g, {"x": np.random.uniform(0.5, 1.0, (8, n_in))})

        X = np.random.uniform(0.5, 1.0, (1, n_in))
        stdout = _run_sequential_rtl(g, X)
        assert "PASS" in stdout, f"sequential RTL did not pass:\n{stdout}"
        assert "FAIL" not in stdout, f"sequential RTL reported FAIL:\n{stdout}"
