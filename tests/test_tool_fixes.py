"""
Regression tests for confirmed bugs in the NN->Verilog tool.

Each test below targets one specific confirmed bug. Every test is written to
FAIL against the original (buggy) code and PASS against the fix.

  1. sparsity._prune_to_target overshoots target on quantized (tied) weights.
  2. sparsity.detect_structured_2_4 mislabels plain unstructured sparsity.
  3. wrapper.generate_tiny_tapeout_wrapper silently truncates bits>8.
  4. wrapper serial FSM deadlocks for single-input (n_in==1) designs.
  5. pipeline.build reports SUCCESS / runs downstream stages when required
     upstream stages were only SKIPPED (not actually run).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from w2s.core import ComputeGraph, QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.sparsity import (
    analyze_sparsity,
    prune_weights,
    detect_structured_2_4,
)
from w2s.wrapper import (
    generate_serial_wrapper,
    generate_tiny_tapeout_wrapper,
)
from w2s.graph import compile_graph
from w2s.pipeline import build, detect_tools, ToolStatus


# ---------------------------------------------------------------------------
#  Fix #1 — exact-count pruning on quantized (tied) weights
# ---------------------------------------------------------------------------

def _tied_quantized_graph(bits=4):
    """A graph whose quantized weights have MANY ties at low magnitudes."""
    np.random.seed(0)
    W1 = np.random.randn(16, 16) * 0.3
    b1 = np.zeros(16)
    W2 = np.random.randn(4, 16) * 0.3
    b2 = np.zeros(4)
    gb = GraphBuilder("ties")
    inp = gb.input("x", shape=(16,))
    h = gb.dense(inp, W1, b1, activation="relu", name="h")
    o = gb.dense(h, W2, b2, name="o")
    gb.output(o)
    g = gb.build()
    g.quant_config = QuantConfig(bits=bits)
    quantize_graph(g, {"x": np.random.randn(20, 16)})
    return g


class TestPruneToTargetExact:
    def test_realized_sparsity_matches_target_on_tied_weights(self):
        g = _tied_quantized_graph(bits=4)
        before = analyze_sparsity(g).overall_sparsity
        prune_weights(g, target_sparsity=0.25)
        after = analyze_sparsity(g).overall_sparsity
        # Old threshold-based code overshoots badly (~0.42 for this graph);
        # the fix prunes EXACTLY n weights, landing essentially on target.
        assert after >= before
        assert abs(after - 0.25) <= 0.02, (
            f"realized sparsity {after:.3f} should track requested 0.25"
        )

    def test_high_target(self):
        g = _tied_quantized_graph(bits=4)
        prune_weights(g, target_sparsity=0.6)
        after = analyze_sparsity(g).overall_sparsity
        assert abs(after - 0.6) <= 0.02


# ---------------------------------------------------------------------------
#  Fix #2 — structured 2:4 detection must not flag unstructured sparsity
# ---------------------------------------------------------------------------

class TestDetect24NotFooledByUnstructured:
    def test_unstructured_high_sparsity_not_flagged(self):
        rng = np.random.RandomState(7)
        # ~76% sparse, UNSTRUCTURED: trivially has <=2 non-zeros in almost
        # every group of 4, so the old loose >=90% test wrongly flags it.
        m = (rng.rand(8, 16) < 0.30).astype(np.int64) * rng.randint(1, 8, size=(8, 16))
        sparsity = 1 - np.count_nonzero(m) / m.size
        assert sparsity > 0.6
        assert detect_structured_2_4(m) is False

    def test_genuine_2_4_still_detected(self):
        rng = np.random.RandomState(7)
        W = rng.randn(8, 16)
        flat = W.reshape(-1, 16).copy()
        for r in range(flat.shape[0]):
            for gi in range(4):
                s = gi * 4
                grp = flat[r, s:s + 4]
                idx = np.argsort(np.abs(grp))
                flat[r, s + idx[0]] = 0
                flat[r, s + idx[1]] = 0
        W24 = flat.reshape(8, 16)
        assert detect_structured_2_4(W24) is True


# ---------------------------------------------------------------------------
#  Fix #3 — Tiny Tapeout wrapper must not silently truncate bits>8
# ---------------------------------------------------------------------------

def _single_dense_graph(bits=8):
    gb = GraphBuilder("ttgraph")
    inp = gb.input("x", shape=(2,))
    o = gb.dense(inp, np.eye(2), np.zeros(2), name="o")
    gb.output(o)
    g = gb.build()
    g.quant_config = QuantConfig(bits=bits)
    quantize_graph(g, {"x": np.random.randn(8, 2)})
    return g


class TestTinyTapeoutBitGuard:
    def test_bits_gt_8_raises(self, output_dir):
        g = _single_dense_graph(bits=16)
        with pytest.raises(ValueError) as exc:
            generate_tiny_tapeout_wrapper(g, output_dir=output_dir)
        assert "8" in str(exc.value)

    def test_bits_le_8_ok(self, output_dir):
        g = _single_dense_graph(bits=8)
        path = generate_tiny_tapeout_wrapper(g, output_dir=output_dir)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
#  Fix #4 — serial FSM must reach COMPUTE/DONE for single-input designs
# ---------------------------------------------------------------------------

def _single_input_graph():
    """A 1-input, 1-output quantized graph (n_in == 1)."""
    gb = GraphBuilder("oneinput")
    inp = gb.input("x", shape=(1,))
    o = gb.dense(inp, np.array([[1.0]]), np.array([0.0]), name="o")
    gb.output(o)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"x": np.random.randn(8, 1)})
    return g


class TestSerialSingleInputFSM:
    def test_idle_transitions_to_compute(self, output_dir):
        g = _single_input_graph()
        wpath = generate_serial_wrapper(g, output_dir=output_dir)
        text = Path(wpath).read_text()
        # The IDLE state (which captures in_buf[0]) must move straight to
        # COMPUTE for a single input — never stall in LOADING.
        idle = text.split("IDLE: begin", 1)[1].split("end", 1)[0]
        assert "COMPUTE" in idle

    @pytest.mark.skipif(
        not (shutil.which("iverilog") and shutil.which("vvp")),
        reason="iverilog/vvp not installed",
    )
    def test_fsm_reaches_done_in_simulation(self, output_dir):
        g = _single_input_graph()
        # Combinational core + serial wrapper
        core_v = compile_graph(g, output_dir=output_dir, mode="combinational")
        wrap_v = generate_serial_wrapper(g, output_dir=output_dir)

        wrapper_name = f"{g.name}_serial"
        tb = f"""
`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0, data_valid = 0;
  reg signed [7:0] data_in = 0;
  wire signed [7:0] data_out;
  wire out_valid, done, ready;
  integer i;
  reg done_seen = 0;
  {wrapper_name} dut (.clk(clk), .rst_n(rst_n), .data_in(data_in),
      .data_valid(data_valid), .data_out(data_out),
      .out_valid(out_valid), .done(done), .ready(ready));
  always #5 clk = ~clk;
  initial begin
    rst_n = 0; repeat (3) @(posedge clk);
    rst_n = 1; @(posedge clk);
    data_in = 8'sd3; data_valid = 1; @(posedge clk); data_valid = 0;
    for (i = 0; i < 60; i = i + 1) begin
      @(posedge clk);
      if (done) done_seen = 1;
    end
    if (done_seen) $display("REACHED_DONE");
    else $display("DEADLOCK_NO_DONE");
    $finish;
  end
endmodule
"""
        out = Path(output_dir)
        tb_path = out / "tb_single.v"
        tb_path.write_text(tb)
        sim_bin = str(out / "sim_single")
        comp = subprocess.run(
            ["iverilog", "-g2012", "-o", sim_bin, core_v, wrap_v, str(tb_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert comp.returncode == 0, comp.stderr
        run = subprocess.run(["vvp", sim_bin], capture_output=True, text=True, timeout=60)
        assert "REACHED_DONE" in run.stdout, run.stdout + run.stderr


# ---------------------------------------------------------------------------
#  Fix #5 — skipped upstream stages must not yield SUCCESS / bogus downstream
# ---------------------------------------------------------------------------

def _all_tools_absent():
    names = ["yosys", "nextpnr-ice40", "nextpnr-ecp5",
             "icepack", "iceprog", "ecppack", "iverilog", "vvp"]
    return {n: ToolStatus(name=n, path=None, available=False) for n in names}


class TestSkippedStagesNotSuccess:
    def _graph_and_calib(self, xor_trained_weights):
        W1, b1, W2, b2, X, _y = xor_trained_weights
        gb = GraphBuilder("skip_xor")
        inp = gb.input("x", shape=(2,))
        h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
        out = gb.dense(h, W2, b2, name="output")
        gb.output(out)
        return gb.build(), {"x": X}

    def test_skipped_synth_blocks_success_and_route(
        self, xor_trained_weights, output_dir, monkeypatch
    ):
        graph, calib = self._graph_and_calib(xor_trained_weights)
        # Pretend no FPGA tools are installed.
        monkeypatch.setattr(
            "w2s.pipeline.detect_tools", _all_tools_absent
        )
        result = build(
            graph, calib,
            output_dir=output_dir,
            mode="combinational",
            bits=8,
            target="ice40up5k",
            simulate=False,
            synthesize=True,
            route=True,
            bitstream=True,
        )

        # No stage should have actually FAILED.
        assert not any((not s.passed) for s in result.stages), str(result)

        by_name = {s.name: s for s in result.stages}

        # Synthesis was requested but yosys is absent -> SKIPPED (not run).
        assert by_name["Synthesize"].skipped is True

        # Route must NOT have been run against a missing design.json; it
        # should be recorded as SKIPPED because synthesis did not run.
        assert "Place & Route" in by_name
        assert by_name["Place & Route"].skipped is True
        assert "synth" in by_name["Place & Route"].message.lower() \
            or "no synthesis" in by_name["Place & Route"].message.lower()

        # The crux: with required stages skipped, this is NOT a success.
        assert result.success is False
        s = str(result)
        assert "BUILD SUCCEEDED" not in s
        assert "INCOMPLETE" in s or "SKIP" in s

    def test_pure_python_stages_still_succeed(
        self, xor_trained_weights, output_dir
    ):
        # Sanity: when synth/route/bitstream are not requested, the
        # pure-Python flow still reports success.
        graph, calib = self._graph_and_calib(xor_trained_weights)
        result = build(
            graph, calib,
            output_dir=output_dir,
            mode="combinational",
            bits=8,
            target="ice40up5k",
            simulate=False,
            synthesize=False,
        )
        assert result.success is True
