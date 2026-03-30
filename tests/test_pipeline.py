"""Tests for the build pipeline.

Tests the pure-Python stages (quantize, compile, testbench) end-to-end,
and the tool detection logic. External tools (yosys, iverilog) are not
required — those stages are tested for correct skip behavior.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.pipeline import (
    detect_tools,
    build,
    ToolStatus,
    PipelineResult,
    _stage_quantize,
    _stage_compile,
    _stage_testbench,
)


# ---------------------------------------------------------------------------
#  Fixture: fresh unquantized graph (pipeline needs to quantize it)
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_xor_graph(xor_trained_weights):
    """Build a fresh (unquantized) XOR graph for pipeline tests."""
    W1, b1, W2, b2, _X, _y = xor_trained_weights
    gb = GraphBuilder("pipeline_xor")
    inp = gb.input("x", shape=(2,))
    h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
    out = gb.dense(h, W2, b2, name="output")
    gb.output(out)
    return gb.build()


@pytest.fixture
def xor_calib(xor_trained_weights):
    _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
    return {"x": X}


# ---------------------------------------------------------------------------
#  Tool detection
# ---------------------------------------------------------------------------

class TestToolDetection:
    def test_returns_dict(self):
        tools = detect_tools()
        assert isinstance(tools, dict)
        assert "yosys" in tools
        assert "iverilog" in tools
        assert "vvp" in tools
        assert "nextpnr-ice40" in tools
        assert "nextpnr-ecp5" in tools
        assert "icepack" in tools
        assert "ecppack" in tools

    def test_tool_status_fields(self):
        tools = detect_tools()
        for name, status in tools.items():
            assert isinstance(status, ToolStatus)
            assert isinstance(status.available, bool)
            assert status.name == name


# ---------------------------------------------------------------------------
#  Individual stage tests
# ---------------------------------------------------------------------------

class TestStageQuantize:
    def test_passes(self, fresh_xor_graph, xor_calib):
        stage = _stage_quantize(fresh_xor_graph, xor_calib, 8, None)
        assert stage.passed
        assert stage.name == "Quantize"
        assert "params" in stage.message

    def test_populates_q_weights(self, fresh_xor_graph, xor_calib):
        _stage_quantize(fresh_xor_graph, xor_calib, 8, None)
        weighted = [op for op in fresh_xor_graph.operations if op.q_weights]
        assert len(weighted) > 0

    def test_different_bit_widths(self, xor_trained_weights):
        for bits in [4, 8, 16]:
            W1, b1, W2, b2, X, _y = xor_trained_weights
            gb = GraphBuilder(f"test_{bits}")
            inp = gb.input("x", shape=(2,))
            h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
            out = gb.dense(h, W2, b2, name="output")
            gb.output(out)
            graph = gb.build()
            stage = _stage_quantize(graph, {"x": X}, bits, None)
            assert stage.passed, f"Failed at bits={bits}"


class TestStageCompile:
    def test_combinational(self, xor_quantized_graph, output_dir):
        stage = _stage_compile(xor_quantized_graph, output_dir, "combinational")
        assert stage.passed
        assert len(stage.output_files) > 0
        assert any(f.endswith(".v") for f in stage.output_files)
        assert "lines" in stage.message

    def test_sequential(self, xor_quantized_graph, output_dir):
        stage = _stage_compile(xor_quantized_graph, output_dir, "sequential")
        assert stage.passed
        assert len(stage.output_files) > 0


class TestStageTestbench:
    def test_generates_testbench(self, xor_quantized_graph, xor_trained_weights, output_dir):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        # First compile so the output dir has the Verilog
        _stage_compile(xor_quantized_graph, output_dir, "combinational")
        stage = _stage_testbench(xor_quantized_graph, {"x": X}, output_dir, 8)
        assert stage.passed
        assert len(stage.output_files) > 0
        assert "vectors" in stage.message


# ---------------------------------------------------------------------------
#  Full pipeline (pure-Python stages only)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_xor_build_no_external_tools(self, fresh_xor_graph, xor_calib, output_dir):
        """Run the full pipeline with simulate/synthesize disabled."""
        result = build(
            fresh_xor_graph, xor_calib,
            output_dir=output_dir,
            mode="combinational",
            bits=8,
            target="ice40up5k",
            simulate=False,
            synthesize=False,
        )
        assert isinstance(result, PipelineResult)
        # Quantize + Compile + Testbench should pass
        passed_stages = [s for s in result.stages if s.passed]
        assert len(passed_stages) >= 3
        # All stages should pass (no external tools needed)
        assert result.success

    def test_sequential_mode(self, fresh_xor_graph, xor_calib, output_dir):
        result = build(
            fresh_xor_graph, xor_calib,
            output_dir=output_dir,
            mode="sequential",
            bits=8,
            target="ice40up5k",
            simulate=False,
            synthesize=False,
        )
        assert result.success

    def test_pipeline_result_str(self, fresh_xor_graph, xor_calib, output_dir):
        result = build(
            fresh_xor_graph, xor_calib,
            output_dir=output_dir,
            mode="combinational",
            bits=8,
            target="ice40up5k",
            simulate=False,
            synthesize=False,
        )
        s = str(result)
        assert "Pipeline" in s
        assert "PASS" in s

    def test_invalid_device_fails_gracefully(self, fresh_xor_graph, xor_calib, output_dir):
        result = build(
            fresh_xor_graph, xor_calib,
            output_dir=output_dir,
            target="nonexistent_fpga",
            simulate=False,
            synthesize=False,
        )
        assert not result.success
        assert any("Unknown device" in s.message for s in result.stages)

    def test_4bit_build(self, fresh_xor_graph, xor_calib, output_dir):
        result = build(
            fresh_xor_graph, xor_calib,
            output_dir=output_dir,
            mode="combinational",
            bits=4,
            target="ice40up5k",
            simulate=False,
            synthesize=False,
        )
        assert result.success
