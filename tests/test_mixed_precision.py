"""Tests for mixed-precision per-layer quantization."""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph


class TestMixedPrecision:
    def _build_graph(self):
        """Build a simple 2-layer network for mixed precision testing."""
        np.random.seed(42)
        W1 = np.random.randn(8, 2).astype(np.float64) * 0.5
        b1 = np.zeros(8)
        W2 = np.random.randn(1, 8).astype(np.float64) * 0.5
        b2 = np.zeros(1)

        gb = GraphBuilder("mixed_test")
        inp = gb.input("x", shape=(2,))
        h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
        out = gb.dense(h, W2, b2, name="output")
        gb.output(out)
        return gb.build()

    def test_bits_map_quantization(self):
        graph = self._build_graph()
        config = QuantConfig(bits=8)
        graph.quant_config = config

        X = np.random.randn(4, 2).astype(np.float32)
        bits_map = {"hidden": 4, "output": 16}
        quantize_graph(graph, {"x": X}, config, bits_map=bits_map)

        # Check that per-op bits are stored
        hidden_op = graph.get_op("hidden")
        output_op = graph.get_op("output")
        assert hidden_op.attrs.get("bits") == 4
        assert output_op.attrs.get("bits") == 16

    def test_mixed_compiles(self, output_dir):
        graph = self._build_graph()
        config = QuantConfig(bits=8)
        graph.quant_config = config

        X = np.random.randn(4, 2).astype(np.float32)
        bits_map = {"hidden": 4, "output": 16}
        quantize_graph(graph, {"x": X}, config, bits_map=bits_map)

        path = compile_graph(graph, output_dir=output_dir, mode="combinational")
        assert path.endswith(".v")
        with open(path) as f:
            content = f.read()
        # compile_graph writes "mixed [int4/int16]" in the Verilog header
        assert "mixed" in content

    def test_no_bits_map_is_default(self):
        """Without bits_map, all ops use the default bit width."""
        graph = self._build_graph()
        config = QuantConfig(bits=8)
        graph.quant_config = config

        X = np.random.randn(4, 2).astype(np.float32)
        quantize_graph(graph, {"x": X}, config, bits_map=None)

        # No per-op bits should be set
        for op in graph.operations:
            assert "bits" not in op.attrs
