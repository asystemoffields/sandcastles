"""
Regression tests for graph-level audit fixes:
  * Verilog-keyword graph names must produce a legal module (not `module xor`).
  * An output tensor that no op produces must raise, not emit a constant-0 port.
  * _safe_ident sanitises identifiers.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph, _safe_ident
from tests.rtl_harness import have_iverilog, simulate


def _xor_graph(name):
    np.random.seed(0)
    W1 = np.random.randn(4, 2) * 0.5
    b1 = np.zeros(4)
    W2 = np.random.randn(1, 4) * 0.5
    b2 = np.zeros(1)
    gb = GraphBuilder(name)
    x = gb.input("x", shape=(2,))
    h = gb.dense(x, W1, b1, activation="relu", name="h")
    o = gb.dense(h, W2, b2, name="o")
    gb.output(o)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"x": np.random.randn(8, 2)})
    return g


def test_safe_ident_avoids_keywords_and_bad_chars():
    assert _safe_ident("xor") != "xor"
    assert _safe_ident("and") != "and"
    assert _safe_ident("123net")[0].isalpha() or _safe_ident("123net")[0] == "_"
    assert _safe_ident("my-net.v2") == "my_net_v2"
    assert _safe_ident("dense_block") == "dense_block"  # already fine


@pytest.mark.skipif(not have_iverilog(), reason="iverilog/vvp not installed")
def test_keyword_graph_name_compiles():
    g = _xor_graph("xor")  # 'xor' is a Verilog gate primitive
    out = simulate(g, {"x": np.array([1.0, 0.0])})
    assert len(out) == 1  # compiled + simulated without an iverilog syntax error


def test_unproduced_output_raises():
    g = _xor_graph("net")
    g.output_names = list(g.output_names) + ["does_not_exist"]
    with pytest.raises(ValueError, match="not produced by any operation"):
        compile_graph(g, "/tmp/w2s_unproduced")


def test_embedding_table_quantized_in_output_scale():
    """The embedding ROM must hold values in the output scale (identity requant),
    so downstream layers read it correctly."""
    np.random.seed(1)
    V, D = 20, 8
    table = np.random.randn(V, D) * 0.5
    gb = GraphBuilder("emb")
    idx = gb.input("idx", shape=(1,))
    e = gb.embedding(idx, table, name="emb")
    gb.output(e)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"idx": np.zeros((4, 1))})
    op = [o for o in g.topological_order() if o.op_type.name == "EMBEDDING"][0]
    osc = op.q_params["output_scale"]
    expected = np.clip(np.round(table * osc), -127, 127)
    assert np.array_equal(op.q_weights["weight"], expected)
