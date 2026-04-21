"""
Regression tests for the robustness fix pass on 2026-04-21.

Each test pins one specific bug that used to compile silently but produce
wrong Verilog (or crash on a realistic model).  Don't collapse these into
broader tests — the point is that a future refactor that reintroduces any
one of these bugs must be caught by a named failure.
"""

import numpy as np
import pytest

from w2s.core import ComputeGraph, QuantConfig, QuantGranularity, QuantScheme
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph


# ---------------------------------------------------------------------------
#  ComputeGraph name sanitization (was: hyphens leaked into Verilog module name)
# ---------------------------------------------------------------------------

def test_graph_name_sanitized_for_verilog():
    g = ComputeGraph(name="mnist-12")
    assert g.name == "mnist_12"
    # Leading digit gets a prefix so the identifier is legal.
    g2 = ComputeGraph(name="12_model")
    assert not g2.name[0].isdigit()
    # Every produced name must be a legal Verilog identifier.
    for raw in ("mnist-12", "12_model", "!!!", "foo.bar", "a b c"):
        name = ComputeGraph(name=raw).name
        assert name, f"empty sanitized name for {raw!r}"
        assert name[0].isalpha() or name[0] == "_", (
            f"sanitized name {name!r} (from {raw!r}) is not a valid "
            f"Verilog identifier (must start with letter or underscore)"
        )
        assert all(c.isalnum() or c == "_" for c in name), (
            f"sanitized name {name!r} (from {raw!r}) contains an illegal "
            f"character for a Verilog identifier"
        )


# ---------------------------------------------------------------------------
#  Verilog >>> vs + precedence in LayerNorm generator
#  (was: `(a * b) >>> bits + bias` parsed as `>>> (bits + bias)`)
# ---------------------------------------------------------------------------

def test_layernorm_verilog_shift_bias_grouping(output_dir):
    """The emitted affine must parenthesize the shift before adding bias."""
    from w2s.graph import compile_graph

    gb = GraphBuilder("ln_test")
    x = gb.input("x", shape=(8,))
    ln = gb.layernorm(
        x,
        scale=np.ones(8, dtype=np.float32),
        bias=np.arange(8, dtype=np.float32) * 0.1 + 0.5,   # non-zero bias
        eps=1e-5,
        name="ln1",
    )
    gb.output(ln)
    graph = gb.build()
    graph.quant_config = QuantConfig(bits=8)
    quantize_graph(graph, {"x": np.random.randn(16, 8).astype(np.float32)})
    v_path = compile_graph(graph, output_dir, mode="combinational")
    src = open(v_path, encoding="utf-8").read()
    # Look for the specific pattern the bug produced.  Broken form was
    # `) >>> 16 + (32'sdN)`.  Fixed form wraps the shift in parens:
    # `((x * y) >>> 16) + (32'sdN)`.
    import re
    broken = re.search(r"\) >>> \d+\s*\n?\s*\+\s*\(", src)
    assert broken is None, (
        f"LayerNorm still emits the unparenthesized shift-then-add "
        f"precedence bug.  Check generators/norm.py."
    )


# ---------------------------------------------------------------------------
#  Conv2D input-shape normalization for batched ONNX inputs
#  (was: (1,C,H,W) wire shape treated as (C,H,W) -> H,W swapped with C,H)
# ---------------------------------------------------------------------------

def test_conv2d_generator_accepts_leading_batch_dim():
    """Direct generator call with a 4-D wire shape (N,C,H,W).

    Tests the conv.py fix in isolation — no quantize/forward-pass path.
    Uses the same shape shape the ONNX importer produces for batched inputs.
    """
    from w2s.core import Operation, OpType, TensorWires
    from w2s.generators.conv import generate_conv2d

    C_out, C_in, kH, kW = 2, 1, 3, 3
    weight = np.ones((C_out, C_in, kH, kW), dtype=np.int64)
    op = Operation(
        op_type=OpType.CONV2D,
        name="c1",
        inputs=["image"],
        outputs=["y"],
        attrs={"kernel_size": (kH, kW), "stride": (1, 1), "padding": (0, 0)},
        weights={"weight": weight.astype(np.float32)},
    )
    op.q_weights = {"weight": weight}
    op.q_params = {"requant_mult": 256, "requant_shift": 8, "acc_bits": 32}

    # 4-D wire map — mimics an ONNX-imported graph input that retained its
    # leading batch dim.
    wires = [f"image_{i}" for i in range(1 * 1 * 8 * 8)]
    wire_map = {"image": TensorWires(wires, (1, 1, 8, 8), 8)}
    lines, new_wires = generate_conv2d(op, wire_map, bits=8)
    assert any("c1_out_0" in l for l in lines), "conv generator produced no outputs"
    # The output tensor shape must reflect spatial dims (8-3+1=6), not a
    # collapsed-to-channels size.
    assert new_wires["y"].shape == (C_out, 6, 6)

    # Mismatched leading dim should raise, not silently misindex.
    bad_map = {"image": TensorWires(wires, (3, 1, 8, 8), 8)}
    with pytest.raises(ValueError, match="channel count"):
        generate_conv2d(op, bad_map, bits=8)


# ---------------------------------------------------------------------------
#  ONNX importer: unknown op raises instead of silently dropping
#  (was: any op not in _OPTYPE_MAP was skipped, producing a malformed graph)
# ---------------------------------------------------------------------------

def test_onnx_unknown_op_raises(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto
    from w2s.importers.onnx_import import load_onnx

    # Graph: input -> SomeNonsenseOp -> output.  Shape inference can't see
    # the output shape but the importer should still notice the unknown op.
    node = helper.make_node("ThisOpDoesNotExist",
                            inputs=["x"], outputs=["y"], name="weird")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    g = helper.make_graph([node], "g", [x], [y], initializer=[])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "unknown.onnx")
    onnx.save(model, path)

    with pytest.raises(NotImplementedError, match="ThisOpDoesNotExist"):
        load_onnx(path)


# ---------------------------------------------------------------------------
#  ONNX importer: Identity is absorbed, Transpose-of-initializer folded
# ---------------------------------------------------------------------------

def test_onnx_identity_absorbed(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto
    from w2s.importers.onnx_import import load_onnx

    w = np.random.randn(3, 4).astype(np.float32)
    w_init = numpy_helper.from_array(w, name="W")
    # Identity -> MatMul chain where Identity passes through the input.
    n1 = helper.make_node("Identity", ["x"], ["x_id"], name="id1")
    n2 = helper.make_node("MatMul", ["x_id", "W"], ["y"], name="mm")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    g = helper.make_graph([n1, n2], "g", [x], [y], initializer=[w_init])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "identity.onnx")
    onnx.save(model, path)

    graph = load_onnx(path)
    op_types = [op.op_type.value for op in graph.operations]
    assert "dense" in op_types, f"Identity should be absorbed, got {op_types}"


# ---------------------------------------------------------------------------
#  Sequential mode: refuses ops it can't faithfully compile
#  (was: silently dropped MaxPool/LayerNorm/Conv2D and emitted the wrong net)
# ---------------------------------------------------------------------------

def test_sequential_refuses_conv(cnn_quantized_graph, output_dir):
    from w2s.graph import compile_graph

    with pytest.raises(NotImplementedError, match="Sequential compile"):
        compile_graph(cnn_quantized_graph, output_dir, mode="sequential")


def test_sequential_refuses_per_channel_requant(output_dir):
    from w2s.graph import compile_graph

    gb = GraphBuilder("per_ch")
    x = gb.input("x", shape=(8,))
    w = np.random.randn(4, 8).astype(np.float32) * 0.3
    b = np.random.randn(4).astype(np.float32) * 0.01
    y = gb.dense(x, weight=w, bias=b, name="fc")
    gb.output(y)
    g = gb.build()
    g.quant_config = QuantConfig(
        bits=8, scheme=QuantScheme.SYMMETRIC,
        granularity=QuantGranularity.PER_CHANNEL,
    )
    quantize_graph(g, {"x": np.random.randn(16, 8).astype(np.float32)})
    with pytest.raises(NotImplementedError, match="per-channel"):
        compile_graph(g, output_dir, mode="sequential")


# ---------------------------------------------------------------------------
#  CLI calibration: ONNX-style batched input shapes don't produce 5-D calib
# ---------------------------------------------------------------------------

def test_batchnorm_axis_is_explicit_not_heuristic(output_dir):
    """Batch equal to channel count no longer flips the channel axis."""
    from w2s.graph import compile_graph

    C = 4                                   # match calibration batch of 4 on purpose
    gb = GraphBuilder("bn_axis")
    x = gb.input("x", shape=(C, 2, 2))      # (C, H, W) — c_axis 0
    y = gb.batchnorm(
        x,
        scale=np.ones(C, dtype=np.float32),
        bias=np.zeros(C, dtype=np.float32),
        running_mean=np.zeros(C, dtype=np.float32),
        running_var=np.ones(C, dtype=np.float32),
        c_axis=0,
        name="bn",
    )
    gb.output(y)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    # Calibration batch == C; the old heuristic used this coincidence to
    # declare that dim 0 was batch, flipping c_axis to 1 and silently
    # miscomputing the ranges.  With the explicit attr this must work.
    calib = np.random.randn(C, C, 2, 2).astype(np.float32)
    quantize_graph(g, {"x": calib})
    v_path = compile_graph(g, output_dir, mode="combinational")
    assert "c_axis=0" in open(v_path, encoding="utf-8").read()


def test_embedding_large_vocab_uses_readmemh(output_dir):
    """A ~65K-entry embedding table must not be emitted as a deep ternary."""
    from w2s.graph import compile_graph
    from pathlib import Path

    gb = GraphBuilder("emb_big")
    idx = gb.input("idx", shape=(1,))
    V, D = 4096, 16
    np.random.seed(7)
    w = np.random.randn(V, D).astype(np.float32)
    e = gb.embedding(idx, w, name="emb")
    gb.output(e)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"idx": np.array([[0]], dtype=np.float32)})
    v_path = compile_graph(g, output_dir, mode="combinational")
    src = open(v_path, encoding="utf-8").read()
    assert "$readmemh" in src, "large embedding should use $readmemh, not inline"
    # Verilog body must be tiny (line count should be O(seq_len * D), not O(V*D)).
    assert src.count("\n") < 1000, (
        f"Verilog is {src.count(chr(10))} lines for {V*D} entries; expected "
        f"compact $readmemh output."
    )
    hex_files = list(Path(output_dir).glob("*.hex"))
    assert len(hex_files) == 1, f"expected one hex file, got {hex_files}"
    assert sum(1 for _ in open(hex_files[0])) == V * D


def test_embedding_small_vocab_stays_inline(output_dir):
    """Small tables don't pay the hex-file + readmemh overhead."""
    from w2s.graph import compile_graph
    from pathlib import Path

    gb = GraphBuilder("emb_small")
    idx = gb.input("idx", shape=(1,))
    w = np.random.randn(16, 4).astype(np.float32)
    e = gb.embedding(idx, w, name="emb")
    gb.output(e)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"idx": np.array([[0]], dtype=np.float32)})
    v_path = compile_graph(g, output_dir, mode="combinational")
    src = open(v_path, encoding="utf-8").read()
    assert "$readmemh" not in src
    assert "function signed" in src               # case-in-function ROM
    assert not list(Path(output_dir).glob("*.hex"))


def test_onnx_transpose_of_initializer_folded_into_matmul(tmp_path):
    """PyTorch nn.Linear → Transpose(W)+MatMul idiom collapses cleanly."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto
    from w2s.importers.onnx_import import load_onnx

    W = np.random.randn(3, 4).astype(np.float32)  # stored (out, in)
    W_init = numpy_helper.from_array(W, name="W")
    # Transpose stored weight to (in, out) at runtime, then MatMul.
    n1 = helper.make_node("Transpose", ["W"], ["W_T"], perm=[1, 0], name="t")
    n2 = helper.make_node("MatMul", ["x", "W_T"], ["y"], name="mm")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])
    g = helper.make_graph([n1, n2], "g", [x], [y], initializer=[W_init])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "linear.onnx")
    onnx.save(model, path)

    graph = load_onnx(path)
    assert [op.op_type.value for op in graph.operations] == ["dense"], (
        f"Transpose should have been constant-folded; got "
        f"{[op.op_type.value for op in graph.operations]}"
    )
    # Dense generator expects weight shape (n_out, n_in).
    assert graph.operations[0].weights["weight"].shape == (3, 4)


def test_cli_make_calibration_strips_batched_leading_dim():
    from w2s.__main__ import _make_calibration_data
    from w2s.core import ComputeGraph

    g = ComputeGraph(name="tmp")
    g.input_names = ["img"]
    g.input_shapes = {"img": (1, 1, 28, 28)}       # mnist-12 shape
    calib = _make_calibration_data(g, batch=4)
    assert calib["img"].shape == (4, 1, 28, 28), (
        f"Expected leading batch replaced to 4, got {calib['img'].shape}"
    )

    # Dynamic leading dim (-1) also replaced, no negative-dim crash.
    g.input_shapes = {"img": (-1, 3, 16, 16)}
    calib = _make_calibration_data(g, batch=2)
    assert calib["img"].shape == (2, 3, 16, 16)

    # Genuinely unbatched shape just gets a batch prepended.
    g.input_shapes = {"img": (784,)}
    calib = _make_calibration_data(g, batch=4)
    assert calib["img"].shape == (4, 784)
