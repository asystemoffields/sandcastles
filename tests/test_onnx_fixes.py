"""
Regression tests for three confirmed bugs in w2s.importers.onnx_import.

These exercise the importer *helper functions* in isolation (no real onnx
package, which is intentionally not installed on this RAM-constrained box).  A
tiny fake ``onnx`` module is injected only long enough to import the importer,
then removed from ``sys.modules`` so that other test files'
``pytest.importorskip("onnx")`` calls still skip correctly.

Each test FAILS on the pre-fix code and PASSES on the fixed code:

1. ``_extract_weights_gemm`` — inverted transB transpose, plus ignored
   alpha/beta and silently-ignored transA.
2. ``_build_attrs_conv`` — dropped dilations, silently kept only "begin" of
   asymmetric pads, silently emitted (0,0) for auto_pad=SAME.
3. ``_build_attrs_embedding`` — Gather blanket-mapped to EMBEDDING regardless
   of data rank / axis.
"""

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
#  Import the importer module behind a minimal fake `onnx`, then clean up.
# ---------------------------------------------------------------------------

def _import_onnx_import():
    added = []

    fake_onnx = types.ModuleType("onnx")

    class _AttributeProto:
        FLOAT = 1
        INT = 2
        STRING = 3
        TENSOR = 4
        FLOATS = 6
        INTS = 7

    fake_onnx.AttributeProto = _AttributeProto

    class _TensorProto:
        FLOAT = 1

    fake_onnx.TensorProto = _TensorProto

    fake_numpy_helper = types.ModuleType("onnx.numpy_helper")
    fake_numpy_helper.to_array = lambda t: t
    fake_shape_inference = types.ModuleType("onnx.shape_inference")
    fake_shape_inference.infer_shapes = lambda m: m

    fake_onnx.numpy_helper = fake_numpy_helper
    fake_onnx.shape_inference = fake_shape_inference

    for name, mod in (
        ("onnx", fake_onnx),
        ("onnx.numpy_helper", fake_numpy_helper),
        ("onnx.shape_inference", fake_shape_inference),
    ):
        if name not in sys.modules:
            added.append(name)
        sys.modules[name] = mod

    try:
        import w2s.importers.onnx_import as oi
    finally:
        # Remove the fakes so a later importorskip("onnx") attempts (and fails)
        # a real import and skips, rather than seeing our stub.
        for name in added:
            sys.modules.pop(name, None)
    return oi


oi = _import_onnx_import()


# A node stub matching what the helpers touch: `.input` (list[str]) and an
# `_attrs` dict read via the monkeypatched `_onnx_attr`.
class StubNode:
    def __init__(self, inputs, attrs=None):
        self.input = list(inputs)
        self.attribute = []
        self._attrs = dict(attrs or {})


# Replace the onnx-proto-driven attribute reader with a plain dict lookup.
oi._onnx_attr = lambda node, name, default=None: getattr(
    node, "_attrs", {}
).get(name, default)


# ---------------------------------------------------------------------------
#  Fix 1: _extract_weights_gemm transpose / alpha / beta / transA
# ---------------------------------------------------------------------------

def test_gemm_transB1_no_transpose_and_alpha_beta():
    """transB=1 => B already [out, in] => NO transpose; honor alpha & beta."""
    out, in_ = 3, 4
    B = np.arange(out * in_, dtype=np.float32).reshape(out, in_)  # [out, in]
    C = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    inits = {"B": B, "C": C}
    node = StubNode(
        ["A", "B", "C"],
        attrs={"transB": 1, "transA": 0, "alpha": 2.0, "beta": 3.0},
    )
    w = oi._extract_weights_gemm(node, inits)
    # Old code transposed when transB truthy -> (4, 3); fixed keeps (3, 4).
    assert w["weight"].shape == (out, in_)
    assert np.allclose(w["weight"], B * 2.0)          # alpha applied
    assert np.allclose(w["bias"], C * 3.0)            # beta applied


def test_gemm_transB0_transposes():
    """transB=0 => B is [in, out] => must transpose to [out, in]."""
    out, in_ = 3, 4
    B = np.arange(in_ * out, dtype=np.float32).reshape(in_, out)  # [in, out]
    inits = {"B": B}
    node = StubNode(["A", "B"], attrs={"transB": 0})
    w = oi._extract_weights_gemm(node, inits)
    # Old code did NOT transpose here -> wrong (4, 3); fixed gives (3, 4).
    assert w["weight"].shape == (out, in_)
    assert np.allclose(w["weight"], B.T)


def test_gemm_transA_raises():
    out, in_ = 3, 4
    inits = {"B": np.zeros((out, in_), dtype=np.float32)}
    node = StubNode(["A", "B"], attrs={"transB": 1, "transA": 1})
    with pytest.raises(NotImplementedError):
        oi._extract_weights_gemm(node, inits)


# ---------------------------------------------------------------------------
#  Fix 2: _build_attrs_conv dilations / asymmetric pads / auto_pad
# ---------------------------------------------------------------------------

def test_conv_parses_dilations():
    node = StubNode([], attrs={"kernel_shape": (3, 3), "dilations": (2, 2)})
    attrs = oi._build_attrs_conv(node, {})
    # Old code never emitted a "dilations" key.
    assert attrs["dilations"] == (2, 2)


def test_conv_default_dilations_when_absent():
    node = StubNode([], attrs={"kernel_shape": (3, 3)})
    attrs = oi._build_attrs_conv(node, {})
    assert attrs["dilations"] == (1, 1)


def test_conv_asymmetric_pads_raise():
    # pads = [h_begin, w_begin, h_end, w_end] with begin != end.
    node = StubNode([], attrs={"kernel_shape": (3, 3), "pads": (1, 1, 2, 2)})
    # Old code silently kept (1, 1) as padding; fixed raises.
    with pytest.raises(NotImplementedError):
        oi._build_attrs_conv(node, {})


def test_conv_symmetric_pads_ok():
    node = StubNode([], attrs={"kernel_shape": (3, 3), "pads": (1, 1, 1, 1)})
    attrs = oi._build_attrs_conv(node, {})
    assert attrs["padding"] == (1, 1)


def test_conv_auto_pad_same_upper_stride1():
    node = StubNode(
        [], attrs={"kernel_shape": (3, 3), "strides": (1, 1),
                   "auto_pad": "SAME_UPPER"}
    )
    attrs = oi._build_attrs_conv(node, {})
    # Old code ignored auto_pad and emitted (0, 0); fixed computes (1, 1).
    assert attrs["padding"] == (1, 1)


def test_conv_auto_pad_valid_zero():
    node = StubNode([], attrs={"kernel_shape": (3, 3), "auto_pad": "VALID"})
    attrs = oi._build_attrs_conv(node, {})
    assert attrs["padding"] == (0, 0)


def test_conv_auto_pad_same_stride2_raises():
    node = StubNode(
        [], attrs={"kernel_shape": (3, 3), "strides": (2, 2),
                   "auto_pad": "SAME_UPPER"}
    )
    with pytest.raises(NotImplementedError):
        oi._build_attrs_conv(node, {})


# ---------------------------------------------------------------------------
#  Fix 3: _build_attrs_embedding only for 2-D initializer + axis 0
# ---------------------------------------------------------------------------

def test_gather_embedding_valid_2d_axis0():
    table = np.random.randn(16, 4).astype(np.float32)
    node = StubNode(["W", "idx"], attrs={"axis": 0})
    attrs = oi._build_attrs_embedding(node, {"W": table})
    assert attrs == {"num_embeddings": 16, "embedding_dim": 4}


def test_gather_nonzero_axis_raises():
    table = np.random.randn(16, 4).astype(np.float32)
    node = StubNode(["W", "idx"], attrs={"axis": 1})
    # Old code ignored axis and emitted a (16, 4) embedding regardless.
    with pytest.raises(NotImplementedError):
        oi._build_attrs_embedding(node, {"W": table})


def test_gather_non2d_data_raises():
    table = np.random.randn(16, 4, 2).astype(np.float32)  # 3-D
    node = StubNode(["W", "idx"], attrs={"axis": 0})
    # Old code did w.shape[0], w.shape[1] and produced a malformed embedding.
    with pytest.raises(NotImplementedError):
        oi._build_attrs_embedding(node, {"W": table})


def test_gather_non_initializer_data_raises():
    node = StubNode(["runtime_x", "idx"], attrs={"axis": 0})
    with pytest.raises(NotImplementedError):
        oi._build_attrs_embedding(node, {})
