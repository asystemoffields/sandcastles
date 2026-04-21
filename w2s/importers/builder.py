"""
builder.py — Fluent API for building ComputeGraphs programmatically.

Usage:
    from w2s.importers.builder import GraphBuilder

    g = (GraphBuilder("my_net")
         .input("x", (1, 784))
         .dense("x", W1, b1, activation="relu", name="fc1")
         .dense("fc1_out", W2, b2, name="fc2")
         .output("fc2_out")
         .build())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from w2s.core import ComputeGraph, OpType, Operation


class GraphBuilder:
    """Fluent API for building compute graphs programmatically.

    Each layer method creates an :class:`Operation`, appends it to the
    internal graph, and returns the **output tensor name** so that it can
    be threaded into the next call.
    """

    def __init__(self, name: str):
        self.graph = ComputeGraph(name=name)
        self._counter: int = 0

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        idx = self._counter
        self._counter += 1
        return idx

    def _auto_name(self, prefix: str, name: Optional[str]) -> str:
        if name is not None:
            return name
        return f"{prefix}_{self._next_id()}"

    def _add_op(
        self,
        op_type: OpType,
        name: str,
        inputs: List[str],
        attrs: Dict[str, Any] | None = None,
        weights: Dict[str, np.ndarray] | None = None,
    ) -> str:
        """Create an Operation, append it, return the output tensor name."""
        out_name = f"{name}_out"
        op = Operation(
            op_type=op_type,
            name=name,
            inputs=inputs,
            outputs=[out_name],
            attrs=attrs or {},
            weights=weights or {},
        )
        self.graph.add(op)
        return out_name

    # ------------------------------------------------------------------
    #  Graph-level declarations
    # ------------------------------------------------------------------

    def input(self, name: str, shape: Tuple[int, ...]) -> str:
        """Declare a graph input.  Returns the tensor name (``name`` itself)."""
        self.graph.input_names.append(name)
        self.graph.input_shapes[name] = shape
        return name

    def output(self, tensor_name: str) -> "GraphBuilder":
        """Mark a tensor as a graph output.  Returns *self* for chaining."""
        self.graph.output_names.append(tensor_name)
        return self

    def build(self) -> ComputeGraph:
        """Return the constructed :class:`ComputeGraph`."""
        return self.graph

    # ------------------------------------------------------------------
    #  Dense / Linear
    # ------------------------------------------------------------------

    def dense(
        self,
        input: str,
        weight: np.ndarray,
        bias: np.ndarray = None,
        activation: str = "none",
        name: str = None,
    ) -> str:
        """Add a fully-connected (dense) layer.  Returns output tensor name."""
        op_name = self._auto_name("dense", name)
        weights: Dict[str, np.ndarray] = {"weight": np.asarray(weight)}
        if bias is not None:
            weights["bias"] = np.asarray(bias)
        attrs: Dict[str, Any] = {}
        if activation != "none":
            attrs["activation"] = activation
        return self._add_op(OpType.DENSE, op_name, [input], attrs, weights)

    # ------------------------------------------------------------------
    #  Convolution
    # ------------------------------------------------------------------

    def conv2d(
        self,
        input: str,
        weight: np.ndarray,
        bias: np.ndarray = None,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        groups: int = 1,
        activation: str = "none",
        name: str = None,
    ) -> str:
        """Add a 2-D convolution.  Returns output tensor name."""
        op_name = self._auto_name("conv2d", name)
        w = np.asarray(weight)
        weights: Dict[str, np.ndarray] = {"weight": w}
        if bias is not None:
            weights["bias"] = np.asarray(bias)
        kernel_size = (w.shape[2], w.shape[3]) if w.ndim == 4 else tuple(w.shape[-2:])
        attrs: Dict[str, Any] = {
            "kernel_size": kernel_size,
            "stride": tuple(stride),
            "padding": tuple(padding),
            "groups": groups,
        }
        if activation != "none":
            attrs["activation"] = activation
        return self._add_op(OpType.CONV2D, op_name, [input], attrs, weights)

    # ------------------------------------------------------------------
    #  Activations
    # ------------------------------------------------------------------

    def relu(self, input: str, name: str = None) -> str:
        return self._add_op(OpType.RELU, self._auto_name("relu", name), [input])

    def gelu(self, input: str, name: str = None) -> str:
        return self._add_op(OpType.GELU, self._auto_name("gelu", name), [input])

    def sigmoid(self, input: str, name: str = None) -> str:
        return self._add_op(OpType.SIGMOID, self._auto_name("sigmoid", name), [input])

    def tanh(self, input: str, name: str = None) -> str:
        return self._add_op(OpType.TANH, self._auto_name("tanh", name), [input])

    def silu(self, input: str, name: str = None) -> str:
        return self._add_op(OpType.SILU, self._auto_name("silu", name), [input])

    def softmax(self, input: str, axis: int = -1, name: str = None) -> str:
        return self._add_op(
            OpType.SOFTMAX,
            self._auto_name("softmax", name),
            [input],
            attrs={"axis": axis},
        )

    # ------------------------------------------------------------------
    #  Normalization
    # ------------------------------------------------------------------

    def layernorm(
        self,
        input: str,
        scale: np.ndarray,
        bias: np.ndarray,
        eps: float = 1e-5,
        name: str = None,
    ) -> str:
        op_name = self._auto_name("layernorm", name)
        s = np.asarray(scale)
        return self._add_op(
            OpType.LAYERNORM,
            op_name,
            [input],
            attrs={"eps": eps, "normalized_shape": tuple(s.shape)},
            weights={"scale": s, "bias": np.asarray(bias)},
        )

    def rmsnorm(
        self,
        input: str,
        scale: np.ndarray,
        eps: float = 1e-5,
        name: str = None,
    ) -> str:
        op_name = self._auto_name("rmsnorm", name)
        return self._add_op(
            OpType.RMSNORM,
            op_name,
            [input],
            attrs={"eps": eps},
            weights={"scale": np.asarray(scale)},
        )

    def batchnorm(
        self,
        input: str,
        scale: np.ndarray,
        bias: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        eps: float = 1e-5,
        c_axis: int = 1,
        name: str = None,
    ) -> str:
        """
        Construct a BatchNorm op.

        ``c_axis`` is the channel axis of the runtime input tensor (ONNX
        convention: 1 for (N, C, ...); use 0 for bare (C, ...) inputs).
        Making the axis explicit avoids a shape-dependent heuristic at
        calibration time.
        """
        op_name = self._auto_name("batchnorm", name)
        return self._add_op(
            OpType.BATCHNORM,
            op_name,
            [input],
            attrs={"eps": eps, "c_axis": int(c_axis)},
            weights={
                "scale": np.asarray(scale),
                "bias": np.asarray(bias),
                "running_mean": np.asarray(running_mean),
                "running_var": np.asarray(running_var),
            },
        )

    # ------------------------------------------------------------------
    #  Embedding
    # ------------------------------------------------------------------

    def embedding(self, input: str, weight: np.ndarray, name: str = None) -> str:
        op_name = self._auto_name("embedding", name)
        w = np.asarray(weight)
        return self._add_op(
            OpType.EMBEDDING,
            op_name,
            [input],
            attrs={"num_embeddings": w.shape[0], "embedding_dim": w.shape[1]},
            weights={"weight": w},
        )

    # ------------------------------------------------------------------
    #  Element-wise / structural
    # ------------------------------------------------------------------

    def add(self, a: str, b: str, name: str = None) -> str:
        return self._add_op(OpType.ADD, self._auto_name("add", name), [a, b])

    def multiply(self, a: str, b: str, name: str = None) -> str:
        return self._add_op(OpType.MULTIPLY, self._auto_name("mul", name), [a, b])

    def concat(self, inputs: List[str], axis: int = 0, name: str = None) -> str:
        return self._add_op(
            OpType.CONCAT,
            self._auto_name("concat", name),
            inputs,
            attrs={"axis": axis},
        )

    # ------------------------------------------------------------------
    #  Pooling
    # ------------------------------------------------------------------

    def maxpool2d(
        self,
        input: str,
        kernel_size: tuple,
        stride: tuple = None,
        name: str = None,
    ) -> str:
        if stride is None:
            stride = kernel_size
        return self._add_op(
            OpType.MAXPOOL2D,
            self._auto_name("maxpool2d", name),
            [input],
            attrs={"kernel_size": tuple(kernel_size), "stride": tuple(stride)},
        )

    def avgpool2d(
        self,
        input: str,
        kernel_size: tuple,
        stride: tuple = None,
        name: str = None,
    ) -> str:
        if stride is None:
            stride = kernel_size
        return self._add_op(
            OpType.AVGPOOL2D,
            self._auto_name("avgpool2d", name),
            [input],
            attrs={"kernel_size": tuple(kernel_size), "stride": tuple(stride)},
        )

    def global_avgpool(self, input: str, name: str = None) -> str:
        return self._add_op(
            OpType.GLOBAL_AVGPOOL,
            self._auto_name("global_avgpool", name),
            [input],
        )

    # ------------------------------------------------------------------
    #  Shape manipulation
    # ------------------------------------------------------------------

    def reshape(self, input: str, target_shape: tuple, name: str = None) -> str:
        return self._add_op(
            OpType.RESHAPE,
            self._auto_name("reshape", name),
            [input],
            attrs={"target_shape": tuple(target_shape)},
        )

    def flatten(self, input: str, name: str = None) -> str:
        return self._add_op(
            OpType.FLATTEN,
            self._auto_name("flatten", name),
            [input],
        )

    # ------------------------------------------------------------------
    #  Multi-head attention
    # ------------------------------------------------------------------

    def mha(
        self,
        input: str,
        q_weight: np.ndarray,
        q_bias: np.ndarray,
        k_weight: np.ndarray,
        k_bias: np.ndarray,
        v_weight: np.ndarray,
        v_bias: np.ndarray,
        out_weight: np.ndarray,
        out_bias: np.ndarray,
        num_heads: int,
        seq_len: int = 1,
        name: str = None,
    ) -> str:
        op_name = self._auto_name("mha", name)
        qw = np.asarray(q_weight)
        embed_dim = qw.shape[0]
        head_dim = embed_dim // num_heads
        return self._add_op(
            OpType.MULTI_HEAD_ATTENTION,
            op_name,
            [input],
            attrs={
                "num_heads": num_heads,
                "head_dim": head_dim,
                "embed_dim": embed_dim,
                "seq_len": seq_len,
            },
            weights={
                "q_weight": qw,
                "q_bias": np.asarray(q_bias),
                "k_weight": np.asarray(k_weight),
                "k_bias": np.asarray(k_bias),
                "v_weight": np.asarray(v_weight),
                "v_bias": np.asarray(v_bias),
                "out_weight": np.asarray(out_weight),
                "out_bias": np.asarray(out_bias),
            },
        )

    # ------------------------------------------------------------------
    #  Modern transformer ops (DeepSeek, Llama, Qwen, Mistral, Gemma)
    # ------------------------------------------------------------------

    def gqa(
        self, input: str,
        q_weight, q_bias, k_weight, k_bias,
        v_weight, v_bias, out_weight, out_bias,
        num_heads: int, num_kv_heads: int,
        seq_len: int = 1, name: str = None,
    ) -> str:
        """Grouped-query attention (fewer K/V heads than Q heads)."""
        op_name = self._auto_name("gqa", name)
        qw = np.asarray(q_weight)
        embed_dim = qw.shape[0]
        head_dim = embed_dim // num_heads
        return self._add_op(
            OpType.GROUPED_QUERY_ATTENTION, op_name, [input],
            attrs={"num_heads": num_heads, "num_kv_heads": num_kv_heads,
                   "head_dim": head_dim, "embed_dim": embed_dim,
                   "seq_len": seq_len},
            weights={"q_weight": qw, "q_bias": np.asarray(q_bias),
                     "k_weight": np.asarray(k_weight),
                     "k_bias": np.asarray(k_bias),
                     "v_weight": np.asarray(v_weight),
                     "v_bias": np.asarray(v_bias),
                     "out_weight": np.asarray(out_weight),
                     "out_bias": np.asarray(out_bias)},
        )

    def swiglu(
        self, input: str,
        gate_weight, up_weight, down_weight,
        gate_bias=None, up_bias=None, down_bias=None,
        name: str = None,
    ) -> str:
        """SwiGLU FFN block: SiLU(x @ Wg) * (x @ Wu) then @ Wd."""
        op_name = self._auto_name("swiglu", name)
        weights = {
            "gate_weight": np.asarray(gate_weight),
            "up_weight": np.asarray(up_weight),
            "down_weight": np.asarray(down_weight),
        }
        if gate_bias is not None:
            weights["gate_bias"] = np.asarray(gate_bias)
        if up_bias is not None:
            weights["up_bias"] = np.asarray(up_bias)
        if down_bias is not None:
            weights["down_bias"] = np.asarray(down_bias)
        return self._add_op(
            OpType.SWIGLU, op_name, [input], weights=weights,
        )

    def rope(
        self, input: str, cos_table, sin_table,
        dim: int = None, max_seq_len: int = None,
        base: float = 10000.0,
        position_input: str = None, name: str = None,
    ) -> str:
        """Rotary position embedding with precomputed sin/cos tables."""
        op_name = self._auto_name("rope", name)
        cos_t = np.asarray(cos_table)
        sin_t = np.asarray(sin_table)
        d = dim or cos_t.shape[-1] * 2
        msl = max_seq_len or cos_t.shape[0]
        inputs = [input]
        if position_input is not None:
            inputs.append(position_input)
        return self._add_op(
            OpType.ROPE, op_name, inputs,
            attrs={"dim": d, "max_seq_len": msl, "base": base},
            weights={"cos_table": cos_t, "sin_table": sin_t},
        )

    def kv_cache(
        self, input: str, position_input: str,
        num_heads: int, head_dim: int, max_seq_len: int,
        name: str = None,
    ) -> str:
        """KV cache register file for sequential token-by-token inference."""
        op_name = self._auto_name("kv_cache", name)
        return self._add_op(
            OpType.KV_CACHE, op_name, [input, position_input],
            attrs={"num_heads": num_heads, "head_dim": head_dim,
                   "max_seq_len": max_seq_len},
        )
