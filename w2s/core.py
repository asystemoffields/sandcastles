"""
core.py — Data structures for the weights2silicon compute graph.

Every component (generators, quantizer, importers) depends on these types.
This is the single source of truth for the IR (intermediate representation).
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


# ---------------------------------------------------------------------------
#  Operation types
# ---------------------------------------------------------------------------

class OpType(Enum):
    # Linear
    DENSE = "dense"

    # Convolution
    CONV1D = "conv1d"
    CONV2D = "conv2d"

    # Activation (applied element-wise)
    RELU = "relu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SILU = "silu"
    SOFTMAX = "softmax"

    # Normalization
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    BATCHNORM = "batchnorm"

    # Gated activations
    SWIGLU = "swiglu"

    # Position encoding
    ROPE = "rope"

    # Attention
    MULTI_HEAD_ATTENTION = "mha"
    GROUPED_QUERY_ATTENTION = "gqa"

    # Embedding
    EMBEDDING = "embedding"

    # Runtime state (sequential mode)
    KV_CACHE = "kv_cache"

    # Structural / element-wise
    ADD = "add"
    MULTIPLY = "multiply"
    RESHAPE = "reshape"
    FLATTEN = "flatten"
    CONCAT = "concat"

    # Pooling
    MAXPOOL2D = "maxpool2d"
    AVGPOOL2D = "avgpool2d"
    GLOBAL_AVGPOOL = "global_avgpool"


# ---------------------------------------------------------------------------
#  Quantization configuration
# ---------------------------------------------------------------------------

class QuantScheme(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantGranularity(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"


@dataclass
class QuantConfig:
    """Controls how weights and activations are quantized."""
    bits: int = 8                                        # 4, 8, or 16
    scheme: QuantScheme = QuantScheme.SYMMETRIC
    granularity: QuantGranularity = QuantGranularity.PER_TENSOR


# ---------------------------------------------------------------------------
#  Compute graph nodes
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    """
    A single operation (layer) in the compute graph.

    Attributes:
        op_type   – What kind of operation (dense, conv2d, relu, …).
        name      – Unique name for this op (used in Verilog wire naming).
        inputs    – Tensor names this op reads from.
        outputs   – Tensor names this op writes to.
        attrs     – Op-specific attributes:
                      dense:   {}
                      conv2d:  {'kernel_size': (3,3), 'stride': (1,1),
                                'padding': (1,1), 'groups': 1}
                      relu:    {}
                      gelu:    {'approx': 'tanh'}  # or 'none'
                      layernorm: {'eps': 1e-5, 'normalized_shape': (768,)}
                      rmsnorm: {'eps': 1e-5}
                      mha:     {'num_heads': 8, 'head_dim': 64, 'embed_dim': 512}
                      gqa:     {'num_heads': 32, 'num_kv_heads': 8,
                                'head_dim': 64, 'embed_dim': 2048,
                                'seq_len': 128}
                      swiglu:  {}  (uses two dense sub-projections + silu gate)
                      rope:    {'dim': 64, 'max_seq_len': 2048, 'base': 10000.0}
                      kv_cache: {'num_heads': 8, 'head_dim': 64,
                                 'max_seq_len': 2048}
                      embedding: {'num_embeddings': 50257, 'embedding_dim': 768}
                      maxpool2d: {'kernel_size': (2,2), 'stride': (2,2)}
                      avgpool2d: {'kernel_size': (2,2), 'stride': (2,2)}
                      reshape: {'target_shape': (16, 4, 4)}
                      concat:  {'axis': 0}
        weights   – Named float weight arrays:
                      dense:   {'weight': (n_out, n_in), 'bias': (n_out,)}
                      conv2d:  {'weight': (C_out, C_in, kH, kW), 'bias': (C_out,)}
                      layernorm: {'scale': (D,), 'bias': (D,)}
                      rmsnorm: {'scale': (D,)}
                      batchnorm: {'scale': (C,), 'bias': (C,),
                                  'running_mean': (C,), 'running_var': (C,)}
                      mha:     {'q_weight': (E,E), 'q_bias': (E,),
                                'k_weight': (E,E), 'k_bias': (E,),
                                'v_weight': (E,E), 'v_bias': (E,),
                                'out_weight': (E,E), 'out_bias': (E,)}
                      gqa:     same as mha but k/v weights are
                               (num_kv_heads*head_dim, embed_dim)
                      swiglu:  {'gate_weight': (ffn_dim, dim),
                                'up_weight': (ffn_dim, dim),
                                'down_weight': (dim, ffn_dim),
                                'gate_bias': optional, 'up_bias': optional,
                                'down_bias': optional}
                      rope:    {'cos_table': (max_seq_len, dim//2),
                                'sin_table': (max_seq_len, dim//2)}
                      kv_cache: {} (no weights — runtime state)
                      embedding: {'weight': (V, D)}

    After quantization the following fields are populated:
        q_weights – Quantized integer arrays (same keys as weights).
        q_params  – Dict with quantization metadata:
                      'requant_mult':   int or np.ndarray (per-channel)
                      'requant_shift':  int
                      'input_scale':    float
                      'weight_scale':   float or np.ndarray
                      'output_scale':   float
                      'acc_bits':       int
    """
    op_type: OpType
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    q_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    q_params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Compute graph
# ---------------------------------------------------------------------------

@dataclass
class ComputeGraph:
    """
    A neural network as a directed acyclic graph of Operations.

    Tensor names connect operations: an op's output tensor name is another
    op's input tensor name.  Graph inputs/outputs are the entry/exit points.
    """
    name: str
    operations: List[Operation] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    output_names: List[str] = field(default_factory=list)
    quant_config: QuantConfig = field(default_factory=QuantConfig)

    # Filled during quantization / calibration:
    tensor_scales: Dict[str, float] = field(default_factory=dict)
    tensor_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        # The graph name is used as a Verilog module name and as a file stem,
        # so it must be a valid identifier.  Sanitize hyphens/dots/etc.
        raw = self.name
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
        if safe and safe[0].isdigit():
            safe = "m_" + safe
        self.name = safe or "model"

    def add(self, op: Operation) -> "ComputeGraph":
        """Add an operation.  Returns self for chaining."""
        self.operations.append(op)
        return self

    def topological_order(self) -> List[Operation]:
        """Return operations in a valid execution order (Kahn's algorithm)."""
        produced_by: Dict[str, str] = {}
        for op in self.operations:
            for t in op.outputs:
                produced_by[t] = op.name

        in_degree: Dict[str, int] = {}
        fwd: Dict[str, List[str]] = {}
        for op in self.operations:
            deg = 0
            for t in op.inputs:
                if t in produced_by:
                    producer = produced_by[t]
                    if producer != op.name:
                        deg += 1
                        fwd.setdefault(producer, []).append(op.name)
            in_degree[op.name] = deg

        op_map = {op.name: op for op in self.operations}
        queue = deque(n for n, d in in_degree.items() if d == 0)
        order: List[Operation] = []

        while queue:
            n = queue.popleft()
            order.append(op_map[n])
            for dep in fwd.get(n, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(self.operations):
            raise ValueError("Cycle detected in compute graph")
        return order

    def get_op(self, name: str) -> Optional[Operation]:
        for op in self.operations:
            if op.name == name:
                return op
        return None


# ---------------------------------------------------------------------------
#  Verilog wire tracking
# ---------------------------------------------------------------------------

@dataclass
class TensorWires:
    """
    Maps a logical tensor to its Verilog wire representation.

    For a tensor of shape (8,) at 8-bit precision, this stores 8 wire
    names like ['l0_out_0', 'l0_out_1', …, 'l0_out_7'], each being a
    ``wire signed [7:0]``.
    """
    wire_names: List[str]
    shape: Tuple[int, ...]
    bits: int
    signed: bool = True

    @property
    def numel(self) -> int:
        r = 1
        for s in self.shape:
            r *= s
        return r

    def flat(self, *indices) -> str:
        """Get wire name by multi-dimensional index (row-major / C order)."""
        idx = 0
        stride = 1
        for i in reversed(range(len(indices))):
            idx += indices[i] * stride
            stride *= self.shape[i]
        return self.wire_names[idx]


# ---------------------------------------------------------------------------
#  Generator protocol
# ---------------------------------------------------------------------------
#
#  Every file in w2s/generators/ must expose one or more functions with
#  this signature:
#
#    def generate_XXX(
#        op:       Operation,
#        wire_map: Dict[str, TensorWires],   # tensor_name -> wires
#        bits:     int,                        # quantization bit width
#    ) -> Tuple[List[str], Dict[str, TensorWires]]:
#        '''
#        Returns:
#            verilog_lines  – list of Verilog source lines
#            new_wires      – dict mapping each output tensor name
#                             to its TensorWires (to be merged into wire_map)
#        '''
#
#  Generators should use helpers from w2s.emit for Verilog emission.
#  They must NOT modify wire_map in place; return new entries only.
# ---------------------------------------------------------------------------
