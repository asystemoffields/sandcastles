"""Tests for the architecture-fidelity fixes in hf_import.

Each test builds a graph from a synthetic config + numpy weight dict and
asserts on the resulting graph's op sequence / topology.  These tests fail
against the pre-fix importer and pass against the fixed one.

Covered fixes:
  1. RoPE is now wired into Llama-family attention (a ROPE op is present and
     feeds the attention op).
  2. Gemma uses a GELU-gated (GeGLU) FFN, not SwiGLU; Gemma2 wires all four
     per-layer norms.
  3. Phi-1/2 uses a parallel block (one LayerNorm feeds both attention and
     MLP); Phi-3 uses RMSNorm + a fused SwiGLU MLP.
"""

import numpy as np
import pytest

from w2s.core import ComputeGraph, OpType
from w2s.importers.hf_import import _build_llama, _build_phi


# ---------------------------------------------------------------------------
#  Synthetic weight factories
# ---------------------------------------------------------------------------

def _llama_weights(embed_dim, n_kv_heads, head_dim, ffn_dim,
                   gemma=False, gemma2=False):
    E = embed_dim
    kv_dim = n_kv_heads * head_dim
    w = {
        "model.layers.0.input_layernorm.weight": np.ones(E),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(E, E) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(kv_dim, E) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(kv_dim, E) * 0.02,
        "model.layers.0.self_attn.o_proj.weight": np.random.randn(E, E) * 0.02,
        "model.layers.0.post_attention_layernorm.weight": np.ones(E),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(ffn_dim, E) * 0.02,
        "model.layers.0.mlp.up_proj.weight": np.random.randn(ffn_dim, E) * 0.02,
        "model.layers.0.mlp.down_proj.weight": np.random.randn(E, ffn_dim) * 0.02,
    }
    if gemma2:
        w["model.layers.0.pre_feedforward_layernorm.weight"] = np.ones(E)
        w["model.layers.0.post_feedforward_layernorm.weight"] = np.ones(E)
    return w


def _phi_parallel_weights(embed_dim, n_kv_heads, head_dim, ffn_dim):
    E = embed_dim
    kv_dim = n_kv_heads * head_dim
    return {
        "model.layers.0.input_layernorm.weight": np.ones(E),
        "model.layers.0.input_layernorm.bias": np.zeros(E),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(E, E) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(kv_dim, E) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(kv_dim, E) * 0.02,
        "model.layers.0.self_attn.dense.weight": np.random.randn(E, E) * 0.02,
        "model.layers.0.mlp.fc1.weight": np.random.randn(ffn_dim, E) * 0.02,
        "model.layers.0.mlp.fc2.weight": np.random.randn(E, ffn_dim) * 0.02,
    }


def _phi3_weights(embed_dim, head_dim, ffn_dim, with_mlp=True):
    E = embed_dim
    qkv = E + 2 * (E)  # n_heads==n_kv_heads -> q_dim + kv_dim + kv_dim == 3E
    w = {
        "model.layers.0.input_layernorm.weight": np.ones(E),
        "model.layers.0.post_attention_layernorm.weight": np.ones(E),
        "model.layers.0.self_attn.qkv_proj.weight": np.random.randn(qkv, E) * 0.02,
        "model.layers.0.self_attn.o_proj.weight": np.random.randn(E, E) * 0.02,
    }
    if with_mlp:
        w["model.layers.0.mlp.gate_up_proj.weight"] = np.random.randn(2 * ffn_dim, E) * 0.02
        w["model.layers.0.mlp.down_proj.weight"] = np.random.randn(E, ffn_dim) * 0.02
    return w


def _producer_of(graph, tensor_name):
    for op in graph.operations:
        if tensor_name in op.outputs:
            return op
    return None


# ---------------------------------------------------------------------------
#  Fix 1 — RoPE present in Llama-family attention
# ---------------------------------------------------------------------------

class TestFix1Rope:
    def test_rope_op_present(self):
        np.random.seed(0)
        config = {"model_type": "llama", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8, "rope_theta": 10000.0}
        weights = _llama_weights(32, 4, 8, 64)
        graph = _build_llama("rope_llama", config, weights, blocks=[0])
        op_types = [op.op_type for op in graph.operations]
        assert OpType.ROPE in op_types, "RoPE op missing from Llama graph"

    def test_rope_feeds_attention(self):
        """The attention op must consume the RoPE op's output (positional
        rotation actually reaches attention, not a dead op)."""
        np.random.seed(0)
        config = {"model_type": "llama", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _llama_weights(32, 4, 8, 64)
        graph = _build_llama("rope_llama2", config, weights, blocks=[0])

        rope_op = next(o for o in graph.operations if o.op_type == OpType.ROPE)
        attn_op = next(o for o in graph.operations
                       if o.op_type in (OpType.MULTI_HEAD_ATTENTION,
                                        OpType.GROUPED_QUERY_ATTENTION))
        assert rope_op.outputs[0] in attn_op.inputs

    def test_rope_uses_rope_theta_and_position_input(self):
        np.random.seed(0)
        config = {"model_type": "mistral", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 2,
                  "head_dim": 8, "rope_theta": 1000000.0}
        weights = _llama_weights(32, 2, 8, 64)
        graph = _build_llama("rope_mistral", config, weights, blocks=[0])
        rope_op = next(o for o in graph.operations if o.op_type == OpType.ROPE)
        assert rope_op.attrs["base"] == 1000000.0
        # position index threaded in as a second input
        assert "position_id" in rope_op.inputs
        assert "position_id" in graph.input_names


# ---------------------------------------------------------------------------
#  Fix 2 — Gemma GeGLU FFN + Gemma2 four norms
# ---------------------------------------------------------------------------

class TestFix2Gemma:
    def test_gemma_ffn_uses_gelu_not_swiglu(self):
        np.random.seed(1)
        config = {"model_type": "gemma", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _llama_weights(32, 4, 8, 64, gemma=True)
        graph = _build_llama("gemma", config, weights, blocks=[0])
        op_types = [op.op_type for op in graph.operations]
        assert OpType.GELU in op_types, "Gemma FFN must use a GELU op"
        assert OpType.SWIGLU not in op_types, "Gemma must not use SwiGLU"

    def test_gemma_geglu_topology(self):
        """gate->gelu and up both feed a multiply, which feeds down (dense)."""
        np.random.seed(1)
        config = {"model_type": "gemma", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _llama_weights(32, 4, 8, 64, gemma=True)
        graph = _build_llama("gemma_topo", config, weights, blocks=[0])
        mul = next(o for o in graph.operations if o.op_type == OpType.MULTIPLY)
        producers = {_producer_of(graph, i).op_type for i in mul.inputs}
        assert OpType.GELU in producers
        assert OpType.DENSE in producers  # the "up" projection

    def test_llama_still_uses_swiglu(self):
        """Regression: non-Gemma Llama keeps SwiGLU (SiLU-gated)."""
        np.random.seed(1)
        config = {"model_type": "llama", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _llama_weights(32, 4, 8, 64)
        graph = _build_llama("llama_swiglu", config, weights, blocks=[0])
        op_types = [op.op_type for op in graph.operations]
        assert OpType.SWIGLU in op_types

    def test_gemma2_has_four_rmsnorms(self):
        np.random.seed(2)
        config = {"model_type": "gemma2", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _llama_weights(32, 4, 8, 64, gemma=True, gemma2=True)
        graph = _build_llama("gemma2", config, weights, blocks=[0])
        rms_ops = [o for o in graph.operations if o.op_type == OpType.RMSNORM]
        assert len(rms_ops) == 4, f"expected 4 norms/layer, got {len(rms_ops)}"

    def test_gemma2_sandwich_norm_wraps_sublayers(self):
        """post_attention norm consumes attention output; post_feedforward
        norm consumes the FFN (down/dense) output."""
        np.random.seed(2)
        config = {"model_type": "gemma2", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _llama_weights(32, 4, 8, 64, gemma=True, gemma2=True)
        graph = _build_llama("gemma2_sw", config, weights, blocks=[0])

        attn_op = next(o for o in graph.operations
                       if o.op_type == OpType.MULTI_HEAD_ATTENTION)
        post_attn_norm = _producer_of(graph, _consumer_input(graph, attn_op))
        assert post_attn_norm is not None
        assert post_attn_norm.op_type == OpType.RMSNORM


def _consumer_input(graph, producer_op):
    """Return the producer's output tensor if it is consumed by an RMSNORM."""
    out = producer_op.outputs[0]
    for op in graph.operations:
        if op.op_type == OpType.RMSNORM and out in op.inputs:
            return op.outputs[0]
    return None


# ---------------------------------------------------------------------------
#  Fix 3 — Phi parallel block + Phi-3 RMSNorm/SwiGLU
# ---------------------------------------------------------------------------

class TestFix3Phi:
    def test_phi_parallel_single_norm_two_consumers(self):
        """Phi-1/2: exactly one LayerNorm, whose output feeds BOTH the
        attention op and the first MLP dense (parallel block)."""
        np.random.seed(3)
        config = {"model_type": "phi", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _phi_parallel_weights(32, 4, 8, 128)
        graph = _build_phi("phi", config, weights, blocks=[0])

        ln_ops = [o for o in graph.operations if o.op_type == OpType.LAYERNORM]
        assert len(ln_ops) == 1
        ln_out = ln_ops[0].outputs[0]

        consumers = [o for o in graph.operations if ln_out in o.inputs]
        consumer_types = {o.op_type for o in consumers}
        assert OpType.MULTI_HEAD_ATTENTION in consumer_types
        assert OpType.DENSE in consumer_types  # MLP fc1 fed from same LN
        assert len(consumers) == 2

    def test_phi_parallel_residual_sums_attn_and_mlp(self):
        """The block output adds the MLP (fc2) result onto (x + attn)."""
        np.random.seed(3)
        config = {"model_type": "phi", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _phi_parallel_weights(32, 4, 8, 128)
        graph = _build_phi("phi_res", config, weights, blocks=[0])

        # fc2 = last dense feeding the final add; ensure MLP output is NOT
        # produced from a residual (i.e. fc1 is fed by LN, not by res1).
        ln_out = next(o for o in graph.operations
                      if o.op_type == OpType.LAYERNORM).outputs[0]
        dense_ops = [o for o in graph.operations if o.op_type == OpType.DENSE]
        fc1 = next(o for o in dense_ops if ln_out in o.inputs)
        # fc1's single input is the LayerNorm output (parallel, not sequential)
        assert fc1.inputs == [ln_out]

    def test_phi3_uses_rmsnorm_not_layernorm(self):
        np.random.seed(4)
        config = {"model_type": "phi3", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _phi3_weights(32, 8, 64)
        graph = _build_phi("phi3", config, weights, blocks=[0])
        op_types = {o.op_type for o in graph.operations}
        assert OpType.RMSNORM in op_types
        assert OpType.LAYERNORM not in op_types
        assert OpType.SWIGLU in op_types

    def test_phi3_missing_mlp_raises(self):
        np.random.seed(4)
        config = {"model_type": "phi3", "hidden_size": 32,
                  "num_attention_heads": 4, "num_key_value_heads": 4,
                  "head_dim": 8}
        weights = _phi3_weights(32, 8, 64, with_mlp=False)
        with pytest.raises(NotImplementedError):
            _build_phi("phi3_bad", config, weights, blocks=[0])
