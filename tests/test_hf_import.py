"""Tests for the HuggingFace import module.

Tests build graphs from synthetic weight dicts that match the key patterns
of real HuggingFace models, without requiring network access.
"""

import numpy as np
import pytest

from w2s.importers.hf_import import (
    supported_architectures,
    _ARCH_MAP,
    _build_gpt2,
    _build_llama,
    _build_phi,
)
from w2s.core import ComputeGraph, OpType


class TestArchitectureMap:
    def test_supported_architectures_not_empty(self):
        archs = supported_architectures()
        assert len(archs) > 0

    def test_gpt2_supported(self):
        assert "gpt2" in _ARCH_MAP

    def test_llama_supported(self):
        assert "llama" in _ARCH_MAP

    def test_mistral_supported(self):
        assert "mistral" in _ARCH_MAP

    def test_all_builders_exist(self):
        import w2s.importers.hf_import as mod
        for arch, fn_name in _ARCH_MAP.items():
            assert hasattr(mod, fn_name), f"Missing builder {fn_name} for {arch}"


# ---------------------------------------------------------------------------
#  Synthetic weight factories
# ---------------------------------------------------------------------------

def _make_gpt2_block_weights(block_idx, embed_dim, ffn_dim):
    """Create weights for a single GPT-2 block."""
    E, F = embed_dim, ffn_dim
    p = f"h.{block_idx}"
    return {
        f"{p}.ln_1.weight": np.ones(E),
        f"{p}.ln_1.bias": np.zeros(E),
        f"{p}.attn.c_attn.weight": np.random.randn(E, 3 * E).astype(np.float64) * 0.02,
        f"{p}.attn.c_attn.bias": np.zeros(3 * E),
        f"{p}.attn.c_proj.weight": np.random.randn(E, E).astype(np.float64) * 0.02,
        f"{p}.attn.c_proj.bias": np.zeros(E),
        f"{p}.ln_2.weight": np.ones(E),
        f"{p}.ln_2.bias": np.zeros(E),
        f"{p}.mlp.c_fc.weight": np.random.randn(E, F).astype(np.float64) * 0.02,
        f"{p}.mlp.c_fc.bias": np.zeros(F),
        f"{p}.mlp.c_proj.weight": np.random.randn(F, E).astype(np.float64) * 0.02,
        f"{p}.mlp.c_proj.bias": np.zeros(E),
    }


def _make_gpt2_weights(embed_dim=32, n_heads=4, ffn_dim=128):
    """Create a minimal fake GPT-2 block 0 weight dict."""
    return _make_gpt2_block_weights(0, embed_dim, ffn_dim)


def _make_llama_weights(embed_dim=32, n_heads=4, n_kv_heads=4,
                         head_dim=8, ffn_dim=64):
    """Create a minimal fake Llama block 0 weight dict."""
    E = embed_dim
    kv_dim = n_kv_heads * head_dim
    return {
        "model.layers.0.input_layernorm.weight": np.ones(E),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(E, E).astype(np.float64) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(kv_dim, E).astype(np.float64) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(kv_dim, E).astype(np.float64) * 0.02,
        "model.layers.0.self_attn.o_proj.weight": np.random.randn(E, E).astype(np.float64) * 0.02,
        "model.layers.0.post_attention_layernorm.weight": np.ones(E),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(ffn_dim, E).astype(np.float64) * 0.02,
        "model.layers.0.mlp.up_proj.weight": np.random.randn(ffn_dim, E).astype(np.float64) * 0.02,
        "model.layers.0.mlp.down_proj.weight": np.random.randn(E, ffn_dim).astype(np.float64) * 0.02,
    }


def _make_phi_weights(embed_dim=32, n_heads=4, n_kv_heads=4,
                       head_dim=8, ffn_dim=128):
    """Create a minimal fake Phi block 0 weight dict."""
    E = embed_dim
    kv_dim = n_kv_heads * head_dim
    return {
        "model.layers.0.input_layernorm.weight": np.ones(E),
        "model.layers.0.input_layernorm.bias": np.zeros(E),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(E, E).astype(np.float64) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(kv_dim, E).astype(np.float64) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(kv_dim, E).astype(np.float64) * 0.02,
        "model.layers.0.self_attn.dense.weight": np.random.randn(E, E).astype(np.float64) * 0.02,
        "model.layers.0.mlp.fc1.weight": np.random.randn(ffn_dim, E).astype(np.float64) * 0.02,
        "model.layers.0.mlp.fc2.weight": np.random.randn(E, ffn_dim).astype(np.float64) * 0.02,
    }


# ---------------------------------------------------------------------------
#  GPT-2 builder tests
# ---------------------------------------------------------------------------

class TestBuildGPT2:
    def test_builds_graph(self):
        np.random.seed(42)
        config = {"n_embd": 32, "n_head": 4, "layer_norm_epsilon": 1e-5}
        weights = _make_gpt2_weights(embed_dim=32, n_heads=4, ffn_dim=128)
        graph = _build_gpt2("test_gpt2", config, weights, blocks=[0])
        assert isinstance(graph, ComputeGraph)
        assert graph.name == "test_gpt2"

    def test_has_expected_ops(self):
        np.random.seed(42)
        config = {"n_embd": 32, "n_head": 4}
        weights = _make_gpt2_weights(embed_dim=32, n_heads=4, ffn_dim=128)
        graph = _build_gpt2("test_gpt2", config, weights, blocks=[0])
        op_types = {op.op_type for op in graph.operations}
        assert OpType.LAYERNORM in op_types
        assert OpType.DENSE in op_types
        assert OpType.GELU in op_types
        assert OpType.ADD in op_types

    def test_input_output_shapes(self):
        np.random.seed(42)
        config = {"n_embd": 32, "n_head": 4}
        weights = _make_gpt2_weights(embed_dim=32, n_heads=4, ffn_dim=128)
        graph = _build_gpt2("test_gpt2", config, weights, blocks=[0])
        assert "token_embed" in graph.input_names
        assert graph.input_shapes["token_embed"] == (32,)
        assert len(graph.output_names) == 1

    def test_quantizes_without_error(self):
        np.random.seed(42)
        config = {"n_embd": 32, "n_head": 4}
        weights = _make_gpt2_weights(embed_dim=32, n_heads=4, ffn_dim=128)
        graph = _build_gpt2("test_gpt2", config, weights, blocks=[0])
        from w2s.core import QuantConfig
        from w2s.quantize import quantize_graph
        graph.quant_config = QuantConfig(bits=8)
        calib = {"token_embed": np.random.randn(4, 32).astype(np.float32)}
        quantize_graph(graph, calib)
        # Should have q_weights on weighted ops
        weighted = [op for op in graph.operations if op.q_weights]
        assert len(weighted) > 0

    def test_multi_block(self):
        np.random.seed(42)
        E, F = 32, 128
        weights = _make_gpt2_block_weights(0, E, F)
        weights.update(_make_gpt2_block_weights(1, E, F))
        config = {"n_embd": E, "n_head": 4}
        graph = _build_gpt2("test_gpt2_2b", config, weights, blocks=[0, 1])
        # Should have roughly twice as many ops
        assert len(graph.operations) > 10


# ---------------------------------------------------------------------------
#  Llama builder tests
# ---------------------------------------------------------------------------

class TestBuildLlama:
    def test_builds_graph(self):
        np.random.seed(42)
        config = {
            "model_type": "llama",
            "hidden_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 4, "head_dim": 8,
            "intermediate_size": 64, "rms_norm_eps": 1e-5,
        }
        weights = _make_llama_weights(embed_dim=32, n_heads=4,
                                       n_kv_heads=4, head_dim=8, ffn_dim=64)
        graph = _build_llama("test_llama", config, weights, blocks=[0])
        assert isinstance(graph, ComputeGraph)

    def test_has_rmsnorm_and_swiglu(self):
        np.random.seed(42)
        config = {
            "hidden_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 4, "head_dim": 8,
            "intermediate_size": 64, "rms_norm_eps": 1e-5,
        }
        weights = _make_llama_weights(embed_dim=32, n_heads=4,
                                       n_kv_heads=4, head_dim=8, ffn_dim=64)
        graph = _build_llama("test_llama", config, weights, blocks=[0])
        op_types = {op.op_type for op in graph.operations}
        assert OpType.RMSNORM in op_types
        assert OpType.SWIGLU in op_types

    def test_gqa_when_kv_heads_differ(self):
        np.random.seed(42)
        config = {
            "hidden_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "intermediate_size": 64, "rms_norm_eps": 1e-5,
        }
        kv_dim = 2 * 8  # n_kv_heads * head_dim = 16
        weights = _make_llama_weights(embed_dim=32, n_heads=4,
                                       n_kv_heads=2, head_dim=8, ffn_dim=64)
        graph = _build_llama("test_llama_gqa", config, weights, blocks=[0])
        op_types = {op.op_type for op in graph.operations}
        assert OpType.GROUPED_QUERY_ATTENTION in op_types


# ---------------------------------------------------------------------------
#  Phi builder tests
# ---------------------------------------------------------------------------

class TestBuildPhi:
    def test_builds_graph(self):
        np.random.seed(42)
        config = {
            "hidden_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 4, "head_dim": 8,
            "intermediate_size": 128, "rms_norm_eps": 1e-5,
        }
        weights = _make_phi_weights(embed_dim=32, n_heads=4, ffn_dim=128)
        graph = _build_phi("test_phi", config, weights, blocks=[0])
        assert isinstance(graph, ComputeGraph)

    def test_uses_layernorm_and_gelu(self):
        np.random.seed(42)
        config = {
            "hidden_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 4, "head_dim": 8,
            "intermediate_size": 128,
        }
        weights = _make_phi_weights(embed_dim=32, n_heads=4, ffn_dim=128)
        graph = _build_phi("test_phi", config, weights, blocks=[0])
        op_types = {op.op_type for op in graph.operations}
        assert OpType.LAYERNORM in op_types
        assert OpType.GELU in op_types
