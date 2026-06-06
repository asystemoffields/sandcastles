"""
hf_import.py — Load models directly from HuggingFace into the w2s IR.

Usage:
    from w2s.importers.hf_import import load_hf
    graph = load_hf("openai-community/gpt2")
    graph = load_hf("TinyLlama/TinyLlama-1.1B-Chat-v1.0", blocks=[0])

Supports automatic architecture detection for:
  - GPT-2 family (openai-community/gpt2, gpt2-medium, etc.)
  - Llama family (TinyLlama, Llama-2, Llama-3, etc.)
  - Mistral family
  - Qwen2 family
  - Phi family
  - Gemma family

The importer downloads safetensors weights, detects the model architecture
from config.json, and builds a ComputeGraph using the GraphBuilder API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from w2s.core import ComputeGraph, QuantConfig
from w2s.importers.builder import GraphBuilder


# ---------------------------------------------------------------------------
#  Architecture detection
# ---------------------------------------------------------------------------

# Maps HuggingFace model_type -> our builder function name
_ARCH_MAP = {
    "gpt2": "_build_gpt2",
    "llama": "_build_llama",
    "mistral": "_build_llama",     # same structure as llama
    "qwen2": "_build_llama",       # same structure as llama
    "phi": "_build_phi",
    "phi3": "_build_phi",
    "gemma": "_build_llama",       # same structure as llama
    "gemma2": "_build_llama",
}


def supported_architectures() -> List[str]:
    """Return list of supported HuggingFace model_type values."""
    return sorted(set(_ARCH_MAP.keys()))


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def load_hf(
    model_id: str,
    blocks: Optional[List[int]] = None,
    name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> ComputeGraph:
    """
    Load a model from HuggingFace and return a ComputeGraph.

    Args:
        model_id:   HuggingFace model ID (e.g., "openai-community/gpt2").
        blocks:     Which transformer blocks to include (default: [0]).
                    Use None or [0] for the first block only.
        name:       Verilog module name (default: derived from model_id).
        cache_dir:  HuggingFace cache directory (default: system default).

    Returns:
        A ComputeGraph with float weights ready for quantization.
    """
    if blocks is None:
        blocks = [0]

    if name is None:
        # "openai-community/gpt2" -> "gpt2"
        name = model_id.split("/")[-1].replace("-", "_").replace(".", "_").lower()

    # Download and load
    config, weights = _download_model(model_id, cache_dir)
    model_type = config.get("model_type", "").lower()

    builder_fn = _ARCH_BUILDERS.get(model_type)
    if builder_fn is None:
        raise ValueError(
            f"Unsupported model architecture: '{model_type}'. "
            f"Supported: {', '.join(supported_architectures())}"
        )

    graph = builder_fn(name, config, weights, blocks)
    return graph


def load_hf_weights(
    model_id: str,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Download and return (config, weights) without building a graph.

    Useful for inspecting model structure before compilation.
    """
    return _download_model(model_id, cache_dir)


# ---------------------------------------------------------------------------
#  Download helpers
# ---------------------------------------------------------------------------

def _download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Download config and weights from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "The 'huggingface_hub' package is required to load models from HuggingFace.\n"
            "Install it with:  pip install huggingface_hub safetensors"
        )

    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError(
            "The 'safetensors' package is required to load model weights.\n"
            "Install it with:  pip install safetensors"
        )

    # Download config
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    config_path = hf_hub_download(model_id, "config.json", **kwargs)
    with open(config_path) as f:
        config = json.load(f)

    # Download weights -- try safetensors first
    weights = {}
    try:
        # Try single-file safetensors
        st_path = hf_hub_download(model_id, "model.safetensors", **kwargs)
        with safe_open(st_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    except (OSError, ValueError):
        # Try sharded safetensors
        try:
            index_path = hf_hub_download(
                model_id, "model.safetensors.index.json", **kwargs)
            with open(index_path) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            for shard_name in sorted(shard_files):
                shard_path = hf_hub_download(model_id, shard_name, **kwargs)
                with safe_open(shard_path, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
        except Exception as e:
            raise RuntimeError(
                f"Could not load weights from '{model_id}'. "
                f"Ensure the model has safetensors weights.\n"
                f"Error: {e}"
            )

    return config, weights


# ---------------------------------------------------------------------------
#  Architecture-specific builders
# ---------------------------------------------------------------------------

def _get_weight(weights: Dict[str, np.ndarray], key: str) -> np.ndarray:
    """Get a weight tensor, raising a clear error if missing."""
    if key not in weights:
        raise KeyError(
            f"Weight '{key}' not found. Available keys: "
            + ", ".join(sorted(k for k in weights if key.split(".")[0] in k)[:10])
        )
    return weights[key].astype(np.float64)


def _get_weight_optional(
    weights: Dict[str, np.ndarray], key: str
) -> Optional[np.ndarray]:
    """Get a weight tensor or None if missing."""
    if key not in weights:
        return None
    return weights[key].astype(np.float64)


def _rope_tables(
    dim: int, max_seq_len: int, base: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute (cos, sin) rotary tables of shape (max_seq_len, dim // 2).

    inv_freq[i] = base ** (-2i / dim);  angle[pos, i] = pos * inv_freq[i].
    These match the layout expected by ``generators.transformer.generate_rope``
    (which indexes cos/sin as ``[pos, half_dim]``).
    """
    half = dim // 2
    inv_freq = base ** (-(np.arange(0, half, dtype=np.float64) * 2.0) / dim)
    pos = np.arange(max_seq_len, dtype=np.float64)[:, None]      # (S, 1)
    angle = pos * inv_freq[None, :]                              # (S, half)
    return np.cos(angle), np.sin(angle)


# ---------------------------------------------------------------------------
#  GPT-2
# ---------------------------------------------------------------------------

def _build_gpt2(
    name: str,
    config: Dict[str, Any],
    weights: Dict[str, np.ndarray],
    blocks: List[int],
) -> ComputeGraph:
    """Build a ComputeGraph for GPT-2 family models."""
    embed_dim = config.get("n_embd", 768)
    n_heads = config.get("n_head", 12)
    eps = config.get("layer_norm_epsilon", 1e-5)

    gb = GraphBuilder(name)
    x = gb.input("token_embed", shape=(embed_dim,))

    for block_idx in blocks:
        prefix = f"h.{block_idx}"

        # LayerNorm 1
        ln1_s = _get_weight(weights, f"{prefix}.ln_1.weight")
        ln1_b = _get_weight(weights, f"{prefix}.ln_1.bias")
        ln1 = gb.layernorm(x, ln1_s, ln1_b, eps=eps, name=f"b{block_idx}_ln1")

        # Attention: GPT-2 stores Q/K/V as a single (embed, 3*embed) matrix
        attn_w = _get_weight(weights, f"{prefix}.attn.c_attn.weight")  # (E, 3E)
        attn_b = _get_weight(weights, f"{prefix}.attn.c_attn.bias")    # (3E,)

        q_w = attn_w[:, :embed_dim].T
        k_w = attn_w[:, embed_dim:2*embed_dim].T
        v_w = attn_w[:, 2*embed_dim:].T
        q_b = attn_b[:embed_dim]
        k_b = attn_b[embed_dim:2*embed_dim]
        v_b = attn_b[2*embed_dim:]

        out_w = _get_weight(weights, f"{prefix}.attn.c_proj.weight").T
        out_b = _get_weight(weights, f"{prefix}.attn.c_proj.bias")

        attn_out = gb.mha(
            ln1,
            q_weight=q_w, q_bias=q_b,
            k_weight=k_w, k_bias=k_b,
            v_weight=v_w, v_bias=v_b,
            out_weight=out_w, out_bias=out_b,
            num_heads=n_heads,
            causal=True,   # GPT-2 is an autoregressive decoder
            name=f"b{block_idx}_attn",
        )

        proj = attn_out  # MHA includes output projection

        # Residual 1
        res1 = gb.add(x, proj, name=f"b{block_idx}_res1")

        # LayerNorm 2
        ln2_s = _get_weight(weights, f"{prefix}.ln_2.weight")
        ln2_b = _get_weight(weights, f"{prefix}.ln_2.bias")
        ln2 = gb.layernorm(res1, ln2_s, ln2_b, eps=eps, name=f"b{block_idx}_ln2")

        # FFN
        ffn1_w = _get_weight(weights, f"{prefix}.mlp.c_fc.weight").T
        ffn1_b = _get_weight(weights, f"{prefix}.mlp.c_fc.bias")
        ffn1 = gb.dense(ln2, ffn1_w, ffn1_b, name=f"b{block_idx}_ffn1")
        ffn1_act = gb.gelu(ffn1, name=f"b{block_idx}_gelu")

        ffn2_w = _get_weight(weights, f"{prefix}.mlp.c_proj.weight").T
        ffn2_b = _get_weight(weights, f"{prefix}.mlp.c_proj.bias")
        ffn2 = gb.dense(ffn1_act, ffn2_w, ffn2_b, name=f"b{block_idx}_ffn2")

        # Residual 2
        x = gb.add(res1, ffn2, name=f"b{block_idx}_res2")

    gb.output(x)
    return gb.build()


# ---------------------------------------------------------------------------
#  Llama / Mistral / Qwen2 / Gemma (all share the same structure)
# ---------------------------------------------------------------------------

def _geglu_ffn(
    gb: GraphBuilder,
    inp: str,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
    name: str,
) -> str:
    """GELU-gated FFN (GeGLU), built from gate/up/down dense + GELU.

    out = (gelu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T

    Used by Gemma/Gemma2 whose MLP is GeGLU (gelu_pytorch_tanh), in contrast
    to the SiLU-gated SwiGLU used by Llama/Mistral/Qwen2.  There is no fused
    GeGLU builder op, so it is composed from existing primitives.
    """
    gate = gb.dense(inp, gate_w, name=f"{name}_gate")
    gate_act = gb.gelu(gate, name=f"{name}_gelu")
    up = gb.dense(inp, up_w, name=f"{name}_up")
    hidden = gb.multiply(gate_act, up, name=f"{name}_mul")
    return gb.dense(hidden, down_w, name=f"{name}_down")


def _build_llama(
    name: str,
    config: Dict[str, Any],
    weights: Dict[str, np.ndarray],
    blocks: List[int],
) -> ComputeGraph:
    """Build a ComputeGraph for Llama-family models (Llama/Mistral/Qwen2/Gemma).

    Fidelity notes:
      * RoPE — Llama-family attention is rotary.  The fused ``mha``/``gqa``
        builder ops compute Q/K/V projections *internally* and expose no Q/K
        tensor, so RoPE cannot be wired between projection and attention.
        Rather than ship a model with NO positional information, a RoPE op is
        wired onto the (post-norm) hidden state feeding attention, using config
        ``rope_theta`` / ``head_dim`` to size the rotation.  This is a
        documented approximation forced by the monolithic attention op: exact
        per-head rotation of Q/K would require extending the attention
        generator (not permitted here).  The rotation is applied at full
        ``embed_dim`` width so the vector flows through attention unchanged in
        shape.
      * Gemma uses GeGLU (GELU-gated) FFN, not SwiGLU.
      * Gemma2 uses four norms/layer (sandwich norm); all four are wired.
      * Gemma RMSNorm weights carry the (1 + w) convention (preserved).
    """
    embed_dim = config.get("hidden_size", 4096)
    n_heads = config.get("num_attention_heads", 32)
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim", embed_dim // n_heads)
    eps = config.get("rms_norm_eps", 1e-5)
    model_type = config.get("model_type", "llama")
    rope_theta = float(config.get("rope_theta", 10000.0))
    # Cap the rotary ROM at a synthesizable size (the generator emits one ROM
    # entry per position); synthetic test configs are far smaller than this.
    max_seq_len = min(int(config.get("max_position_embeddings", 2048)), 2048)

    is_gemma = model_type in ("gemma", "gemma2")
    is_gemma2 = model_type == "gemma2"
    norm_off = 1.0 if is_gemma else 0.0  # Gemma RMSNorm uses (1 + weight)

    prefix_base = "model.layers"

    gb = GraphBuilder(name)
    x = gb.input("hidden_state", shape=(embed_dim,))
    pos = gb.input("position_id", shape=(1,))

    # RoPE cos/sin tables (full-width rotation on the hidden state; see notes).
    cos_t, sin_t = _rope_tables(embed_dim, max_seq_len, rope_theta)

    for block_idx in blocks:
        prefix = f"{prefix_base}.{block_idx}"

        # --- Input RMSNorm ---
        rms_w = _get_weight(weights, f"{prefix}.input_layernorm.weight") + norm_off
        rms = gb.rmsnorm(x, rms_w, eps=eps, name=f"b{block_idx}_rms1")

        # --- RoPE on the normalized hidden state feeding attention ---
        roped = gb.rope(
            rms, cos_t, sin_t,
            dim=embed_dim, max_seq_len=max_seq_len, base=rope_theta,
            position_input=pos, name=f"b{block_idx}_rope",
        )

        # --- Self-attention projections ---
        q_w = _get_weight(weights, f"{prefix}.self_attn.q_proj.weight")
        k_w = _get_weight(weights, f"{prefix}.self_attn.k_proj.weight")
        v_w = _get_weight(weights, f"{prefix}.self_attn.v_proj.weight")
        o_w = _get_weight(weights, f"{prefix}.self_attn.o_proj.weight")
        q_b = _get_weight_optional(weights, f"{prefix}.self_attn.q_proj.bias")
        k_b = _get_weight_optional(weights, f"{prefix}.self_attn.k_proj.bias")
        v_b = _get_weight_optional(weights, f"{prefix}.self_attn.v_proj.bias")
        o_b = _get_weight_optional(weights, f"{prefix}.self_attn.o_proj.bias")

        # Default biases to zeros if missing
        if q_b is None: q_b = np.zeros(q_w.shape[0])
        if k_b is None: k_b = np.zeros(k_w.shape[0])
        if v_b is None: v_b = np.zeros(v_w.shape[0])
        if o_b is None: o_b = np.zeros(o_w.shape[0])

        if n_kv_heads == n_heads:
            attn_out = gb.mha(
                roped,
                q_weight=q_w, q_bias=q_b,
                k_weight=k_w, k_bias=k_b,
                v_weight=v_w, v_bias=v_b,
                out_weight=o_w, out_bias=o_b,
                num_heads=n_heads,
                causal=True,   # Llama/Mistral/Qwen/Gemma are decoders
                name=f"b{block_idx}_attn",
            )
        else:
            attn_out = gb.gqa(
                roped,
                q_weight=q_w, q_bias=q_b,
                k_weight=k_w, k_bias=k_b,
                v_weight=v_w, v_bias=v_b,
                out_weight=o_w, out_bias=o_b,
                num_heads=n_heads, num_kv_heads=n_kv_heads,
                causal=True,
                name=f"b{block_idx}_attn",
            )

        # Gemma2 sandwich norm: normalize attention output before residual.
        if is_gemma2:
            pa_w = _get_weight(
                weights, f"{prefix}.post_attention_layernorm.weight") + norm_off
            attn_out = gb.rmsnorm(
                attn_out, pa_w, eps=eps, name=f"b{block_idx}_post_attn_norm")

        # Residual 1
        res1 = gb.add(x, attn_out, name=f"b{block_idx}_res1")

        # --- Pre-FFN norm ---
        # Llama/Mistral/Qwen2/Gemma1: post_attention_layernorm is the FFN's
        # pre-norm.  Gemma2: a dedicated pre_feedforward_layernorm is used.
        if is_gemma2:
            ffn_norm_w = _get_weight(
                weights, f"{prefix}.pre_feedforward_layernorm.weight") + norm_off
        else:
            ffn_norm_w = _get_weight(
                weights, f"{prefix}.post_attention_layernorm.weight") + norm_off
        rms2 = gb.rmsnorm(res1, ffn_norm_w, eps=eps, name=f"b{block_idx}_rms2")

        # --- Gated FFN ---
        gate_w = _get_weight(weights, f"{prefix}.mlp.gate_proj.weight")
        up_w = _get_weight(weights, f"{prefix}.mlp.up_proj.weight")
        down_w = _get_weight(weights, f"{prefix}.mlp.down_proj.weight")

        if is_gemma:
            ffn = _geglu_ffn(gb, rms2, gate_w, up_w, down_w,
                             name=f"b{block_idx}_ffn")
        else:
            ffn = gb.swiglu(rms2, gate_w, up_w, down_w, name=f"b{block_idx}_ffn")

        # Gemma2 sandwich norm: normalize FFN output before residual.
        if is_gemma2:
            pf_w = _get_weight(
                weights, f"{prefix}.post_feedforward_layernorm.weight") + norm_off
            ffn = gb.rmsnorm(ffn, pf_w, eps=eps, name=f"b{block_idx}_post_ffn_norm")

        # Residual 2
        x = gb.add(res1, ffn, name=f"b{block_idx}_res2")

    gb.output(x)
    return gb.build()


# ---------------------------------------------------------------------------
#  Phi family
# ---------------------------------------------------------------------------

def _load_phi_attn_weights(
    weights: Dict[str, np.ndarray],
    prefix: str,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
):
    """Load Q/K/V/O projection weights+biases for a Phi-family attention block.

    Handles both fused ``qkv_proj`` (Phi-3) and separate ``q/k/v_proj``
    (Phi-1/2), and both ``self_attn.dense`` (Phi-1/2) and ``self_attn.o_proj``
    (Phi-3) output projections.  Biases default to zeros when absent.
    """
    qkv_key = f"{prefix}.self_attn.qkv_proj.weight"
    if qkv_key in weights:
        qkv_w = _get_weight(weights, qkv_key)
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        q_w = qkv_w[:q_dim]
        k_w = qkv_w[q_dim:q_dim + kv_dim]
        v_w = qkv_w[q_dim + kv_dim:]

        qkv_bias_key = f"{prefix}.self_attn.qkv_proj.bias"
        if qkv_bias_key in weights:
            qkv_b = _get_weight(weights, qkv_bias_key)
            q_b = qkv_b[:q_dim]
            k_b = qkv_b[q_dim:q_dim + kv_dim]
            v_b = qkv_b[q_dim + kv_dim:]
        else:
            q_b = _get_weight_optional(weights, f"{prefix}.self_attn.q_proj.bias")
            k_b = _get_weight_optional(weights, f"{prefix}.self_attn.k_proj.bias")
            v_b = _get_weight_optional(weights, f"{prefix}.self_attn.v_proj.bias")
    else:
        q_w = _get_weight(weights, f"{prefix}.self_attn.q_proj.weight")
        k_w = _get_weight(weights, f"{prefix}.self_attn.k_proj.weight")
        v_w = _get_weight(weights, f"{prefix}.self_attn.v_proj.weight")
        q_b = _get_weight_optional(weights, f"{prefix}.self_attn.q_proj.bias")
        k_b = _get_weight_optional(weights, f"{prefix}.self_attn.k_proj.bias")
        v_b = _get_weight_optional(weights, f"{prefix}.self_attn.v_proj.bias")

    # Output projection: Phi-1/2 use `dense`, Phi-3 uses `o_proj`.
    o_w = _get_weight_optional(weights, f"{prefix}.self_attn.dense.weight")
    o_b = _get_weight_optional(weights, f"{prefix}.self_attn.dense.bias")
    if o_w is None:
        o_w = _get_weight(weights, f"{prefix}.self_attn.o_proj.weight")
        o_b = _get_weight_optional(weights, f"{prefix}.self_attn.o_proj.bias")

    if q_b is None: q_b = np.zeros(q_w.shape[0])
    if k_b is None: k_b = np.zeros(k_w.shape[0])
    if v_b is None: v_b = np.zeros(v_w.shape[0])
    if o_b is None: o_b = np.zeros(o_w.shape[0])
    return q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b


def _emit_phi_attn(
    gb: GraphBuilder, inp: str, attn_w, n_heads: int, n_kv_heads: int, name: str
) -> str:
    """Emit an MHA or GQA op from pre-loaded attention weights."""
    q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b = attn_w
    if n_kv_heads == n_heads:
        return gb.mha(
            inp,
            q_weight=q_w, q_bias=q_b, k_weight=k_w, k_bias=k_b,
            v_weight=v_w, v_bias=v_b, out_weight=o_w, out_bias=o_b,
            num_heads=n_heads, name=name, causal=True,
        )
    return gb.gqa(
        inp,
        q_weight=q_w, q_bias=q_b, k_weight=k_w, k_bias=k_b,
        v_weight=v_w, v_bias=v_b, out_weight=o_w, out_bias=o_b,
        num_heads=n_heads, num_kv_heads=n_kv_heads, name=name, causal=True,
    )


def _build_phi(
    name: str,
    config: Dict[str, Any],
    weights: Dict[str, np.ndarray],
    blocks: List[int],
) -> ComputeGraph:
    """Build a ComputeGraph for Phi-family models.

    Two distinct architectures share this builder:

      * model_type "phi" (Phi-1/Phi-2): a PARALLEL block with a single
        LayerNorm whose output feeds BOTH attention and MLP:
            out = x + attn(LN(x)) + mlp(LN(x))
      * model_type "phi3": a sequential pre-norm block using RMSNorm (not
        LayerNorm), with a fused gate_up_proj/down_proj SwiGLU MLP.
    """
    embed_dim = config.get("hidden_size", 2560)
    n_heads = config.get("num_attention_heads", 32)
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim", embed_dim // n_heads)
    eps = config.get("rms_norm_eps", 1e-5)
    model_type = config.get("model_type", "phi")

    prefix_base = "model.layers"

    gb = GraphBuilder(name)
    x = gb.input("hidden_state", shape=(embed_dim,))

    for block_idx in blocks:
        prefix = f"{prefix_base}.{block_idx}"
        attn_w = _load_phi_attn_weights(
            weights, prefix, n_heads, n_kv_heads, head_dim)

        if model_type == "phi3":
            # --- Phi-3: sequential pre-norm, RMSNorm, SwiGLU MLP ---
            rms_w = _get_weight(weights, f"{prefix}.input_layernorm.weight")
            rms = gb.rmsnorm(x, rms_w, eps=eps, name=f"b{block_idx}_rms1")

            attn_out = _emit_phi_attn(
                gb, rms, attn_w, n_heads, n_kv_heads, name=f"b{block_idx}_attn")
            res1 = gb.add(x, attn_out, name=f"b{block_idx}_res1")

            post_key = f"{prefix}.post_attention_layernorm.weight"
            if post_key not in weights:
                raise NotImplementedError(
                    f"phi3 block {block_idx}: expected RMSNorm "
                    f"'post_attention_layernorm' weight; not found."
                )
            rms2 = gb.rmsnorm(
                res1, _get_weight(weights, post_key), eps=eps,
                name=f"b{block_idx}_rms2",
            )

            # Phi-3 MLP is SwiGLU with a fused gate_up_proj.
            gate_up_key = f"{prefix}.mlp.gate_up_proj.weight"
            down_key = f"{prefix}.mlp.down_proj.weight"
            if gate_up_key not in weights or down_key not in weights:
                raise NotImplementedError(
                    f"phi3 block {block_idx}: expected fused SwiGLU MLP weights "
                    f"'mlp.gate_up_proj.weight' and 'mlp.down_proj.weight'; "
                    f"found neither — unhandled Phi-3 MLP variant."
                )
            gate_up = _get_weight(weights, gate_up_key)   # (2 * ffn_dim, embed)
            ffn_dim = gate_up.shape[0] // 2
            gate_w = gate_up[:ffn_dim]
            up_w = gate_up[ffn_dim:]
            down_w = _get_weight(weights, down_key)
            ffn = gb.swiglu(rms2, gate_w, up_w, down_w, name=f"b{block_idx}_ffn")
            x = gb.add(res1, ffn, name=f"b{block_idx}_res2")
            continue

        # --- Phi-1/Phi-2: PARALLEL block with a shared LayerNorm ---
        ln_w = _get_weight(weights, f"{prefix}.input_layernorm.weight")
        ln_b = _get_weight_optional(weights, f"{prefix}.input_layernorm.bias")
        if ln_b is None:
            ln_b = np.zeros_like(ln_w)
        ln = gb.layernorm(x, ln_w, ln_b, eps=eps, name=f"b{block_idx}_ln")

        # Attention sublayer fed from LN(x)
        attn_out = _emit_phi_attn(
            gb, ln, attn_w, n_heads, n_kv_heads, name=f"b{block_idx}_attn")

        # MLP sublayer fed from the SAME LN(x)  (parallel, not sequential)
        fc1_w = _get_weight(weights, f"{prefix}.mlp.fc1.weight")
        fc1_b = _get_weight_optional(weights, f"{prefix}.mlp.fc1.bias")
        fc1 = gb.dense(ln, fc1_w, fc1_b, name=f"b{block_idx}_fc1")
        act = gb.gelu(fc1, name=f"b{block_idx}_gelu")
        fc2_w = _get_weight(weights, f"{prefix}.mlp.fc2.weight")
        fc2_b = _get_weight_optional(weights, f"{prefix}.mlp.fc2.bias")
        fc2 = gb.dense(act, fc2_w, fc2_b, name=f"b{block_idx}_fc2")

        # Single residual sums both sublayers: out = x + attn + mlp
        partial = gb.add(x, attn_out, name=f"b{block_idx}_res1")
        x = gb.add(partial, fc2, name=f"b{block_idx}_res2")

    gb.output(x)
    return gb.build()


# ---------------------------------------------------------------------------
#  Architecture builder dispatch
# ---------------------------------------------------------------------------

_ARCH_BUILDERS = {
    "gpt2": _build_gpt2,
    "llama": _build_llama,
    "mistral": _build_llama,
    "qwen2": _build_llama,
    "phi": _build_phi,
    "phi3": _build_phi,
    "gemma": _build_llama,
    "gemma2": _build_llama,
}


# ---------------------------------------------------------------------------
#  Inspection helpers
# ---------------------------------------------------------------------------

def inspect_hf(model_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Download and inspect a HuggingFace model without building a graph.

    Returns a human-readable summary of the model's architecture,
    parameters, and weight tensor shapes.
    """
    config, weights = _download_model(model_id, cache_dir)
    model_type = config.get("model_type", "unknown")
    supported = model_type.lower() in _ARCH_MAP

    total_params = sum(int(np.prod(v.shape)) for v in weights.values())

    lines = [
        f"Model: {model_id}",
        f"Architecture: {model_type}" + (" (supported)" if supported else " (NOT SUPPORTED)"),
        f"Parameters: {total_params:,}",
        "",
        "Config:",
    ]

    # Show relevant config entries
    for key in ("hidden_size", "n_embd", "num_attention_heads", "n_head",
                "num_key_value_heads", "num_hidden_layers", "n_layer",
                "intermediate_size", "vocab_size", "max_position_embeddings"):
        if key in config:
            lines.append(f"  {key}: {config[key]}")

    lines.append("")
    lines.append(f"Weight tensors: {len(weights)}")

    # Group by layer prefix
    prefixes = {}
    for key in sorted(weights.keys()):
        parts = key.split(".")
        prefix = ".".join(parts[:3]) if len(parts) > 3 else key
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append((key, weights[key].shape))

    for prefix, entries in list(prefixes.items())[:15]:
        lines.append(f"  {prefix}:")
        for key, shape in entries[:5]:
            short_key = key[len(prefix)+1:] if key.startswith(prefix + ".") else key
            lines.append(f"    {short_key}: {shape}")
        if len(entries) > 5:
            lines.append(f"    ... and {len(entries) - 5} more")

    if len(prefixes) > 15:
        lines.append(f"  ... and {len(prefixes) - 15} more layer groups")

    return "\n".join(lines)
