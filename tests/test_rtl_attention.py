"""
RTL regression tests for multi-head attention.

Covers two audit fixes:
  * #3  the context (weighted-sum) stage was missing the rescale by the
        attention-weight fixed-point scale (2**(bits-1)), so every context
        element overflowed int{bits} and saturated to +/-max.
  * #6  no causal masking -- a position attended to *future* tokens.

forward_int only models attention at seq_len==1 (output == V projection), and
ReLU-attention equals softmax there only when the single score is positive, so
the seq=1 check forces a positive score (q_weight == k_weight => Q.K = |Q|^2).
The causal check is behavioural: perturbing a later token must not change the
output at earlier positions.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import forward_int
from tests.rtl_harness import simulate, have_iverilog

pytestmark = pytest.mark.skipif(not have_iverilog(),
                                reason="iverilog/vvp not installed")


def _proj(E, rng):
    return rng.standard_normal((E, E)) * 0.3, rng.standard_normal(E) * 0.1


def test_mha_seq1_context_rescale_matches_golden():
    """seq_len=1, positive score => attn weight 1.0 => context == V; the
    output must match forward_int.  Before the rescale fix the context
    saturated and the answer was garbage."""
    rng = np.random.default_rng(7)
    E, H = 8, 2
    W = rng.standard_normal((E, E)) * 0.3
    z = np.zeros(E)
    wv, bv = _proj(E, rng)
    wo, bo = _proj(E, rng)
    gb = GraphBuilder("mha_seq1")
    x = gb.input("x", shape=(E,))
    # q_weight == k_weight, zero q/k bias  =>  Q == K  =>  score = |Q|^2 >= 0
    y = gb.mha(x, num_heads=H, q_weight=W, q_bias=z, k_weight=W, k_bias=z,
               v_weight=wv, v_bias=bv, out_weight=wo, out_bias=bo, name="a")
    gb.output(y)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"x": rng.standard_normal((40, E))})

    worst = 0
    for _ in range(8):
        xin = rng.standard_normal(E)
        fi = forward_int(g, {"x": xin})[g.output_names[0]].ravel()
        rtl = simulate(g, {"x": xin})[g.output_names[0]]
        worst = max(worst, int(np.abs(rtl - fi).max()))
    assert worst <= 6, f"seq=1 MHA RTL vs golden off by {worst}"


@pytest.mark.parametrize("causal", [False, True])
def test_mha_causal_masking(causal):
    """With causal=True, changing the last token must NOT change the output at
    earlier positions.  With causal=False the earlier positions are allowed to
    (and generally do) depend on later tokens."""
    rng = np.random.default_rng(11)
    E, H, S = 8, 2, 3
    wq, bq = _proj(E, rng); wk, bk = _proj(E, rng)
    wv, bv = _proj(E, rng); wo, bo = _proj(E, rng)
    gb = GraphBuilder(f"mha_causal_{int(causal)}")
    x = gb.input("x", shape=(S, E))
    y = gb.mha(x, num_heads=H, seq_len=S, q_weight=wq, q_bias=bq,
               k_weight=wk, k_bias=bk, v_weight=wv, v_bias=bv,
               out_weight=wo, out_bias=bo, name="a")
    gb.output(y)
    g = gb.build()
    for op in g.topological_order():
        if op.op_type.name == "MULTI_HEAD_ATTENTION":
            op.attrs["causal"] = causal
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"x": rng.standard_normal((40, S, E))})

    base = rng.standard_normal((S, E))
    pert = base.copy()
    pert[S - 1] += 5.0  # perturb only the last token
    o0 = simulate(g, {"x": base})[g.output_names[0]].reshape(S, E)
    o1 = simulate(g, {"x": pert})[g.output_names[0]].reshape(S, E)

    if causal:
        # positions 0..S-2 must be unaffected by the last token
        assert np.array_equal(o0[:S - 1], o1[:S - 1]), \
            "causal mask leaked: an earlier position changed with a future token"
    else:
        # sanity: the model is actually wired up and reacts somewhere
        assert not np.array_equal(o0, o1)
