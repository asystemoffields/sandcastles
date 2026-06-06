"""
Regression tests for three confirmed resource-estimation bugs.

Each test is written to FAIL on the pre-fix code and PASS after the fix:

  1. fpga.estimate_fpga counted multipliers only for DENSE/CONV 'weight',
     so attention / SwiGLU / embedding contributed ZERO multipliers and a
     transformer was reported to "FIT" at ~0 LUTs.
  2. autofit.autofit_fpga estimated area with the Tiny Tapeout ASIC model
     (whole weight ROM stuffed into LUT fabric) and ignored BRAM, wrongly
     declaring BRAM-resident models "too large".
  3. estimate.estimate sequential path never populated sparsity /
     multipliers_eliminated, even for a pruned model.
"""

import numpy as np

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.sparsity import prune_weights
from w2s.estimate import estimate
from w2s.fpga import estimate_fpga, ICE40_UP5K
from w2s.autofit import autofit_fpga


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _mha_graph(embed_dim=8, num_heads=2, seq_len=4, seed=0):
    """A single multi-head-attention layer (the transformer path)."""
    rng = np.random.default_rng(seed)

    def W():
        return rng.standard_normal((embed_dim, embed_dim)) * 0.3

    def b():
        return np.zeros(embed_dim)

    gb = GraphBuilder("mha_only")
    x = gb.input("x", shape=(seq_len, embed_dim))
    y = gb.mha(x, W(), b(), W(), b(), W(), b(), W(), b(),
               num_heads=num_heads, seq_len=seq_len, name="attn")
    gb.output(y)
    g = gb.build()
    g.quant_config = QuantConfig(bits=8)
    quantize_graph(g, {"x": rng.standard_normal((3, seq_len, embed_dim))})
    return g


def _dense_graph(n_in, n_out, seed=0, name="dense_only"):
    rng = np.random.default_rng(seed)
    gb = GraphBuilder(name)
    x = gb.input("x", shape=(n_in,))
    y = gb.dense(x, rng.standard_normal((n_out, n_in)) * 0.1,
                 np.zeros(n_out), name="fc")
    gb.output(y)
    g = gb.build()
    return g


# ---------------------------------------------------------------------------
#  Fix 1 — attention weights must count toward FPGA multipliers
# ---------------------------------------------------------------------------

class TestFix1AttentionMultipliers:
    def test_mha_yields_multipliers_and_nontrivial_luts(self):
        g = _mha_graph()
        report = estimate_fpga(g, ICE40_UP5K, "combinational")

        # Old code: the four attention projections were ignored, so
        # lut4s_used == 0, mul_luts == 0, and the layer falsely "FITS".
        mul_luts = report._breakdown.get("mul_luts", 0)
        assert mul_luts > 0, "attention projections must produce multipliers"
        assert report.lut4s_used > 0, "transformer must have a non-trivial LUT cost"

        # The combinational TT estimator counts these same weights; the FPGA
        # multiplier total must therefore be > 0 as well.
        tt = estimate(g, mode="combinational")
        assert tt.total_multipliers > 0
        # ~4 projections x 8x8 dense = ~256 nonzero multipliers (a couple may
        # quantize to zero).  The FPGA estimator must count the same matmul
        # weights as the TT estimator: mul_luts == multipliers * 24 (8-bit).
        assert tt.total_multipliers >= 250
        assert mul_luts == tt.total_multipliers * 24

    def test_mha_sequential_weights_go_to_bram(self):
        g = _mha_graph()
        report = estimate_fpga(g, ICE40_UP5K, "sequential")
        # Attention weights now land in the weight ROM / BRAM.
        assert report.bram_bits_used > 0


# ---------------------------------------------------------------------------
#  Fix 2 — autofit_fpga must use the BRAM-aware FPGA estimator
# ---------------------------------------------------------------------------

class TestFix2AutofitUsesFPGAEstimator:
    def test_bram_resident_model_is_not_too_large(self):
        # ~8200 params.  Sequential FPGA: ROM = 8190*8 = 65,520 bits fits in
        # the iCE40UP5K's 122,880 BRAM bits and the logic is a few hundred
        # LUTs -> FITS.  The old ASIC estimator stuffed the ROM into LUTs
        # (65,520 // 5 ~= 13,104 LUTs >> 5,280) so autofit declared it too big.
        g = _dense_graph(90, 90, name="bram_model")
        calib = {"x": np.random.default_rng(1).standard_normal((4, 90))}

        # The FPGA estimator itself says this FITS in sequential mode...
        gq = _dense_graph(90, 90, name="bram_model")
        gq.quant_config = QuantConfig(bits=8)
        quantize_graph(gq, calib)
        fpga_seq = estimate_fpga(gq, ICE40_UP5K, "sequential")
        assert fpga_seq.fits, "precondition: model fits per estimate_fpga"

        # ...so autofit_fpga must NOT report it as too large.
        result = autofit_fpga(g, calib, ICE40_UP5K)
        assert result.fits
        assert "too large" not in result.config_summary.lower()
        assert result.device_luts == ICE40_UP5K.lut4s


# ---------------------------------------------------------------------------
#  Fix 3 — sequential estimate must populate sparsity stats
# ---------------------------------------------------------------------------

class TestFix3SequentialSparsity:
    def test_sequential_reports_sparsity_for_pruned_model(self):
        g = _dense_graph(32, 32, name="pruned_model")
        g.quant_config = QuantConfig(bits=8)
        calib = {"x": np.random.default_rng(2).standard_normal((4, 32))}
        quantize_graph(g, calib)
        prune_weights(g, target_sparsity=0.5)

        seq = estimate(g, mode="sequential")
        comb = estimate(g, mode="combinational")

        # Old code: sequential left both at 0 even for a pruned model.
        assert seq.sparsity > 0.0
        assert seq.multipliers_eliminated > 0
        # Consistent with the combinational path on the same graph.
        assert abs(seq.sparsity - comb.sparsity) < 1e-9
        assert seq.multipliers_eliminated == comb.multipliers_eliminated
