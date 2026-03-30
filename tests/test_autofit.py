"""Tests for the auto-fit engine.

Exercises sensitivity analysis, greedy search, mixed-precision downgrade,
and sparsity paths using models of different sizes.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.autofit import (
    analyze_sensitivity,
    autofit,
    autofit_fpga,
    FitResult,
    SensitivityReport,
)


# ---------------------------------------------------------------------------
#  Fixtures: small and large models
# ---------------------------------------------------------------------------

@pytest.fixture
def small_graph():
    """A tiny 2->4->1 network that fits on anything."""
    np.random.seed(42)
    gb = GraphBuilder("tiny")
    inp = gb.input("x", shape=(2,))
    h = gb.dense(inp, np.random.randn(4, 2) * 0.5, np.zeros(4),
                 activation="relu", name="hidden")
    out = gb.dense(h, np.random.randn(1, 4) * 0.5, np.zeros(1), name="output")
    gb.output(out)
    return gb.build()


@pytest.fixture
def medium_graph():
    """A 16->64->32->8 network large enough to stress a small LUT budget."""
    np.random.seed(42)
    gb = GraphBuilder("medium")
    inp = gb.input("x", shape=(16,))
    h1 = gb.dense(inp, np.random.randn(64, 16) * 0.1, np.zeros(64),
                  activation="relu", name="fc1")
    h2 = gb.dense(h1, np.random.randn(32, 64) * 0.1, np.zeros(32),
                  activation="relu", name="fc2")
    out = gb.dense(h2, np.random.randn(8, 32) * 0.1, np.zeros(8), name="fc3")
    gb.output(out)
    return gb.build()


@pytest.fixture
def calib_small():
    np.random.seed(42)
    return {"x": np.random.randn(4, 2).astype(np.float32)}


@pytest.fixture
def calib_medium():
    np.random.seed(42)
    return {"x": np.random.randn(4, 16).astype(np.float32)}


# ---------------------------------------------------------------------------
#  Sensitivity analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_returns_report(self, xor_graph, xor_trained_weights):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        report = analyze_sensitivity(xor_graph, {"x": X})
        assert isinstance(report, SensitivityReport)
        assert len(report.layers) > 0

    def test_layers_have_positive_params(self, xor_graph, xor_trained_weights):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        report = analyze_sensitivity(xor_graph, {"x": X})
        for ls in report.layers:
            assert ls.n_params > 0

    def test_report_str(self, xor_graph, xor_trained_weights):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        report = analyze_sensitivity(xor_graph, {"x": X})
        s = str(report)
        assert "Sensitivity" in s
        assert "hidden" in s or "output" in s

    def test_ranked_is_sorted(self, xor_graph, xor_trained_weights):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        report = analyze_sensitivity(xor_graph, {"x": X})
        ranked = report.ranked()
        sensitivities = [ls.sensitivity for ls in ranked]
        assert sensitivities == sorted(sensitivities)

    def test_int4_error_gte_int8_error(self, medium_graph, calib_medium):
        """int4 should generally introduce more error than int8."""
        report = analyze_sensitivity(medium_graph, calib_medium)
        for ls in report.layers:
            # int4 error should be >= baseline (int8) in most cases
            # Allow small floating point tolerance
            assert ls.error_int4 >= ls.error_int8 - 1e-6


# ---------------------------------------------------------------------------
#  Auto-fit: basic behavior
# ---------------------------------------------------------------------------

class TestAutofitBasic:
    def test_tiny_fits_on_large_device(self, small_graph, calib_small):
        result = autofit(small_graph, calib_small, device_luts=100000)
        assert isinstance(result, FitResult)
        assert result.fits
        assert result.mode in ("combinational", "sequential")

    def test_tiny_does_not_fit_on_zero_luts(self, small_graph, calib_small):
        result = autofit(small_graph, calib_small, device_luts=1)
        assert not result.fits

    def test_result_str(self, small_graph, calib_small):
        result = autofit(small_graph, calib_small, device_luts=100000)
        s = str(result)
        assert "Auto-Fit" in s
        assert "FITS" in s

    def test_not_fit_result_str(self, small_graph, calib_small):
        result = autofit(small_graph, calib_small, device_luts=1)
        s = str(result)
        assert "DOES NOT FIT" in s


# ---------------------------------------------------------------------------
#  Auto-fit: exercises downgrade and sparsity paths
# ---------------------------------------------------------------------------

class TestAutofitDowngrade:
    def test_medium_needs_downgrade_for_tiny_device(self, medium_graph, calib_medium):
        """With a tight LUT budget, auto-fit should downgrade to int4 or mixed."""
        # medium_graph ~3700 params.  int8 sequential ~6280 LUTs, int4 ~3320.
        # Budget of 4000 forces a downgrade from int8 but allows int4.
        result = autofit(medium_graph, calib_medium, device_luts=4000)
        assert result.fits
        assert result.bits == 4 or result.bits_map is not None

    def test_medium_sequential_fits_more_easily(self, medium_graph, calib_medium):
        """Sequential mode should fit with a smaller LUT budget than combinational."""
        # Budget of 4000: int8 sequential ~6280 doesn't fit, but int4
        # sequential ~3320 does — and sequential is cheaper than combinational.
        result = autofit(medium_graph, calib_medium, device_luts=4000)
        assert result.fits
        assert result.mode == "sequential"

    def test_prefer_combinational_flag(self, small_graph, calib_small):
        result = autofit(small_graph, calib_small, device_luts=100000,
                         prefer_combinational=True)
        assert result.fits
        assert result.mode == "combinational"

    def test_sparsity_path(self, medium_graph, calib_medium):
        """With an impossibly tight budget, verify graceful failure."""
        result = autofit(medium_graph, calib_medium, device_luts=50,
                         max_sparsity=0.9,
                         sparsity_steps=[0.5, 0.75, 0.9])
        assert isinstance(result, FitResult)
        # Nothing fits on 50 LUTs — verify the search completed gracefully
        assert not result.fits
        assert "too large" in result.config_summary.lower() or result.sparsity > 0

    def test_config_summary_has_mode(self, small_graph, calib_small):
        result = autofit(small_graph, calib_small, device_luts=100000)
        assert result.mode in result.config_summary


# ---------------------------------------------------------------------------
#  Auto-fit FPGA wrapper
# ---------------------------------------------------------------------------

class TestAutofitFPGA:
    def test_default_device(self, small_graph, calib_small):
        result = autofit_fpga(small_graph, calib_small)
        assert isinstance(result, FitResult)

    def test_specific_device(self, small_graph, calib_small):
        from w2s.fpga import ECP5_25K
        result = autofit_fpga(small_graph, calib_small, ECP5_25K)
        assert result.fits
        assert result.device_luts == ECP5_25K.lut4s
