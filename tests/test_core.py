"""
Tests for turboquant_tools core compression engine.
"""

import numpy as np
import pytest

from turboquant_tools import compress, decompress, estimate_savings


class TestCompressDecompress:
    """Round-trip compress/decompress for various dimensions and bit widths."""

    @pytest.mark.parametrize("d", [64, 128, 384])
    @pytest.mark.parametrize("bits", [3, 4])
    def test_roundtrip_polarquant(self, d, bits):
        """PolarQuant-only roundtrip preserves shape and produces reasonable error."""
        n = 20
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=bits, use_qjl=False)
        r = decompress(c)
        assert r.shape == (n, d), f"Shape mismatch: {r.shape} != ({n}, {d})"
        assert r.dtype == np.float32
        err = np.abs(vecs - r).mean()
        assert np.isfinite(err), f"Non-finite error: {err}"
        assert err < 20.0, f"Error too high: {err}"

    @pytest.mark.parametrize("d", [64, 128])
    def test_roundtrip_turboquant(self, d):
        """TurboQuant (PQ+QJL) roundtrip."""
        n = 10
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=True)
        r = decompress(c)
        assert r.shape == (n, d)
        err = np.abs(vecs - r).mean()
        assert np.isfinite(err)

    def test_large_batch(self):
        """500 vectors at 384 dims."""
        n, d = 500, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)
        r = decompress(c)
        assert r.shape == (n, d)
        err = np.abs(vecs - r).mean()
        assert err < 10.0

    def test_compression_ratio(self):
        """Compressed data should be smaller than original for PolarQuant 3-bit."""
        n, d = 1000, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)
        assert c.nbytes < vecs.nbytes, f"Compressed larger: {c.nbytes} > {vecs.nbytes}"
        ratio = c.memory.ratio
        assert ratio > 1.0, f"Ratio should be >1, got {ratio}"

    def test_empty_vectors(self):
        """Zero vectors should not crash."""
        vecs = np.zeros((0, 128), dtype=np.float32)
        with pytest.raises((RuntimeError, ValueError)):
            compress(vecs, bits=3, use_qjl=False)

    def test_padding_restoration(self):
        """Non-power-of-2 dims should restore original shape exactly."""
        n, d = 15, 100
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)
        r = decompress(c)
        assert r.shape == (n, d), f"Padding broke shape: {r.shape} != ({n}, {d})"

    def test_deterministic(self):
        """Same input → same compressed output (seed-based)."""
        vecs = np.random.randn(10, 128).astype(np.float32)
        c1 = compress(vecs, bits=3, seed=42)
        c2 = compress(vecs, bits=3, seed=42)
        assert c1.data == c2.data, "Deterministic compression failed"


class TestEstimateSavings:
    """Estimate savings calculations."""

    def test_estimate_shape(self):
        """estimate_savings returns a MemoryBytes with reasonable values."""
        est = estimate_savings(100000, 384, bits=3)
        assert est.original > 0
        assert est.compressed > 0
        assert est.ratio > 1.0
        assert est.saved_percent > 50

    def test_estimate_vs_actual(self):
        """Estimate should be in the same ballpark as actual for PolarQuant."""
        n, d = 100, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)

        est = estimate_savings(n, d, bits=3)
        actual_ratio = vecs.nbytes / c.nbytes
        assert abs(actual_ratio - est.ratio) / est.ratio < 1.0, (
            f"Estimate {est.ratio:.1f}x vs actual {actual_ratio:.1f}x too different"
        )
