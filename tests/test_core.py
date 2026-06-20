"""
Tests for turboquant_tools core compression engine.
"""
import numpy as np
import pytest
from turboquant_tools import compress, decompress, estimate_savings

class TestCompressDecompress:
    @pytest.mark.parametrize("d", [64, 128, 384])
    @pytest.mark.parametrize("bits", [3, 4])
    def test_roundtrip_polarquant(self, d, bits):
        n = 20
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=bits, use_qjl=False)
        r = decompress(c)
        assert r.shape == (n, d)
        assert np.isfinite(np.abs(vecs - r).mean())

    @pytest.mark.parametrize("d", [64, 128])
    def test_roundtrip_turboquant(self, d):
        n = 10
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=True)
        r = decompress(c)
        assert r.shape == (n, d)

    def test_large_batch(self):
        n, d = 500, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)
        r = decompress(c)
        assert r.shape == (n, d)
        assert np.abs(vecs - r).mean() < 10.0

    def test_compression_ratio(self):
        n, d = 1000, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)
        assert c.nbytes < vecs.nbytes
        assert c.memory.ratio > 1.0

    def test_padding_restoration(self):
        n, d = 15, 100
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)
        r = decompress(c)
        assert r.shape == (n, d)

    def test_deterministic(self):
        vecs = np.random.randn(10, 128).astype(np.float32)
        c1 = compress(vecs, bits=3, seed=42)
        c2 = compress(vecs, bits=3, seed=42)
        assert c1.data == c2.data

class TestEstimateSavings:
    def test_estimate_shape(self):
        est = estimate_savings(100000, 384, bits=3)
        assert est.ratio > 1.0
        assert est.saved_percent > 50

    def test_estimate_vs_actual(self):
        n, d = 100, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3, use_qjl=False)
        est = estimate_savings(n, d, bits=3)
        actual_ratio = vecs.nbytes / c.nbytes
        assert abs(actual_ratio - est.ratio) / est.ratio < 1.0
