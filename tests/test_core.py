"""
Tests for turboquant_tools core compression engine.
"""
import numpy as np
import pytest
from turboquant_tools import compress, decompress, estimate_savings


class TestCompressDecompress:
    @pytest.mark.parametrize("d", [384, 768])
    @pytest.mark.parametrize("bits", [3, 4])
    def test_roundtrip_polarquant(self, d, bits):
        n = 20
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=bits)
        r = decompress(c)
        assert r.shape == (n, d)
        assert np.isfinite(np.abs(vecs - r).mean())

    def test_large_batch(self):
        n, d = 500, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3)
        r = decompress(c)
        assert r.shape == (n, d)
        assert np.abs(vecs - r).mean() < 10.0

    def test_compression_ratio(self):
        n, d = 1000, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3)
        assert c.nbytes < vecs.nbytes
        assert c.memory.ratio > 1.0

    def test_non_power_of_2_dim(self):
        n, d = 15, 100
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3)
        r = decompress(c)
        assert r.shape == (n, d)

    def test_single_vector(self):
        vecs = np.random.randn(1, 384).astype(np.float32)
        c = compress(vecs, bits=3)
        r = decompress(c)
        assert r.shape == (1, 384)

    def test_deterministic(self):
        vecs = np.random.randn(10, 384).astype(np.float32)
        c1 = compress(vecs, bits=3, seed=42)
        c2 = compress(vecs, bits=3, seed=42)
        assert c1.data == c2.data

    def test_binary_format_roundtrip(self):
        vecs = np.random.randn(10, 384).astype(np.float32)
        c = compress(vecs, bits=3)
        import struct
        magic = struct.unpack_from("<4s", c.data, 0)[0]
        assert magic == b"TQT2"
        from turboquant_tools.core import CompressedVectors
        c2 = CompressedVectors(data=c.data, shape=c.shape, bits=c.bits)
        r = decompress(c2)
        assert r.shape == (10, 384)


class TestEstimateSavings:
    def test_estimate_shape(self):
        est = estimate_savings(100000, 384, bits=3)
        assert est.ratio > 1.0
        assert est.saved_percent > 50

    def test_estimate_vs_actual(self):
        n, d = 100, 384
        vecs = np.random.randn(n, d).astype(np.float32)
        c = compress(vecs, bits=3)
        est = estimate_savings(n, d, bits=3)
        assert c.memory.ratio > 1.0
        assert est.ratio > c.memory.ratio  # estimate is upper bound

    def test_estimate_str(self):
        est = estimate_savings(100, 384, bits=3)
        s = str(est)
        assert "Original" in s
        assert "Compressed" in s
