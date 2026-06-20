"""
Core compression engine for TurboQuant tools.

Pure numpy implementation of PolarQuant — no PyTorch, no GPU needed.
Inspired by Google's TurboQuant: Random Hadamard rotation + scalar quantization.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────

@dataclass
class MemoryBytes:
    original: int
    compressed: int
    ratio: float

    @property
    def saved_bytes(self) -> int:
        return self.original - self.compressed

    @property
    def saved_percent(self) -> float:
        if self.original == 0:
            return 0.0
        return (1 - self.compressed / self.original) * 100

    def __str__(self) -> str:
        return (
            f"Original: {self.original / 1e6:.2f} MB -> "
            f"Compressed: {self.compressed / 1e6:.2f} MB "
            f"({self.ratio:.2f}x, save {self.saved_percent:.0f}%)"
        )


@dataclass
class CompressedVectors:
    data: bytes
    shape: tuple[int, int]
    bits: int
    _original_bytes: int = 0

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def n_vectors(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[1]

    @property
    def memory(self) -> MemoryBytes:
        return MemoryBytes(
            original=self._original_bytes or self.n_vectors * self.dim * 4,
            compressed=self.nbytes,
            ratio=self._original_bytes / self.nbytes if self.nbytes > 0 else 0.0,
        )


# ── Walsh-Hadamard Transform (iterative, in-place) ────────────────────────

def _fwht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform on last axis. x.shape = (n, d), d must be power of 2."""
    n, d = x.shape
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                u = x[:, j].copy()
                v = x[:, j + h].copy()
                x[:, j] = u + v
                x[:, j + h] = u - v
        h *= 2
    return x


# ── Codebook ───────────────────────────────────────────────────────────────

def _make_codebook(bits: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate scalar codebook: boundaries + centroids from normal samples."""
    K = 2 ** bits
    n_bins = max(100000, K * 100)
    rng = np.random.RandomState(seed)
    samples = np.sort(rng.randn(n_bins))
    boundaries = np.array([samples[(k + 1) * n_bins // K] for k in range(K - 1)])
    centroids = np.zeros(K)
    prev = -np.inf
    for k in range(K):
        nxt = boundaries[k] if k < K - 1 else np.inf
        mask = (samples >= prev) & (samples < nxt)
        centroids[k] = samples[mask].mean() if mask.sum() > 0 else 0.0
        prev = nxt
    return boundaries.astype(np.float32), centroids.astype(np.float32)


# ── Public API ─────────────────────────────────────────────────────────────

def compress(vectors: np.ndarray, bits: int = 3, use_qjl: bool = False, seed: int = 42) -> CompressedVectors:
    """
    Compress float32 embedding vectors using PolarQuant.

    Args:
        vectors: (n, d) float32 array of embeddings.
        bits: Target bit width (3 or 4).
        use_qjl: Ignored in numpy-only mode (QJL requires PyTorch).
        seed: Random seed for Hadamard rotation.

    Returns:
        CompressedVectors with .tq binary data.
    """
    if use_qjl:
        import warnings
        warnings.warn("QJL not available in numpy-only mode. Falling back to PolarQuant.")

    n, d = vectors.shape
    arr = np.ascontiguousarray(vectors, dtype=np.float32)

    # Pad to next power of 2 for FWHT
    d_padded = 1 << (d - 1).bit_length()
    if d_padded != d:
        padded = np.zeros((n, d_padded), dtype=np.float32)
        padded[:, :d] = arr
        arr = padded

    # Random rotation: arr *= diag, then FWHT, then / sqrt(d)
    rng = np.random.RandomState(seed)
    diag = rng.choice([-1.0, 1.0], size=d_padded).astype(np.float32)
    arr *= diag[None, :]
    _fwht(arr)
    arr /= math.sqrt(d_padded)

    # Split into norm + direction
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm_safe = np.where(norm > 0, norm, 1.0)
    direction = arr / norm_safe

    # Quantize direction components
    boundaries, centroids = _make_codebook(bits, seed=0)
    indices_flat = np.searchsorted(boundaries, direction.ravel()).astype(np.uint8)
    indices = indices_flat.reshape(n, d_padded)

    # Serialize
    pq_norm = norm.astype(np.float16)
    magic = b"TQT2"
    fmt_type = 1  # PolarQuant only
    pq_norm_len = len(pq_norm.tobytes())

    header = struct.pack(
        "<4s B B I I I I I I",
        magic, fmt_type, bits, seed, n, d, d_padded,
        pq_norm_len, 0,
    )
    data = header + pq_norm.tobytes() + indices.tobytes()

    return CompressedVectors(
        data=data,
        shape=(n, d),
        bits=bits,
        _original_bytes=vectors.nbytes,
    )


def decompress(compressed: CompressedVectors) -> np.ndarray:
    """
    Decompress .tq data back to float32 embeddings.

    Args:
        compressed: CompressedVectors from compress().

    Returns:
        (n, d) float32 numpy array.
    """
    data = compressed.data
    magic, fmt_type, bits, seed, n, d, d_padded, pq_norm_len, _ = struct.unpack_from(
        "<4s B B I I I I I I", data, 0
    )
    assert magic == b"TQT2", f"Invalid magic: {magic}"

    offset = struct.calcsize("<4s B B I I I I I I")
    pq_norm = np.frombuffer(data, dtype=np.float16, count=n, offset=offset).copy()
    offset += pq_norm_len
    pq_indices = np.frombuffer(data, dtype=np.uint8, count=n * d_padded, offset=offset).reshape(n, d_padded).copy()

    # Dequantize
    _, centroids = _make_codebook(bits, seed=0)
    direction = centroids[pq_indices.astype(np.int32)]

    # Apply norm
    restored = direction * pq_norm[:, None]

    # Inverse rotation: diag * FWHT * sqrt(d)
    rng = np.random.RandomState(seed)
    diag = rng.choice([-1.0, 1.0], size=d_padded).astype(np.float32)
    _fwht(restored)
    restored *= math.sqrt(d_padded)
    restored *= diag[None, :]

    # Unpad
    if d_padded != d:
        restored = restored[:, :d]

    return np.ascontiguousarray(restored, dtype=np.float32)


def estimate_savings(n_vectors: int, dim: int, bits: int = 3) -> MemoryBytes:
    """
    Estimate compression savings without running the algorithm.

    Args:
        n_vectors: Number of embedding vectors.
        dim: Dimension of each vector.
        bits: Target bit width (3 or 4).

    Returns:
        MemoryBytes with original/compressed sizes and ratio.
    """
    d_padded = 1 << (dim - 1).bit_length()
    # Per vector: norm (2 bytes float16) + indices (d_padded bytes uint8) + header share
    header_size = 32
    per_vector = 2 + d_padded
    original = n_vectors * dim * 4
    compressed = n_vectors * per_vector + header_size
    ratio = original / compressed if compressed > 0 else 1.0
    return MemoryBytes(original=original, compressed=int(compressed), ratio=ratio)
