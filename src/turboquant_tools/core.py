"""
Core compression engine for TurboQuant tools.

Wraps the OnlyTerp/turboquant library into a simple compress/decompress API
for numpy embedding vectors. No PyTorch knowledge needed by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


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
    dtype: str = "float32"
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


def _make_codebook(d: int, b: int, device=torch.device("cpu"), seed: int = 0):
    import turboquant as tq
    K = 2**b
    n_bins = max(100000, K * 100)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    samples = torch.randn(n_bins, generator=g)
    sorted_v = samples.sort().values
    boundaries = torch.tensor([sorted_v[(k + 1) * n_bins // K].item() for k in range(K - 1)])
    centroids = []
    prev = -float("inf")
    for k in range(K):
        nxt = boundaries[k].item() if k < K - 1 else float("inf")
        mask = (samples >= prev) & (samples < nxt)
        centroids.append(samples[mask].mean().item() if mask.sum() > 0 else 0.0)
        prev = nxt
    return tq.cache.Codebook(
        centroids=torch.tensor(centroids, device=device),
        boundaries=boundaries.to(device),
        d=d, b=b, K=K,
    )


def _get_rotation(d: int, seed: int = 42, device=torch.device("cpu")):
    import turboquant as tq
    return tq.cache.RandomHadamardRotation(d=d, seed=seed, device=device)


def compress(vectors: np.ndarray, bits: int = 3, use_qjl: bool = True, seed: int = 42) -> CompressedVectors:
    import turboquant as tq
    n, d = vectors.shape
    device = torch.device("cpu")
    d_padded = 1 << (d - 1).bit_length()
    if d_padded != d:
        vec_t = torch.zeros(n, d_padded, dtype=torch.float32)
        vec_t[:, :d] = torch.from_numpy(vectors.astype(np.float32).copy())
    else:
        vec_t = torch.from_numpy(vectors.astype(np.float32).copy())
    codebook = _make_codebook(d_padded, bits, device, seed=0)
    rotation = _get_rotation(d_padded, seed, device)
    if use_qjl:
        S = tq.generate_qjl_matrix(d_padded, seed=seed, device=device)
        compressed = tq.turboquant_encode_internal(vec_t, codebook, rotation, S)
    else:
        compressed = tq.polarquant_encode(vec_t, codebook, rotation)
    import struct
    pq = compressed.pq if isinstance(compressed, tq.cache.TurboQuantCompressed) else compressed
    pq_indices_np = pq.indices.cpu().numpy().astype(np.uint8)
    pq_norm_np = pq.norm.cpu().numpy().astype(np.float16)
    if isinstance(compressed, tq.cache.TurboQuantCompressed):
        qjl = compressed.qjl
        fmt_type = 2
        header = struct.pack(
            "<4s B B I I I I I I",
            b"TQT2", fmt_type, bits, seed, n, d, d_padded,
            len(pq_norm_np.tobytes()), len(qjl.r_norm.cpu().numpy().astype(np.float32).tobytes()),
        )
        data = header + pq_norm_np.tobytes() + pq_indices_np.tobytes() + qjl.r_norm.cpu().numpy().astype(np.float32).tobytes() + qjl.signs.cpu().numpy().astype(np.uint8).tobytes()
    else:
        fmt_type = 1
        header = struct.pack(
            "<4s B B I I I I I I",
            b"TQT2", fmt_type, bits, seed, n, d, d_padded,
            len(pq_norm_np.tobytes()), 0,
        )
        data = header + pq_norm_np.tobytes() + pq_indices_np.tobytes()
    return CompressedVectors(data=data, shape=(n, d), bits=bits, _original_bytes=vectors.nbytes)


def decompress(compressed: CompressedVectors) -> np.ndarray:
    import struct, turboquant as tq
    data = compressed.data
    magic, fmt_type, bits, seed, n, d, d_padded, pq_norm_len, qjl_norm_len = struct.unpack_from(
        "<4s B B I I I I I I", data, 0
    )
    assert magic == b"TQT2"
    offset = struct.calcsize("<4s B B I I I I I I")
    pq_norm = np.frombuffer(data, dtype=np.float16, count=n, offset=offset).copy()
    offset += pq_norm_len
    pq_indices = np.frombuffer(data, dtype=np.uint8, count=n * d_padded, offset=offset).reshape(n, d_padded).copy()
    offset += n * d_padded
    pq_norm_t = torch.from_numpy(pq_norm)
    pq_indices_t = torch.from_numpy(pq_indices)
    codebook = _make_codebook(d_padded, bits, seed=0)
    rotation = _get_rotation(d_padded, seed)
    if fmt_type == 2:
        qjl_norm = np.frombuffer(data, dtype=np.float32, count=n, offset=offset).copy()
        offset += qjl_norm_len
        qjl_signs = np.frombuffer(data, dtype=np.uint8, count=n * d_padded, offset=offset).reshape(n, d_padded).copy()
        qjl_signs_t = torch.from_numpy(qjl_signs)
        qjl_norm_t = torch.from_numpy(qjl_norm)
        S = tq.generate_qjl_matrix(d_padded, seed=seed)
        pq_comp = tq.cache.PolarQuantCompressed(norm=pq_norm_t, indices=pq_indices_t, codebook=codebook, rotation=rotation)
        qjl_comp = tq.cache.QJLCompressed(signs=qjl_signs_t, r_norm=qjl_norm_t, S=S)
        restored = tq.turboquant_decode_single(tq.cache.TurboQuantCompressed(pq=pq_comp, qjl=qjl_comp))
    else:
        pq_comp = tq.cache.PolarQuantCompressed(norm=pq_norm_t, indices=pq_indices_t, codebook=codebook, rotation=rotation)
        restored = tq.polarquant_decode(pq_comp)
    result = restored.cpu().numpy().astype(np.float32)
    if d_padded != d:
        result = result[:, :d]
    return result


def estimate_savings(n_vectors: int, dim: int, bits: int = 3) -> MemoryBytes:
    import turboquant as tq
    original = n_vectors * dim * 4
    compressed = n_vectors * tq.memory_bytes_per_vector(dim, bits)[0]
    return MemoryBytes(original=original, compressed=compressed, ratio=original / compressed if compressed > 0 else 1.0)
