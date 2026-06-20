"""
MCP server for turboquant-tools.

Provides AI agents with tools to compress, decompress, and estimate
embedding vectors using TurboQuant.

Usage:
    pip install turboquant-tools[mcp]
    turboquant mcp-server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from turboquant_tools import compress, decompress, estimate_savings
from turboquant_tools.core import CompressedVectors


def _serve_stdio():
    """Run MCP server over stdio using FastMCP."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        print("Need: pip install turboquant-tools[mcp]", file=sys.stderr)
        sys.exit(1)

    import os
    os.environ["FASTMCP_LOG_LEVEL"] = "WARNING"
    mcp = FastMCP("turboquant-tools")

    @mcp.tool()
    def compress_embeddings(
        vectors: list[list[float]],
        bits: int = 3,
        use_qjl: bool = False,
    ) -> dict:
        """
        Compress a list of embedding vectors using TurboQuant.

        Args:
            vectors: List of vectors, each a list of floats (all same length).
            bits: Target bit width (3 or 4). Default 3.
            use_qjl: Whether to apply QJL correction. Default False.

        Returns:
            Dict with compressed data, shape, ratio.
        """
        import base64
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2:
            return {"error": f"Expected 2D array, got {arr.ndim}D"}
        c = compress(arr, bits=bits, use_qjl=use_qjl)
        return {
            "compressed": base64.b64encode(c.data).decode(),
            "shape": list(c.shape),
            "bits": bits,
            "ratio": round(c.memory.ratio, 2),
            "original_bytes": c.memory.original,
            "compressed_bytes": c.nbytes,
            "saved_percent": round(c.memory.saved_percent, 1),
        }

    @mcp.tool()
    def decompress_embeddings(compressed_b64: str, shape: list[int]) -> list[list[float]]:
        """
        Restore compressed vectors.

        Args:
            compressed_b64: Base64 .tq data from compress_embeddings().
            shape: Original shape [n_vectors, dim].

        Returns:
            List of restored vectors.
        """
        import base64, struct
        data = base64.b64decode(compressed_b64)
        magic = struct.unpack_from("<4s", data, 0)[0]
        if magic != b"TQT2":
            return {"error": "Invalid .tq data"}
        cv = CompressedVectors(data=data, shape=(shape[0], shape[1]), bits=0)
        restored = decompress(cv)
        return restored.tolist()

    @mcp.tool()
    def estimate_savings_mcp(
        n_vectors: int,
        dim: int,
        bits: int = 3,
    ) -> dict:
        """
        Estimate compression savings.

        Args:
            n_vectors: Number of embedding vectors.
            dim: Dimension of each vector.
            bits: Target bit width (3 or 4).

        Returns:
            Dict with sizes and ratio.
        """
        est = estimate_savings(n_vectors, dim, bits)
        return {
            "original_mb": round(est.original / 1e6, 2),
            "compressed_mb": round(est.compressed / 1e6, 2),
            "ratio": round(est.ratio, 2),
            "saved_percent": round(est.saved_percent, 1),
        }

    @mcp.tool()
    def embed_and_compress(
        texts: list[str],
        model: str = "text-embedding-3-small",
        api_key: str = "",
        bits: int = 3,
    ) -> dict:
        """
        Embed texts via API, then compress the vectors.

        Args:
            texts: List of text strings.
            model: Embedding model name.
            api_key: API key (or use OPENAI_API_KEY env var).
            bits: Target bit width.

        Returns:
            Dict with compressed data.
        """
        import os, urllib.request
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return {"error": "No API key. Set OPENAI_API_KEY or pass api_key."}
        payload = json.dumps({"input": texts, "model": model}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        embeddings = [d["embedding"] for d in result["data"]]
        arr = np.array(embeddings, dtype=np.float32)
        c = compress(arr, bits=bits, use_qjl=False)
        import base64
        return {
            "n_texts": len(texts),
            "dim": arr.shape[1],
            "compressed": base64.b64encode(c.data).decode(),
            "ratio": round(c.memory.ratio, 2),
            "saved_percent": round(c.memory.saved_percent, 1),
        }

    mcp.run()


def run_server():
    """Entry point for the MCP server."""
    _serve_stdio()


if __name__ == "__main__":
    run_server()
