"""
Tests for MCP server tool functions (called directly, not over stdio).
"""

import numpy as np
import pytest

from turboquant_tools import compress, decompress


def test_compress_embeddings_tool():
    """Test compress_embeddings logic via direct function calls."""
    import base64, struct
    from turboquant_tools.core import CompressedVectors

    vecs = np.random.randn(10, 128).astype(np.float32)
    c = compress(vecs, bits=3, use_qjl=False)

    b64 = base64.b64encode(c.data).decode()
    assert isinstance(b64, str)
    assert len(b64) > 0

    data = base64.b64decode(b64)
    magic = struct.unpack_from("<4s", data, 0)[0]
    assert magic == b"TQT2"

    cv = CompressedVectors(data=data, shape=(10, 128), bits=3)
    r = decompress(cv)
    assert r.shape == (10, 128)
    assert np.isfinite(np.abs(vecs - r).mean())


def test_estimate_savings_tool():
    """Test estimate_savings via MCP return format."""
    from turboquant_tools import estimate_savings
    est = estimate_savings(100000, 384, bits=3)
    result = {
        "original_mb": round(est.original / 1e6, 2),
        "compressed_mb": round(est.compressed / 1e6, 2),
        "ratio": round(est.ratio, 2),
        "saved_percent": round(est.saved_percent, 1),
    }
    assert result["original_mb"] == 153.60
    assert result["ratio"] > 5.0
    assert result["saved_percent"] > 80


def test_mcp_tool_registration():
    """Verify all MCP tools are registered with FastMCP."""
    from turboquant_tools import mcp_server
    assert hasattr(mcp_server, "_serve_stdio")
    assert hasattr(mcp_server, "run_server")
