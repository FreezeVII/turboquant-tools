"""TurboQuant Tools — CLI + MCP Server + Python Library for embedding compression."""

from turboquant_tools.core import compress, decompress, estimate_savings, CompressedVectors

__version__ = "0.1.2"
__all__ = ["compress", "decompress", "estimate_savings", "CompressedVectors"]