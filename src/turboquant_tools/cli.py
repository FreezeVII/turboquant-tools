"""
CLI for turboquant-tools.
"""
from __future__ import annotations
import sys
from pathlib import Path
import click
import numpy as np
from turboquant_tools import compress, decompress, estimate_savings


@click.group()
def main():
    """TurboQuant Tools - compress AI embeddings with 5x memory reduction."""
    pass


@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False), required=False)
@click.option("--bits", "-b", default=3, type=int, help="Target bit width (default: 3)")
@click.option("--output", "-o", default=None, help="Output .tq file path (alternative to positional OUTPUT)")
@click.option("--no-qjl", is_flag=True, default=False, help="Skip QJL correction (faster but lower quality)")
def compress_cmd(input, output, bits, no_qjl):
    """Compress .npy embedding vectors to .tq format.

    INPUT is a .npy file with float32 embeddings (n_vectors x dimensions).
    OUTPUT is the destination .tq file. If omitted, auto-names based on input.
    """
    vectors = np.load(input)
    if vectors.ndim != 2:
        click.echo(f"Error: expected 2D array, got {vectors.ndim}D", err=True)
        sys.exit(1)
    n, d = vectors.shape
    click.echo(f"Vectors: {n} x {d} ({vectors.nbytes / 1e6:.2f} MB)", err=True)
    compressed = compress(vectors, bits=bits, use_qjl=not no_qjl)
    out_path = output or click.get_current_context().params.get("output")
    if out_path is None:
        out_path = f"{Path(input).stem}_tq{bits}.tq"
    with open(out_path, "wb") as f:
        f.write(compressed.data)
    click.echo(f"Compressed: {compressed.nbytes / 1e6:.2f} MB ({compressed.memory.ratio:.1f}x)")
    click.echo(f"Saved to: {out_path}")


@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False), required=False)
@click.option("--output", "-o", default=None, help="Output .npy file path (alternative to positional OUTPUT)")
def decompress_cmd(input, output):
    """Restore compressed .tq file to .npy.

    INPUT is a .tq compressed file.
    OUTPUT is the destination .npy file. If omitted, auto-names based on input.
    """
    from turboquant_tools.core import CompressedVectors
    with open(input, "rb") as f:
        data = f.read()
    import struct
    magic = struct.unpack_from("<4s", data, 0)[0]
    if magic != b"TQT2":
        click.echo(f"Error: not a valid .tq file", err=True)
        sys.exit(1)
    compressed = CompressedVectors(data=data, shape=(0, 0), bits=0)
    restored = decompress(compressed)
    out_path = output or click.get_current_context().params.get("output")
    if out_path is None:
        out_path = f"{Path(input).stem}_restored.npy"
    np.save(out_path, restored)
    click.echo(f"Restored: {restored.shape} ({restored.nbytes / 1e6:.2f} MB)")
    click.echo(f"Saved to: {out_path}")


@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--bits", "-b", default=3, type=int, help="Target bit width (default: 3)")
def estimate_cmd(input, bits):
    """Estimate compression savings without running the algorithm."""
    arr = np.load(input, mmap_mode='r')
    if arr.ndim != 2:
        click.echo(f"Error: expected 2D array", err=True)
        sys.exit(1)
    n, d = arr.shape
    del arr
    click.echo(str(estimate_savings(n, d, bits=bits)))


@main.command()
def mcp_server():
    """Start the MCP protocol server (stdio transport for Hermes AI agents)."""
    try:
        from turboquant_tools.mcp_server import run_server
        run_server()
    except ImportError:
        click.echo("MCP server requires: pip install turboquant-tools[mcp]", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
