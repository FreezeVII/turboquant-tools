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
@click.option("--bits", "-b", default=3, type=int)
@click.option("--output", "-o", default=None)
@click.option("--no-qjl", is_flag=True, default=False)
def compress_cmd(input, bits, output, no_qjl):
    vectors = np.load(input)
    if vectors.ndim != 2:
        click.echo(f"Error: expected 2D array, got {vectors.ndim}D", err=True)
        sys.exit(1)
    n, d = vectors.shape
    click.echo(f"Vectors: {n} x {d} ({vectors.nbytes / 1e6:.2f} MB)", err=True)
    compressed = compress(vectors, bits=bits, use_qjl=not no_qjl)
    if output is None:
        output = f"{Path(input).stem}_tq{bits}.tq"
    with open(output, "wb") as f:
        f.write(compressed.data)
    click.echo(f"Compressed: {compressed.nbytes / 1e6:.2f} MB ({compressed.memory.ratio:.1f}x)")
    click.echo(f"Saved to: {output}")

@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default=None)
def decompress_cmd(input, output):
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
    if output is None:
        output = f"{Path(input).stem}_restored.npy"
    np.save(output, restored)
    click.echo(f"Restored: {restored.shape} ({restored.nbytes / 1e6:.2f} MB)")
    click.echo(f"Saved to: {output}")

@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--bits", "-b", default=3, type=int)
def estimate_cmd(input, bits):
    arr = np.load(input, mmap_mode='r')
    if arr.ndim != 2:
        click.echo(f"Error: expected 2D array", err=True)
        sys.exit(1)
    n, d = arr.shape
    del arr
    click.echo(str(estimate_savings(n, d, bits=bits)))

@main.command()
def mcp_server():
    try:
        from turboquant_tools.mcp_server import run_server
        run_server()
    except ImportError:
        click.echo("MCP server requires: pip install turboquant-tools[mcp]", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
