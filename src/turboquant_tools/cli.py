"""
CLI for turboquant-tools.

Usage:
    turboquant compress <input.npy> [--bits 3] [--output <file.tq>] [--no-qjl]
    turboquant decompress <file.tq> [--output <restored.npy>]
    turboquant estimate <input.npy> [--bits 3]
    turboquant mcp-server
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import numpy as np

from turboquant_tools import compress, decompress, estimate_savings
from turboquant_tools.core import CompressedVectors


@click.group()
def main():
    """TurboQuant Tools — compress AI embeddings with 5x memory reduction."""
    pass


@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--bits", "-b", default=3, type=int, help="Target bit width (default: 3)")
@click.option("--output", "-o", default=None, help="Output .tq file path")
@click.option("--no-qjl", is_flag=True, default=False, help="Skip QJL correction (faster)")
def compress_cmd(input, bits, output, no_qjl):
    """Compress .npy embedding vectors using TurboQuant."""
    click.echo(f"Loading {input}...", err=True)
    vectors = np.load(input)
    if vectors.ndim != 2:
        click.echo(f"Error: expected 2D array, got {vectors.ndim}D", err=True)
        sys.exit(1)

    n, d = vectors.shape
    click.echo(f"Vectors: {n} × {d} ({vectors.nbytes / 1e6:.2f} MB)", err=True)

    click.echo(f"Compressing ({bits}-bit{' QJL' if not no_qjl else ''})...", err=True)
    compressed = compress(vectors, bits=bits, use_qjl=not no_qjl)

    if output is None:
        stem = Path(input).stem
        output = f"{stem}_tq{bits}.tq"

    with open(output, "wb") as f:
        f.write(compressed.data)

    ratio = compressed.memory.ratio
    click.echo(
        f"Compressed: {compressed.nbytes / 1e6:.2f} MB "
        f"({ratio:.1f}x compression, saved {compressed.memory.saved_percent:.0f}%)"
    )
    click.echo(f"Saved to: {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default=None, help="Output .npy file path")
def decompress_cmd(input, output):
    """Restore compressed .tq file to .npy."""

    click.echo(f"Loading {input}...", err=True)
    with open(input, "rb") as f:
        data = f.read()

    import struct
    magic, fmt_type, bits, seed, n, d, d_padded, pq_norm_len, qjl_norm_len = struct.unpack_from(
        "<4s B B I I I I I I", data, 0
    )
    if magic != b"TQT2":
        click.echo(f"Error: not a valid .tq file (magic: {magic!r})", err=True)
        sys.exit(1)
    if fmt_type not in (1, 2):
        click.echo(f"Error: unknown format type {fmt_type}", err=True)
        sys.exit(1)

    kind = "TurboQuant (PQ+QJL)" if fmt_type == 2 else "PolarQuant"
    click.echo(f"Format: {kind} | {n}×{d} vectors | {bits}-bit | seed={seed}", err=True)

    compressed = CompressedVectors(data=data, shape=(n, d), bits=bits)

    click.echo("Decompressing...", err=True)
    restored = decompress(compressed)

    if output is None:
        stem = Path(input).stem
        output = f"{stem}_restored.npy"

    np.save(output, restored)
    click.echo(f"Restored: {restored.shape} ({restored.nbytes / 1e6:.2f} MB)")
    click.echo(f"Saved to: {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--bits", "-b", default=3, type=int, help="Target bit width")
def estimate_cmd(input, bits):
    """Estimate compression savings without running the algorithm."""
    arr = np.load(input, mmap_mode='r')
    if arr.ndim != 2:
        click.echo(f"Error: expected 2D array, got {arr.ndim}D", err=True)
        sys.exit(1)
    n, d = arr.shape
    del arr

    est = estimate_savings(n, d, bits=bits)
    click.echo(str(est))


@main.command()
def mcp_server():
    """Start MCP server for AI agent integration."""
    try:
        from turboquant_tools.mcp_server import run_server
        run_server()
    except ImportError:
        click.echo(
            "MCP server requires: pip install turboquant-tools[mcp]",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
