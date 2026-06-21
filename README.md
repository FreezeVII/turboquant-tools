# 🧊 TurboQuant Tools

> **Compress AI embeddings by 5–7× with near-lossless quality.**

CLI + Python Library + [MCP](https://modelcontextprotocol.io) Server for extreme vector compression using [Google's TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (PolarQuant + QJL) — wrapped in a clean numpy-first API.

[![PyPI](https://img.shields.io/pypi/v/turboquant-tools)](https://pypi.org/project/turboquant-tools/)
[![Python](https://img.shields.io/pypi/pyversions/turboquant-tools)](https://www.python.org)
[![License](https://img.shields.io/github/license/FreezeVII/turboquant-tools)](LICENSE)
[![Tests](https://github.com/FreezeVII/turboquant-tools/actions/workflows/python-tests.yml/badge.svg)](https://github.com/FreezeVII/turboquant-tools/actions)

---

## Quick Start

```bash
pip install turboquant-tools
turboquant compress embeddings.npy --bits 3
```

```python
from turboquant_tools import compress, decompress
import numpy as np

vectors = np.random.randn(1000, 384).astype(np.float32)
compressed = compress(vectors, bits=3)
print(f"Original: {vectors.nbytes / 1e6:.1f} MB")
print(f"Compressed: {compressed.nbytes / 1e6:.1f} MB")
```

## CLI

```bash
# Compress embeddings
turboquant compress embeddings.npy --bits 3 --output compressed.tq

# Estimate savings without compressing
turboquant estimate embeddings.npy

# Decompress
turboquant decompress compressed.tq --output restored.npy
```

## MCP Server

```bash
turboquant mcp-server
```

Exposes `compress_embeddings`, `decompress_embeddings`, `estimate_savings`, `embed_and_compress`.

## How It Works

1. **PolarQuant** — Random rotation + polar coordinate quantization (3-bit)
2. **QJL** — Quantized Johnson-Lindenstrauss error correction (1-bit)

Result: **~5x compression** with near-zero accuracy loss, no training needed.

## Use Cases

- **RAG pipelines** — Store 5x more documents in the same RAM
- **Local LLMs** — Fit larger vector stores on your GPU/CPU
- **Edge devices** — Deploy vector search with minimal memory
- **AI Agents** — Compress embeddings between agent calls

## License

MIT
