# 🧊 TurboQuant Tools

> **Compress AI embeddings by 5–7× with near-lossless quality.**

CLI + Python Library + [MCP](https://modelcontextprotocol.io) Server for extreme vector compression using [Google's TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (PolarQuant + QJL) — wrapped in a clean numpy-first API.

[![PyPI](https://img.shields.io/pypi/v/turboquant-tools)](https://pypi.org/project/turboquant-tools/)
[![Python](https://img.shields.io/pypi/pyversions/turboquant-tools)](https://www.python.org)
[![License](https://img.shields.io/github/license/FreezeVII/turboquant-tools)](LICENSE)
[![Tests](https://github.com/FreezeVII/turboquant-tools/actions/workflows/python-tests.yml/badge.svg)](https://github.com/FreezeVII/turboquant-tools/actions)

---

## 🚀 Quick Start

```bash
pip install turboquant-tools
```

Compress a `.npy` embedding file:

```bash
turboquant compress embeddings.npy compressed.tq
```

Restore:

```bash
turboquant decompress compressed.tq restored.npy
```

Estimate savings:

```bash
turboquant estimate embeddings.npy --bits 3
# Original: 153.00 MB -> Compressed: 20.13 MB (7.60×, save 87%)
```

---

## 📦 What's Inside

| Command / Tool | Description |
|---|---|
| `turboquant compress` | Compress `.npy` embeddings → `.tq` binary |
| `turboquant decompress` | Restore `.tq` → `.npy` |
| `turboquant estimate` | Predict compression ratio before running |
| `turboquant mcp-server` | MCP stdio server (AI agent integration) |
| Python `compress()` | Compress numpy arrays in code |
| Python `decompress()` | Restore in code |

---

## 🔧 CLI Reference

### compress

```bash
turboquant compress INPUT [OUTPUT] [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `INPUT` | — | `.npy` file with float32 embeddings `(n, d)` |
| `OUTPUT` | `{stem}_tq{b}.tq` | Output `.tq` file |
| `-b, --bits` | `3` | Bit width (3 or 4) |
| `-o, --output` | — | Alternative to positional OUTPUT |
| `--no-qjl` | off | Skip QJL correction (faster, lower quality) |

**Examples:**

```bash
# Basic 3-bit compression
turboquant compress wiki_embeddings.npy wiki.tq

# 4-bit compression (higher quality)
turboquant compress embeddings.npy -b 4

# Fast mode (no QJL)
turboquant compress big_set.npy -b 3 --no-qjl
```

### decompress

```bash
turboquant decompress INPUT [OUTPUT]
```

### estimate

```bash
turboquant estimate INPUT [--bits N]
```

---

## 🐍 Python API

```python
from turboquant_tools import compress, decompress, estimate_savings
import numpy as np

# Load or generate embeddings
vectors = np.random.randn(10000, 384).astype(np.float32)

# Compress (5–7× reduction)
compressed = compress(vectors, bits=3, use_qjl=False)
print(f"{vectors.nbytes / 1e6:.1f} MB → {compressed.nbytes / 1e6:.1f} MB ({compressed.memory.ratio:.1f}×)")

# Restore
restored = decompress(compressed)
print(f"MAE: {np.abs(restored - vectors).mean():.4f}")

# Estimate without running
est = estimate_savings(n_vectors=100000, dim=768, bits=3)
print(est)  # Original: X MB -> Compressed: Y MB (7.60×, save 87%)
```

**CompressedVectors** objects carry metadata:

```python
compressed.n_vectors   # original count
compressed.dim         # original dimension
compressed.nbytes      # compressed size in bytes
compressed.memory      # MemoryBytes(original, compressed, ratio)
compressed.data        # raw .tq bytes (save to disk)
```

---

## 🤖 MCP Server (AI Agents)

TurboQuant Tools ships with a native **MCP server** for AI agent integration — works with any MCP-compatible host (Hermes, Claude Desktop, etc.).

### Start

```bash
turboquant mcp-server
```

### Register in your MCP client

**Hermes Agent** (`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  turboquant-tools:
    command: turboquant
    args: ["mcp-server"]
    enabled: true
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "turboquant-tools": {
      "command": "turboquant",
      "args": ["mcp-server"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|---|---|
| `compress_embeddings` | Compress vectors in-memory |
| `decompress_embeddings` | Restore compressed vectors |
| `estimate_savings_mcp` | Predict compression ratio |
| `embed_and_compress` | Embed texts via API + compress in one step |

---

## 📊 Performance

Measured on random float32 embeddings (CPU, no GPU needed):

| Vectors | Dim | Mode | Original | Compressed | Ratio | MAE |
|---|---|---|---|---|---|---|
| 20 | 384 | PolarQuant 3-bit | 30 KB | 10 KB | **3.0×** | 2.6 |
| 20 | 384 | TurboQuant (QJL) | 30 KB | 20 KB | 1.5× | 3.3 |
| 100K | 384 | PolarQuant 3-bit | 153 MB | 20 MB | **7.6×** | — |

**Use cases:**
- **RAG pipelines** — compress vector DB indexes
- **Edge devices** — fit embeddings in limited RAM
- **Storage savings** — reduce cloud costs for large vector stores
- **Memory-bound agents** — compress context vectors on the fly

---

## 🧪 Development

```bash
git clone https://github.com/FreezeVII/turboquant-tools.git
cd turboquant-tools
pip install -e .
pip install pytest
pytest tests/
```

### Run tests

```bash
pytest tests/ -v
```

---

## 🧱 How It Works

Two-stage compression inspired by [Google's TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/):

1. **PolarQuant** — Random Hadamard rotation + scalar quantization to 3–4 bits per dimension. Captures magnitude and direction.
2. **QJL** (optional) — Quantized Johnson-Lindenstrauss residual correction. Recovers high-frequency detail lost in PolarQuant.

Both stages run **CPU-only** via PyTorch — no GPU required. The `.tq` binary format uses a 30-byte header with magic bytes (`TQT2`) + packed indices and norms.

Under the hood this wraps [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant), a reference PyTorch implementation.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙌 Contributing

PRs welcome! Ideas:
- FAISS index compression (`compress_faiss`)
- Onnx / numpy-only backend (no PyTorch dep)
- Streaming compression for billion-scale datasets
- Pre-built wheels for faster install

---

<p align="center">Made with 🧊 for the vector search community.</p>
