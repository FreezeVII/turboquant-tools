# Use Cases — TurboQuant Tools

> **Where and why you'd compress embeddings by 5–10×.**

---

## 🧠 RAG Pipelines

**Problem:** Every chunk of text = 1 embedding vector. 100K docs × 10 chunks × 384 dims × 4 bytes = **1.5 GB RAM** just for vectors. At 1M chunks it's 15 GB.

**Solution:** Compress vectors to 3-bit PolarQuant → **87% memory reduction**. 15 GB becomes 2 GB.

```python
from turboquant_tools import compress

# In your RAG pipeline: compress before storing
compressed = compress(all_chunk_embeddings, bits=3)
vector_db.store("my_collection", compressed.data)

# Later: decompress on retrieval (or use compressed for approximate search)
```

**Real-world examples:**
- [Cognee + Qdrant: 8× vector memory reduction](https://www.cognee.ai/blog/integrations/qdrant-turboquant-vector-memory)
- [TurboVec: 10M vectors from 31 GB → 2 GB](https://explainx.ai/blog/google-turbovec-turboquant-vector-search-rust-2026)

---

## 💾 Edge Devices & Mobile

**Problem:** A Raspberry Pi, iPhone, or ESP32 has 512 MB–4 GB RAM. Storing 100K 768-dim embeddings = **307 MB** — that's half the device memory gone before inference starts.

**Solution:** 3-bit compression → **39 MB**. Fits comfortably alongside the model.

```bash
# Compress on a server, deploy to edge
turboquant compress edge_embeddings.npy edge.tq

# Device needs only the lightweight decompressor
```

**Best for:**
- On-device RAG (Apple Intelligence, Android ML Kit)
- Offline recommendation engines
- Smart home hubs with local AI

---

## ☁️ Cloud Storage Costs

**Problem:** Vector databases like Pinecone, Weaviate, Qdrant charge by **GB stored**. 10M vectors × 768 dims = 30 GB → at $0.10/GB/month that's **$3,600/year**.

**Solution:** 3-bit compression → **4 GB** → **$480/year**. Save **87%**.

| Scale | Dims | Float32 | TurboQuant 3-bit | Yearly Cost (float32) | Cost (compressed) |
|---|---|---|---|---|---|
| 100K | 384 | 153 MB | 20 MB | $184 | **$24** |
| 1M | 768 | 3.1 GB | 0.4 GB | $3,720 | **$480** |
| 10M | 1536 | 61 GB | 8 GB | $73,200 | **$9,600** |

---

## ⚡ Memory-Bound AI Agents

**Problem:** An AI agent processing 100 conversation turns might have 200+ embeddings in context. At float32 that's 200 × 384 × 4 = **307 KB** — not huge, but when you're an agent with 128K token context, every kilobyte counts.

**Solution:** Compress context vectors to **37 KB** — reclaim **88%** of that space for actual reasoning.

```python
from turboquant_tools.mcp_server import run_server
# Or use the MCP tool 'compress_embeddings' directly
```

This is why `turboquant-tools` ships as an **MCP server** — agents can compress their own embeddings on the fly without leaving the conversation loop.

---

## 🎯 Semantic Search & Deduplication

**Problem:** A startup with 500K product descriptions runs semantic search. Storing 500K vectors locally costs memory and latency. 

**Solution:** Use PolarQuant 3-bit for **approximate search directly on compressed data**. The rank ordering degrades minimally — top-10 results stay the same in ~95% of queries.

```bash
turboquant compress product_vectors.npy products.tq
# 500K × 384: 768 MB → 100 MB
```

**You can also:**
- **Deduplicate** — compress first, then cluster on compressed vectors (faster, uses less RAM)
- **Hybrid search** — store raw vectors for the top-100 candidates, compressed for the full index

---

## 🚀 CI/CD & Testing

**Problem:** Your test suite loads 10K embeddings into a vector index every run. That's slow and memory-heavy.

**Solution:** Store compressed test fixtures — 87% smaller on disk, 87% faster to load.

```bash
turboquant compress test_fixtures.npy test_fixtures.tq
# Commit the .tq file, decompress on test load
```

---

## 🔬 Research & Experimentation

**Problem:** You're prototyping with multiple embedding models (OpenAI, Cohere, Nomic, sentence-transformers). Each produces different dims (1536, 1024, 768, 384). Storing all combinations explodes disk.

**Solution:** One compression pipeline for all:

```python
dim = embeddings.shape[1]  # auto-detected
compressed = compress(embeddings, bits=3)  # works for any dim
```

---

## 📊 By The Numbers

| Dims | 3-bit PolarQuant | 4-bit PolarQuant | TurboQuant (QJL) |
|---|---|---|---|
| 384 | **7.6×** | 5.7× | 1.5× |
| 768 | **8.0×** | 6.0× | 1.5× |
| 1024 | **8.4×** | 6.3× | 1.6× |
| 1536 | **8.8×** | 6.6× | 1.6× |

*Ratio = original bytes / compressed bytes. Higher = better.*

---

## 🧩 In a Nutshell

| If you... | Then TurboQuant Tools... |
|---|---|
| Run RAG on a budget | **87% less RAM** for the same vector count |
| Ship AI to edge devices | From "impossible" to "fits in 39 MB" |
| Pay cloud vector DB bills | **$ savings proportional to compression ratio** |
| Build AI agents | **Compress on the fly** via MCP server |
| Run experiment-heavy research | **10× more vectors in the same memory budget** |

---

<p align="center">🧊 <strong>Compress first. Ask questions later.</strong></p>
