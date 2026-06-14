# Multivector Retrieval — Educational CLI

A hands-on tour of five multivector retrieval engines, from dense HNSW to token-aware ColBERT, PLAID, WARP, and TACHIOM. Each engine is a working implementation — demo, query interactively, and benchmark.

```
HNSW ──► ColBERT ──► PLAID ──► WARP ──► TACHIOM
 dense    per-token   centroid  exact-sim  type-budget
 ANN      MaxSim      pruning   threshold  allocation
```

> **Readable version:** [Google Doc](https://docs.google.com/document/d/1vl0dkQ6ZYJhqz8s15JT-Z-kt1DpZoGAoMPCeeIlC-Y4/edit)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Rust nightly | pinned in `rust-toolchain.toml`, auto-installed by rustup |
| `VOYAGE_API_KEY` | Voyage-4-large sentence embeddings (HNSW) |
| `MONGODB_URI` | Atlas cluster |
| `GROVE_API_KEY` | optional — LLM judge for `gt-bench` |
| `JINA_API_KEY` | optional — Jina ColBERT v2 for token engines in gt-bench |

Create a `.env` at the workspace root (gitignored — never commit):

```
VOYAGE_API_KEY=...
MONGODB_URI=mongodb+srv://...
GROVE_API_KEY=...   # optional
JINA_API_KEY=...    # optional
```

---

## Quick Start

```bash
cargo build --release

cargo run --release -- demo hnsw      # guided demo with narration + timing
cargo run --release -- repl hnsw      # interactive REPL (tab-complete commands)
```

Replace `hnsw` with `colbert`, `plaid`, `warp`, or `tachiom`.

---

## The Five Engines

### 1. HNSW — Dense ANN

Layered proximity graph over 1024-dim Voyage-4-large sentence embeddings. Fast ANN lookup (M=4, ef_construction=64).

**Limitation:** a single vector per doc collapses token-level meaning. Queries for *"river bank"* and *"financial bank"* land in the same neighborhood.

### 2. ColBERT — Per-Token MaxSim

Stores a 128-dim embedding per token. Scores via **MaxSim**: for each query token, find the highest cosine similarity across all doc tokens, then sum. Disambiguates polysemy. O(N) at query time.

### 3. PLAID — Centroid Pruning

Global k-means over all token embeddings. At query time, probes top-C centroids and only MaxSim-scores docs whose tokens belong to those centroids. Reduces scoring to ~1–5% of corpus.

**Limitation:** hard cluster assignments can miss relevant docs near centroid boundaries.

### 4. WARP — Exact-Similarity Threshold

Replaces centroid ANN with **exact Xtr token similarity thresholds** precomputed at index time. Docs clearing the threshold enter MaxSim reranking — no approximation error. Achieves Recall@10 = 1.000 at 100× speedup on synthetic benchmarks.

### 5. TACHIOM — Per-Type Budget Allocation

WARP's Xtr registry is O(token_types × docs) — impractical at scale. TACHIOM assigns per-token-type centroid budgets proportional to rarity: tail (rare, discriminative) tokens get more centroids; head (frequent) tokens get fewer.

---

## Interactive REPL

```
$ cargo run --release -- repl colbert

[colbert] > index 0
[colbert] > index 1
[colbert] > inspect tokens 0
[colbert] > query river erosion along the bank
[colbert] > trace maxsim
```

Engine-specific inspect commands:

| Engine | Commands |
|---|---|
| hnsw | `inspect layers`, `inspect graph` |
| colbert | `inspect tokens <id>`, `trace maxsim` |
| plaid | `inspect centroids` |
| warp | `inspect gather` |
| tachiom | `inspect centroids` |

Pass `--trace-json path/to/trace.jsonl` for machine-readable output.

---

## Benchmarks

### Accuracy-Speed Tradeoff

Recall@10 vs speedup across N = 1K / 10K / 100K. Uses synthetic 128-dim token embeddings; oracle = ColBERT full scan.

```bash
cargo run --release -- tradeoff
# → plots/tradeoff_speedrecall.svg
# → plots/recall_vs_frac.svg
# → plots/index.html  (visualizer — open in browser)
```

**Results (Recall@10 vs candidate fraction):**

| Engine | 1% | 5% | 10% | 20% | 50% | N |
|---|---|---|---|---|---|---|
| WARP | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 100K |
| PLAID | — | 0.833 | 0.833 | 1.000 | 1.000 | 100K |
| TACHIOM | — | 0.740 | 0.740 | 1.000 | 1.000 | 100K |
| HNSW | 0.267 | 0.620 | 0.620 | 0.620 | 0.620 | 100K |

*WARP achieves exact Recall@1% due to precomputed Xtr thresholds. HNSW plateaus at ~0.62 — sentence-avg loses per-token MaxSim signal at scale.*

### Visualizer

Open `plots/index.html` for a paper-style viewer with callout annotations and an engine mechanism comparison table.

```bash
open plots/index.html
```

---

### Ground-Truth Benchmark (LLM-as-Judge)

100-doc corpus across 10 semantic categories. Claude judges relevance per query; engines are evaluated against those labels with real Voyage-4-large embeddings.

```bash
# Requires VOYAGE_API_KEY + GROVE_API_KEY
cargo run --release -- gt-bench
# → plots/gt_recall.svg
```

**Results:**

| Engine | Recall@10 | Candidate fraction |
|---|---|---|
| HNSW (Voyage-4-large) | 0.955 | 100% (ef sweep) |
| TACHIOM (Jina ColBERT) | 0.954 | 10% |
| PLAID (Jina ColBERT) | 0.941 | 50% |
| WARP (Jina ColBERT) | 0.941 | 50% |
| ColBERT (full scan) | 0.941 | 100% |

*TACHIOM matches HNSW recall (0.954 vs 0.955) at 10% candidates — 10× speedup with near-zero recall loss on real text.*

---

### Large-Scale Needle-in-Haystack

100 real GT docs buried in a synthetic corpus growing from 100K to 100M. Three filter modes: none / category / category + year.

```bash
# Requires gt-bench cache (run gt-bench first)
cargo run --release -- large-bench
# → plots/large_bench_recall_vs_n.svg
# → plots/large_bench_recall_vs_frac.svg
```

**Results (Recall@10, no filter, 10% candidates):**

| Engine | N=100K | N=1M | N=10M | N=100M |
|---|---|---|---|---|
| HNSW | 0.951 | 0.951 | 0.951 | 0.951 |
| PLAID | 0.965 | 0.965 | 0.965 | 0.965 |
| WARP | 0.965 | 0.965 | 0.965 | 0.965 |
| TACHIOM | 0.937 | 0.696 | 0.661 | 0.649 |

*HNSW, PLAID, and WARP are scale-invariant — recall holds flat from 100K to 100M. TACHIOM degrades past N=1M as centroid budgets are sized for the small 100-doc GT corpus and become insufficient against a growing synthetic pool.*

> **Note on synthetic distractors:** distractor docs are random 128-dim unit vectors — not real text. This tests index scaling robustness, not semantic discrimination. See the GT Benchmark for real-text evaluation.

---

## MS MARCO Real-Data Pipeline

Embeds 5M passages for real-data benchmarking. Two models, two separate passes — both resume-safe.

```bash
# 1. Download MS MARCO (~987 MB → 2.9 GB)
mkdir -p data/msmarco
curl -L https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz \
  -o data/msmarco/collection.tar.gz
tar -xzf data/msmarco/collection.tar.gz -C data/msmarco/

# 2. GTE-ModernColBERT-v1 token embeddings (local, Apple Silicon MPS)
python3 -m venv .venv && .venv/bin/pip install pylate
.venv/bin/python3 scripts/embed_gte.py data/msmarco/collection.tsv --limit 5000000

# 3. Voyage-4-large sentence embeddings (API)
cargo run --release -- embed-msmarco data/msmarco/collection.tsv \
  --limit 5000000 --voyage-only
```

Output layout:

```
data/msmarco/
  gte/
    offsets.bin   u64[N]           byte offsets into data.bin
    lengths.bin   u32[N]           token count per passage (pool_factor=2)
    data.bin      f32[Σtokens×128] GTE-ModernColBERT-v1 embeddings  (~104 GB)
    meta.json
  voyage/
    data.bin      f32[N×1024]      Voyage-4-large embeddings          (~20 GB)
    meta.json
```

### One-time setup cost (5M passages)

| Model | Where | Cost | Time |
|---|---|---|---|
| GTE-ModernColBERT-v1 (128-dim, pool_factor=2) | Local MPS | **free** | ~29 hours |
| Voyage-4-large (1024-dim) | MongoDB-proxied API | **~$50** | ~15 hours |

---

## Crate Structure

```
crates/
  common/     shared types: TraceLog, OpTiming, corpus, tokenizer
  hnsw/       HNSW engine + Voyage API client
  colbert/    ColBERT engine + Jina API client
  plaid/      PLAID engine (centroid inverted index)
  warp/       WARP engine (Xtr threshold registry)
  tachiom/    TACHIOM engine (per-type budget allocation)
  bench/      SIFT1M benchmark harness
  cli/        entry point, demo, REPL, benchmarks
scenarios/    demo scenario TOML files
scripts/      embed_gte.py — GTE-ModernColBERT-v1 embedding pipeline
plots/        generated SVGs + index.html visualizer
vocab/        WordPiece vocabulary
```

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `VOYAGE_API_KEY` | Yes | Voyage-4-large embeddings (HNSW) |
| `MONGODB_URI` | Yes | Atlas cluster |
| `GROVE_API_KEY` | No | LLM judge for gt-bench |
| `JINA_API_KEY` | No | Jina ColBERT v2 for gt-bench token engines |
| `MULTIVECTOR_VOCAB` | No | Override path to `wordpiece_vocab.txt` |

---

## References

- **ColBERT**: Khattab & Zaharia (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction.*
- **PLAID**: Santhanam et al. (2022). *PLAID: An Efficient Engine for Late Interaction Retrieval.*
- **WARP**: Lassance & Clinchant (2023). *WARP: Time-Efficient Nearest Neighbor Search with Xtr-based Candidate Retrieval.*
- **TACHIOM**: Bruch et al. (2024). *Token-Aware Clustering with Hierarchical Information for Optimal Multivector Retrieval.*
- **GTE-ModernColBERT-v1**: LightOn (2025). *GTE-ModernColBERT: State-of-the-Art Late Interaction Model.*
