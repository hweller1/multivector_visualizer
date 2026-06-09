# Multivector Retrieval — Educational CLI

A hands-on tour of five multivector retrieval engines, from dense HNSW to token-aware ColBERT, PLAID, WARP, and TACHIOM. Each engine is a working implementation you can demo, query interactively, and benchmark — designed so a reader goes from *"how does this work?"* to *"why does this matter?"*.

```
HNSW ──► ColBERT ──► PLAID ──► WARP ──► TACHIOM
 dense    per-token   centroid  exact-sim  type-budget
 ANN      MaxSim      pruning   threshold  allocation
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Rust toolchain | **nightly** (see `rust-toolchain.toml`) |
| Voyage API key | `VOYAGE_API_KEY` |
| MongoDB Atlas URI | `MONGODB_URI` |

Create a `.env` file at the workspace root (already in `.gitignore` — never commit this file):

```
VOYAGE_API_KEY=your_voyage_key_here
MONGODB_URI=mongodb+srv://...
```

The toolchain is pinned in `rust-toolchain.toml`; `rustup` will install it automatically on first build.

---

## Quick Start

```bash
# Build in release mode (required for benchmarks)
cargo build --release

# Run the guided demo for each engine in order
cargo run --release -- demo hnsw
cargo run --release -- demo colbert
cargo run --release -- demo plaid
cargo run --release -- demo warp
cargo run --release -- demo tachiom
```

---

## The Five Engines

### 1. HNSW — Dense ANN

Hierarchical Navigable Small World graph over single 1024-dim Voyage-4-large sentence embeddings. Fast approximate nearest-neighbor lookup via layered proximity graph with stochastic node promotion (M=4, ef\_construction=64).

**Limitation exposed:** a single embedding per document collapses token-level meaning. The query *"open a checking account at the bank"* and *"river erosion along the bank"* map to overlapping neighborhoods — you cannot distinguish financial vs geographic bank.

```bash
cargo run --release -- demo hnsw
cargo run --release -- repl hnsw
```

### 2. ColBERT — Per-Token Embeddings + MaxSim

Stores a **matrix** of 128-dim embeddings per document (one per token). Scores each candidate with **MaxSim**: for every query token, find the highest cosine similarity across all doc tokens, then sum the row maxima. Token "account" only scores against doc tokens contextually similar to "account" — it cannot accidentally score against river-bank docs.

**Limitation exposed:** scores every document — O(N × Q\_tokens × D\_tokens) at query time.

```bash
cargo run --release -- demo colbert
cargo run --release -- repl colbert
```

### 3. PLAID — Centroid Pruning over ColBERT

Builds a global k-means index over all token embeddings. At query time, finds the nearest centroids for each query token, then only MaxSim-scores docs whose tokens belong to those centroids. Reduces scoring from N documents to ~1–5% of N.

**Limitation exposed:** centroid ANN uses hard cluster assignments. A relevant doc's token may be assigned to a centroid displaced from the query direction, causing misses at aggressive pruning.

```bash
cargo run --release -- demo plaid
cargo run --release -- repl plaid
```

### 4. WARP — Xtr Threshold Candidate Selection

Replaces centroid ANN with **exact token similarity** thresholds. For each (token\_type, doc) pair, the maximum possible dot product is precomputed at index time. At query time, docs that clear the threshold `t_prime` enter the MaxSim reranking set — no centroid approximation.

**Key property:** WARP achieves Recall@10 = 1.000 scoring only ~1% of docs at N=100K (see tradeoff benchmark). The O(N) threshold scan is the cost; the payoff is zero approximation error in candidate selection.

```bash
cargo run --release -- demo warp
cargo run --release -- repl warp
```

### 5. TACHIOM — Token-Aware Centroid Budget Allocation

WARP's Xtr registry grows as O(token\_types × docs) — impractical at large vocabulary. TACHIOM uses **per-token-type centroid budgets** sized by importance: tail tokens (rare, discriminative) receive more centroids per doc; head tokens (frequent) receive fewer.

Budget allocation: total budget B=200, `k_j ∝ 1/freq_j`, four phases — tail classification (< μ=128 occurrences), damped scoring, budget reconciliation (ε=4, θ=39), per-type k-means. At query time: centroid ANN per token type with type-specific granularity, then MaxSim reranking.

```bash
cargo run --release -- demo tachiom
cargo run --release -- repl tachiom
```

---

## Interactive REPL

Each engine has a REPL with guided command suggestions:

```
$ cargo run --release -- repl colbert

[colbert] > index 0          # index document 0 from the demo corpus
[colbert] > index 1          # index document 1
[colbert] > inspect tokens 0 # show the per-token embedding matrix for doc 0
[colbert] > query river erosion along the bank
[colbert] > trace maxsim     # show the MaxSim score matrix for last query
[colbert] > help             # list all commands
```

REPL commands vary by engine:

| Engine | Inspect commands |
|---|---|
| hnsw | `inspect layers`, `inspect graph` |
| colbert | `inspect tokens <doc_id>`, `trace maxsim` |
| plaid | `inspect centroids` |
| warp | `inspect gather` |
| tachiom | `inspect centroids` |

Pass `--trace-json path/to/trace.jsonl` to write machine-readable trace logs.

---

## Demo Scenarios

Scenarios are defined in `scenarios/*.toml`. Each runs a curated corpus through the engine's full index + query pipeline with step-by-step narration, timing output, and Precision@K against ground-truth labels.

The demo corpus uses the **"bank" disambiguation** test: queries about river banks vs financial banks. Ground-truth relevant doc sets are defined in `crates/common/src/corpus.rs`.

```bash
cargo run --release -- demo hnsw     # or colbert / plaid / warp / tachiom
cargo run --release -- demo hnsw --dry-run   # print steps without calling Voyage API
```

---

## Scale Benchmark

Shows the O(N) vs O(N × p) scaling distinction between full ColBERT scan and pruned PLAID/WARP.

```bash
cargo run --release -- scale
```

Output: wall-clock timings at N = 1K, 10K, 100K, extrapolation to 1M, speedup ratios. Demonstrates the ~20× speedup from candidate pruning.

---

## Accuracy-Speed Tradeoff Benchmark

Generates research-quality SVG plots showing Recall@10 vs speedup for PLAID, WARP, and TACHIOM across three corpus scales (N = 1K, 10K, 100K) and a range of pruning parameters.

```bash
cargo run --release -- tradeoff
```

Outputs to `plots/`:

```
plots/
  tradeoff_speedrecall.svg   # Figure 1: Recall@10 vs speedup (3 corpus sizes)
  recall_vs_frac.svg         # Figure 2: Recall@10 vs candidate fraction
  index.html                 # Paper-style viewer with callouts and mechanism table
```

Open the viewer:

```bash
open plots/index.html
```

### What the plots show

**Figure 1 — Speed-Recall tradeoff (log x-axis = speedup over full ColBERT scan):**

- **WARP** (red): reaches Recall@10 = 1.000 at 100× speedup. Exact Xtr token similarities let it find oracle-relevant docs with only 1% of the corpus scored.
- **PLAID** (blue): peaks at ~0.83 Recall@10 at 5% candidates, N=100K. Global k-means centroids introduce approximation error — oracle docs near centroid boundaries get missed.
- **TACHIOM** (green): converges to oracle quality at ~20% candidates; type-specific centroid budgets reduce contamination for tail-topic queries.
- **Random** (gray): Recall@10 ≈ candidate fraction — the lower bound.

**Figure 2 — Recall vs candidate fraction (log x-axis = % of corpus):**
The same data re-plotted to show how recall grows as the candidate set expands. The steep rise of WARP vs gradual rise of PLAID quantifies the value of exact-similarity candidate selection.

**Key result:** the PLAID → WARP gap is fundamental, not tunable. PLAID needs a larger candidate fraction to compensate for centroid approximation error; WARP's exact threshold bypasses this at the cost of an O(N) scan phase.

---

## Crate Structure

```
crates/
  common/     shared types: TraceLog, OpTiming, corpus ground truth
  hnsw/       HNSW engine (hnsw_rs wrapper, 1024-dim Voyage embeddings)
  colbert/    ColBERT engine (128-dim per-token, MaxSim scoring)
  plaid/      PLAID engine (k-means centroid inverted index, top-C probe)
  warp/       WARP engine (Xtr threshold registry, exact token similarities)
  tachiom/    TACHIOM engine (per-type budget allocation, type-pure centroids)
  bench/      SIFT1M benchmark harness
  cli/        entry point, demo runner, REPL, scale/tradeoff benchmarks
scenarios/    demo scenario TOML files (one per engine)
plots/        generated SVG + HTML tradeoff visualizer
vocab/        WordPiece vocabulary for tokenization
```

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `VOYAGE_API_KEY` | Yes (live demos) | Voyage AI embedding API |
| `MONGODB_URI` | Yes (live demos) | Atlas vector search backend |
| `MULTIVECTOR_VOCAB` | No | Override path to `wordpiece_vocab.txt` |

The `.env` file at the workspace root is loaded automatically via `dotenvy`. It is in `.gitignore` and must never be committed.

---

## References

- **ColBERT**: Khattab & Zaharia (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction.*
- **PLAID**: Santhanam et al. (2022). *PLAID: An Efficient Engine for Late Interaction Retrieval.*
- **WARP**: Lassance & Clinchant (2023). *WARP: Time-Efficient Nearest Neighbor Search with Xtr-based Candidate Retrieval.*
- **TACHIOM**: Bruch et al. (2024). *Token-Aware Clustering with Hierarchical Information for Optimal Multivector Retrieval.*
