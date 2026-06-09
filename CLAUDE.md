# CLAUDE.md — Multivector Retrieval Educational CLI

## Project overview

Rust workspace teaching five multivector retrieval engines step by step:
`HNSW → ColBERT → PLAID → WARP → TACHIOM`

Each engine is a working implementation runnable via `cargo run -- demo <engine>` or `cargo run -- repl <engine>`.

## Toolchain

**Nightly required.** `rust-toolchain.toml` pins the channel. Run `rustup show` to confirm.

## Secrets — NEVER commit `.env`

`.env` at workspace root contains live credentials:
```
VOYAGE_API_KEY=...   # MongoDB-proxied Voyage AI (https://ai.mongodb.com/v1)
MONGODB_URI=...      # Atlas cluster
ANTHROPIC_API_KEY=...  # optional — needed for gt-bench LLM judging
```

`.env` is in `.gitignore`. Never `git add .env`, `git add -A`, or `git add .`. Stage files explicitly by name.

## Key commands

```bash
cargo run --release -- demo <hnsw|colbert|plaid|warp|tachiom>
cargo run --release -- repl <engine>
cargo run --release -- scale        # O(N) vs O(N×p) scaling demo
cargo run --release -- tradeoff     # Recall@10 vs speedup plots (plots/)
cargo run --release -- gt-bench     # LLM-judged ground truth benchmark (plots/)
```

## Architecture

- `crates/common/` — `TraceLog`, `OpTiming`, `RandomProjection` (token_id → 128-dim), `WordPieceTokenizer`, corpus
- `crates/colbert/src/encoder.rs` — `ColBertEncoder` using `RandomProjection` (deterministic hash, NOT a learned model)
- `crates/hnsw/src/voyage.rs` — `VoyageClient` calls `https://ai.mongodb.com/v1/embeddings`, caches to `cache/voyage_*.json`
- `crates/cli/src/tradeoffbench.rs` — synthetic 128-dim benchmark (HNSW + PLAID + WARP + TACHIOM)
- `crates/cli/src/gtbench.rs` — real-text 100-doc benchmark with Anthropic judge + Voyage HNSW embeddings

## Per-token embeddings are random projections

The ColBERT/PLAID/WARP/TACHIOM engines use `RandomProjection`: a **deterministic hash** from WordPiece vocab_id → 128-dim unit vector. This is NOT a learned semantic embedding. The word "bank" always maps to the same vector in every document. Disambiguation works through multi-token MaxSim (river+bank vs account+bank), not semantic meaning. HNSW uses real Voyage sentence embeddings.

## Benchmark notes

**tradeoff benchmark** (synthetic, N=1K/10K/100K):
- HNSW recall ceiling at N=10K: ~0.572 regardless of ef — proves sentence-avg loses MaxSim per-token interactions
- WARP: 1.000 Recall@10 at 100× speedup (exact Xtr threshold)
- PLAID: ~0.833 ceiling at 5% candidates (centroid approximation error)

**gt-bench** (real text, N=100):
- With `ANTHROPIC_API_KEY`: calls `claude-haiku-4-5-20251001` to judge all 100 docs per query
- Without key: falls back to category-membership heuristic (20 river, 20 finance, etc.)
- Results cached in `cache/llm_gt.json` and `cache/gt_voyage_*.json`
- HNSW with Voyage: 0.940 Recall@10 at 10% candidates (heuristic GT)
- ColBERT full scan: 0.700 (lexical mismatch vs topic GT)

## Noise calibration for synthetic data

`DOC_NOISE=0.20`, `QUERY_NOISE=0.06` in tradeoffbench. These satisfy `noise << 1/sqrt(DIM) ≈ 0.088`
to keep topic signal after L2 normalization. Same-topic dot product ≈ 0.57. Do not increase without recalibrating WARP threshold sweep.

## plotters notes

Use `SVGBackend`. Log axes: manual `log2` transform — do NOT use `.log_scale()` API. Features required: `svg_backend`, `line_series`, `point_series`, `area_series`. Do NOT add `full_palette` or `series_labels` (they don't exist in v0.3).

## hnsw_rs API (v0.3)

```rust
let h = Hnsw::<f32, DistCosine>::new(m, max_elem, max_layer, ef_construction, DistCosine{});
h.insert((slice, index));                         // index is DataId: usize
let results: Vec<Neighbour> = h.search(query, nb_res, ef_arg);
// Neighbour has .d_id: usize and .distance: f32 (= 1 - cosine_sim)
```

## Voyage API endpoint

Uses MongoDB proxy: `https://ai.mongodb.com/v1/embeddings` (not `api.voyageai.com`). Bearer auth with `VOYAGE_API_KEY`. Default model: `voyage-4-large` (1024-dim). Override with `VOYAGE_MODEL` env var.

## Anthropic API (for gt-bench)

```
POST https://api.anthropic.com/v1/messages
Headers: x-api-key: $ANTHROPIC_API_KEY, anthropic-version: 2023-06-01
Body: {"model": "claude-haiku-4-5-20251001", "max_tokens": 512, "messages": [...]}
Response: {"content": [{"type": "text", "text": "[0, 5, 14]"}]}
```
