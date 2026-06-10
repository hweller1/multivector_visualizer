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
GROVE_API_KEY=...    # optional — needed for gt-bench LLM judging
JINA_API_KEY=...     # optional — Jina ColBERT v2 per-token embeddings for token engines
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
- `crates/colbert/src/encoder.rs` — `ColBertEncoder`: auto-loads Jina ColBERT cache on construction; falls back to `RandomProjection` if no cache
- `crates/colbert/src/jina.rs` — `JinaColBertClient`: fetches per-token 128-dim embeddings from `https://api.jina.ai/v1/embeddings`, caches to `cache/jina_colbert_embeddings.json`
- `crates/hnsw/src/voyage.rs` — `VoyageClient` calls `https://ai.mongodb.com/v1/embeddings`, caches to `cache/voyage_*.json`
- `crates/cli/src/tradeoffbench.rs` — synthetic 128-dim benchmark (HNSW + PLAID + WARP + TACHIOM), uses RandomProjection (no Jina — N=100K would be too expensive)
- `crates/cli/src/gtbench.rs` — real-text 100-doc benchmark with Anthropic judge + Voyage HNSW + Jina token embeddings

## Per-token embeddings

**With JINA_API_KEY** (recommended): `ColBertEncoder` uses Jina ColBERT v2 per-token embeddings (128-dim, learned, semantically meaningful). "bank" in a river context gets a different representation than "bank" in a finance context because the model is trained with cross-attention. Enables apples-to-apples comparison with HNSW.

**Without JINA_API_KEY** (fallback): `RandomProjection` — a **deterministic hash** from WordPiece vocab_id → 128-dim unit vector. NOT a learned model. "bank" always maps to the same vector in every document. Disambiguation only works through multi-token MaxSim patterns.

## Benchmark notes

**tradeoff benchmark** (synthetic, N=1K/10K/100K, oracle = ColBERT full scan on random projections):
- HNSW ceiling at N=100K: ~0.747 regardless of numCandidates — sentence-avg loses per-token MaxSim signal
- HNSW at N=10K: gradual curve, reaches 0.920 at 50% candidates (ceiling less severe at smaller scale)
- WARP: 1.000 Recall@10 at 100× speedup (exact Xtr threshold)
- PLAID: ~0.833 ceiling at 5% candidates (centroid approximation error)
- ef_vals for HNSW sweep extend to 5000 (50% of 10K, ~5% of 100K) to prove plateau

**gt-bench** (real text, N=100, Jina ColBERT per-token embeddings):
- With `GROVE_API_KEY`: calls `claude-sonnet-4-6` via Grove gateway to judge all 100 docs per query
- Without key: falls back to category-membership heuristic (20 river, 20 finance, etc.)
- With `JINA_API_KEY`: token engines use Jina ColBERT v2 128-dim per-token embeddings (cached in `cache/jina_colbert_embeddings.json`)
- Results cached in `cache/llm_gt.json`, `cache/gt_voyage_*.json`, `cache/jina_colbert_embeddings.json`
- HNSW (Voyage): 0.955 Recall@10
- ColBERT full scan (Jina): 0.941 — competitive with HNSW, true apples-to-apples
- TACHIOM (Jina, 10% candidates): 0.954 — category routing achieves near-HNSW recall at 10× speedup
- PLAID/WARP (Jina, 50% candidates): 0.941
- HNSW ef sweep: always returns K_EVAL=10 results, varies ef as numCandidates; frac = ef/n

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

## Jina ColBERT API (for token engines)

```
POST https://api.jina.ai/v1/embeddings
Headers: Authorization: Bearer $JINA_API_KEY
Body: {"model": "jina-colbert-v2", "input": ["text"], "output_format": "token_embeddings", "truncate_dim": 128}
Response: {"data": [{"embeddings": [[...128 floats per token...], ...]}]}
```
Cache path: `cache/jina_colbert_embeddings.json` (keyed by text string).
`ColBertEncoder::new()` auto-loads this cache; engines pick up Jina embeddings transparently.
Demo/repl: `warm_jina_for_corpus()` in main.rs prefetches SHARED_CORPUS before engine construction.

## Anthropic API (for gt-bench)

Uses MongoDB's internal Grove gateway (not api.anthropic.com directly):
```
POST https://grove-gateway-prod.azure-api.net/grove-foundry-prod/anthropic/v1/messages
Headers: api-key: $GROVE_API_KEY, anthropic-version: 2023-06-01
Body: {"model": "claude-sonnet-4-6", "max_tokens": 512, "messages": [...]}
Response: {"content": [{"type": "text", "text": "[0, 5, 14]"}]}
```
`GROVE_API_KEY` holds the Grove gateway key (32-char hex), not an Anthropic sk-ant key.
