# Requirements: Multivector Retrieval Educational CLI

## Goal

An interactive Rust CLI that teaches information retrieval engineers exactly how five retrieval engines — HNSW, ColBERTv2, PLAID, WARP, and TACHIOM — index documents and answer queries. The reader finishes with a working mental model of why each engine exists and what tradeoff it resolves over its predecessor.

---

## User Stories

### US-1: Step Through HNSW Dense Retrieval

**As a** developer new to vector search
**I want to** watch a sentence embedding get indexed into an HNSW graph and a query navigate that graph
**So that** I understand why ANN over single-vector representations is fast but loses token-level precision

**Acceptance Criteria:**
- [ ] AC-1.1: `cargo demo hnsw` runs a canned TOML scenario that prints each HNSW layer traversal step with candidate IDs and distances
- [ ] AC-1.2: `index` command in the HNSW REPL ingests a user-supplied sentence, embeds it via voyage-4-large (Atlas cluster), and shows the graph insertion path
- [ ] AC-1.3: `query` command shows the greedy ANN walk: entry node → neighbor exploration → final top-k, with cosine scores at each hop
- [ ] AC-1.4: `inspect` command shows the HNSW graph layer structure: node count per layer, average out-degree, ef_construction value
- [ ] AC-1.5: A narrative annotation after each step explains the precision loss: "one vector per document means token 'bank' and token 'river' share one point"

---

### US-2: Step Through ColBERTv2 Late Interaction

**As a** developer who understands dense retrieval
**I want to** see per-token embeddings generated, indexed, and scored via MaxSim
**So that** I understand how late interaction preserves token-level signal without full cross-attention

**Acceptance Criteria:**
- [ ] AC-2.1: `cargo demo colbert` runs a TOML scenario that walks through: tokenization → per-token embedding → HNSW insertion per token → MaxSim query scoring
- [ ] AC-2.2: `index` command shows each token's embedding coordinates (truncated to 3 dims in viz) and which HNSW shard it lands in
- [ ] AC-2.3: `query` command shows the MaxSim matrix: rows = query tokens, columns = document tokens, max per row highlighted, final score = sum of row maxima
- [ ] AC-2.4: `trace` command replays the full indexing event log for one document, showing TraceEvent names and payloads in order
- [ ] AC-2.5: Narrative contrasts ColBERT with HNSW: "instead of one point per doc, each token gets its own point; MaxSim lets each query token find its best match independently"

---

### US-3: Step Through PLAID Centroid Pruning

**As a** developer who understands ColBERT scoring
**I want to** see how PLAID reduces the candidate set before MaxSim using centroid proximity
**So that** I understand why PLAID is faster than vanilla ColBERT at scale without recall collapse

**Acceptance Criteria:**
- [ ] AC-3.1: `cargo demo plaid` runs a TOML scenario that shows (a) centroid assignment during indexing, (b) centroid ANN for query tokens, (c) candidate doc expansion, (d) MaxSim only on candidates
- [ ] AC-3.2: `inspect centroids` shows centroid count, average cluster size, and which centroids are activated for a sample query
- [ ] AC-3.3: `query` with `--verbose` prints the number of docs pruned at the centroid stage vs. the number scored by MaxSim, with percentage reduction
- [ ] AC-3.4: Scenario narration explicitly shows a document that PLAID would skip and explains why (no query token's centroid is close enough)
- [ ] AC-3.5: Demo shares the same ColBERT token embeddings from US-2 — PLAID is presented as a pruning layer added on top, not a separate engine

---

### US-4: Step Through WARP Xtr-Based Gather

**As a** developer who understands PLAID pruning
**I want to** see how WARP's Xtr-based lightweight gather phase differs from PLAID's centroid ANN
**So that** I understand when WARP's gather-then-refine is preferable and why

**Acceptance Criteria:**
- [ ] AC-4.1: `cargo demo warp` runs a TOML scenario covering: Xtr scoring phase → candidate doc gather → token-level MaxSim refinement
- [ ] AC-4.2: `trace` shows named TraceEvents: `XtrScore`, `CandidateGather`, `MaxSimRefine` with token IDs and scores at each stage
- [ ] AC-4.3: `inspect` shows gather phase statistics: candidates gathered, overlap with ground truth, fraction promoted to refinement
- [ ] AC-4.4: Narrative explicitly contrasts WARP gather vs PLAID centroid ANN: "WARP avoids per-query centroid distance computation; instead Xtr scores guide which token embeddings to fetch"
- [ ] AC-4.5: `bench warp` command calls the xtr-warp binary (must be installed separately) and prints latency and recall output in the terminal

---

### US-5: Step Through TACHIOM TAC Indexing

**As a** developer who understands PLAID and WARP
**I want to** walk through TACHIOM's Token-Aware Clustering pipeline step by step
**So that** I understand how TAC scales to 4M centroids with hierarchical PQ and why it outperforms PLAID at very large scale

**Acceptance Criteria:**
- [ ] AC-5.1: `cargo demo tachiom` runs a TOML scenario covering all four TAC phases: tail handling → damped scoring → floor/ceiling bounding → budget reconciliation → parallel κⱼ-means
- [ ] AC-5.2: `trace tail-handling` shows which token types fall below μ=128 (tail) and which exceed τ=256 (heavy), with their frequency counts
- [ ] AC-5.3: `trace damped-scoring` shows per-token-type variance (sⱼ) and weight (wⱼ = √nⱼ·sⱼ) before and after damping
- [ ] AC-5.4: `trace budget` shows ε=4 floor, θ=39 ceiling application, and the reconciliation step that redistributes leftover centroid budget
- [ ] AC-5.5: `inspect pq` shows the 3-level hierarchical PQ distance table layout: level dimensions, sub-quantizer count, code sizes
- [ ] AC-5.6: `bench tachiom` calls the tachiom binary (must be installed separately) and prints SIFT 1B recall@10 and QPS
- [ ] AC-5.7: Narrative explicitly positions TACHIOM: "PLAID uses one centroid set for all tokens; TAC gives each token type its own κⱼ centroids tuned to that token's distribution"

---

### US-6: Compare Engines Side by Side

**As a** developer who has walked through all five engines
**I want to** see a summary comparison of their indexing strategies and query pipelines
**So that** I can articulate the tradeoff each engine makes and choose the right one for a given workload

**Acceptance Criteria:**
- [ ] AC-6.1: `cargo demo compare` prints a structured table: engine, index structure, query pipeline stages, time complexity per query, memory per token
- [ ] AC-6.2: The table is also emitted as a JSON file to `output/compare.json` for use in downstream plotting
- [ ] AC-6.3: `cargo bench all` runs all benchmark suites and writes a unified `output/bench_results.json` with per-engine recall@k and latency percentiles
- [ ] AC-6.4: A `scripts/plot.py` renders recall@k vs. latency Pareto curves from `bench_results.json` using matplotlib, saving to `output/pareto.png`

---

### US-7: Scenario Authoring via TOML

**As a** contributor or instructor
**I want to** write a new engine walkthrough in TOML without modifying Rust source
**So that** the tool can be extended with new scenarios or adapted for different teaching contexts

**Acceptance Criteria:**
- [ ] AC-7.1: Each scenario file under `scenarios/` defines: `[meta]` (title, engine, description), `[[steps]]` (action, args, narration), and `[corpus]` (inline docs or path to file)
- [ ] AC-7.2: `cargo demo <name>` resolves `scenarios/<name>.toml`, validates the schema at startup, and aborts with a clear error if the file is malformed
- [ ] AC-7.3: Each step's `narration` field is printed to stdout before the step executes, clearly separated from engine output
- [ ] AC-7.4: A `--dry-run` flag prints all step narrations and actions without executing any engine code
- [ ] AC-7.5: An example scenario `scenarios/colbert.toml` is included and used as the reference for the schema

---

### US-8: Verification Harness Per Engine

**As a** developer contributing changes to an engine's educational implementation
**I want to** run a deterministic correctness check against a small known corpus
**So that** I can confirm my change did not break the educational implementation's retrieval accuracy

**Acceptance Criteria:**
- [ ] AC-8.1: Each engine has a `verify` module with a fixed corpus of 20–50 documents and ≥5 queries with ground-truth document IDs
- [ ] AC-8.2: Verification asserts recall@1 = 1.0 for all test queries
- [ ] AC-8.3: Verification asserts recall@10 ≥ 0.9 for all test queries
- [ ] AC-8.4: Verification asserts index output is byte-identical across two runs using the same fixed seed
- [ ] AC-8.5: Verification asserts p99 query latency < 100ms on the small corpus (CPU-only, single thread)
- [ ] AC-8.6: `cargo test --workspace` runs all engine verification modules and fails with a named assertion on the first failing criterion
- [ ] AC-8.7: For HNSW, verification uses mock embeddings (fixed vectors) rather than live Atlas calls, so the harness runs offline

---

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Cargo workspace with crates: `common`, `hnsw`, `colbert`, `plaid`, `warp`, `tachiom`, `bench`, `cli` | High | `cargo build --workspace` succeeds with no warnings |
| FR-2 | `common` crate exports: `TraceEvent`, `TraceLog`, `VizRepl`, `ScenarioRunner`, `PlotData`, `BenchResult` types | High | All engine crates import from `common` without duplication |
| FR-3 | `TraceEvent` has named variant per pipeline stage; `TraceLog` serializes to JSON | High | `trace` REPL command prints JSON-serializable event stream |
| FR-4 | REPL commands: `index <text>`, `query <text>`, `inspect [target]`, `trace [filter]`, `help`, `quit` | High | All commands parse, return structured output, and print usage on invalid input |
| FR-5 | REPL provides context-aware suggestions after each command (e.g., "try: query bank") | Medium | Suggestion text appears after each command output |
| FR-6 | HNSW engine connects to Atlas cluster via `MONGODB_URI` env var; all other engines are self-contained | High | Startup fails with clear error if `MONGODB_URI` is unset and engine is `hnsw` |
| FR-7 | ColBERT educational impl: tokenize → embed (fixed 128-dim random projection for demo, real model for bench) | High | MaxSim scores are computed correctly; verified by verification harness |
| FR-8 | PLAID educational impl extends ColBERT: centroid assignment, centroid ANN lookup, candidate expansion | High | Candidate set from centroid stage is a superset of ground truth top-10 on small corpus |
| FR-9 | WARP educational impl: Xtr gather stage then MaxSim refinement; wraps xtr-warp binary for bench mode | High | `bench warp` executes xtr-warp binary and parses its stdout into `BenchResult` |
| FR-10 | TACHIOM educational impl: all four TAC phases are implemented as separate Rust functions with matching parameter names (μ, τ, sⱼ, wⱼ, ε, θ, κⱼ) | High | `trace` shows each TAC phase output; verified by verification harness |
| FR-11 | TACHIOM bench mode wraps the tachiom binary; `bench tachiom` parses its output into `BenchResult` | High | `bench tachiom` exits 0 and writes to `output/bench_results.json` |
| FR-12 | Benchmark crate validates that SIFT files exist at `MULTIVECTOR_SIFT_PATH`; prints manual FTP download instructions if missing | Medium | `cargo bench check-sift` exits 0 if files are present, exits 1 with instructions if `MULTIVECTOR_SIFT_PATH` is unset or files are missing |
| FR-13 | Benchmark metrics: recall@1, recall@10, recall@100, p50/p95/p99 query latency, throughput (QPS), index build time, peak RAM | High | All metrics appear in `BenchResult` struct and in JSON output |
| FR-14 | `scripts/plot.py` reads `output/bench_results.json` and renders recall@k vs. latency Pareto PNG | Medium | Script runs with `python3 scripts/plot.py` and writes `output/pareto.png` |
| FR-15 | All scenario TOML files validated at load time against a typed schema | High | Malformed TOML prints field-level error and exits 1 |
| FR-16 | `--dry-run` flag on `cargo demo` suppresses engine execution, prints narrations only | Medium | No embeddings computed, no Atlas connection attempted in dry-run |

---

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Educational impl query latency (small corpus, CPU) | p99 latency | < 100ms per query |
| NFR-2 | REPL startup time | Wall time from command to first prompt | < 2s (excluding Atlas handshake) |
| NFR-3 | Atlas HNSW latency (small corpus) | p99 latency | < 500ms per query (network included) |
| NFR-4 | Index determinism | Byte equality across two runs, same seed | Must pass on CI with no flakiness |
| NFR-5 | Codebase compilation | `cargo build --workspace --release` | Completes in < 5 minutes on M-series Mac |
| NFR-6 | Test coverage | Verification harness line coverage | ≥ 80% per engine crate |
| NFR-7 | Dependency footprint | No large ML runtime (PyTorch, ONNX runtime) in Rust crates | `Cargo.lock` contains no `ort`, `tch`, `candle-*` for educational impls |
| NFR-8 | SIFT 1B benchmark RAM | Peak RSS during full SIFT index build | Documented in README with measured value |

---

## Glossary

- **MaxSim**: Scoring function in ColBERT; for each query token, take the maximum cosine similarity to any document token; sum across query tokens.
- **HNSW**: Hierarchical Navigable Small World graph; ANN data structure that supports O(log n) approximate nearest-neighbor search.
- **Late interaction**: Query and document are encoded independently; interaction (MaxSim) happens at query time, not encoding time.
- **Centroid pruning** (PLAID): Before MaxSim, each query token finds its nearest centroids; only documents assigned to those centroids are scored.
- **TAC**: Token-Aware Clustering; TACHIOM's approach of learning separate centroid sets per token type rather than one global centroid set.
- **Xtr gather** (WARP): Lightweight phase that uses Xtr scores to identify which token embeddings to fetch before MaxSim refinement.
- **PQ**: Product Quantization; compresses high-dimensional vectors into compact codes for fast approximate distance computation.
- **Hierarchical PQ** (TACHIOM): 3-level PQ layout where distance tables are structured to exploit token-type clustering.
- **μ (mu)**: TACHIOM tail threshold; token types with fewer than μ=128 occurrences are treated as tail tokens.
- **τ (tau)**: TACHIOM heavy threshold; token types with more than τ=256 occurrences get their own centroid budget.
- **sⱼ, wⱼ**: Per-token-type variance and damped weight in TACHIOM budget allocation; wⱼ = √nⱼ·sⱼ.
- **ε, θ**: Floor (ε=4) and ceiling (θ=39) bounds on per-token-type centroid count in TACHIOM.
- **κⱼ**: Per-token-type centroid count in TACHIOM after budget reconciliation.
- **TraceEvent**: Named struct emitted by each engine pipeline stage; carries stage name, payload, and timestamp.
- **VizRepl**: Shared REPL scaffolding in `common` crate; handles command parsing, suggestion printing, and output formatting.
- **ScenarioRunner**: Component that reads a TOML scenario file and executes steps sequentially, printing narrations between steps.
- **SIFT 1B**: Standard ANN benchmark dataset: 1 billion 128-dim float vectors derived from SIFT image descriptors.

---

## Out of Scope

- GPU acceleration for any educational implementation (CPU-only is a hard constraint for verification harness portability)
- Training or fine-tuning embedding models (use pre-trained voyage-4-large via API for HNSW; fixed random projection for educational ColBERT/PLAID/WARP/TACHIOM impls)
- A web UI or browser-based visualization (terminal-only)
- Multi-user or networked REPL sessions
- Persistent index storage between sessions for educational impls (in-memory only; HNSW/Atlas is the exception)
- Streaming or real-time document ingestion pipelines
- Evaluation on datasets other than SIFT 1B in the benchmark suite (additional datasets are future work)
- Production-hardened implementations of ColBERT, PLAID, WARP, or TACHIOM (educational impls are for understanding, not deployment)
- Automatic download or build of the xtr-warp or tachiom binaries (user installs them separately; `bench` mode checks for their presence and fails with a clear message)
- Windows support (macOS and Linux only)

---

## Dependencies

| Dependency | Type | Notes |
|------------|------|-------|
| `MONGODB_URI` env var pointing to an Atlas cluster | Runtime (HNSW only) | Must support Atlas Vector Search; collection schema specified in design doc |
| voyage-4-large embedding API | Runtime (HNSW only) | Called via Atlas; not called during educational ColBERT/PLAID/WARP/TACHIOM demos |
| xtr-warp binary on PATH | Runtime (WARP bench only) | Installed by user from https://github.com/jlscheerer/xtr-warp |
| tachiom binary on PATH | Runtime (TACHIOM bench only) | Installed by user from https://github.com/TusKANNy/tachiom |
| Python 3 with matplotlib | Dev/bench | Required only for `scripts/plot.py`; not a Rust dependency |
| SIFT 1B dataset | Bench | ~100GB; fetched by `cargo bench download-sift`; not committed to repo |
| Rust stable toolchain | Build | Minimum version TBD in design doc based on dependencies |

---

## Success Criteria

- A developer with no prior ColBERT knowledge can run `cargo demo colbert` and correctly explain MaxSim scoring to a peer within 20 minutes
- All five `cargo demo <engine>` scenarios complete end-to-end without errors on a fresh macOS checkout (HNSW requires Atlas URI)
- `cargo test --workspace` passes with recall assertions holding on every engine's verification corpus
- `cargo bench all` produces a `bench_results.json` with valid recall@10 and p99 latency entries for all five engines (requires xtr-warp and tachiom binaries)
- The `scripts/plot.py` Pareto chart shows a clear frontier with TACHIOM at higher recall and lower latency than PLAID for large-scale settings

---

## Unresolved Questions

1. **Embedding model for educational ColBERT/PLAID/WARP/TACHIOM demos**: Using a fixed random projection keeps the demo offline and deterministic, but the vectors are not semantically meaningful — users may be confused when "bank" and "river" end up near each other. Alternative: use a real tokenizer + a tiny pre-trained model (e.g., `bert-base-uncased` via `candle`). Decision needed before design starts.

2. **Atlas collection schema**: Which Atlas collection name, index name, and field names does the HNSW engine use? This needs to be fixed so TOML scenarios reference consistent names.

3. **ColBERT token embedding dimensionality**: Real ColBERTv2 uses 128-dim. Should educational impl match (128-dim random projection) or use a smaller dim (e.g., 16 or 32) to make viz output readable? Affects NFR-7 (no ONNX runtime) and AC-2.2 (3-dim truncation for viz).

4. **xtr-warp and tachiom binary interfaces**: What are the exact CLI flags and stdout formats for both binaries? The WARP and TACHIOM bench parsers depend on this. If the formats change between binary releases, the parser breaks silently.

5. **Scenario corpus**: Should each scenario use its own tiny corpus (e.g., 10 sentences about rivers and banks) or a shared corpus across all engines? A shared corpus lets users see the same document ranked differently by each engine, which is more educational but complicates TOML authoring.

6. **SIFT 1B download source and licensing**: The original SIFT 1B URL (`ftp://ftp.irisa.fr/...`) is unreliable. Should the bench crate use a mirror, or support a user-supplied path? Affects FR-12.

---

## Next Steps

1. Resolve Unresolved Questions 1 (embedding model choice) and 3 (token dim) — these gate the ColBERT educational impl design
2. Produce `design.md`: workspace structure, crate interfaces, `TraceEvent` schema, REPL command dispatch, TOML schema, Atlas collection spec
3. Produce `tasks.md`: ordered implementation tasks with file-level targets and test stubs
4. Confirm xtr-warp and tachiom binary CLI interfaces (UQ-4) before designing the bench parsers
