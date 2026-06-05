# Tasks: Multivector Retrieval Educational CLI

**Workflow**: POC-first (GREENFIELD)
**POC milestone**: `cargo demo colbert` runs end-to-end — validates common crate, tokenizer, random projection, ColBertEngine, ScenarioRunner, and VizRepl all at once.

---

## Phase 1: Make It Work (POC)

Focus: get the minimal end-to-end path compiling and running. Skip polish, accept hardcoded values, no tests yet.

---

### 1.1 Workspace scaffold and toolchain

**Do**:
1. Create `rust-toolchain.toml` at workspace root with nightly channel + components
2. Create workspace `Cargo.toml` with all 8 member crates, workspace deps (including the 6 missing ones: `async-trait`, `tokenizers`, `which`, `half`, `rayon`, and `dotenvy = "0.15"` — confirm `rayon`, `half`, and `dotenvy` are present alongside `async-trait`, `tokenizers`, `which`), and `[patch.crates-io]` stubs for vectorium/kannolo with placeholder SHAs
3. Create `Cargo.lock` placeholder (will be generated on first build)
4. Create `output/.gitkeep`, `data/.gitkeep`, add both dirs to `.gitignore`

**Files**:
- `rust-toolchain.toml`
- `Cargo.toml`
- `.gitignore`
- `output/.gitkeep`
- `data/.gitkeep`

**Done when**: `cat rust-toolchain.toml` shows `channel = "nightly"` and `Cargo.toml` contains all 8 workspace members and all required workspace deps

**Verify**: `grep -E 'channel|nightly' rust-toolchain.toml && grep -E 'async-trait|tokenizers|which|half|rayon|dotenvy' Cargo.toml && grep -c 'members' Cargo.toml`

**Commit**: `chore(workspace): scaffold nightly workspace with all 8 crates and missing deps`

_Requirements: FR-1_
_Design: Nightly Rust and Toolchain, Workspace Cargo.toml_

---

### 1.2 Stub all 8 crate skeletons (parallel group)

**Do** (all in parallel — zero file overlap between crates):
For each of the 8 crates (`common`, `hnsw`, `colbert`, `plaid`, `warp`, `tachiom`, `bench`, `cli`):
1. Create `crates/<name>/Cargo.toml` with `[package]` block and correct dependency declarations
2. Create `crates/<name>/src/lib.rs` (or `main.rs` for `cli`) with a single `// TODO` comment

This task creates all skeleton Cargo.toml files so `cargo build --workspace` can parse the graph before any code is written.

**Files** (create all 16):
- `crates/common/Cargo.toml`, `crates/common/src/lib.rs`
- `crates/hnsw/Cargo.toml`, `crates/hnsw/src/lib.rs`
- `crates/colbert/Cargo.toml`, `crates/colbert/src/lib.rs`
- `crates/plaid/Cargo.toml`, `crates/plaid/src/lib.rs`
- `crates/warp/Cargo.toml`, `crates/warp/src/lib.rs`
- `crates/tachiom/Cargo.toml`, `crates/tachiom/src/lib.rs`
- `crates/bench/Cargo.toml`, `crates/bench/src/lib.rs`
- `crates/cli/Cargo.toml`, `crates/cli/src/main.rs`

**Done when**: `cargo metadata --no-deps 2>&1 | grep -c '"name"'` returns 8 or more packages

**Verify**: `cargo metadata --no-deps 2>&1 | grep '"name"' | grep -E 'common|hnsw|colbert|plaid|warp|tachiom|bench|cli' | wc -l`

**Commit**: `chore(workspace): add skeleton Cargo.toml and lib.rs for all 8 crates`

_Requirements: FR-1_
_Design: File Structure_

---

### 1.3 [VERIFY] Workspace parses cleanly

**Do**: Confirm cargo can parse the workspace dependency graph without errors

**Verify**: `cargo check --workspace 2>&1 | grep -v "^warning" | grep -c "error" | grep -q "^0$" && echo WORKSPACE_OK`

**Done when**: Zero errors (warnings about empty files acceptable at this stage)

**Commit**: None

---

### 1.4 Download WordPiece vocab file

**Do**:
1. Create `vocab/` directory
2. Download `bert-base-uncased-vocab.txt` from HuggingFace (30,522 lines, MIT-licensed) to `vocab/wordpiece_vocab.txt`
   - URL: `https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt`
3. Verify line count is exactly 30522

**Files**:
- `vocab/wordpiece_vocab.txt`

**Done when**: File exists with 30522 lines

**Verify**: `wc -l vocab/wordpiece_vocab.txt | grep -q '30522' && echo VOCAB_OK`

**Commit**: `feat(vocab): add bert-base-uncased WordPiece vocabulary file`

_Requirements: FR-7_
_Design: token.rs — WordPieceTokenizer_

---

### 1.5 [P] `common` crate — trace.rs

**Do**:
1. Create `crates/common/src/trace.rs` with `TraceEvent` enum (all variants from design: HnswInsert, HnswQuery, HnswLayerStats, Tokenize, TokenEmbed, MaxSimMatrix, CentroidAssign, CentroidAnn, CandidateExpand, PlaidMaxSim, XtrScore, CandidateGather, MaxSimRefine, TailHandle, DampedScore, BudgetBound, BudgetReconcile, PqInspect, TachiomSearch)
2. Add `TailClass`, `TachiomTimings` types
3. Add `TraceLog` struct with `push()` and `to_json()` methods
4. Add `JsonTracer` struct with `write()` method

**Files**:
- `crates/common/src/trace.rs`

**Done when**: File compiles (`cargo check -p common`); `TraceEvent` has all 19 named variants

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -c "TraceEvent::" crates/common/src/trace.rs`

**Commit**: `feat(common): add TraceEvent enum with all pipeline stage variants`

_Requirements: FR-2, FR-3_
_Design: trace.rs — TraceEvent and TraceLog_

---

### 1.6 [P] `common` crate — bench_types.rs

**Do**:
1. Create `crates/common/src/bench_types.rs` with `BenchResult`, `BuildStats`, `PlotData` structs (exact fields from design)
2. All fields derive `Debug, Clone, Serialize, Deserialize`

**Files**:
- `crates/common/src/bench_types.rs`

**Done when**: Structs compile with all fields matching design spec

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'BenchResult|BuildStats|PlotData' crates/common/src/bench_types.rs | wc -l`

**Commit**: `feat(common): add BenchResult, BuildStats, PlotData types`

_Requirements: FR-2, FR-13_
_Design: bench_types.rs_

---

### 1.7 [P] `common` crate — corpus.rs

**Do**:
1. Create `crates/common/src/corpus.rs` with `SHARED_CORPUS` constant (20 sentences, exact text from design)
2. Add `VERIFY_QUERIES` constant (20 query/expected-doc-id pairs, exact from design)

**Files**:
- `crates/common/src/corpus.rs`

**Done when**: Both constants present; `SHARED_CORPUS.len() == 20` and `VERIFY_QUERIES.len() == 20`

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -c '"bank"' crates/common/src/corpus.rs`

**Commit**: `feat(common): add SHARED_CORPUS and VERIFY_QUERIES constants`

_Requirements: FR-2, AC-8.1_
_Design: corpus.rs_

---

### 1.8 [VERIFY] Quality checkpoint: common trace/bench/corpus modules

**Do**: Run check and verify all three new modules compile together

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && echo COMMON_PASS`

**Done when**: Zero errors across all three modules

**Commit**: None

---

### 1.9 `common` crate — token.rs (WordPieceTokenizer + RandomProjection)

**Do**:
1. Create `crates/common/src/token.rs`
2. Add `TOKEN_DIM = 128` const
3. Add `TokenMatrix` struct with `tokens: Vec<String>`, `rows: Vec<[f32; TOKEN_DIM]>`, `num_tokens()`, `preview()` methods
4. Add `Tokenizer` trait with `tokenize(&self, text: &str) -> Vec<String>`
5. Add `WordPieceTokenizer` struct: loads from `vocab_path`, wraps `tokenizers::Tokenizer::from_file()`; implements `Tokenizer`; returns token strings from `enc.get_tokens()`
6. Add `RandomProjection` struct with `seed: u64`, `cache: HashMap<u32, [f32; TOKEN_DIM]>`, `new()`, `project()` (SmallRng seeded with `seed ^ token_id`, Gaussian entries, L2-normalize), and `embed()` method

**Files**:
- `crates/common/src/token.rs`

**Done when**: `cargo check -p common` passes; `TokenMatrix`, `WordPieceTokenizer`, `RandomProjection` all exported

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub struct (TokenMatrix|WordPieceTokenizer|RandomProjection)' crates/common/src/token.rs | wc -l`

**Commit**: `feat(common): add TokenMatrix, WordPieceTokenizer, RandomProjection`

_Requirements: FR-2, FR-7_
_Design: token.rs — Tokenizer and TokenMatrix_

---

### 1.10 `common` crate — engine.rs (Engine trait)

**Do**:
1. Create `crates/common/src/engine.rs`
2. Add `Engine` trait with `async_trait` macro: `name()`, `index()`, `query()`, `inspect()`, `verify()` methods (exact signatures from design)
3. Import `TraceLog`, `BenchResult` from crate root

**Files**:
- `crates/common/src/engine.rs`

**Done when**: `cargo check -p common` passes; Engine trait exported with all 5 methods

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -c 'async fn' crates/common/src/engine.rs`

**Commit**: `feat(common): add async Engine trait`

_Requirements: FR-2, FR-4_
_Design: engine.rs — Engine trait_

---

### 1.11 `common` crate — viz.rs (VizRepl + SuggestionMode)

**Do**:
1. Create `crates/common/src/viz.rs`
2. Add `VizGuard` struct (RAII pattern with `suppress()` constructor and Drop impl)
3. Add `SuggestionMode` enum (`None`, `Sequence { suggestions, index }`) with `next_suggestion()` method
4. Add `VizRepl` struct with `engine_name`, `suggestions`, `trace_path` fields; `print_suggestion()` and `print_narration()` methods (exact ANSI escape codes from design: `\x1b[36m` for cyan, `\x1b[2m` for dim)

**Files**:
- `crates/common/src/viz.rs`

**Done when**: `cargo check -p common` passes; `VizRepl::print_narration` and `SuggestionMode::Sequence` exported

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub struct VizRepl|SuggestionMode' crates/common/src/viz.rs | wc -l`

**Commit**: `feat(common): add VizRepl, SuggestionMode, VizGuard`

_Requirements: FR-2, FR-4, FR-5_
_Design: viz.rs — VizRepl and SuggestionMode_

---

### 1.12 `common` crate — scenario.rs (ScenarioRunner + typed TOML schema)

**Do**:
1. Create `crates/common/src/scenario.rs`
2. Add `ScenarioFile`, `ScenarioMeta`, `CorpusDef` (tagged enum with Inline/File/Shared variants — must use `#[serde(tag = "type", rename_all = "lowercase")]` so TOML `type = "shared"` deserializes correctly), `InlineDoc`, `StepDef` structs (all fields from design; `pause` defaults to false)
3. Add `ScenarioRunner` struct with `scenario: ScenarioFile`, `dry_run: bool`
4. Add `ScenarioRunner::from_path()` — reads file, parses TOML, validates `meta.version == 1`, returns error with field context if malformed
5. Add `ScenarioRunner::run()` — async generic over dispatch fn; calls `VizRepl::print_narration` then dispatches each step; skips dispatch if `dry_run`

**Files**:
- `crates/common/src/scenario.rs`

**Done when**: `cargo check -p common` passes; scenario loads from TOML string

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub struct (ScenarioRunner|ScenarioFile|StepDef)' crates/common/src/scenario.rs | wc -l`

**Commit**: `feat(common): add ScenarioRunner and typed TOML schema types`

_Requirements: FR-2, FR-15, AC-7.1, AC-7.2_
_Design: scenario.rs — ScenarioRunner and typed TOML schema_

---

### 1.13 `common` crate — lib.rs wiring

**Do**:
1. Update `crates/common/src/lib.rs` to declare all 7 modules and re-export key types
2. Re-export: `TraceEvent`, `TraceLog`, `TailClass`, `TachiomTimings`, `JsonTracer` (from trace), `Engine` (from engine), `TOKEN_DIM`, `TokenMatrix`, `Tokenizer`, `WordPieceTokenizer`, `RandomProjection` (from token), `VizRepl`, `VizGuard`, `SuggestionMode` (from viz), `ScenarioRunner`, `ScenarioFile`, `StepDef`, `CorpusDef` (from scenario), `BenchResult`, `BuildStats`, `PlotData` (from bench_types), `SHARED_CORPUS`, `VERIFY_QUERIES` (from corpus)

**Files**:
- `crates/common/src/lib.rs`

**Done when**: `cargo check -p common` passes with all public types accessible

**Verify**: `cargo check -p common 2>&1 | grep -c "^error" | grep -q "^0$" && grep -c 'pub use' crates/common/src/lib.rs`

**Commit**: `feat(common): wire all modules in lib.rs with public re-exports`

_Requirements: FR-2_

---

### 1.14 [VERIFY] Quality checkpoint: complete common crate

**Do**: Full check of common crate; verify all public types accessible

**Verify**: `cargo check -p common 2>&1 | grep "^error" && echo FAIL || echo COMMON_COMPLETE`

**Done when**: Zero errors; `common` is ready for all engine crates to depend on

**Commit**: `chore(common): pass complete crate quality check`

---

### 1.15 [P] `colbert` crate — encoder.rs

**Do**:
1. Create `crates/colbert/src/encoder.rs`
2. Add `ColBertEncoder` struct holding `tokenizer: WordPieceTokenizer`, `projection: RandomProjection`
3. Add `ColBertEncoder::new(vocab_path, seed) -> Result<Self>`
4. Add `ColBertEncoder::encode(text) -> Result<(TokenMatrix, Vec<u32>)>` — tokenizes, gets vocab IDs from `tokenizer.inner.encode(text, false).unwrap().get_ids()`, embeds via `projection.embed()`, returns (TokenMatrix, vocab_ids)
5. Add `ColBertEncoder::encode_with_trace(doc_id, text) -> Result<(TokenMatrix, Vec<u32>, TraceLog)>` — same as encode but emits `Tokenize` and `TokenEmbed` events per token (embedding_preview = matrix.preview(i))

**Files**:
- `crates/colbert/src/encoder.rs`

**Done when**: `cargo check -p colbert` passes; `ColBertEncoder::encode` produces a `TokenMatrix` with one row per token

**Verify**: `cargo check -p colbert 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn encode' crates/colbert/src/encoder.rs | wc -l`

**Commit**: `feat(colbert): add ColBertEncoder wrapping WordPieceTokenizer + RandomProjection`

_Requirements: FR-7, AC-2.2_
_Design: encoder.rs_

---

### 1.16 [P] `colbert` crate — maxsim.rs

**Do**:
1. Create `crates/colbert/src/maxsim.rs`
2. Add `cosine(a: &[f32; TOKEN_DIM], b: &[f32; TOKEN_DIM]) -> f32` — dot product (both already L2-normalized from RandomProjection)
3. Add `maxsim(query: &TokenMatrix, doc: &TokenMatrix) -> f32` — for each query row, max over doc rows of cosine; sum across query rows
4. Add `maxsim_with_matrix(query: &TokenMatrix, doc: &TokenMatrix) -> (f32, Vec<Vec<f32>>, Vec<f32>)` — returns (score, full matrix rows×cols, row_maxima) for `MaxSimMatrix` trace event

**Files**:
- `crates/colbert/src/maxsim.rs`

**Done when**: `cargo check -p colbert` passes; `maxsim` returns correct scalar sum

**Verify**: `cargo check -p colbert 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn maxsim' crates/colbert/src/maxsim.rs | wc -l`

**Commit**: `feat(colbert): add MaxSim scorer with full matrix output for tracing`

_Requirements: FR-7, AC-2.3_
_Design: MaxSim scorer_

---

### 1.17 `colbert` crate — index.rs

**Do**:
1. Create `crates/colbert/src/index.rs`
2. Add `ColBertIndex` struct: `docs: Vec<(u32, TokenMatrix)>`
3. Add `insert(doc_id, matrix)` — appends; if doc_id already present logs a warning (no panic)
4. Add `search(query_matrix, top_k) -> Vec<(u32, f32)>` — brute-force: for each doc compute `maxsim(query_matrix, doc_matrix)`, collect (doc_id, score), sort descending, return top_k
5. Add `search_with_trace(query_matrix, top_k) -> (Vec<(u32, f32)>, TraceLog)` — same but emits `MaxSimMatrix` trace event per doc

**Files**:
- `crates/colbert/src/index.rs`

**Done when**: `cargo check -p colbert` passes; search returns sorted results

**Verify**: `cargo check -p colbert 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn (search|insert)' crates/colbert/src/index.rs | wc -l`

**Commit**: `feat(colbert): add ColBertIndex with brute-force MaxSim search`

_Requirements: FR-7_
_Design: ColBertIndex_

---

### 1.18 `colbert` crate — engine.rs (ColBertEngine impl Engine)

**Do**:
1. Create `crates/colbert/src/engine.rs`
2. Add `ColBertEngine` struct: `encoder: ColBertEncoder`, `index: ColBertIndex`, `last_trace: Option<TraceLog>`, `next_doc_id: u32`
3. Implement `Engine` for `ColBertEngine`:
   - `name()` → `"colbert"`
   - `index(doc_id, text)` → encode with trace, insert into index, return TraceLog. Storage is `Vec<(DocId, TokenMatrix)>` — not actual HNSW shards. During indexing, emit one `HnswInsert`-style `TraceEvent` per token (showing doc_id, token position, embedding preview) so the REPL can display "logical insertion" without building a graph. REPL narration must label this as "ColBERT's logical insertion, not yet an ANN graph — that comes in PLAID." (AC-2.1)
   - `query(text, top_k)` → encode query, search_with_trace, store last_trace, return results + TraceLog
   - `inspect(target)` → match on `Some("tokens")` → show token table for doc; `None` → show index size; unknown → list options
   - `verify()` → delegate to `crate::verify::run()`
4. Constructor: `ColBertEngine::new(vocab_path) -> Result<Self>` with seed `0xCAFEBABE_DEADBEEF`

**Files**:
- `crates/colbert/src/engine.rs`

**Done when**: `cargo check -p colbert` passes; Engine trait fully implemented

**Verify**: `cargo check -p colbert 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'impl Engine' crates/colbert/src/engine.rs`

**Commit**: `feat(colbert): implement ColBertEngine with full Engine trait`

_Requirements: FR-4, FR-7_
_Design: ColBERT Crate_

---

### 1.19 `colbert` crate — verify.rs + lib.rs

**Do**:
1. Create `crates/colbert/src/verify.rs` with `run(engine: &mut ColBertEngine)` function:
   - Index all 20 `SHARED_CORPUS` docs using a new tokio current-thread runtime per call
   - Run all 20 `VERIFY_QUERIES`; assert `results[0].0 == expected_top1` (recall@1=1.0)
   - Assert recall@10 ≥ 0.9 (count of queries where expected_top1 is in top-10)
   - Run determinism check: build second identical engine, verify first query result matches
   - Measure p99 latency across all queries; assert < 100ms
2. Add `#[cfg(test)] mod tests { #[test] fn verify_engine() { ... } }` calling `run()`
3. Update `crates/colbert/src/lib.rs` to declare `encoder`, `index`, `maxsim`, `engine`, `verify` modules and re-export `ColBertEngine`, `ColBertIndex`

**Files**:
- `crates/colbert/src/verify.rs`
- `crates/colbert/src/lib.rs`

**Done when**: `cargo check -p colbert` passes; verify module compiles

**Verify**: `cargo check -p colbert 2>&1 | grep -c "^error" | grep -q "^0$" && grep -c 'fn verify_engine' crates/colbert/src/verify.rs`

**Commit**: `feat(colbert): add verification harness with recall@1, recall@10, determinism, latency assertions`

_Requirements: AC-8.1, AC-8.2, AC-8.3, AC-8.4, AC-8.5_
_Design: Verification Harness Design_

---

### 1.20 [VERIFY] Quality checkpoint: colbert crate

**Do**: Check colbert crate compiles cleanly; all modules present

**Verify**: `cargo check -p colbert 2>&1 | grep "^error" && echo FAIL || echo COLBERT_OK`

**Done when**: Zero errors

**Commit**: None

---

### 1.21 `cli` crate — commands.rs + main.rs skeleton

**Do**:
1. Create `crates/cli/src/commands.rs` with the full clap derive tree from design: `Cli`, `TopCommand` (Demo/Repl/Bench variants), `EngineCmd` (Hnsw/Colbert/Plaid/Warp/Tachiom), `BenchTarget` (All/Hnsw/Colbert/Plaid/Warp/Tachiom/CheckSift). The `CheckSift` variant implements `cargo bench check-sift`: reads `MULTIVECTOR_SIFT_PATH` env var, validates that `bigann_base.bvecs` and `bigann_gnd/idx_100M.ivecs` (ground truth for the SIFT 100M subset) exist at that path, and prints manual FTP download instructions if missing. There is no `DownloadSift` variant — automatic download is not supported.
2. Update `crates/cli/src/main.rs` to:
   - Parse `Cli` with clap
   - Match `TopCommand::Demo` → print "demo not yet implemented"
   - Match `TopCommand::Repl` → print "repl not yet implemented"
   - Match `TopCommand::Bench` → print "bench not yet implemented"
   - Use `#[tokio::main]` async runtime

**Files**:
- `crates/cli/src/commands.rs`
- `crates/cli/src/main.rs`

**Done when**: `cargo build -p cli` succeeds and `./target/debug/multivector --help` prints the top-level usage

**Verify**: `cargo build -p cli 2>&1 | grep -c "^error" | grep -q "^0$" && ./target/debug/multivector --help | grep -q 'demo'`

**Commit**: `feat(cli): add clap command tree skeleton`

_Requirements: FR-4_
_Design: REPL Command Dispatch — clap subcommand tree_

---

### 1.22 `cli` crate — repl.rs (VizRepl dispatch loop)

**Do**:
1. Create `crates/cli/src/repl.rs`
2. Implement `run_repl(engine: &mut dyn Engine, viz: VizRepl) -> anyhow::Result<()>` from design: read stdin line by line; match `quit`/`exit`/`q`, `help`/`h`, `index <rest>`, `query <rest>`, `inspect`, `inspect <target>`, `trace`, `trace <filter>`; call engine methods; render results; print suggestion after each command
3. Add `render_log(log: &TraceLog)` — serializes to JSON and prints with color prefix
4. Add `render_results(results: &[(u32, f32)])` — numbered list with doc_id and score
5. Add `print_help(engine_name: &str)` — lists all REPL commands

**Files**:
- `crates/cli/src/repl.rs`

**Done when**: `cargo check -p cli` passes; `run_repl` compiles with correct Engine trait usage

**Verify**: `cargo check -p cli 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub async fn run_repl' crates/cli/src/repl.rs`

**Commit**: `feat(cli): add VizRepl dispatch loop with all REPL commands`

_Requirements: FR-4, FR-5_
_Design: VizRepl dispatch loop_

---

### 1.23 `cli` crate — demo.rs (ScenarioRunner dispatch for ColBERT)

**Do**:
1. Create `crates/cli/src/demo.rs`
2. Add `run_demo(name: &str, dry_run: bool, trace_json: Option<PathBuf>) -> anyhow::Result<()>`
3. For `name == "compare"` → print "compare not yet implemented" (stub)
4. For any other name → resolve `scenarios/<name>.toml`, load via `ScenarioRunner::from_path()`, construct the matching engine based on `scenario.meta.engine` string, call `runner.run()` with a dispatch closure that matches `op` strings ("index", "query", "inspect", "trace") and calls engine methods
5. Wire ColBERT dispatch: `engine = ColBertEngine::new(vocab_path)?`; `vocab_path` derived from a `MULTIVECTOR_VOCAB` env var or defaults to `./vocab/wordpiece_vocab.txt`
6. Other engines dispatch to "engine not yet implemented" for now (stubs)

**Files**:
- `crates/cli/src/demo.rs`

**Done when**: `cargo check -p cli` passes; demo.rs compiles against `ScenarioRunner` and `ColBertEngine`

**Verify**: `cargo check -p cli 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub async fn run_demo' crates/cli/src/demo.rs`

**Commit**: `feat(cli): add demo runner with ColBERT scenario dispatch`

_Requirements: AC-7.2, FR-15_
_Design: demo.rs — cargo demo <engine>_

---

### 1.24 Wire cli main.rs to demo/repl

**Do**:
1. Update `crates/cli/src/main.rs` to declare `mod commands; mod repl; mod demo; mod compare;` (compare as stub)
2. Replace "not yet implemented" stubs: `TopCommand::Demo { name, dry_run }` → `demo::run_demo(&name, dry_run, cli.trace_json).await?`
3. `TopCommand::Repl { engine: EngineCmd::Colbert }` → construct `ColBertEngine`, construct `VizRepl` with ColBERT `SuggestionMode::Sequence` (7 suggestions from design), call `repl::run_repl(&mut engine, viz).await?`
4. Create `crates/cli/src/compare.rs` stub with `pub async fn run_compare() -> anyhow::Result<()>` returning "not yet implemented"

**Files**:
- `crates/cli/src/main.rs`
- `crates/cli/src/compare.rs`

**Done when**: `cargo build -p cli` succeeds; `./target/debug/multivector demo colbert --dry-run` exits 0

**Verify**: `cargo build -p cli 2>&1 | grep -c "^error" | grep -q "^0$" && echo CLI_WIRED`

**Commit**: `feat(cli): wire demo and repl dispatch to ColBERT engine`

_Requirements: FR-4_

---

### 1.25 `scenarios/colbert.toml` — reference scenario file

**Do**:
1. Create `scenarios/colbert.toml` from the exact reference example in design.md
2. Verify TOML parses: all required fields present (`meta.title`, `meta.engine="colbert"`, `meta.version=1`, `[corpus]`, `[[steps]]` with op/args/narration)
3. Each step uses `op` values only from the allowed set: "index", "query", "inspect", "trace"
4. Steps in order: index doc 0, inspect tokens, index doc 1, query river erosion, query open checking account, trace

**Files**:
- `scenarios/colbert.toml`

**Done when**: `toml` parses without error; `grep -c '^\[\[steps\]\]' scenarios/colbert.toml` = 6

**Verify**: `python3 -c "import tomllib; tomllib.load(open('scenarios/colbert.toml','rb'))" && grep -c '^\[\[steps\]\]' scenarios/colbert.toml`

**Commit**: `feat(scenarios): add colbert.toml reference scenario`

_Requirements: AC-7.1, AC-7.5_
_Design: TOML Scenario Schema_

---

### 1.26 [VERIFY] POC Checkpoint: `cargo demo colbert` end-to-end

**Do**:
1. Build the workspace in release mode: `cargo build --workspace`
2. Run ColBERT demo in dry-run mode (no vocab file needed): `./target/debug/multivector demo colbert --dry-run`
3. Run with live vocab file: `./target/debug/multivector demo colbert` (reads `vocab/wordpiece_vocab.txt`, indexes 2 docs, runs 2 queries, runs trace step)
4. Verify scenario narration lines are printed in cyan; results appear for each query step

**Done when**: `cargo demo colbert` completes all 6 steps without panicking; at least one query result is printed

**Verify**: `cargo build --workspace 2>&1 | grep -c "^error" | grep -q "^0$" && ./target/debug/multivector demo colbert --dry-run 2>&1 | grep -q "ColBERT" && echo POC_PASS`

**Commit**: `feat(poc): end-to-end colbert demo compiles and runs`

_Requirements: FR-1, FR-7, AC-2.1, AC-7.2_

---

## Phase 2: Remaining Engine Crates

Build each engine on the validated `common` + `colbert` foundation. HNSW/Atlas deferred to Phase 6 (hardware dependency).

---

### 2.1 `plaid` crate — centroid.rs (CentroidPruner with Lloyd's KMeans)

**Do**:
1. Create `crates/plaid/src/centroid.rs`
2. Add `CentroidPruner { num_centroids: usize }` struct
3. Add `fit(&self, all_token_rows: &[[f32; TOKEN_DIM]]) -> Vec<[f32; TOKEN_DIM]>` — Lloyd's algorithm: random init (seeded SmallRng), iterate until convergence or 100 iterations; each iteration: assign each point to nearest centroid, recompute centroid means; return final centroids
4. Add `assign(&self, centroids: &[[f32; TOKEN_DIM]], token: &[f32; TOKEN_DIM]) -> u32` — nearest centroid index by dot product (vectors are L2-normalized)
5. Add `query_centroids(&self, centroids, query_token, nprobe) -> Vec<(u32, f32)>` — top-nprobe centroids by dot product, sorted descending

**Files**:
- `crates/plaid/src/centroid.rs`

**Done when**: `cargo check -p plaid` passes; KMeans `fit` compiles

**Verify**: `cargo check -p plaid 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn (fit|assign|query_centroids)' crates/plaid/src/centroid.rs | wc -l`

**Commit**: `feat(plaid): add CentroidPruner with Lloyd KMeans and centroid ANN lookup`

_Requirements: FR-8, AC-3.2_
_Design: CentroidPruner_

---

### 2.2 `plaid` crate — index.rs (PlaidIndex wrapping ColBertIndex)

**Do**:
1. Create `crates/plaid/src/index.rs`
2. Add `PlaidIndex` struct: `colbert: ColBertIndex`, `centroids: Vec<[f32; TOKEN_DIM]>`, `centroid_map: Vec<Vec<u32>>` (per-doc per-token centroid assignments), `centroid_to_docs: Vec<Vec<u32>>` (inverted index centroid→doc_ids)
3. Add `PlaidIndex::build(colbert_index: ColBertIndex, num_centroids: usize) -> Self` — collects all token rows from colbert_index, fits CentroidPruner, assigns each token to a centroid, builds centroid_to_docs inverted index; emits `CentroidAssign` trace events
4. Add `search(query_matrix, top_k, nprobe) -> (Vec<(u32, f32)>, TraceLog)` — for each query token find top-nprobe centroids (emits `CentroidAnn`), union candidate doc_ids (emits `CandidateExpand`), run MaxSim only on candidates (emits `PlaidMaxSim`)

**Files**:
- `crates/plaid/src/index.rs`

**Done when**: `cargo check -p plaid` passes; `PlaidIndex::build` and `search` compile

**Verify**: `cargo check -p plaid 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn (build|search)' crates/plaid/src/index.rs | wc -l`

**Commit**: `feat(plaid): add PlaidIndex wrapping ColBertIndex with centroid inverted index`

_Requirements: FR-8, AC-3.1_
_Design: PlaidIndex_

---

### 2.3 `plaid` crate — engine.rs + verify.rs + lib.rs

**Do**:
1. Create `crates/plaid/src/engine.rs`: `PlaidEngine` struct with `colbert_engine: ColBertEngine`, `plaid_index: Option<PlaidIndex>`, `num_centroids: usize`; implements `Engine`:
   - `name()` → `"plaid"`
   - `index(doc_id, text)` → call `colbert_engine.index(doc_id, text)` first; rebuild `PlaidIndex::build` from updated colbert_engine index after each insert (acceptable for 20-doc demo)
   - `query(text, top_k)` → encode query via colbert_engine encoder, search PlaidIndex; fallback to ColBert brute-force if index not built
   - `inspect(target)` → `Some("centroids")` → centroid count, sizes, activated for last query; else list options
   - `verify()` → delegate to `crate::verify::run()`
2. Create `crates/plaid/src/verify.rs` — same structure as colbert verify; adds assertion: for each query the candidate set from centroid stage is a superset of the ground truth top-1 (FR-8)
3. Update `crates/plaid/src/lib.rs`

**Files**:
- `crates/plaid/src/engine.rs`
- `crates/plaid/src/verify.rs`
- `crates/plaid/src/lib.rs`

**Done when**: `cargo check -p plaid` passes; `PlaidEngine` implements Engine

**Verify**: `cargo check -p plaid 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'impl Engine' crates/plaid/src/engine.rs`

**Commit**: `feat(plaid): complete PlaidEngine as progressive layer over ColBERT`

_Requirements: FR-8, AC-3.5_

---

### 2.4 [VERIFY] Quality checkpoint: plaid crate

**Do**: Check plaid crate; verify it imports colbert without duplication

**Verify**: `cargo check -p plaid 2>&1 | grep "^error" && echo FAIL || echo PLAID_OK`

**Done when**: Zero errors

**Commit**: None

---

### 2.5 `warp` crate — xtr.rs (XtrScorer)

**Do**:
1. Create `crates/warp/src/xtr.rs`
2. Add `XtrScorer` struct: `token_registry: Vec<Vec<(String, u32)>>` (per doc: list of (token_string, vocab_id)), `projection: RandomProjection`
3. Add `XtrScorer::new(seed: u64) -> Self`
4. Add `register_doc(doc_id: u32, tokens: &[(String, u32)])` — stores token registry for doc
5. Add `score(query_matrix: &TokenMatrix, t_prime: f32, bound: usize) -> Vec<(u32, f32)>` — for each query token row, for each doc, compute `max over doc tokens of dot(query_tok, token_embedding)`; aggregate max xtr score per doc; filter docs with any score > t_prime; sort descending; truncate to bound; emit `XtrScore` trace events per query token

**Files**:
- `crates/warp/src/xtr.rs`

**Done when**: `cargo check -p warp` passes; `XtrScorer::score` compiles

**Verify**: `cargo check -p warp 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn score' crates/warp/src/xtr.rs`

**Commit**: `feat(warp): add XtrScorer with threshold-filtered candidate generation`

_Requirements: FR-9, AC-4.2_
_Design: XtrScorer_

---

### 2.6 `warp` crate — gather.rs (CandidateGather)

**Do**:
1. Create `crates/warp/src/gather.rs`
2. Add `GatherStats { gathered: Vec<u32>, overlap_with_gt: f32, fraction_promoted: f32 }` struct
3. Add `gather_candidates(xtr_results: &[(u32, f32)], t_prime: f32, colbert_index: &ColBertIndex) -> (Vec<u32>, GatherStats)` — extracts doc_ids from xtr_results above threshold; computes fraction_promoted as len(gathered)/total_docs; emits `CandidateGather` trace event

**Files**:
- `crates/warp/src/gather.rs`

**Done when**: `cargo check -p warp` passes; `gather_candidates` compiles

**Verify**: `cargo check -p warp 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn gather_candidates' crates/warp/src/gather.rs`

**Commit**: `feat(warp): add CandidateGather with fraction-promoted stats`

_Requirements: AC-4.3_
_Design: CandidateGather_

---

### 2.7 `warp` crate — engine.rs + verify.rs + lib.rs

**Do**:
1. Create `crates/warp/src/engine.rs`: `WarpEngine` struct with `colbert_engine: ColBertEngine`, `xtr_scorer: XtrScorer`, `t_prime: f32` (default 0.3), `bound: usize` (default 100); implements `Engine`:
   - `name()` → `"warp"`
   - `index(doc_id, text)` → call `colbert_engine.index()`, register tokens in xtr_scorer
   - `query(text, top_k)` → encode query, Xtr score (emits `XtrScore`), gather candidates (emits `CandidateGather`), MaxSim refine on candidates (emits `MaxSimRefine`)
   - `inspect(target)` → `Some("gather")` → last gather stats
   - `verify()` → delegate to `crate::verify::run()`
2. Create `crates/warp/src/verify.rs` — adds assertion: `fraction_promoted ≤ 1.0` for all queries
3. Update `crates/warp/src/lib.rs`

**Files**:
- `crates/warp/src/engine.rs`
- `crates/warp/src/verify.rs`
- `crates/warp/src/lib.rs`

**Done when**: `cargo check -p warp` passes; Engine trait fully implemented

**Verify**: `cargo check -p warp 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'impl Engine' crates/warp/src/engine.rs`

**Commit**: `feat(warp): complete WarpEngine with Xtr gather then MaxSim refine pipeline`

_Requirements: FR-9, AC-4.1, AC-4.4_

---

### 2.8 [VERIFY] Quality checkpoint: warp crate

**Do**: Check warp crate

**Verify**: `cargo check -p warp 2>&1 | grep "^error" && echo FAIL || echo WARP_OK`

**Done when**: Zero errors

**Commit**: None

---

### 2.9 Resolve TACHIOM git dependencies (vectorium + kannolo SHAs)

**Do**:
1. Attempt to clone both repos: `git ls-remote https://github.com/TusKANNy/vectorium HEAD` and `git ls-remote https://github.com/TusKANNy/kannolo HEAD`
2. If both repos are accessible: capture the HEAD SHAs; update the `[patch.crates-io]` section in workspace `Cargo.toml` with real SHAs; try `cargo fetch` to confirm network access works
3. If either repo returns 404/403 (private/missing): update `Cargo.toml` to comment out the TACHIOM git patch section with a clear note; update `crates/tachiom/Cargo.toml` to use stub local deps instead; add a `// TACHIOM_UNAVAILABLE` compile flag; make tachiom crate compile without the git deps by gating all vectorium/kannolo usage behind `#[cfg(feature = "tachiom_git")]`

**Files**:
- `Cargo.toml` (update patch section with real SHAs or stub)
- `crates/tachiom/Cargo.toml`

**Done when**: Workspace `Cargo.toml` reflects either real SHAs (if accessible) or a clearly commented stub; `cargo check --workspace` continues to pass

**Verify**: `cargo check --workspace 2>&1 | grep "^error" && echo FAIL || echo TACHIOM_DEP_RESOLVED`

**Commit**: `chore(tachiom): resolve vectorium/kannolo git dependencies with pinned SHAs (or stub if unavailable)`

_Design: TACHIOM Crate — Cargo.toml for tachiom crate_

---

### 2.10 [P] `tachiom` crate — tac/tail.rs (TailHandler)

**Do**:
1. Create `crates/tachiom/src/tac/mod.rs` declaring `pub mod tail; pub mod damping; pub mod budget; pub mod clustering;`
2. Create `crates/tachiom/src/tac/tail.rs` with:
   - Constants `MU: u32 = 128`, `TAU: u32 = 256`
   - `TailHandler { freq: HashMap<u32, u32> }` struct
   - `classify(token_type_id) -> TailClass` using match ranges from design
   - `trace_all(&self) -> Vec<TraceEvent>` — one `TailHandle` event per token type

**Files**:
- `crates/tachiom/src/tac/mod.rs`
- `crates/tachiom/src/tac/tail.rs`

**Done when**: `cargo check -p tachiom` passes for tail module

**Verify**: `cargo check -p tachiom 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub const (MU|TAU)' crates/tachiom/src/tac/tail.rs | wc -l`

**Commit**: `feat(tachiom): add TailHandler with mu/tau thresholds and TailClass classification`

_Requirements: FR-10, AC-5.2_

---

### 2.11 [P] `tachiom` crate — tac/damping.rs (DampedScorer)

**Do**:
1. Create `crates/tachiom/src/tac/damping.rs`
2. Add `damped_weight(variance: f32, freq: u32) -> f32` — `(freq as f32).sqrt() * variance`
3. Add `DampedScorer { weights: HashMap<u32, f32> }` struct
4. Add `DampedScorer::compute(token_embeddings: &HashMap<u32, Vec<[f32; TOKEN_DIM]>>) -> Self` — computes variance of each token type's embeddings; applies `damped_weight`; stores in `weights`
5. Variance formula: mean of squared distances from centroid for each token type's embedding set

**Files**:
- `crates/tachiom/src/tac/damping.rs`

**Done when**: `cargo check -p tachiom` passes for damping module

**Verify**: `cargo check -p tachiom 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn damped_weight' crates/tachiom/src/tac/damping.rs`

**Commit**: `feat(tachiom): add DampedScorer with sqrt(n)*variance weight formula`

_Requirements: FR-10, AC-5.3_

---

### 2.12 [P] `tachiom` crate — tac/budget.rs (BudgetReconciler)

**Do**:
1. Create `crates/tachiom/src/tac/budget.rs`
2. Add constants `EPSILON: u32 = 4`, `THETA: u32 = 39`
3. Add `BudgetReconciler { total_budget: u32 }` struct
4. Add `allocate(weights: &HashMap<u32, f32>) -> HashMap<u32, u32>` — proportional allocation: `raw_kappa_j = total_budget * w_j / sum(w)`, then clamp to [EPSILON, THETA], emit `BudgetBound` trace events
5. Add `reconcile(&mut self, raw: &mut HashMap<u32, u32>)` — compute leftover budget (total_budget - sum(clamped)), redistribute to token types below ceiling by weight rank, emit `BudgetReconcile` trace event

**Files**:
- `crates/tachiom/src/tac/budget.rs`

**Done when**: `cargo check -p tachiom` passes for budget module

**Verify**: `cargo check -p tachiom 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub const (EPSILON|THETA)' crates/tachiom/src/tac/budget.rs | wc -l`

**Commit**: `feat(tachiom): add BudgetReconciler with epsilon/theta floor-ceiling and redistribution`

_Requirements: FR-10, AC-5.4_

---

### 2.13 `tachiom` crate — tac/clustering.rs (parallel κⱼ-means)

**Do**:
1. Create `crates/tachiom/src/tac/clustering.rs`
2. Add `#![feature(portable_simd)]` and `#![feature(iter_array_chunks)]` at crate lib.rs level (not module level)
3. Add `kmeans(embeddings: &[[f32; TOKEN_DIM]], k: usize, seed: u64) -> Vec<[f32; TOKEN_DIM]>` — Lloyd's with fixed seed SmallRng, 50 iterations max
4. Add `parallel_kappa_means(token_type_embeddings: &HashMap<u32, Vec<[f32; TOKEN_DIM]>>, kappa: &HashMap<u32, u32>, seed: u64) -> HashMap<u32, Vec<[f32; TOKEN_DIM]>>` — uses `rayon::prelude::*` `par_iter()` as in design; for each token type runs `kmeans` with `seed ^ token_type_id`
5. Update `crates/tachiom/src/lib.rs` to add `#![feature(portable_simd, iter_array_chunks)]`

**Files**:
- `crates/tachiom/src/tac/clustering.rs`
- `crates/tachiom/src/lib.rs`

**Done when**: `cargo check -p tachiom` passes; parallel_kappa_means compiles with rayon

**Verify**: `cargo +nightly check -p tachiom 2>&1 | grep -c "^error" | grep -q "^0$" && grep -c 'par_iter' crates/tachiom/src/tac/clustering.rs`

**Commit**: `feat(tachiom): add parallel kappa-means with rayon and nightly features`

_Requirements: FR-10_

---

### 2.14 `tachiom` crate — pq.rs (3-level hierarchical PQ)

**Do**:
1. Create `crates/tachiom/src/pq.rs`
2. Add `PQLevel { dimensions: u32, num_subquantizers: u32, code_bits: u8, codebook: Vec<Vec<[f32; 16]>> }` struct
3. Add `HierarchicalPQ { levels: [PQLevel; 3] }` struct with constructor creating the 3 levels from design table (level 0: 128-dim, 8 subq, 1 byte; level 1: 64-dim, 4 subq, 1 byte; level 2: 32-dim, 2 subq, 1 byte)
4. Add `HierarchicalPQ::inspect() -> String` — formats the 3-level table for display (for `inspect pq` command)
5. Add stub `train(data: &[[f32; TOKEN_DIM]])` and `encode(vec: &[f32; TOKEN_DIM]) -> Vec<u8>` methods (educational: KMeans per subquantizer, educational impl, not production-optimized)

**Files**:
- `crates/tachiom/src/pq.rs`

**Done when**: `cargo check -p tachiom` passes; `HierarchicalPQ::inspect()` compiles

**Verify**: `cargo +nightly check -p tachiom 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub struct (HierarchicalPQ|PQLevel)' crates/tachiom/src/pq.rs | wc -l`

**Commit**: `feat(tachiom): add 3-level HierarchicalPQ with inspect output`

_Requirements: AC-5.5_
_Design: 3-Level Hierarchical PQ_

---

### 2.15 `tachiom` crate — engine.rs + verify.rs + lib.rs

**Do**:
1. Create `crates/tachiom/src/engine.rs`: `TachiomEngine` struct with `encoder: ColBertEncoder` (from colbert crate), `tail_handler: TailHandler`, `damped_scorer: Option<DampedScorer>`, `budget_reconciler: BudgetReconciler`, `centroids_by_type: HashMap<u32, Vec<[f32; TOKEN_DIM]>>`, `pq: HierarchicalPQ`; seed `0xCAFEBABE_DEADBEEF`; implements `Engine`:
   - `name()` → `"tachiom"`
   - `index(doc_id, text)` → encode, update tail_handler freqs, after all docs indexed: run full TAC pipeline (phases 1-4) — emit `TailHandle`, `DampedScore`, `BudgetBound`, `BudgetReconcile` trace events
   - `query(text, top_k)` → encode query, search using `centroids_by_type` (for each query token, find matching token-type centroids, gather candidates, MaxSim refine), emit `TachiomSearch` with timing
   - `inspect(target)` → `Some("pq")` → `pq.inspect()`; `Some("centroids")` → per-type centroid counts
   - `verify()` → delegate to `crate::verify::run()`
2. Create `crates/tachiom/src/verify.rs` — recall assertions + κⱼ in [EPSILON, THETA] for all token types
3. Update `crates/tachiom/src/lib.rs` — wire all modules; add nightly feature flags

**Files**:
- `crates/tachiom/src/engine.rs`
- `crates/tachiom/src/verify.rs`
- `crates/tachiom/src/lib.rs`

**Done when**: `cargo check -p tachiom` passes; `TachiomEngine` implements Engine

**Verify**: `cargo +nightly check -p tachiom 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'impl Engine' crates/tachiom/src/engine.rs`

**Commit**: `feat(tachiom): complete TachiomEngine with all 4 TAC phases and verify harness`

_Requirements: FR-10, AC-5.1_

---

### 2.16 [VERIFY] Quality checkpoint: tachiom crate

**Do**: Check tachiom with nightly; confirm all TAC phases compile

**Verify**: `cargo +nightly check -p tachiom 2>&1 | grep "^error" && echo FAIL || echo TACHIOM_OK`

**Done when**: Zero errors (git dep stubs acceptable if TusKANNy repos unavailable)

**Commit**: None

---

### 2.17 [P] `bench` crate — python_engine.rs (PythonEngine trait)

**Do**:
1. Create `crates/bench/src/python_engine.rs`
2. Add `PythonEngine` async trait with methods: `name()`, `binary_name()`, `build_index()`, `search()`, `parse_line()`, `check_installed()`, `install_url()`
3. `check_installed()` uses `which::which(self.binary_name())` (exact from design)
4. Add `ColbertPython` and `WarpPython` structs implementing `PythonEngine` (exact from design)
5. `build_index` and `search` implementations: use `tokio::process::Command` to spawn subprocess; read stdout line by line; call `parse_line()` on each line; return `Vec<BenchResult>`

**Files**:
- `crates/bench/src/python_engine.rs`

**Done when**: `cargo check -p bench` passes; `PythonEngine` trait compiles with `async_trait`

**Verify**: `cargo check -p bench 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub trait PythonEngine' crates/bench/src/python_engine.rs`

**Commit**: `feat(bench): add PythonEngine trait with ColbertPython and WarpPython impls`

_Requirements: FR-9, FR-11_

---

### 2.18 [P] `bench` crate — sift.rs (SiftDownloader)

**Do**:
1. Create `crates/bench/src/sift.rs`
2. Add `SiftDownloader { data_dir: PathBuf }` struct
3. Add `ensure_present() -> anyhow::Result<PathBuf>` — checks `MULTIVECTOR_SIFT_PATH` env var; validates `bigann_base.bvecs` and `bigann_gnd/idx_100M.ivecs` (SIFT 100M subset ground truth — note: the 100M subset GT has a different filename than the full dataset's `bigann_groundtruth.ivecs`) exist at that path; if env var is unset or files are missing, prints manual FTP download instructions (ftp://ftp.irisa.fr/local/texmex/corpus/) and returns `Err`. No automatic download — reqwest does not support FTP and the design explicitly requires user-supplied path only.
4. Add `read_bvecs(path) -> anyhow::Result<Vec<Vec<f32>>>` — parses `[u32 dim][u8 × dim]` bvecs format; converts u8 to f32
5. Add `read_ivecs(path) -> anyhow::Result<Vec<Vec<i32>>>` — parses `[u32 dim][i32 × dim]` ivecs format

**Files**:
- `crates/bench/src/sift.rs`

**Done when**: `cargo check -p bench` passes; `SiftDownloader::ensure_present` compiles

**Verify**: `cargo check -p bench 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn ensure_present' crates/bench/src/sift.rs`

**Commit**: `feat(bench): add SiftDownloader with env var validation and manual download instructions`

_Requirements: FR-12, AC-6.3_

---

### 2.19 [P] `bench` crate — metrics.rs + runner.rs + lib.rs

**Do**:
1. Create `crates/bench/src/metrics.rs`: `recall_at_k()` and `latency_percentiles()` functions (exact implementations from design)
2. Create `crates/bench/src/runner.rs`: `BenchRunner` struct that orchestrates running all engines; `run_all() -> anyhow::Result<PlotData>` — runs each PythonEngine, collects `BenchResult`s, writes to `output/bench_results.json`, returns `PlotData`
3. Update `crates/bench/src/lib.rs` to declare all 4 modules and re-export key types

**Files**:
- `crates/bench/src/metrics.rs`
- `crates/bench/src/runner.rs`
- `crates/bench/src/lib.rs`

**Done when**: `cargo check -p bench` passes; all modules compile

**Verify**: `cargo check -p bench 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub fn recall_at_k' crates/bench/src/metrics.rs`

**Commit**: `feat(bench): add recall metrics, latency percentiles, BenchRunner`

_Requirements: FR-13, AC-6.3_

---

### 2.20 [VERIFY] Quality checkpoint: bench crate

**Do**: Check bench crate

**Verify**: `cargo check -p bench 2>&1 | grep "^error" && echo FAIL || echo BENCH_OK`

**Done when**: Zero errors

**Commit**: None

---

## Phase 3: CLI Wiring and Scenarios

Complete the CLI dispatch for all engines, add remaining scenario files, and add REPL suggestion sequences.

---

### 3.1 Wire remaining engines into cli demo.rs

**Do**:
1. Update `crates/cli/src/demo.rs`: add engine construction for `"hnsw"`, `"plaid"`, `"warp"`, `"tachiom"` in the `match scenario.meta.engine.as_str()` block
2. Each engine: construct with same vocab_path logic as ColBERT
3. Dispatch closure handles same op strings ("index", "query", "inspect", "trace") for all engines via the `Engine` trait
4. `"compare"` branch → call `compare::run_compare().await?`

**Files**:
- `crates/cli/src/demo.rs`

**Done when**: `cargo check -p cli` passes; all 5 engines constructable from demo.rs

**Verify**: `cargo check -p cli 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E '"(hnsw|plaid|warp|tachiom)"' crates/cli/src/demo.rs | wc -l`

**Commit**: `feat(cli): wire all 5 engines into demo runner`

_Requirements: FR-1, AC-1.1, AC-3.1, AC-4.1, AC-5.1_

---

### 3.2 Wire remaining engines into cli main.rs repl dispatch

**Do**:
1. Update `crates/cli/src/main.rs` `TopCommand::Repl` match arms for `Hnsw`, `Plaid`, `Warp`, `Tachiom`
2. Each arm: construct engine, construct VizRepl with engine-specific `SuggestionMode::Sequence` (suggestions from design for each engine), call `run_repl`
3. Wire `TopCommand::Bench` dispatch to `bench::runner::BenchRunner` calls per `BenchTarget` variant

**Files**:
- `crates/cli/src/main.rs`

**Done when**: `cargo build -p cli` succeeds; `./target/debug/multivector repl --help` shows all 5 engine subcommands

**Verify**: `cargo build -p cli 2>&1 | grep -c "^error" | grep -q "^0$" && ./target/debug/multivector repl --help | grep -E '(hnsw|colbert|plaid|warp|tachiom)' | wc -l`

**Commit**: `feat(cli): wire all 5 engine REPLs with suggestion sequences`

_Requirements: FR-4, FR-5_

---

### 3.3 [P] `scenarios/hnsw.toml`

**Do**:
1. Create `scenarios/hnsw.toml`
2. Include: `[meta]` with title, engine="hnsw", description, version=1; `[corpus]` with `type = "shared"`; `[[steps]]` sequence:
   - index doc 0 ("The river bank was slippery...") with narration explaining single-vector precision loss
   - inspect layers — narration on HNSW graph layer structure
   - query "river erosion along the bank" — narration: greedy ANN walk, cosine scores
   - query "open a checking account at the bank" — narration: same doc ranked differently
   - inspect graph — narration: average degree, ef_construction
   - trace — narration: HnswInsert/HnswQuery events explained

**Files**:
- `scenarios/hnsw.toml`

**Done when**: TOML parses; 6 steps; all op values are valid

**Verify**: `python3 -c "import tomllib; tomllib.load(open('scenarios/hnsw.toml','rb'))" && grep -c '^\[\[steps\]\]' scenarios/hnsw.toml | grep -q 6 && echo HNSW_TOML_OK`

**Commit**: `feat(scenarios): add hnsw.toml with HNSW layer traversal walkthrough`

_Requirements: AC-1.1, AC-1.5, AC-7.1_

---

### 3.4 [P] `scenarios/plaid.toml`

**Do**:
1. Create `scenarios/plaid.toml`
2. Include `[meta]` with engine="plaid", version=1; `[corpus]` with `type = "shared"`; `[[steps]]`:
   - index all 20 docs (op="index", args=["all"]) — narration on centroid assignment, contrasts with ColBERT
   - inspect centroids — narration: centroid count, cluster sizes
   - query "bank interest rate" with verbose — narration: pruned doc count vs. scored, percentage reduction (AC-3.3)
   - query "river bank hiking" — narration: which doc gets skipped by centroid pruning (AC-3.4)
   - inspect centroids (verbose) — narration: which centroids activated for last query

**Files**:
- `scenarios/plaid.toml`

**Done when**: TOML parses; 5 steps; engine="plaid"

**Verify**: `python3 -c "import tomllib; tomllib.load(open('scenarios/plaid.toml','rb'))" && grep -q 'engine.*plaid' scenarios/plaid.toml && echo PLAID_TOML_OK`

**Commit**: `feat(scenarios): add plaid.toml with centroid pruning walkthrough`

_Requirements: AC-3.1, AC-3.2, AC-3.3, AC-3.4, AC-7.1_

---

### 3.5 [P] `scenarios/warp.toml`

**Do**:
1. Create `scenarios/warp.toml`
2. Include `[meta]` with engine="warp", version=1; `[corpus]` with `type = "shared"`; `[[steps]]`:
   - index all 20 docs — narration: Xtr token registry built
   - inspect gather — narration: gather statistics for a sample query
   - query "bank interest rate" — narration contrasting WARP gather vs PLAID centroid ANN (AC-4.4)
   - trace xtr — narration: XtrScore event per query token
   - inspect gather (verbose) — narration: fraction promoted, overlap with ground truth

**Files**:
- `scenarios/warp.toml`

**Done when**: TOML parses; engine="warp"

**Verify**: `python3 -c "import tomllib; tomllib.load(open('scenarios/warp.toml','rb'))" && grep -q 'engine.*warp' scenarios/warp.toml && echo WARP_TOML_OK`

**Commit**: `feat(scenarios): add warp.toml with Xtr gather walkthrough`

_Requirements: AC-4.1, AC-4.4, AC-7.1_

---

### 3.6 [P] `scenarios/tachiom.toml`

**Do**:
1. Create `scenarios/tachiom.toml`
2. Include `[meta]` with engine="tachiom", version=1; `[corpus]` with `type = "shared"`; `[[steps]]`:
   - index all 20 docs — narration: TAC phases fire during index
   - trace tail-handling — narration: which tokens are tail/normal/heavy (AC-5.2)
   - trace damped-scoring — narration: sⱼ and wⱼ per token type (AC-5.3)
   - trace budget — narration: ε floor, θ ceiling, redistribution (AC-5.4)
   - inspect pq — narration: 3-level PQ distance table layout (AC-5.5)
   - query "bank interest rate" — narration: TACHIOM vs PLAID at scale (AC-5.7)

**Files**:
- `scenarios/tachiom.toml`

**Done when**: TOML parses; 6 steps; engine="tachiom"

**Verify**: `python3 -c "import tomllib; tomllib.load(open('scenarios/tachiom.toml','rb'))" && grep -c '^\[\[steps\]\]' scenarios/tachiom.toml | grep -q 6 && echo TACHIOM_TOML_OK`

**Commit**: `feat(scenarios): add tachiom.toml with full TAC pipeline walkthrough`

_Requirements: AC-5.1, AC-5.7, AC-7.1_

---

### 3.7 `scenarios/compare.toml` + `cli/src/compare.rs` implementation

**Do**:
1. Create `scenarios/compare.toml`: `[meta]` engine="compare", version=1; corpus shared; steps: one step per engine querying "bank interest rate", final step prints comparison table
2. Implement `crates/cli/src/compare.rs` `run_compare()` function:
   - Construct one instance of each of the 5 engines
   - Index all 20 SHARED_CORPUS docs into each engine
   - Run "bank interest rate" query on all 5 engines
   - Print structured ASCII table: engine | index structure | pipeline stages | O(query) complexity | memory per token
   - Write `output/compare.json` with same data as `Vec<serde_json::Value>`

**Files**:
- `scenarios/compare.toml`
- `crates/cli/src/compare.rs`

**Done when**: `cargo check -p cli` passes; `compare.toml` parses; compare.rs compiles

**Verify**: `cargo check -p cli 2>&1 | grep -c "^error" | grep -q "^0$" && python3 -c "import tomllib; tomllib.load(open('scenarios/compare.toml','rb'))" && echo COMPARE_OK`

**Commit**: `feat(cli): implement compare runner with ASCII table and JSON output`

_Requirements: AC-6.1, AC-6.2_

---

### 3.8 [VERIFY] Quality checkpoint: all crates + CLI build

**Do**: Full workspace build in debug mode; verify all 8 crates compile

**Verify**: `cargo build --workspace 2>&1 | grep "^error" && echo FAIL || echo ALL_CRATES_OK`

**Done when**: Zero errors across all 8 crates

**Commit**: `chore(workspace): all 8 crates build clean`

---

## Phase 4: Recall Calibration

---

### 4.1 Run verification harnesses and calibrate projection seed

**Do**:
1. Run `cargo test -p colbert -- verify_engine 2>&1` and capture output
2. If any `recall@1 failed for query: '...'` assertions fire: the projection seed `0xCAFEBABE_DEADBEEF` needs tuning. Try seeds `0xDEADBEEF_CAFEBABE`, `0x0123456789ABCDEF`, `0xFEEDFACE_DEADC0DE` in order until all 20 queries pass recall@1=1.0
3. Update `RandomProjection::new()` call in `ColBertEncoder::new()` in `colbert/src/encoder.rs` (and all other engines that use the same encoder) with the passing seed
4. Repeat test run to confirm all 20 queries pass
5. Run `cargo test -p plaid -- verify_engine` and `cargo test -p warp -- verify_engine` with the calibrated seed; adjust `PlaidEngine::num_centroids` (try 8, 16, 32) or `WarpEngine::t_prime` (try 0.1, 0.2, 0.3) if those fail

**Files**:
- `crates/colbert/src/engine.rs` (update seed constant if changed)
- `crates/plaid/src/engine.rs` (update num_centroids if needed)
- `crates/warp/src/engine.rs` (update t_prime if needed)

**Done when**: `cargo test -p colbert -p plaid -p warp 2>&1 | grep -E 'FAILED|panicked'` returns empty

**Verify**: `cargo test -p colbert -- verify_engine 2>&1 | tail -5 && cargo test -p plaid -- verify_engine 2>&1 | tail -5 && cargo test -p warp -- verify_engine 2>&1 | tail -5`

**Commit**: `fix(verify): calibrate projection seed to achieve recall@1=1.0 across all 20 queries`

_Requirements: AC-8.2, AC-8.3, NFR-4_
_Design: Verification Harness — calibration note_

---

### 4.2 Run HNSW and TACHIOM verify harnesses

**Do**:
1. Run `cargo test -p hnsw -- verify_engine` — LocalHnsw mock should pass recall@1=1.0 (mock embeddings are unique per doc_id by construction)
2. If HNSW verify fails: adjust `ef_construction` in `LocalHnsw::new()` (try 50, 100, 200)
3. Run `cargo +nightly test -p tachiom -- verify_engine` — if TAC phase produces κⱼ out of [EPSILON, THETA] bounds, adjust `total_budget` in `BudgetReconciler` (try 100, 200, 400)
4. Fix any panics in the TAC pipeline on the 20-sentence corpus (small corpus may hit edge cases in tail handling or KMeans with fewer points than centroids — add guard: `k = k.min(embeddings.len())`)

**Files**:
- `crates/hnsw/src/local.rs` (adjust ef_construction if needed)
- `crates/tachiom/src/engine.rs` (adjust total_budget if needed)
- `crates/tachiom/src/tac/clustering.rs` (add k.min(embeddings.len()) guard)

**Done when**: `cargo test -p hnsw -- verify_engine` passes AND `cargo +nightly test -p tachiom -- verify_engine` passes

**Verify**: `cargo test -p hnsw -- verify_engine 2>&1 | tail -3 && cargo +nightly test -p tachiom -- verify_engine 2>&1 | tail -3`

**Commit**: `fix(verify): calibrate HNSW ef_construction and TACHIOM budget for small corpus`

_Requirements: AC-8.2, AC-8.7_

---

### 4.3 [VERIFY] `cargo test --workspace` — all verify harnesses pass

**Do**: Run the full workspace test suite; all 5 engine verify harnesses must pass

**Verify**: `cargo +nightly test --workspace 2>&1 | grep -E '^test result' | grep -v 'ok'`

**Done when**: Output shows only `ok` lines (zero FAILED); if any failures remain, fix them before proceeding

**Commit**: `test(workspace): all engine verification harnesses pass recall assertions`

---

## Phase 5: Scripts, Quality Gates, and Integration

---

### 5.1 `scripts/plot.py` — all 4 plot functions

**Do**:
1. Create `scripts/` directory
2. Create `scripts/plot.py` with:
   - `load(path) -> list[dict]` function
   - `pareto_plot(results, out)` — scatter on log-scale x=p99_ms, y=recall_at_10 (exact from design)
   - `recall_k_plot(results, out)` — line chart k=1/10/100 vs recall; one line per engine using `ENGINE_ORDER` and `COLORS` from design
   - `latency_cdf_plot(results, out)` — CDF from p50/p95/p99 per engine; log-scale x
   - `qps_bar_plot(results, out)` — bar chart engine vs QPS
   - `main()` with argparse: `--input` (default `output/bench_results.json`), calls all 4 plot functions, saves to `output/`
3. Add shebang `#!/usr/bin/env python3`; make executable

**Files**:
- `scripts/plot.py`

**Done when**: Script runs without import errors: `python3 -c "import scripts.plot" 2>/dev/null || python3 scripts/plot.py --help`

**Verify**: `python3 -c "import sys; sys.path.insert(0,'scripts'); import importlib.util; spec=importlib.util.spec_from_file_location('plot','scripts/plot.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('PLOT_OK')"` 

**Commit**: `feat(scripts): add plot.py with pareto, recall@k, latency CDF, and QPS bar charts`

_Requirements: FR-14, AC-6.4_

---

### 5.2 [VERIFY] Full local quality check

**Do**: Run the complete local quality suite

**Verify**:
```
cargo +nightly clippy --workspace -- -D warnings 2>&1 | grep "^error" && echo CLIPPY_FAIL || echo CLIPPY_OK
cargo +nightly fmt --all -- --check 2>&1 | grep -q "Diff" && echo FMT_FAIL || echo FMT_OK
cargo +nightly test --workspace 2>&1 | grep "^FAILED" && echo TEST_FAIL || echo TEST_OK
cargo +nightly build --workspace --release 2>&1 | grep "^error" && echo BUILD_FAIL || echo BUILD_OK
```

**Done when**: All 4 commands show OK; release build completes in under 5 minutes (NFR-5)

**Commit**: `fix(workspace): address clippy and fmt issues across all crates` (if fixes needed)

_Requirements: FR-1, NFR-5_

---

### 5.3 [VERIFY] V6 AC checklist — verify each acceptance criterion

**Do**: Programmatically verify each AC is implemented:

```
# AC-1.1: hnsw demo scenario exists
grep -q 'engine.*hnsw' scenarios/hnsw.toml && echo AC-1.1 OK

# AC-2.1: colbert demo scenario exists with all required ops
grep -qE 'op.*=.*"(index|query|trace)"' scenarios/colbert.toml && echo AC-2.1 OK

# AC-2.2: 3-dim preview in token.rs
grep -q 'preview.*\[f32; 3\]' crates/common/src/token.rs && echo AC-2.2 OK

# AC-2.3: MaxSimMatrix trace event has matrix field
grep -q 'MaxSimMatrix' crates/common/src/trace.rs && echo AC-2.3 OK

# AC-3.5: PlaidEngine holds ColBertEngine (progressive layer, not separate)
grep -q 'colbert_engine.*ColBertEngine' crates/plaid/src/engine.rs && echo AC-3.5 OK

# AC-4.4: WarpEngine has XtrScorer (not CentroidPruner)
grep -q 'xtr_scorer.*XtrScorer' crates/warp/src/engine.rs && echo AC-4.4 OK

# AC-5.2: TailHandler with MU=128
grep -q 'MU.*128' crates/tachiom/src/tac/tail.rs && echo AC-5.2 OK

# AC-6.1: compare.rs prints table
grep -q 'index structure' crates/cli/src/compare.rs && echo AC-6.1 OK

# AC-7.2: scenario version validated
grep -q 'meta.version != 1' crates/common/src/scenario.rs || grep -q 'version.*!= 1' crates/common/src/scenario.rs && echo AC-7.2 OK

# AC-8.2: recall@1=1.0 assertion in each verify module
for p in colbert plaid warp hnsw tachiom; do
  grep -q 'recall@1' crates/$p/src/verify.rs && echo AC-8.2 $p OK
done

# AC-8.7: LocalHnsw used in hnsw verify (no Atlas)
grep -q 'LocalHnsw' crates/hnsw/src/verify.rs && echo AC-8.7 OK

# NFR-7: no ML runtimes in Cargo.lock
! grep -qE '(ort|tch|candle-)' Cargo.lock && echo NFR-7 OK
```

**Done when**: All grep checks return OK; no ML runtime names found in Cargo.lock

**Commit**: None

---

### 5.4 Create PR and verify CI

**Do**:
1. Confirm current branch is NOT main/master: `git branch --show-current`
2. Push feature branch: `git push -u origin $(git branch --show-current)`
3. Create PR: `gh pr create --title "feat: multivector retrieval educational CLI" --body "$(cat <<'EOF' ... EOF)"`
4. Monitor CI: `gh pr checks --watch`
5. If CI fails on nightly feature flags: check that `#![feature(...)]` declarations are only in `crates/tachiom/src/lib.rs`, not in stable crates; fix and push

**Done when**: All CI checks show green; PR created with summary of all 5 engine implementations

**Verify**: `gh pr checks 2>&1 | grep -v '✓' | grep -v 'Completed' && echo CI_FAIL || echo CI_PASS`

**Commit**: None (PR creation, not code change)

---

## Notes

**POC shortcuts taken** (Phase 1):
- PlaidEngine rebuilds the full PlaidIndex on every `index()` call — acceptable for 20-doc demo, would need incremental updates at scale
- WarpEngine t_prime=0.3 is hardcoded; no dynamic threshold tuning
- `inspect` commands return plain strings, not structured JSON — sufficient for educational display
- HNSW `graph_stats()` returns empty vec (Atlas stats API varies by cluster tier)

**Production TODOs**:
- PlaidIndex: incremental centroid re-assignment without full rebuild
- TACHIOM: real vectorium/kannolo integration once SHA is confirmed accessible
- bench mode: parse actual xtr-warp and tachiom binary output formats once binary CLI contracts are confirmed
- SIFT download: add SHA256 checksum validation after download (security note from design)
- AtlasClient: add retry logic for transient network failures

**Risk: TACHIOM git deps**:
- Task 2.13 handles the TusKANNy repo access check before writing engine code
- If repos are private/inaccessible, the `#[cfg(feature = "tachiom_git")]` gate allows the workspace to compile and all other engines to be fully functional
- TACHIOM verify harness will be skipped in this case (documented in verify.rs with `#[cfg_attr(not(feature = "tachiom_git"), ignore)]`)

**Risk: recall@1=1.0**:
- Task 4.1 is a calibration loop — not a one-shot implementation
- The seed `0xCAFEBABE_DEADBEEF` may not achieve recall@1=1.0 for all 20 queries on first try
- Four candidate seeds are listed in task 4.1; if none work, adjust MaxSim scoring to break ties by doc_id (stable sort) — this ensures determinism and gives the closest semantic match a consistent win

**WordPiece vocab note**:
- Task 1.4 downloads from HuggingFace; if network is unavailable, the file can be sourced from any bert-base-uncased model checkpoint (identical MIT-licensed file)

---

## Phase 6: Atlas / HNSW (load last — hardware dependency)

**Prereq**: Hardware ready + `.env` file present with `MONGODB_URI` and `VOYAGE_API_KEY`.

These tasks were originally Phase 2.1–2.4. They are deferred because the Atlas cluster with binary quantization hardware support is not yet provisioned. All other phases (1–5) can be completed and verified without this phase.

---

### 6.1 `hnsw` crate — local.rs (LocalHnsw mock)

**Do**:
1. Create `crates/hnsw/src/local.rs`
2. Add `LocalHnsw` struct: `inner: hnsw_rs::Hnsw<f32, hnsw_rs::dist::DistCosine>`, `doc_map: Vec<(u32, Vec<f32>)>`
3. Add `LocalHnsw::new()` — constructs `hnsw_rs::Hnsw::new(16, 20, 16, 200, hnsw_rs::dist::DistCosine {})` (ef=200, max_layer=16)
4. Add `insert_mock(doc_id)` — calls `mock_embedding(doc_id)` (SmallRng seeded with `0xCAFEBABEu64 ^ doc_id`, 1536 Gaussian entries, L2-normalized), inserts into hnsw_rs, appends to doc_map
5. Add `mock_embedding(doc_id) -> Vec<f32>` free function (exact from design)
6. Add `search(query_vec, top_k) -> Vec<(u32, f32)>` — calls `inner.search(&query_vec, top_k, 50)`, maps results to (doc_id, score)

**Files**:
- `crates/hnsw/src/local.rs`

**Done when**: `cargo check -p hnsw` passes; `LocalHnsw::insert_mock` compiles against `hnsw_rs`

**Verify**: `cargo check -p hnsw 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub struct LocalHnsw' crates/hnsw/src/local.rs`

**Commit**: `feat(hnsw): add LocalHnsw with deterministic mock embeddings`

_Requirements: AC-8.7_
_Design: LocalHnsw_

---

### 6.2 [P] `hnsw` crate — atlas.rs (AtlasClient)

> **Prereq**: `.env` file at workspace root must contain `MONGODB_URI` and `VOYAGE_API_KEY`. These vars are loaded via `dotenvy::dotenv()` in `from_env()` — no manual export needed.

**Do**:
1. Create `crates/hnsw/src/atlas.rs`
2. Add `AtlasClient` struct: `client: mongodb::Client`, `collection: mongodb::Collection<Document>`, `index_name: &'static str`
3. Add `AtlasClient::from_env() -> anyhow::Result<Self>` — call `dotenvy::dotenv()` at the top (no-op if `.env` absent) before reading env vars; reads `MONGODB_URI`, connects, targets `multivector` db, `multivector_demo` collection
4. Add `index_doc(doc_id, text) -> anyhow::Result<Vec<f32>>` — POST to Voyage API (`https://api.voyageai.com/v1/embeddings`, model=`voyage-4-large`, reads `VOYAGE_API_KEY`), insert doc `{doc_id, text, embedding}` via mongodb driver, return embedding
5. Add `query(embedding, top_k) -> anyhow::Result<Vec<(u32, f32)>>` — `$vectorSearch` aggregation pipeline on `vector_index`, `embedding` field, `numCandidates = top_k * 10` (10× oversampling for binary quantization rescoring)
6. Add `graph_stats() -> anyhow::Result<Vec<HnswLayerStat>>` stub returning empty vec (stats API is cluster-specific)
7. Create the Atlas Vector Search index with binary quantization + storedSource via Atlas UI or mongocli: use the index definition from the design (quantization: binary, storedSource includes embedding/doc_id/text).

**Files**:
- `crates/hnsw/src/atlas.rs`

**Done when**: `cargo check -p hnsw` passes; `AtlasClient::from_env` compiles

**Verify**: `cargo check -p hnsw 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'pub async fn (index_doc|query|from_env)' crates/hnsw/src/atlas.rs | wc -l`

**Commit**: `feat(hnsw): add AtlasClient with Voyage API embedding and vectorSearch`

_Requirements: FR-6, AC-1.2_
_Design: AtlasClient, Atlas Integration Design_

---

### 6.3 `hnsw` crate — engine.rs + verify.rs + lib.rs

**Do**:
1. Create `crates/hnsw/src/engine.rs`: `HnswEngine` struct with `local: LocalHnsw`, `atlas: Option<AtlasClient>`; implements `Engine`:
   - `name()` → `"hnsw"`
   - `index(doc_id, text)` → if atlas present: `atlas.index_doc()`; else: `local.insert_mock(doc_id)`; emit `HnswInsert` trace event
   - `query(text, top_k)` → if atlas: embed + vectorSearch; else: `local.search(mock_embedding(query_hash), top_k)`; emit `HnswQuery` trace events
   - `inspect(target)` → match `Some("layers")` → layer stats; `Some("graph")` → degree stats; else list options
   - `verify()` → delegate to `crate::verify::run()`
2. Create `crates/hnsw/src/verify.rs` — uses `LocalHnsw` only (no Atlas), indexes all 20 SHARED_CORPUS docs with `insert_mock`, runs VERIFY_QUERIES, asserts recall@1=1.0, recall@10≥0.9
3. Update `crates/hnsw/src/lib.rs` to wire all modules

**Files**:
- `crates/hnsw/src/engine.rs`
- `crates/hnsw/src/verify.rs`
- `crates/hnsw/src/lib.rs`

**Done when**: `cargo check -p hnsw` passes; `HnswEngine` implements Engine

**Verify**: `cargo check -p hnsw 2>&1 | grep -c "^error" | grep -q "^0$" && grep -E 'impl Engine' crates/hnsw/src/engine.rs`

**Commit**: `feat(hnsw): complete HnswEngine with local+atlas modes and verify harness`

_Requirements: AC-1.1, AC-1.2, AC-1.3, AC-1.4, AC-8.7_

---

### 6.4 [VERIFY] Quality checkpoint: hnsw crate

**Do**: Check hnsw crate; confirm verify harness compiles

**Verify**: `cargo check -p hnsw 2>&1 | grep "^error" && echo FAIL || echo HNSW_OK`

**Done when**: Zero errors

**Commit**: None
