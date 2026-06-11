use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "multivector",
    version,
    about = "Multivector retrieval educational CLI"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: TopCommand,

    /// Write trace events to JSON file.
    #[arg(long, global = true)]
    pub trace_json: Option<std::path::PathBuf>,
}

#[derive(Subcommand)]
pub enum TopCommand {
    /// Run a TOML scenario end-to-end.
    Demo {
        /// Scenario name (resolves scenarios/<name>.toml) or "compare".
        name: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// Open an interactive REPL for one engine.
    Repl {
        #[command(subcommand)]
        engine: EngineCmd,
    },
    /// Run benchmarks.
    Bench {
        #[command(subcommand)]
        target: BenchTarget,
    },
    /// Synthetic scalability benchmark: ColBERT O(N) vs PLAID/WARP pruned scan.
    /// Runs at 1K / 10K / 100K docs and extrapolates to 1M.
    Scale,
    /// Accuracy-speed tradeoff benchmark: Recall@10 vs speedup for HNSW / PLAID / WARP / TACHIOM.
    /// Generates SVG plots in plots/ directory.
    Tradeoff,
    /// Ground-truth benchmark using LLM-as-judge (Claude Haiku) relevance labels.
    /// 100 real-text docs, all 5 engines, Voyage embeddings for HNSW.
    /// Requires ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env (cached after first run).
    GtBench,
    /// Large-scale "needle in a haystack" benchmark.
    /// 100 real GT docs (Jina ColBERT) buried in a synthetic corpus of N=100K/1M/10M.
    /// Tests HNSW / PLAID / WARP / TACHIOM with no-filter / category / cat+year modes.
    /// Generates SVG plots in plots/ directory.  Requires gt-bench cache to exist first.
    LargeBench,
    /// Embed MS MARCO passages for real-data benchmarking.
    /// Requires JINA_API_KEY and/or VOYAGE_API_KEY in .env.
    /// Resume-safe: file sizes track progress; rerun to continue after interruption.
    EmbedMsMarco {
        /// Path to MS MARCO collection.tsv (tab-separated: passage_id\tpassage_text).
        /// Download: https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
        collection: std::path::PathBuf,

        /// Output directory for binary embedding files.
        #[arg(long, default_value = "data/msmarco")]
        out_dir: std::path::PathBuf,

        /// Embed only with Jina ColBERT v2 (skip Voyage-4-large).
        #[arg(long)]
        jina_only: bool,

        /// Embed only with Voyage-4-large (skip Jina ColBERT v2).
        #[arg(long)]
        voyage_only: bool,
    },
}

#[derive(Subcommand)]
pub enum EngineCmd {
    Hnsw,
    Colbert,
    Plaid,
    Warp,
    Tachiom,
}

#[derive(Subcommand)]
pub enum BenchTarget {
    All,
    Hnsw,
    Colbert,
    Plaid,
    Warp,
    Tachiom,
    /// Validate that SIFT files exist at MULTIVECTOR_SIFT_PATH; prints download instructions if missing.
    CheckSift,
}
