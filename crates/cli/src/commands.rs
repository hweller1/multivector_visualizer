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
    /// Accuracy-speed tradeoff benchmark: Recall@10 vs speedup for PLAID / WARP / TACHIOM.
    /// Generates SVG plots in plots/ directory.
    Tradeoff,
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
