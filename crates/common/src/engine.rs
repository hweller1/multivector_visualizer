use crate::{BenchResult, TraceLog};
use anyhow::Result;

/// Implemented by every retrieval engine (educational + bench modes).
#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    /// Human-readable name shown in REPL prompt and comparison table.
    fn name(&self) -> &'static str;

    /// Index a document; returns a TraceLog of events emitted.
    async fn index(&mut self, doc_id: u32, text: &str) -> Result<TraceLog>;

    /// Run a query; returns top-k (doc_id, score) pairs and a TraceLog.
    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)>;

    /// Engine-specific inspect targets.
    async fn inspect(&self, target: Option<&str>) -> Result<String>;

    /// Run the deterministic verification harness; panics on assertion failure.
    fn verify(&mut self) -> Result<()>;
}

// Suppress unused import warning for BenchResult (used in trait signature context)
const _: fn() = || {
    let _: Option<BenchResult> = None;
};
