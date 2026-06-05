use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub engine: String,
    pub dataset: String,
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub qps: f64,
    pub index_build_s: f64,
    pub peak_ram_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStats {
    pub duration_s: f64,
    pub doc_count: u64,
    pub index_size_bytes: u64,
}

/// Data passed to scripts/plot.py.
#[derive(Debug, Serialize, Deserialize)]
pub struct PlotData {
    pub results: Vec<BenchResult>,
}
