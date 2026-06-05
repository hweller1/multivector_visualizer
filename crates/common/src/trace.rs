use serde::{Deserialize, Serialize};

/// One named event emitted by an engine pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "stage", content = "payload")]
pub enum TraceEvent {
    // ── HNSW ─────────────────────────────────────────────────────────────
    HnswInsert {
        doc_id: u32,
        layer: u8,
        neighbors: Vec<u32>,
    },
    HnswQuery {
        hop: u32,
        current: u32,
        candidates: Vec<(u32, f32)>,
    },
    HnswLayerStats {
        layer: u8,
        node_count: u32,
        avg_degree: f32,
    },

    // ── ColBERT ──────────────────────────────────────────────────────────
    Tokenize {
        doc_id: u32,
        tokens: Vec<String>,
    },
    TokenEmbed {
        doc_id: u32,
        token: String,
        embedding_preview: [f32; 3],
    },
    MaxSimMatrix {
        query_tokens: Vec<String>,
        doc_id: u32,
        matrix: Vec<Vec<f32>>,
        row_maxima: Vec<f32>,
        score: f32,
    },

    // ── PLAID ────────────────────────────────────────────────────────────
    CentroidAssign {
        doc_id: u32,
        token: String,
        centroid_id: u32,
    },
    CentroidAnn {
        query_token: String,
        top_centroids: Vec<(u32, f32)>,
    },
    CandidateExpand {
        centroid_ids: Vec<u32>,
        candidate_doc_ids: Vec<u32>,
        pruned_count: u32,
    },
    PlaidMaxSim {
        candidate_count: u32,
        scored_count: u32,
        top_k: Vec<(u32, f32)>,
    },

    // ── WARP ─────────────────────────────────────────────────────────────
    XtrScore {
        query_token_id: u32,
        token_scores: Vec<(u32, f32)>,
    },
    CandidateGather {
        gathered: Vec<u32>,
        overlap_with_gt: f32,
        fraction_promoted: f32,
    },
    MaxSimRefine {
        candidate_count: u32,
        top_k: Vec<(u32, f32)>,
    },

    // ── TACHIOM ──────────────────────────────────────────────────────────
    TailHandle {
        token_type: String,
        count: u32,
        classification: TailClass,
    },
    DampedScore {
        token_type: String,
        variance: f32,
        weight: f32,
    },
    BudgetBound {
        token_type: String,
        raw_kappa: f32,
        floored: u32,
        ceiled: u32,
        final_kappa: u32,
    },
    BudgetReconcile {
        total_budget: u32,
        allocated: u32,
        redistributed: u32,
    },
    PqInspect {
        level: u8,
        dimensions: u32,
        subquantizer_count: u32,
        code_bits: u8,
    },
    TachiomSearch {
        timings: TachiomTimings,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TailClass {
    Tail,
    Normal,
    Heavy,
}

/// Mirrors TACHIOM's built-in SearchTimings struct for integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TachiomTimings {
    pub gather_ms: f64,
    pub refine_ms: f64,
    pub total_ms: f64,
}

/// Ordered sequence of trace events for one operation.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TraceLog {
    pub events: Vec<(u64 /* epoch_ms */, TraceEvent)>,
}

impl TraceLog {
    pub fn push(&mut self, event: TraceEvent) {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.events.push((ms, event));
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Writes a TraceLog to a file path when `--trace-json <path>` is supplied.
pub struct JsonTracer {
    pub path: std::path::PathBuf,
}

impl JsonTracer {
    pub fn write(&self, log: &TraceLog) -> anyhow::Result<()> {
        std::fs::write(&self.path, log.to_json()?)?;
        Ok(())
    }
}
