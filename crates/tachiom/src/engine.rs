use super::pq::HierarchicalPQ;
use super::tac::{
    budget::BudgetReconciler, clustering::parallel_kappa_means, damping::DampedScorer,
    tail::TailHandler,
};
use anyhow::Result;
use async_trait::async_trait;
use colbert::encoder::ColBertEncoder;
use colbert::maxsim::{cosine, maxsim_weighted};
use common::{token::TOKEN_DIM, trace::TraceEvent, Engine, OpTiming, TokenMatrix, TraceLog};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

pub struct TachiomEngine {
    encoder: RefCell<ColBertEncoder>,
    tail_handler: TailHandler,
    damped_scorer: Option<DampedScorer>,
    budget_reconciler: BudgetReconciler,
    pub centroids_by_type: HashMap<u32, Vec<[f32; TOKEN_DIM]>>,
    pq: HierarchicalPQ,
    pub token_type_embeddings: HashMap<u32, Vec<[f32; TOKEN_DIM]>>,
    /// vocab_id → Vec<(doc_id, embedding)> — tracks per-doc token occurrences for centroid assignment.
    token_type_doc_embs: HashMap<u32, Vec<(u32, [f32; TOKEN_DIM])>>,
    /// vocab_id → Vec<(doc_id, centroid_idx)> — rebuilt after each TAC rebuild.
    doc_centroid_map: HashMap<u32, Vec<(u32, usize)>>,
    /// vocab_id → number of docs containing it (for IDF).
    token_doc_freq: HashMap<u32, usize>,
    colbert_docs: Vec<(u32, TokenMatrix)>,
    doc_vocab_ids: Vec<(u32, Vec<u32>)>,
    /// token_id → token string, built incrementally during indexing
    vocab_map: HashMap<u32, String>,
    /// Phase trace events from the last TAC rebuild — tail/damped/budget
    tac_log: TraceLog,
}

unsafe impl Sync for TachiomEngine {}

impl TachiomEngine {
    pub fn new(vocab_path: impl AsRef<Path>) -> Result<Self> {
        let encoder = ColBertEncoder::new(vocab_path.as_ref(), 0x0123456789ABCDEF)?;
        Ok(Self {
            encoder: RefCell::new(encoder),
            tail_handler: TailHandler::new(),
            damped_scorer: None,
            budget_reconciler: BudgetReconciler::new(200),
            centroids_by_type: HashMap::new(),
            pq: HierarchicalPQ::new(),
            token_type_embeddings: HashMap::new(),
            token_type_doc_embs: HashMap::new(),
            doc_centroid_map: HashMap::new(),
            token_doc_freq: HashMap::new(),
            colbert_docs: Vec::new(),
            doc_vocab_ids: Vec::new(),
            vocab_map: HashMap::new(),
            tac_log: TraceLog::default(),
        })
    }

    fn rebuild_tac(&mut self) {
        // Phase 1: tail events
        let raw_tail = self.tail_handler.trace_all();

        // Phase 2: damped scorer
        let damped = DampedScorer::compute(&self.token_type_embeddings);
        let raw_damped = damped.trace_all(&self.token_type_embeddings);

        // Phase 3: budget allocation
        let mut kappa = self.budget_reconciler.allocate(&damped.weights);
        let raw_budget = self.budget_reconciler.trace_allocate(&damped.weights);

        // Phase 4: reconcile
        let raw_reconcile = self.budget_reconciler.reconcile(&mut kappa, &damped.weights);

        // Build tac_log: replace numeric token_type strings with vocab text
        let vm = &self.vocab_map;
        let mut log = TraceLog::default();
        for e in raw_tail    { log.push(resolve_token_type(e, vm)); }
        for e in raw_damped  { log.push(resolve_token_type(e, vm)); }
        for e in raw_budget  { log.push(resolve_token_type(e, vm)); }
        for e in raw_reconcile { log.push(e); }
        self.tac_log = log;

        // Phase 4: k-means
        let centroids = parallel_kappa_means(&self.token_type_embeddings, &kappa);
        self.centroids_by_type = centroids;
        self.damped_scorer = Some(damped);

        // Build doc_centroid_map: for each vocab_id, assign each doc's token to its nearest centroid.
        let mut dcm: HashMap<u32, Vec<(u32, usize)>> = HashMap::new();
        for (&vid, doc_embs) in &self.token_type_doc_embs {
            if let Some(cents) = self.centroids_by_type.get(&vid) {
                for &(doc_id, emb) in doc_embs {
                    let best_ci = cents.iter().enumerate()
                        .map(|(ci, c)| (ci, cosine(&emb, c)))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(ci, _)| ci)
                        .unwrap_or(0);
                    dcm.entry(vid).or_default().push((doc_id, best_ci));
                }
            }
        }
        self.doc_centroid_map = dcm;
    }
}

/// Replace numeric token_type strings ("12345") with readable vocab text.
fn resolve_token_type(event: TraceEvent, vm: &HashMap<u32, String>) -> TraceEvent {
    let resolve = |s: String| -> String {
        s.parse::<u32>()
            .ok()
            .and_then(|id| vm.get(&id))
            .cloned()
            .unwrap_or(s)
    };
    match event {
        TraceEvent::TailHandle { token_type, count, classification } =>
            TraceEvent::TailHandle { token_type: resolve(token_type), count, classification },
        TraceEvent::DampedScore { token_type, variance, weight } =>
            TraceEvent::DampedScore { token_type: resolve(token_type), variance, weight },
        TraceEvent::BudgetBound { token_type, raw_kappa, floored, ceiled, final_kappa } =>
            TraceEvent::BudgetBound { token_type: resolve(token_type), raw_kappa, floored, ceiled, final_kappa },
        other => other,
    }
}

fn filter_tac_log(log: &TraceLog, filter: &str) -> TraceLog {
    let mut out = TraceLog::default();
    for (ts, event) in &log.events {
        let keep = match (filter, event) {
            ("tail",   TraceEvent::TailHandle { .. })    => true,
            ("damped", TraceEvent::DampedScore { .. })   => true,
            ("budget", TraceEvent::BudgetBound { .. })   => true,
            ("budget", TraceEvent::BudgetReconcile { .. }) => true,
            _ => false,
        };
        if keep {
            out.events.push((*ts, event.clone()));
        }
    }
    out
}

#[async_trait]
impl Engine for TachiomEngine {
    fn name(&self) -> &'static str {
        "tachiom"
    }

    async fn index(&mut self, doc_id: u32, text: &str) -> Result<TraceLog> {
        let (matrix, vocab_ids, log) = self.encoder.borrow_mut().encode_with_trace(doc_id, text)?;

        // Build vocab_map from this doc's tokens
        for (token_str, &vid) in matrix.tokens.iter().zip(vocab_ids.iter()) {
            self.vocab_map.entry(vid).or_insert_with(|| token_str.clone());
        }

        // Clean up per-doc structures if re-indexing an existing doc.
        if self.colbert_docs.iter().any(|(id, _)| *id == doc_id) {
            for entries in self.token_type_doc_embs.values_mut() {
                entries.retain(|(id, _)| *id != doc_id);
            }
            if let Some((_, old_vids)) = self.doc_vocab_ids.iter().find(|(id, _)| *id == doc_id) {
                let unique_old: std::collections::HashSet<u32> = old_vids.iter().copied().collect();
                for vid in unique_old {
                    if let Some(f) = self.token_doc_freq.get_mut(&vid) {
                        *f = f.saturating_sub(1);
                    }
                }
            }
        }
        self.colbert_docs.retain(|(id, _)| *id != doc_id);
        self.doc_vocab_ids.retain(|(id, _)| *id != doc_id);

        // Phase 1: tail handler + store embeddings
        for (&token_id, &row) in vocab_ids.iter().zip(matrix.rows.iter()) {
            self.tail_handler.update(token_id);
            self.token_type_embeddings.entry(token_id).or_default().push(row);
            self.token_type_doc_embs.entry(token_id).or_default().push((doc_id, row));
        }
        // Track per-doc token freq for IDF (count unique vocab_ids per doc).
        let unique_vids: std::collections::HashSet<u32> = vocab_ids.iter().copied().collect();
        for vid in &unique_vids {
            *self.token_doc_freq.entry(*vid).or_insert(0) += 1;
        }

        self.colbert_docs.push((doc_id, matrix));
        self.doc_vocab_ids.push((doc_id, vocab_ids));

        self.rebuild_tac();
        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let start = std::time::Instant::now();
        let (query_matrix, query_vocab_ids) = self.encoder.borrow_mut().encode(text)?;
        let mut log = TraceLog::default();

        let mut candidate_doc_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for (&qvid, query_row) in query_vocab_ids.iter().zip(query_matrix.rows.iter()) {
            if let Some(type_centroids) = self.centroids_by_type.get(&qvid) {
                if !type_centroids.is_empty() {
                    // Probe the top half of centroids (at least 1) for this token type.
                    let nprobe = (type_centroids.len() / 2).max(1);
                    let mut scored: Vec<(usize, f32)> = type_centroids.iter().enumerate()
                        .map(|(ci, c)| (ci, cosine(query_row, c)))
                        .collect();
                    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let top_cis: Vec<usize> = scored.iter().take(nprobe).map(|(ci, _)| *ci).collect();

                    // Include docs whose tokens of this type are assigned to a probed centroid.
                    if let Some(entries) = self.doc_centroid_map.get(&qvid) {
                        for &(doc_id, ci) in entries {
                            if top_cis.contains(&ci) {
                                candidate_doc_ids.insert(doc_id);
                            }
                        }
                    } else {
                        for (doc_id, vocab_ids) in &self.doc_vocab_ids {
                            if vocab_ids.contains(&qvid) {
                                candidate_doc_ids.insert(*doc_id);
                            }
                        }
                    }
                }
            } else {
                for (doc_id, _) in &self.colbert_docs {
                    candidate_doc_ids.insert(*doc_id);
                }
            }
        }

        let gather_ms = start.elapsed().as_secs_f64() * 1000.0;
        let candidates: Vec<u32> = candidate_doc_ids.into_iter().collect();
        let refine_start = std::time::Instant::now();

        // Compute IDF weights for this query.
        let n_docs = self.colbert_docs.len().max(1) as f32;
        let idf: Vec<f32> = query_vocab_ids.iter().map(|vid| {
            let df = *self.token_doc_freq.get(vid).unwrap_or(&0) as f32;
            (n_docs / (1.0 + df)).ln() + 1.0
        }).collect();

        let mut scores: Vec<(u32, f32)> = if candidates.is_empty() {
            self.colbert_docs.iter()
                .map(|(doc_id, doc_matrix)| (*doc_id, maxsim_weighted(&query_matrix, &idf, doc_matrix)))
                .collect()
        } else {
            self.colbert_docs.iter()
                .filter(|(doc_id, _)| candidates.contains(doc_id))
                .map(|(doc_id, doc_matrix)| (*doc_id, maxsim_weighted(&query_matrix, &idf, doc_matrix)))
                .collect()
        };

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        let refine_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        log.push(TraceEvent::TachiomSearch {
            timings: common::trace::TachiomTimings { gather_ms, refine_ms, total_ms },
        });
        let docs_scored = if candidates.is_empty() { self.colbert_docs.len() } else { candidates.len() };
        log.timing = Some(OpTiming {
            embed_ms: None,
            search_ms: Some(total_ms),
            docs_scored: Some(docs_scored),
        });

        Ok((scores, log))
    }

    async fn inspect(&self, target: Option<&str>) -> Result<String> {
        match target {
            // Phase trace logs — returned as JSON for the demo dispatcher to render
            Some("__trace__tail")   => Ok(serde_json::to_string(&filter_tac_log(&self.tac_log, "tail"))?),
            Some("__trace__damped") => Ok(serde_json::to_string(&filter_tac_log(&self.tac_log, "damped"))?),
            Some("__trace__budget") => Ok(serde_json::to_string(&filter_tac_log(&self.tac_log, "budget"))?),

            Some("pq") => Ok(self.pq.inspect()),
            Some("centroids") => {
                let mut out = format!("TACHIOM: {} token types with centroids\n", self.centroids_by_type.len());
                let mut types: Vec<u32> = self.centroids_by_type.keys().copied().collect();
                types.sort();
                for tt in types.iter().take(20) {
                    let n = self.centroids_by_type[tt].len();
                    let tok = self.vocab_map.get(tt).cloned().unwrap_or_else(|| format!("#{tt}"));
                    out.push_str(&format!("  \"{tok}\": {n} centroids\n"));
                }
                if types.len() > 20 {
                    out.push_str(&format!("  ... and {} more\n", types.len() - 20));
                }
                Ok(out)
            }
            None => Ok(format!(
                "TACHIOM: {} docs indexed, {} token types, {} vocab entries",
                self.colbert_docs.len(),
                self.centroids_by_type.len(),
                self.vocab_map.len(),
            )),
            Some(other) => Ok(format!("Unknown target '{other}'. Available: pq, centroids")),
        }
    }

    fn verify(&mut self) -> Result<()> {
        crate::verify::run(self)
    }
}
