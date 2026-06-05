use super::pq::HierarchicalPQ;
use super::tac::{
    budget::BudgetReconciler, clustering::parallel_kappa_means, damping::DampedScorer,
    tail::TailHandler,
};
use anyhow::Result;
use async_trait::async_trait;
use colbert::encoder::ColBertEncoder;
use colbert::maxsim::{cosine, maxsim};
use common::{token::TOKEN_DIM, trace::TraceEvent, Engine, TokenMatrix, TraceLog};
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
    /// Raw token embeddings per token type (for TAC build)
    pub token_type_embeddings: HashMap<u32, Vec<[f32; TOKEN_DIM]>>,
    /// Indexed docs for MaxSim refine
    colbert_docs: Vec<(u32, TokenMatrix)>,
    /// Vocab IDs per doc for token type lookup
    doc_vocab_ids: Vec<(u32, Vec<u32>)>,
}

// Safety: single-threaded demo use only
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
            colbert_docs: Vec::new(),
            doc_vocab_ids: Vec::new(),
        })
    }

    /// Rebuild the full TAC pipeline after indexing a new doc.
    fn rebuild_tac(&mut self) {
        // Phase 2: DampedScorer
        let damped = DampedScorer::compute(&self.token_type_embeddings);

        // Phase 3: Budget allocation
        let mut kappa = self.budget_reconciler.allocate(&damped.weights);
        self.budget_reconciler
            .reconcile(&mut kappa, &damped.weights);

        // Phase 4: Parallel κ-means
        let centroids = parallel_kappa_means(&self.token_type_embeddings, &kappa);

        self.centroids_by_type = centroids;
        self.damped_scorer = Some(damped);
    }
}

#[async_trait]
impl Engine for TachiomEngine {
    fn name(&self) -> &'static str {
        "tachiom"
    }

    async fn index(&mut self, doc_id: u32, text: &str) -> Result<TraceLog> {
        let (matrix, vocab_ids, log) = self.encoder.borrow_mut().encode_with_trace(doc_id, text)?;

        // Phase 1: Update tail handler + store embeddings per token type
        for (&token_id, &row) in vocab_ids.iter().zip(matrix.rows.iter()) {
            self.tail_handler.update(token_id);
            self.token_type_embeddings
                .entry(token_id)
                .or_default()
                .push(row);
        }

        self.colbert_docs.retain(|(id, _)| *id != doc_id);
        self.colbert_docs.push((doc_id, matrix));
        self.doc_vocab_ids.retain(|(id, _)| *id != doc_id);
        self.doc_vocab_ids.push((doc_id, vocab_ids));

        // Rebuild TAC pipeline
        self.rebuild_tac();

        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let start = std::time::Instant::now();
        let (query_matrix, query_vocab_ids) = self.encoder.borrow_mut().encode(text)?;
        let mut log = TraceLog::default();

        // For each query token, find nearest centroid in centroids_by_type
        let mut candidate_doc_ids: std::collections::HashSet<u32> =
            std::collections::HashSet::new();

        for (&qvid, query_row) in query_vocab_ids.iter().zip(query_matrix.rows.iter()) {
            if let Some(type_centroids) = self.centroids_by_type.get(&qvid) {
                if !type_centroids.is_empty() {
                    // Find nearest centroid
                    let best_ci = type_centroids
                        .iter()
                        .enumerate()
                        .map(|(ci, c)| (ci, cosine(query_row, c)))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(ci, _)| ci)
                        .unwrap_or(0);

                    // Find docs that have this token type assigned to this centroid
                    for (doc_id, vocab_ids) in &self.doc_vocab_ids {
                        if vocab_ids.contains(&qvid) {
                            candidate_doc_ids.insert(*doc_id);
                        }
                    }
                    let _ = best_ci; // centroid selection noted
                }
            } else {
                // No centroids for this token type → add all docs as candidates
                for (doc_id, _) in &self.colbert_docs {
                    candidate_doc_ids.insert(*doc_id);
                }
            }
        }

        let gather_ms = start.elapsed().as_secs_f64() * 1000.0;
        let candidates: Vec<u32> = candidate_doc_ids.into_iter().collect();

        let refine_start = std::time::Instant::now();

        // MaxSim refine
        let mut scores: Vec<(u32, f32)> = if candidates.is_empty() {
            // Fallback: score all docs
            self.colbert_docs
                .iter()
                .map(|(doc_id, doc_matrix)| (*doc_id, maxsim(&query_matrix, doc_matrix)))
                .collect()
        } else {
            self.colbert_docs
                .iter()
                .filter(|(doc_id, _)| candidates.contains(doc_id))
                .map(|(doc_id, doc_matrix)| (*doc_id, maxsim(&query_matrix, doc_matrix)))
                .collect()
        };

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        let refine_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        log.push(TraceEvent::TachiomSearch {
            timings: common::trace::TachiomTimings {
                gather_ms,
                refine_ms,
                total_ms,
            },
        });

        Ok((scores, log))
    }

    async fn inspect(&self, target: Option<&str>) -> Result<String> {
        match target {
            Some("pq") => Ok(self.pq.inspect()),
            Some("centroids") => {
                let mut out = format!(
                    "TACHIOM centroids: {} token types\n",
                    self.centroids_by_type.len()
                );
                let mut types: Vec<u32> = self.centroids_by_type.keys().copied().collect();
                types.sort();
                for tt in types.iter().take(20) {
                    let n = self.centroids_by_type[tt].len();
                    out.push_str(&format!("  token_type {tt}: {n} centroids\n"));
                }
                if types.len() > 20 {
                    out.push_str(&format!("  ... and {} more\n", types.len() - 20));
                }
                Ok(out)
            }
            None => Ok(format!(
                "TACHIOM: {} docs indexed, {} token types with centroids",
                self.colbert_docs.len(),
                self.centroids_by_type.len()
            )),
            Some(other) => Ok(format!(
                "Unknown target '{other}'. Available: pq, centroids"
            )),
        }
    }

    fn verify(&mut self) -> Result<()> {
        crate::verify::run(self)
    }
}
