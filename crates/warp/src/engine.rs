use super::gather::{gather_candidates, GatherStats};
use super::xtr::XtrScorer;
use anyhow::Result;
use async_trait::async_trait;
use colbert::{encoder::ColBertEncoder, index::ColBertIndex};
use common::{Engine, TraceEvent, TraceLog};
use std::cell::RefCell;
use std::path::Path;

pub struct WarpEngine {
    pub colbert_encoder: RefCell<ColBertEncoder>,
    pub colbert_index: ColBertIndex,
    pub xtr_scorer: XtrScorer,
    pub t_prime: f32,
    pub bound: usize,
    pub last_gather_stats: Option<GatherStats>,
}

// Safety: single-threaded demo use only
unsafe impl Sync for WarpEngine {}

impl WarpEngine {
    pub fn new(vocab_path: impl AsRef<Path>) -> Result<Self> {
        let encoder = ColBertEncoder::new(vocab_path.as_ref(), 0x0123456789ABCDEF)?;
        Ok(Self {
            colbert_encoder: RefCell::new(encoder),
            colbert_index: ColBertIndex::new(),
            xtr_scorer: XtrScorer::new(),
            t_prime: 0.3,
            bound: 100,
            last_gather_stats: None,
        })
    }
}

#[async_trait]
impl Engine for WarpEngine {
    fn name(&self) -> &'static str {
        "warp"
    }

    async fn index(&mut self, doc_id: u32, text: &str) -> Result<TraceLog> {
        let (matrix, vocab_ids, log) = self
            .colbert_encoder
            .borrow_mut()
            .encode_with_trace(doc_id, text)?;

        // Register in XTR scorer: pair each token embedding with its vocab_id
        let token_pairs: Vec<(u32, [f32; common::token::TOKEN_DIM])> = vocab_ids
            .iter()
            .zip(matrix.rows.iter())
            .map(|(&vid, &emb)| (vid, emb))
            .collect();
        self.xtr_scorer.register_doc(doc_id, &token_pairs);

        self.colbert_index.insert(doc_id, matrix);
        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let (query_matrix, _vocab_ids) = self.colbert_encoder.borrow_mut().encode(text)?;
        let total_docs = self.colbert_index.docs.len();

        // Step 1: XTR scoring
        let (xtr_results, mut log) = self
            .xtr_scorer
            .score(&query_matrix, self.t_prime, self.bound);

        // Step 2: Gather candidates
        let (candidates, _stats, gather_log) = gather_candidates(&xtr_results, total_docs);
        for event in gather_log.events {
            log.events.push(event);
        }

        // Step 3: MaxSim refine on gathered candidates
        let mut scores: Vec<(u32, f32)> = if candidates.is_empty() {
            // If XTR pruned everything, fall back to all docs
            self.colbert_index
                .docs
                .iter()
                .map(|(doc_id, doc_matrix)| {
                    (*doc_id, colbert::maxsim::maxsim(&query_matrix, doc_matrix))
                })
                .collect()
        } else {
            self.colbert_index
                .docs
                .iter()
                .filter(|(doc_id, _)| candidates.contains(doc_id))
                .map(|(doc_id, doc_matrix)| {
                    (*doc_id, colbert::maxsim::maxsim(&query_matrix, doc_matrix))
                })
                .collect()
        };

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_results = scores.iter().take(top_k).copied().collect::<Vec<_>>();

        log.push(TraceEvent::MaxSimRefine {
            candidate_count: candidates.len() as u32,
            top_k: top_k_results.clone(),
        });

        scores.truncate(top_k);
        Ok((scores, log))
    }

    async fn inspect(&self, target: Option<&str>) -> Result<String> {
        match target {
            Some("gather") => {
                if let Some(stats) = &self.last_gather_stats {
                    Ok(format!(
                        "Last gather: {} docs promoted ({:.1}% of index)",
                        stats.gathered.len(),
                        stats.fraction_promoted * 100.0
                    ))
                } else {
                    Ok("No query run yet. Run a query to see gather stats.".to_string())
                }
            }
            None => Ok(format!(
                "WARP index: {} documents, t_prime={}, bound={}",
                self.colbert_index.docs.len(),
                self.t_prime,
                self.bound
            )),
            Some(other) => Ok(format!("Unknown target '{other}'. Available: gather")),
        }
    }

    fn verify(&mut self) -> Result<()> {
        crate::verify::run(self)
    }
}
