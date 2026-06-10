use crate::maxsim::{maxsim_weighted, maxsim_with_matrix};
use common::{TokenMatrix, TraceEvent, TraceLog};
use std::collections::{HashMap, HashSet};

pub struct ColBertIndex {
    /// One TokenMatrix per indexed document.
    pub docs: Vec<(u32 /* doc_id */, TokenMatrix)>,
    /// token string → number of docs containing it (for IDF computation).
    token_doc_freq: HashMap<String, usize>,
}

impl ColBertIndex {
    pub fn new() -> Self {
        Self { docs: Vec::new(), token_doc_freq: HashMap::new() }
    }

    pub fn insert(&mut self, doc_id: u32, matrix: TokenMatrix) {
        if let Some(pos) = self.docs.iter().position(|(id, _)| *id == doc_id) {
            eprintln!("warning: doc_id {doc_id} already indexed; overwriting");
            let old_unique: HashSet<String> = self.docs[pos].1.tokens.iter().cloned().collect();
            for tok in old_unique {
                if let Some(f) = self.token_doc_freq.get_mut(&tok) {
                    *f = f.saturating_sub(1);
                }
            }
            self.docs.remove(pos);
        }
        let unique: HashSet<&String> = matrix.tokens.iter().collect();
        for tok in unique {
            *self.token_doc_freq.entry(tok.clone()).or_insert(0) += 1;
        }
        self.docs.push((doc_id, matrix));
    }

    /// Smoothed IDF weight for each query token: ln(N / (1 + df)) + 1.
    /// Stopwords that appear in every doc approach weight 1.0; rare tokens get higher weights.
    pub fn idf_weights(&self, query_tokens: &[String]) -> Vec<f32> {
        let n = self.docs.len().max(1) as f32;
        query_tokens.iter().map(|tok| {
            let df = *self.token_doc_freq.get(tok).unwrap_or(&0) as f32;
            (n / (1.0 + df)).ln() + 1.0
        }).collect()
    }

    /// IDF-weighted MaxSim over all documents.
    pub fn search(&self, query_matrix: &TokenMatrix, top_k: usize) -> Vec<(u32, f32)> {
        let weights = self.idf_weights(&query_matrix.tokens);
        let mut scores: Vec<(u32, f32)> = self
            .docs
            .iter()
            .map(|(doc_id, doc_matrix)| (*doc_id, maxsim_weighted(query_matrix, &weights, doc_matrix)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Same but emits MaxSimMatrix trace event per doc (IDF-weighted scores).
    pub fn search_with_trace(
        &self,
        query_matrix: &TokenMatrix,
        top_k: usize,
    ) -> (Vec<(u32, f32)>, TraceLog) {
        let weights = self.idf_weights(&query_matrix.tokens);
        let mut log = TraceLog::default();
        let mut scores: Vec<(u32, f32)> = Vec::with_capacity(self.docs.len());

        for (doc_id, doc_matrix) in &self.docs {
            let (_, matrix, row_maxima) = maxsim_with_matrix(query_matrix, doc_matrix);
            // Weighted score: each row maximum scaled by its IDF weight.
            let score: f32 = row_maxima.iter().zip(weights.iter().chain(std::iter::repeat(&1.0f32)))
                .map(|(m, w)| w * m)
                .sum();
            log.push(TraceEvent::MaxSimMatrix {
                query_tokens: query_matrix.tokens.clone(),
                doc_tokens: doc_matrix.tokens.clone(),
                doc_id: *doc_id,
                matrix,
                row_maxima,
                score,
            });
            scores.push((*doc_id, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        (scores, log)
    }
}

impl Default for ColBertIndex {
    fn default() -> Self {
        Self::new()
    }
}
