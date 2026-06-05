use crate::maxsim::{maxsim, maxsim_with_matrix};
use common::{TokenMatrix, TraceEvent, TraceLog};

pub struct ColBertIndex {
    /// One TokenMatrix per indexed document.
    pub docs: Vec<(u32 /* doc_id */, TokenMatrix)>,
}

impl ColBertIndex {
    pub fn new() -> Self {
        Self { docs: Vec::new() }
    }

    pub fn insert(&mut self, doc_id: u32, matrix: TokenMatrix) {
        if self.docs.iter().any(|(id, _)| *id == doc_id) {
            eprintln!("warning: doc_id {doc_id} already indexed; overwriting");
            self.docs.retain(|(id, _)| *id != doc_id);
        }
        self.docs.push((doc_id, matrix));
    }

    /// Brute-force MaxSim over all documents.
    pub fn search(&self, query_matrix: &TokenMatrix, top_k: usize) -> Vec<(u32, f32)> {
        let mut scores: Vec<(u32, f32)> = self
            .docs
            .iter()
            .map(|(doc_id, doc_matrix)| (*doc_id, maxsim(query_matrix, doc_matrix)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Same but emits MaxSimMatrix trace event per doc.
    pub fn search_with_trace(
        &self,
        query_matrix: &TokenMatrix,
        top_k: usize,
    ) -> (Vec<(u32, f32)>, TraceLog) {
        let mut log = TraceLog::default();
        let mut scores: Vec<(u32, f32)> = Vec::with_capacity(self.docs.len());

        for (doc_id, doc_matrix) in &self.docs {
            let (score, matrix, row_maxima) = maxsim_with_matrix(query_matrix, doc_matrix);
            log.push(TraceEvent::MaxSimMatrix {
                query_tokens: query_matrix.tokens.clone(),
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
