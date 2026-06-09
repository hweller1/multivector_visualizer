use colbert::maxsim::cosine;
use common::token::{TokenMatrix, TOKEN_DIM};
use common::trace::{TraceEvent, TraceLog};

type DocTokenList = Vec<(u32, [f32; TOKEN_DIM])>;

pub struct XtrScorer {
    /// (doc_id, [(vocab_id, embedding)])
    doc_tokens: Vec<(u32, DocTokenList)>,
}

impl XtrScorer {
    pub fn new() -> Self {
        Self {
            doc_tokens: Vec::new(),
        }
    }

    pub fn register_doc(&mut self, doc_id: u32, tokens: &[(u32, [f32; TOKEN_DIM])]) {
        // Replace if already registered
        self.doc_tokens.retain(|(id, _)| *id != doc_id);
        self.doc_tokens.push((doc_id, tokens.to_vec()));
    }

    /// Score all docs for the query using XTR approach.
    /// For each query token, for each doc, compute max dot product with any doc token.
    /// Aggregate by taking max across query tokens per doc.
    /// Filter: only docs where max score > t_prime.
    /// Sort desc, truncate to bound.
    pub fn score(
        &self,
        query_matrix: &TokenMatrix,
        t_prime: f32,
        bound: usize,
    ) -> (Vec<(u32, f32)>, TraceLog) {
        let mut log = TraceLog::default();

        // For each doc, track the max xtr score across all query tokens
        let mut doc_max_scores: std::collections::HashMap<u32, f32> =
            std::collections::HashMap::new();

        for (qi, query_row) in query_matrix.rows.iter().enumerate() {
            let query_token_id = qi as u32; // Use index as pseudo vocab_id

            let mut token_scores: Vec<(u32, f32)> = Vec::new();

            for (doc_id, doc_token_list) in &self.doc_tokens {
                // Max dot product of this query token with any doc token
                let max_score = doc_token_list
                    .iter()
                    .map(|(_, emb)| cosine(query_row, emb))
                    .fold(f32::NEG_INFINITY, f32::max);

                token_scores.push((*doc_id, max_score));

                // Update global max for this doc
                let entry = doc_max_scores.entry(*doc_id).or_insert(f32::NEG_INFINITY);
                if max_score > *entry {
                    *entry = max_score;
                }
            }

            log.push(TraceEvent::XtrScore {
                query_token_id,
                query_token: query_matrix.tokens.get(qi).cloned().unwrap_or_default(),
                token_scores,
            });
        }

        // Filter by t_prime
        let mut results: Vec<(u32, f32)> = doc_max_scores
            .into_iter()
            .filter(|(_, score)| *score > t_prime)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(bound);

        (results, log)
    }
}

impl Default for XtrScorer {
    fn default() -> Self {
        Self::new()
    }
}
