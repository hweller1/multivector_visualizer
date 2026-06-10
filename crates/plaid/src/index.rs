use super::centroid::CentroidPruner;
use colbert::index::ColBertIndex;
use colbert::maxsim::maxsim_weighted;
use common::{
    token::{TokenMatrix, TOKEN_DIM},
    trace::{TraceEvent, TraceLog},
};

pub struct PlaidIndex {
    pub colbert: ColBertIndex,
    pub centroids: Vec<[f32; TOKEN_DIM]>,
    /// centroid_id → list of doc_ids that have at least one token assigned there
    pub centroid_to_docs: Vec<Vec<u32>>,
}

impl PlaidIndex {
    /// Build PLAID inverted index from a ColBertIndex.
    pub fn build(colbert_index: ColBertIndex, num_centroids: usize) -> Self {
        // Collect all token rows from all docs
        let all_token_rows: Vec<[f32; TOKEN_DIM]> = colbert_index
            .docs
            .iter()
            .flat_map(|(_, matrix)| matrix.rows.iter().copied())
            .collect();

        let pruner = CentroidPruner::new(num_centroids);
        let centroids = pruner.fit(&all_token_rows);
        let k = centroids.len();

        let mut centroid_to_docs: Vec<Vec<u32>> = vec![Vec::new(); k];

        for (doc_id, matrix) in colbert_index.docs.iter() {
            let mut assigned_centroids = std::collections::HashSet::new();
            for row in &matrix.rows {
                let ci = pruner.assign(&centroids, row) as usize;
                if ci < k {
                    assigned_centroids.insert(ci);
                }
            }
            for ci in assigned_centroids {
                if !centroid_to_docs[ci].contains(doc_id) {
                    centroid_to_docs[ci].push(*doc_id);
                }
            }
        }

        PlaidIndex {
            colbert: colbert_index,
            centroids,
            centroid_to_docs,
        }
    }

    /// PLAID search: centroid ANN → candidate expansion → MaxSim on candidates.
    pub fn search(
        &self,
        query_matrix: &TokenMatrix,
        top_k: usize,
        nprobe: usize,
    ) -> (Vec<(u32, f32)>, TraceLog) {
        let mut log = TraceLog::default();
        let pruner = CentroidPruner::new(self.centroids.len());

        let total_docs = self.colbert.docs.len();
        let mut candidate_doc_ids: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        let mut activated_centroid_ids: Vec<u32> = Vec::new();

        for row in &query_matrix.rows {
            let token_str = "query_token".to_string(); // We don't store token strings in the index
            let top_centroids = pruner.query_centroids(&self.centroids, row, nprobe);

            // Find which token string this is
            let token_idx = query_matrix.rows.iter().position(|r| r == row).unwrap_or(0);
            let token_name = query_matrix
                .tokens
                .get(token_idx)
                .cloned()
                .unwrap_or(token_str);

            log.push(TraceEvent::CentroidAnn {
                query_token: token_name,
                top_centroids: top_centroids.clone(),
            });

            for (ci, _) in &top_centroids {
                let ci_usize = *ci as usize;
                if ci_usize < self.centroid_to_docs.len() {
                    for &doc_id in &self.centroid_to_docs[ci_usize] {
                        candidate_doc_ids.insert(doc_id);
                    }
                    if !activated_centroid_ids.contains(ci) {
                        activated_centroid_ids.push(*ci);
                    }
                }
            }
        }

        let candidates: Vec<u32> = candidate_doc_ids.into_iter().collect();
        let pruned_count = (total_docs as u32).saturating_sub(candidates.len() as u32);

        log.push(TraceEvent::CandidateExpand {
            centroid_ids: activated_centroid_ids,
            candidate_doc_ids: candidates.clone(),
            pruned_count,
        });

        let idf = self.colbert.idf_weights(&query_matrix.tokens);
        // IDF-weighted MaxSim on candidates only
        let mut scores: Vec<(u32, f32)> = self
            .colbert
            .docs
            .iter()
            .filter(|(doc_id, _)| candidates.contains(doc_id))
            .map(|(doc_id, doc_matrix)| (*doc_id, maxsim_weighted(query_matrix, &idf, doc_matrix)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_results = scores.iter().take(top_k).copied().collect::<Vec<_>>();

        log.push(TraceEvent::PlaidMaxSim {
            candidate_count: candidates.len() as u32,
            scored_count: scores.len() as u32,
            top_k: top_k_results.clone(),
        });

        scores.truncate(top_k);
        (scores, log)
    }
}
