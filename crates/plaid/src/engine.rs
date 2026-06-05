use super::index::PlaidIndex;
use anyhow::Result;
use async_trait::async_trait;
use colbert::{encoder::ColBertEncoder, index::ColBertIndex};
use common::{Engine, TraceLog};
use std::cell::RefCell;
use std::path::Path;

pub struct PlaidEngine {
    pub colbert_encoder: RefCell<ColBertEncoder>,
    pub colbert_index: ColBertIndex,
    pub plaid_index: Option<PlaidIndex>,
    pub num_centroids: usize,
}

// Safety: single-threaded demo use only
unsafe impl Sync for PlaidEngine {}

impl PlaidEngine {
    pub fn new(vocab_path: impl AsRef<Path>) -> Result<Self> {
        let encoder = ColBertEncoder::new(vocab_path.as_ref(), 0x0123456789ABCDEF)?;
        Ok(Self {
            colbert_encoder: RefCell::new(encoder),
            colbert_index: ColBertIndex::new(),
            plaid_index: None,
            num_centroids: 8,
        })
    }

    fn rebuild_plaid(&mut self) {
        // Clone the docs to build a fresh ColBertIndex for PlaidIndex::build
        let mut new_colbert = ColBertIndex::new();
        for (doc_id, matrix) in &self.colbert_index.docs {
            new_colbert.insert(*doc_id, matrix.clone());
        }
        self.plaid_index = Some(PlaidIndex::build(new_colbert, self.num_centroids));
    }
}

#[async_trait]
impl Engine for PlaidEngine {
    fn name(&self) -> &'static str {
        "plaid"
    }

    async fn index(&mut self, doc_id: u32, text: &str) -> Result<TraceLog> {
        let (matrix, _vocab_ids, log) = self
            .colbert_encoder
            .borrow_mut()
            .encode_with_trace(doc_id, text)?;
        self.colbert_index.insert(doc_id, matrix);
        self.rebuild_plaid();
        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let (query_matrix, _vocab_ids) = self.colbert_encoder.borrow_mut().encode(text)?;
        if let Some(plaid) = &self.plaid_index {
            let nprobe = (self.num_centroids / 2).max(1);
            Ok(plaid.search(&query_matrix, top_k, nprobe))
        } else {
            // Fallback: brute-force MaxSim
            let (results, log) = self.colbert_index.search_with_trace(&query_matrix, top_k);
            Ok((results, log))
        }
    }

    async fn inspect(&self, target: Option<&str>) -> Result<String> {
        match target {
            Some("centroids") => {
                if let Some(plaid) = &self.plaid_index {
                    let mut out = format!("PLAID centroids: {} total\n", plaid.centroids.len());
                    for (ci, docs) in plaid.centroid_to_docs.iter().enumerate() {
                        out.push_str(&format!("  centroid {ci}: {} docs\n", docs.len()));
                    }
                    Ok(out)
                } else {
                    Ok("No PLAID index built yet. Index some documents first.".to_string())
                }
            }
            None => {
                let doc_count = self.colbert_index.docs.len();
                Ok(format!(
                    "PLAID index: {} documents, {} centroids",
                    doc_count,
                    self.plaid_index
                        .as_ref()
                        .map(|p| p.centroids.len())
                        .unwrap_or(0)
                ))
            }
            Some(other) => Ok(format!("Unknown target '{other}'. Available: centroids")),
        }
    }

    fn verify(&mut self) -> Result<()> {
        crate::verify::run(self)
    }
}
