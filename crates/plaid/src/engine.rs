use super::index::PlaidIndex;
use anyhow::Result;
use async_trait::async_trait;
use colbert::{index::ColBertIndex, ColBertEngine};
use common::{Engine, OpTiming, TraceLog};
use std::path::Path;
use std::time::Instant;

pub struct PlaidEngine {
    pub colbert_engine: ColBertEngine,
    pub plaid_index: Option<PlaidIndex>,
    pub num_centroids: usize,
}

// Safety: single-threaded demo use only
unsafe impl Sync for PlaidEngine {}

impl PlaidEngine {
    pub fn new(vocab_path: impl AsRef<Path>) -> Result<Self> {
        let colbert_engine = ColBertEngine::new(vocab_path.as_ref())?;
        Ok(Self {
            colbert_engine,
            plaid_index: None,
            num_centroids: 8,
        })
    }

    fn rebuild_plaid(&mut self) {
        let mut new_colbert = ColBertIndex::new();
        for (doc_id, matrix) in &self.colbert_engine.index.docs {
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
        let log = self.colbert_engine.index(doc_id, text).await?;
        self.rebuild_plaid();
        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let t = Instant::now();
        let (query_matrix, _vocab_ids) = self.colbert_engine.encoder.borrow_mut().encode(text)?;
        let total_docs = self.colbert_engine.index.docs.len();
        if let Some(plaid) = &self.plaid_index {
            // Probe 75% of centroids — more recall at acceptable cost vs. the old 50%.
            let nprobe = ((self.num_centroids * 3) / 4).max(2);
            let (results, mut log) = plaid.search(&query_matrix, top_k, nprobe);
            // docs_scored = candidate count from CandidateExpand event
            let docs_scored = log.events.iter().find_map(|(_, e)| {
                if let common::TraceEvent::CandidateExpand { candidate_doc_ids, .. } = e {
                    Some(candidate_doc_ids.len())
                } else { None }
            }).unwrap_or(total_docs);
            log.timing = Some(OpTiming {
                embed_ms: None,
                search_ms: Some(t.elapsed().as_secs_f64() * 1000.0),
                docs_scored: Some(docs_scored),
            });
            Ok((results, log))
        } else {
            let (results, mut log) = self
                .colbert_engine
                .index
                .search_with_trace(&query_matrix, top_k);
            log.timing = Some(OpTiming {
                embed_ms: None,
                search_ms: Some(t.elapsed().as_secs_f64() * 1000.0),
                docs_scored: Some(total_docs),
            });
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
                let doc_count = self.colbert_engine.index.docs.len();
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
