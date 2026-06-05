use crate::atlas::AtlasClient;
use crate::local::{mock_embedding, LocalHnsw};
use anyhow::Result;
use async_trait::async_trait;
use common::{Engine, TraceEvent, TraceLog};

pub struct HnswEngine {
    pub local: LocalHnsw,
    pub atlas: Option<AtlasClient>,
}

// Safety: single-threaded demo use only
unsafe impl Sync for HnswEngine {}

impl HnswEngine {
    /// Local-only mode: uses deterministic mock embeddings, no network.
    pub fn new_local() -> Self {
        Self {
            local: LocalHnsw::new(),
            atlas: None,
        }
    }

    /// Atlas mode: requires MONGODB_URI and VOYAGE_API_KEY in env / .env file.
    pub async fn new_atlas() -> Result<Self> {
        let atlas = AtlasClient::from_env().await?;
        Ok(Self {
            local: LocalHnsw::new(),
            atlas: Some(atlas),
        })
    }
}

#[async_trait]
impl Engine for HnswEngine {
    fn name(&self) -> &'static str {
        "hnsw"
    }

    async fn index(&mut self, doc_id: u32, text: &str) -> Result<TraceLog> {
        let mut log = TraceLog::default();

        if let Some(atlas) = &self.atlas {
            let embedding = atlas.index_doc(doc_id, text).await?;
            log.push(TraceEvent::HnswInsert {
                doc_id,
                layer: 0,
                neighbors: vec![],
            });
            self.local.insert_raw(doc_id, embedding);
        } else {
            self.local.insert_mock(doc_id);
            log.push(TraceEvent::HnswInsert {
                doc_id,
                layer: 0,
                neighbors: vec![],
            });
        }

        log.push(TraceEvent::HnswLayerStats {
            layer: 0,
            node_count: self.local.len() as u32,
            avg_degree: 0.0,
        });

        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let mut log = TraceLog::default();

        let results = if let Some(atlas) = &self.atlas {
            let embedding = atlas.embed_query(text).await?;
            let results = atlas.query(&embedding, top_k).await?;
            log.push(TraceEvent::HnswQuery {
                hop: 0,
                current: 0,
                candidates: results
                    .iter()
                    .map(|(id, s)| (*id, *s))
                    .collect(),
            });
            results
        } else {
            // Deterministic local query: hash text to a doc_id seed for mock embedding
            let hash = text
                .bytes()
                .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            let query_vec = mock_embedding(hash);
            let results = self.local.search(&query_vec, top_k);
            log.push(TraceEvent::HnswQuery {
                hop: 0,
                current: hash,
                candidates: results.iter().map(|(id, s)| (*id, *s)).collect(),
            });
            results
        };

        Ok((results, log))
    }

    async fn inspect(&self, target: Option<&str>) -> Result<String> {
        match target {
            Some("layers") => {
                let stats = if let Some(atlas) = &self.atlas {
                    atlas.graph_stats().await?
                } else {
                    vec![]
                };
                if stats.is_empty() {
                    Ok(format!(
                        "HNSW local index: {} docs inserted\n\
                         Layer stats not available (Atlas stats API is cluster-specific).",
                        self.local.len()
                    ))
                } else {
                    let mut out = String::new();
                    for s in stats {
                        out.push_str(&format!(
                            "Layer {}: {} nodes, avg_degree={:.1}\n",
                            s.layer, s.node_count, s.avg_degree
                        ));
                    }
                    Ok(out)
                }
            }
            Some("graph") => Ok(format!(
                "HNSW graph: {} docs, ef_construction=200, max_nb_connection=16\n\
                 (Detailed graph degree stats require direct hnsw_rs introspection)",
                self.local.len()
            )),
            None => Ok(format!(
                "HNSW engine: {} docs indexed. Mode: {}\nAvailable: layers, graph",
                self.local.len(),
                if self.atlas.is_some() { "Atlas" } else { "local mock" }
            )),
            Some(other) => Ok(format!("Unknown target '{other}'. Available: layers, graph")),
        }
    }

    fn verify(&mut self) -> Result<()> {
        crate::verify::run(self)
    }
}
