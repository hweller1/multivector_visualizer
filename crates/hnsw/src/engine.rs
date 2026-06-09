use crate::atlas::AtlasClient;
use crate::local::{mock_embedding, LocalHnsw};
use crate::voyage::VoyageClient;
use anyhow::Result;
use async_trait::async_trait;
use common::{Engine, OpTiming, TraceEvent, TraceLog};
use std::collections::HashMap;
use std::time::Instant;

pub struct HnswEngine {
    pub local: LocalHnsw,
    pub atlas: Option<AtlasClient>,
    doc_texts: HashMap<u32, String>,
    /// Pre-fetched real embeddings for all corpus docs (populated when VOYAGE_API_KEY is set).
    corpus_embeddings: HashMap<u32, Vec<f32>>,
    /// Live Voyage client for embedding query text at search time.
    voyage: Option<VoyageClient>,
}

// Safety: single-threaded demo use only
unsafe impl Sync for HnswEngine {}

impl HnswEngine {
    /// Local index with mock embeddings (no API key required).
    pub fn new_local() -> Self {
        Self {
            local: LocalHnsw::new(),
            atlas: None,
            doc_texts: HashMap::new(),
            corpus_embeddings: HashMap::new(),
            voyage: None,
        }
    }

    /// Local index that fetches real Voyage embeddings for the corpus and caches them on disk.
    /// Falls back silently to mock embeddings if `VOYAGE_API_KEY` is not set.
    pub async fn new_local_with_voyage() -> Self {
        let voyage = VoyageClient::from_env();
        let corpus_embeddings = match &voyage {
            Some(v) => v.load_corpus_embeddings().await.unwrap_or_else(|e| {
                eprintln!("  [voyage] embedding fetch failed: {e} — using mock embeddings");
                HashMap::new()
            }),
            None => HashMap::new(),
        };
        let using_real = !corpus_embeddings.is_empty();
        let eng = Self {
            local: LocalHnsw::new(),
            atlas: None,
            doc_texts: HashMap::new(),
            corpus_embeddings,
            voyage,
        };
        if using_real {
            let model = eng.voyage.as_ref().map(|v| v.model.as_str()).unwrap_or("unknown");
            println!("  [voyage] using real {model} embeddings");
        } else if eng.voyage.is_none() {
            println!("  [hnsw] VOYAGE_API_KEY not set — using deterministic mock embeddings");
        } else {
            println!("  [hnsw] Voyage fetch failed — using deterministic mock embeddings");
        }
        eng
    }

    pub async fn new_atlas() -> Result<Self> {
        let atlas = AtlasClient::from_env().await?;
        Ok(Self {
            local: LocalHnsw::new(),
            atlas: Some(atlas),
            doc_texts: HashMap::new(),
            corpus_embeddings: HashMap::new(),
            voyage: None,
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
        self.doc_texts.insert(doc_id, text.to_string());

        let layers = if let Some(atlas) = &self.atlas {
            let embedding = atlas.index_doc(doc_id, text).await?;
            self.local.insert_raw(doc_id, embedding);
            // Atlas path: no graph introspection available.
            vec![(0u8, vec![])]
        } else if let Some(emb) = self.corpus_embeddings.get(&doc_id).cloned() {
            // Real Voyage embedding pre-fetched at startup.
            self.local.insert_traced_embedding(doc_id, emb)
        } else {
            // Fallback: deterministic mock embedding.
            self.local.insert_traced(doc_id)
        };

        log.push(TraceEvent::HnswInsert {
            doc_id,
            doc_text: text.chars().take(60).collect::<String>()
                + if text.chars().count() > 60 { "…" } else { "" },
            layers,
        });

        // Emit real per-layer stats after insertion.
        let stats = self.local.layer_stats();
        if stats.is_empty() {
            log.push(TraceEvent::HnswLayerStats { layer: 0, node_count: 1, avg_degree: 0.0 });
        } else {
            for (layer, node_count, avg_degree) in &stats {
                log.push(TraceEvent::HnswLayerStats {
                    layer: *layer,
                    node_count: *node_count as u32,
                    avg_degree: *avg_degree as f32,
                });
            }
        }

        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let mut log = TraceLog::default();

        let results = if let Some(atlas) = &self.atlas {
            let embedding = atlas.embed_query(text).await?;
            let results = atlas.query(&embedding, top_k).await?;
            // Atlas: emit a single summary hop (no layer introspection available).
            log.push(TraceEvent::HnswQuery {
                layer: 0,
                entry_doc: u32::MAX,
                candidates: results.clone(),
                greedy_best: results.first().map(|(id, _)| *id).unwrap_or(0),
            });
            results
        } else {
            // Real Voyage query embedding when available; otherwise mock.
            let t_embed = Instant::now();
            let query_vec = if let Some(v) = &self.voyage {
                v.embed_query(text).await.unwrap_or_else(|e| {
                    eprintln!("  [voyage] query embed failed: {e} — using mock");
                    let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
                    mock_embedding(hash)
                })
            } else {
                let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
                mock_embedding(hash)
            };
            let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
            let t_search = Instant::now();
            let (results, hops) = self.local.search_traced(&query_vec, top_k);
            let search_ms = t_search.elapsed().as_secs_f64() * 1000.0;
            for hop in hops {
                let entry_doc = self.local.doc_map.get(hop.entry_idx).map(|(id, _)| *id).unwrap_or(u32::MAX);
                let greedy_best = self.local.doc_map.get(hop.greedy_best_idx).map(|(id, _)| *id).unwrap_or(0);
                let candidates: Vec<(u32, f32)> = hop.candidates
                    .iter()
                    .filter_map(|(did, score)| self.local.doc_map.get(*did).map(|(id, _)| (*id, *score)))
                    .collect();
                log.push(TraceEvent::HnswQuery {
                    layer: hop.layer,
                    entry_doc,
                    candidates,
                    greedy_best,
                });
            }
            log.timing = Some(OpTiming {
                embed_ms: Some(embed_ms),
                search_ms: Some(search_ms),
                docs_scored: Some(top_k.min(self.local.len())),
            });
            results
        };

        Ok((results, log))
    }

    async fn inspect(&self, target: Option<&str>) -> Result<String> {
        match target {
            Some("layers") | Some("graph") => {
                let stats = self.local.layer_stats();
                if stats.is_empty() {
                    return Ok(format!(
                        "HNSW local index: {} docs inserted\nNo layer data yet.",
                        self.local.len()
                    ));
                }
                let mut out = format!(
                    "HNSW local index: {} docs  M=4  ef_construction=64  P(promoted)=25%\n\n",
                    self.local.len()
                );
                out.push_str(&format!("  {:<8} {:<12} {}\n", "layer", "nodes", "avg_degree"));
                out.push_str(&format!("  {:<8} {:<12} {}\n", "-----", "-----", "----------"));
                for (layer, node_count, avg_degree) in &stats {
                    out.push_str(&format!(
                        "  {:<8} {:<12} {:.2}\n",
                        layer, node_count, avg_degree
                    ));
                }
                Ok(out)
            }
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
