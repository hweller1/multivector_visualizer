use crate::encoder::ColBertEncoder;
use crate::index::ColBertIndex;
use anyhow::Result;
use async_trait::async_trait;
use common::{Engine, TraceEvent, TraceLog};
use std::cell::RefCell;

pub struct ColBertEngine {
    pub encoder: RefCell<ColBertEncoder>,
    pub index: ColBertIndex,
    #[allow(dead_code)]
    last_trace: Option<TraceLog>,
    next_doc_id: u32,
}

// Safety: ColBertEngine is only ever used from a single thread in the demo/repl.
unsafe impl Sync for ColBertEngine {}

impl ColBertEngine {
    pub fn new(vocab_path: &std::path::Path) -> Result<Self> {
        let encoder = ColBertEncoder::new(vocab_path, 0x0123456789ABCDEF)?;
        Ok(Self {
            encoder: RefCell::new(encoder),
            index: ColBertIndex::new(),
            last_trace: None,
            next_doc_id: 0,
        })
    }
}

#[async_trait]
impl Engine for ColBertEngine {
    fn name(&self) -> &'static str {
        "colbert"
    }

    async fn index(&mut self, doc_id: u32, text: &str) -> Result<TraceLog> {
        let (matrix, _vocab_ids, mut log) =
            self.encoder.borrow_mut().encode_with_trace(doc_id, text)?;

        // Emit one HnswInsert-style event per token (logical insertion, not an ANN graph).
        // REPL narration: "ColBERT's logical insertion, not yet an ANN graph — that comes in PLAID."
        for (i, row) in matrix.rows.iter().enumerate() {
            let preview: [f32; 3] = [row[0], row[1], row[2]];
            log.push(TraceEvent::HnswInsert {
                doc_id,
                layer: 0,
                neighbors: vec![i as u32],
            });
            // Also emit TokenEmbed for visualization (AC-2.2)
            log.push(TraceEvent::TokenEmbed {
                doc_id,
                token: matrix.tokens[i].clone(),
                embedding_preview: preview,
            });
        }

        self.index.insert(doc_id, matrix);
        self.next_doc_id = self.next_doc_id.max(doc_id + 1);
        Ok(log)
    }

    async fn query(&self, text: &str, top_k: usize) -> Result<(Vec<(u32, f32)>, TraceLog)> {
        let (query_matrix, _vocab_ids) = self.encoder.borrow_mut().encode(text)?;
        let (results, log) = self.index.search_with_trace(&query_matrix, top_k);
        Ok((results, log))
    }

    async fn inspect(&self, target: Option<&str>) -> Result<String> {
        match target {
            Some("tokens") => {
                let mut out = String::new();
                for (doc_id, matrix) in &self.index.docs {
                    out.push_str(&format!("doc {doc_id}: {} tokens\n", matrix.num_tokens()));
                    for (i, tok) in matrix.tokens.iter().enumerate() {
                        let p = matrix.preview(i);
                        out.push_str(&format!(
                            "  [{i}] {tok:20} → [{:.4}, {:.4}, {:.4}, ...]\n",
                            p[0], p[1], p[2]
                        ));
                    }
                }
                if out.is_empty() {
                    out = "No documents indexed yet.".to_string();
                }
                Ok(out)
            }
            None => Ok(format!(
                "ColBERT index: {} documents",
                self.index.docs.len()
            )),
            Some(other) => Ok(format!("Unknown target '{other}'. Available: tokens")),
        }
    }

    fn verify(&mut self) -> Result<()> {
        crate::verify::run(self)
    }
}
