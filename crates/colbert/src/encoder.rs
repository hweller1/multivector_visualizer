use anyhow::Result;
use common::{RandomProjection, TokenMatrix, TraceEvent, TraceLog, WordPieceTokenizer};
use std::collections::HashMap;

pub struct ColBertEncoder {
    pub(crate) tokenizer: WordPieceTokenizer,
    pub(crate) projection: RandomProjection,
    /// Per-text learned token embeddings loaded from Jina ColBERT cache.
    /// When non-empty, `encode()` returns Jina embeddings instead of random projections.
    pub jina_cache: HashMap<String, TokenMatrix>,
}

impl ColBertEncoder {
    pub fn new(vocab_path: &std::path::Path, seed: u64) -> Result<Self> {
        let tokenizer = WordPieceTokenizer::from_vocab(vocab_path)?;
        let projection = RandomProjection::new(seed);
        // Auto-load any cached Jina embeddings from disk so engines pick them up
        // without needing explicit wiring.
        let jina_cache = crate::jina::load_cache();
        Ok(Self { tokenizer, projection, jina_cache })
    }

    /// Replace the Jina cache (called after a refresh_cache() to avoid a second disk read).
    pub fn load_jina_cache(&mut self, cache: HashMap<String, TokenMatrix>) {
        self.jina_cache = cache;
    }

    pub fn is_using_jina(&self) -> bool {
        !self.jina_cache.is_empty()
    }

    /// Tokenize and embed text.
    /// If the text is in the Jina cache, returns learned ColBERT token embeddings.
    /// Otherwise falls back to deterministic RandomProjection.
    pub fn encode(&mut self, text: &str) -> Result<(TokenMatrix, Vec<u32>)> {
        let encoding = self
            .tokenizer
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let wp_tokens: Vec<String> = encoding.get_tokens().to_vec();
        let vocab_ids: Vec<u32> = encoding.get_ids().to_vec();

        if let Some(jina_matrix) = self.jina_cache.get(text) {
            let mut matrix = jina_matrix.clone();
            let n_jina = matrix.rows.len();
            // Substitute WordPiece token strings for display when counts match.
            if wp_tokens.len() == n_jina {
                matrix.tokens = wp_tokens;
            }
            // Align vocab_ids length to Jina token count for IDF consistency.
            let aligned_ids = if vocab_ids.len() == n_jina {
                vocab_ids
            } else {
                let mut v = vocab_ids;
                v.resize(n_jina, 0);
                v
            };
            return Ok((matrix, aligned_ids));
        }

        let matrix = self.projection.embed(&vocab_ids, &wp_tokens);
        Ok((matrix, vocab_ids))
    }

    /// Same as encode but emits Tokenize and TokenEmbed trace events.
    pub fn encode_with_trace(
        &mut self,
        doc_id: u32,
        text: &str,
    ) -> Result<(TokenMatrix, Vec<u32>, TraceLog)> {
        let (matrix, vocab_ids) = self.encode(text)?;
        let mut log = TraceLog::default();

        log.push(TraceEvent::Tokenize {
            doc_id,
            tokens: matrix.tokens.clone(),
        });

        for i in 0..matrix.num_tokens() {
            log.push(TraceEvent::TokenEmbed {
                doc_id,
                token: matrix.tokens[i].clone(),
                embedding_preview: matrix.preview(i),
            });
        }

        Ok((matrix, vocab_ids, log))
    }
}
