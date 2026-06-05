use anyhow::Result;
use common::{RandomProjection, TokenMatrix, TraceEvent, TraceLog, WordPieceTokenizer};

pub struct ColBertEncoder {
    pub(crate) tokenizer: WordPieceTokenizer,
    pub(crate) projection: RandomProjection,
}

impl ColBertEncoder {
    pub fn new(vocab_path: &std::path::Path, seed: u64) -> Result<Self> {
        let tokenizer = WordPieceTokenizer::from_vocab(vocab_path)?;
        let projection = RandomProjection::new(seed);
        Ok(Self {
            tokenizer,
            projection,
        })
    }

    /// Tokenize and embed text, returning (TokenMatrix, vocab_ids).
    pub fn encode(&mut self, text: &str) -> Result<(TokenMatrix, Vec<u32>)> {
        let encoding = self
            .tokenizer
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let tokens: Vec<String> = encoding.get_tokens().to_vec();
        let vocab_ids: Vec<u32> = encoding.get_ids().to_vec();
        let matrix = self.projection.embed(&vocab_ids, &tokens);
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
