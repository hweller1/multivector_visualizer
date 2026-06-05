use anyhow::Result;

/// Dimension of token embeddings — matches ColBERTv2.
pub const TOKEN_DIM: usize = 128;

/// Row-major storage: embedding for token i lives at rows[i][0..TOKEN_DIM].
#[derive(Debug, Clone)]
pub struct TokenMatrix {
    pub tokens: Vec<String>,
    pub rows: Vec<[f32; TOKEN_DIM]>,
}

impl TokenMatrix {
    pub fn num_tokens(&self) -> usize {
        self.rows.len()
    }

    /// Returns the first 3 dimensions of token i for display (AC-2.2).
    pub fn preview(&self, i: usize) -> [f32; 3] {
        [self.rows[i][0], self.rows[i][1], self.rows[i][2]]
    }
}

/// Shared tokenizer contract.
pub trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Real WordPiece tokenizer backed by the shipped vocab file.
pub struct WordPieceTokenizer {
    pub inner: tokenizers::Tokenizer,
}

impl WordPieceTokenizer {
    /// Load vocab from a vocab.txt path (bert-base-uncased style, one token per line).
    pub fn from_vocab(vocab_path: &std::path::Path) -> Result<Self> {
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::normalizers::bert::BertNormalizer;
        use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
        use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;

        let vocab_str = vocab_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid vocab path"))?;

        let wordpiece = WordPiece::from_file(vocab_str)
            .build()
            .map_err(|e| anyhow::anyhow!("wordpiece build: {e}"))?;

        let mut inner = tokenizers::Tokenizer::new(wordpiece);
        inner.with_normalizer(Some(BertNormalizer::default()));
        inner.with_pre_tokenizer(Some(BertPreTokenizer));
        inner.with_decoder(Some(WordPieceDecoder::default()));

        Ok(Self { inner })
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let enc = self.inner.encode(text, false).unwrap();
        enc.get_tokens().to_vec()
    }
}

/// Deterministic fixed-weight linear projection: token_id → 128-dim vector.
/// Uses SmallRng seeded from the token's WordPiece vocab ID — one projection
/// per token type, cached after first computation.
pub struct RandomProjection {
    seed: u64,
    cache: std::collections::HashMap<u32, [f32; TOKEN_DIM]>,
}

impl RandomProjection {
    pub fn new(seed: u64) -> Self {
        Self { seed, cache: Default::default() }
    }

    /// Returns a deterministic 128-dim unit vector for a vocab token ID.
    pub fn project(&mut self, token_id: u32) -> [f32; TOKEN_DIM] {
        if let Some(cached) = self.cache.get(&token_id) {
            return *cached;
        }
        use rand::{Rng, SeedableRng};
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(self.seed ^ token_id as u64);
        let mut v = [0f32; TOKEN_DIM];
        for x in v.iter_mut() {
            *x = rng.gen::<f32>() * 2.0 - 1.0;
        }
        // L2 normalize
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in v.iter_mut() {
            *x /= norm;
        }
        self.cache.insert(token_id, v);
        v
    }

    /// Embed a full token sequence → TokenMatrix.
    pub fn embed(&mut self, vocab_ids: &[u32], tokens: &[String]) -> TokenMatrix {
        let rows: Vec<[f32; TOKEN_DIM]> = vocab_ids.iter()
            .map(|&id| self.project(id))
            .collect();
        TokenMatrix { tokens: tokens.to_vec(), rows }
    }
}
