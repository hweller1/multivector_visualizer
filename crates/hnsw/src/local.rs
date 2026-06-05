use hnsw_rs::prelude::{DistCosine, Hnsw};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::f32::consts::PI;

const EMBEDDING_DIM: usize = 1536;

pub fn mock_embedding(doc_id: u32) -> Vec<f32> {
    let seed = 0xCAFEBABEu64 ^ doc_id as u64;
    let mut rng = SmallRng::seed_from_u64(seed);
    // Box-Muller transform: pairs of uniform → standard normal
    let mut raw = Vec::with_capacity(EMBEDDING_DIM);
    while raw.len() < EMBEDDING_DIM {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen::<f32>();
        let mag = (-2.0 * u1.ln()).sqrt();
        raw.push(mag * (2.0 * PI * u2).cos());
        if raw.len() < EMBEDDING_DIM {
            raw.push(mag * (2.0 * PI * u2).sin());
        }
    }
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    let scale = if norm > 1e-8 { 1.0 / norm } else { 1.0 };
    raw.into_iter().map(|x| x * scale).collect()
}

pub struct LocalHnsw {
    inner: Hnsw<'static, f32, DistCosine>,
    pub doc_map: Vec<(u32, Vec<f32>)>,
}

impl LocalHnsw {
    pub fn new() -> Self {
        // max_nb_connection=16, max_elements=20, max_layer=16, ef_construction=200
        Self {
            inner: Hnsw::<'static, f32, DistCosine>::new(16, 20, 16, 200, DistCosine {}),
            doc_map: Vec::new(),
        }
    }

    pub fn insert_mock(&mut self, doc_id: u32) {
        let embedding = mock_embedding(doc_id);
        let idx = self.doc_map.len();
        self.inner.insert((&embedding, idx));
        self.doc_map.push((doc_id, embedding));
    }

    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        let neighbours = self.inner.search(query_vec, top_k, 50);
        neighbours
            .into_iter()
            .filter_map(|nb| {
                self.doc_map
                    .get(nb.d_id)
                    .map(|(doc_id, _)| (*doc_id, 1.0_f32 - nb.distance))
            })
            .collect()
    }

    /// Insert a pre-computed embedding (e.g. from Voyage API in Atlas mode).
    pub fn insert_raw(&mut self, doc_id: u32, embedding: Vec<f32>) {
        let idx = self.doc_map.len();
        self.inner.insert((&embedding, idx));
        self.doc_map.push((doc_id, embedding));
    }

    pub fn len(&self) -> usize {
        self.doc_map.len()
    }
}

impl Default for LocalHnsw {
    fn default() -> Self {
        Self::new()
    }
}
