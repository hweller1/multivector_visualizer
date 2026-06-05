use common::token::TOKEN_DIM;
use common::trace::TraceEvent;
use std::collections::HashMap;

pub fn damped_weight(variance: f32, freq: u32) -> f32 {
    (freq as f32).sqrt() * variance
}

pub struct DampedScorer {
    pub weights: HashMap<u32, f32>,
}

impl DampedScorer {
    /// Compute damped weights for all token types from their embeddings.
    pub fn compute(token_embeddings: &HashMap<u32, Vec<[f32; TOKEN_DIM]>>) -> Self {
        let mut weights = HashMap::new();

        for (&token_type, embeddings) in token_embeddings {
            if embeddings.is_empty() {
                continue;
            }
            let n = embeddings.len();
            let freq = n as u32;

            // Compute centroid
            let mut centroid = [0f32; TOKEN_DIM];
            for emb in embeddings {
                for (d, x) in centroid.iter_mut().zip(emb.iter()) {
                    *d += x;
                }
            }
            for d in centroid.iter_mut() {
                *d /= n as f32;
            }

            // Compute variance = mean squared distance from centroid
            let variance = embeddings
                .iter()
                .map(|emb| {
                    emb.iter()
                        .zip(centroid.iter())
                        .map(|(x, c)| (x - c) * (x - c))
                        .sum::<f32>()
                })
                .sum::<f32>()
                / n as f32;

            let w = damped_weight(variance, freq);
            weights.insert(token_type, w);
        }

        DampedScorer { weights }
    }

    pub fn trace_all(
        &self,
        token_embeddings: &HashMap<u32, Vec<[f32; TOKEN_DIM]>>,
    ) -> Vec<TraceEvent> {
        self.weights
            .iter()
            .map(|(token_type, &weight)| {
                let embeddings = token_embeddings.get(token_type);
                let n = embeddings.map(|e| e.len()).unwrap_or(0);
                let freq = n as u32;
                let variance = if freq == 0 || weight == 0.0 {
                    0.0
                } else {
                    weight / (freq as f32).sqrt()
                };
                TraceEvent::DampedScore {
                    token_type: token_type.to_string(),
                    variance,
                    weight,
                }
            })
            .collect()
    }
}
