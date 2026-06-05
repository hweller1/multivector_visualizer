use common::token::TOKEN_DIM;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub struct CentroidPruner {
    pub num_centroids: usize,
}

impl CentroidPruner {
    pub fn new(num_centroids: usize) -> Self {
        Self { num_centroids }
    }

    /// Lloyd's KMeans: random init with SmallRng seeded 42, 100 iterations max.
    /// Clamps k = k.min(all_token_rows.len()).
    pub fn fit(&self, all_token_rows: &[[f32; TOKEN_DIM]]) -> Vec<[f32; TOKEN_DIM]> {
        if all_token_rows.is_empty() {
            return Vec::new();
        }
        let k = self.num_centroids.min(all_token_rows.len());
        let mut rng = SmallRng::seed_from_u64(42);

        // Random initialization
        let mut centroids: Vec<[f32; TOKEN_DIM]> = all_token_rows
            .choose_multiple(&mut rng, k)
            .copied()
            .collect();

        for _ in 0..100 {
            // Assignment step
            let assignments: Vec<usize> = all_token_rows
                .iter()
                .map(|row| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(ci, c)| (ci, dot(row, c)))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(ci, _)| ci)
                        .unwrap_or(0)
                })
                .collect();

            // Update step
            let mut new_centroids = vec![[0f32; TOKEN_DIM]; k];
            let mut counts = vec![0usize; k];
            for (row, &ci) in all_token_rows.iter().zip(assignments.iter()) {
                counts[ci] += 1;
                for (d, x) in new_centroids[ci].iter_mut().zip(row.iter()) {
                    *d += x;
                }
            }

            let mut converged = true;
            for (ci, count) in counts.iter().enumerate() {
                if *count == 0 {
                    // Keep old centroid if no assignments
                    new_centroids[ci] = centroids[ci];
                } else {
                    let n = *count as f32;
                    for d in new_centroids[ci].iter_mut() {
                        *d /= n;
                    }
                    // L2 normalize
                    let norm = new_centroids[ci]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-9);
                    for d in new_centroids[ci].iter_mut() {
                        *d /= norm;
                    }
                }
                let diff: f32 = centroids[ci]
                    .iter()
                    .zip(new_centroids[ci].iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                if diff > 1e-6 {
                    converged = false;
                }
            }
            centroids = new_centroids;
            if converged {
                break;
            }
        }

        centroids
    }

    /// Nearest centroid by dot product (L2-normalized vectors).
    pub fn assign(&self, centroids: &[[f32; TOKEN_DIM]], token: &[f32; TOKEN_DIM]) -> u32 {
        centroids
            .iter()
            .enumerate()
            .map(|(ci, c)| (ci as u32, dot(c, token)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ci, _)| ci)
            .unwrap_or(0)
    }

    /// Top-nprobe centroids by dot product, sorted descending.
    pub fn query_centroids(
        &self,
        centroids: &[[f32; TOKEN_DIM]],
        query_token: &[f32; TOKEN_DIM],
        nprobe: usize,
    ) -> Vec<(u32, f32)> {
        let mut scores: Vec<(u32, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(ci, c)| (ci as u32, dot(c, query_token)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(nprobe);
        scores
    }
}

fn dot(a: &[f32; TOKEN_DIM], b: &[f32; TOKEN_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
