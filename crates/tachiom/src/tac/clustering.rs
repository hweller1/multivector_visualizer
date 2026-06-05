use common::token::TOKEN_DIM;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Lloyd's KMeans, 50 iterations, SmallRng seeded.
/// Guard: k = k.min(embeddings.len())
pub fn kmeans(embeddings: &[[f32; TOKEN_DIM]], k: usize, seed: u64) -> Vec<[f32; TOKEN_DIM]> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let k = k.min(embeddings.len());
    if k == 0 {
        return Vec::new();
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut centroids: Vec<[f32; TOKEN_DIM]> =
        embeddings.choose_multiple(&mut rng, k).copied().collect();

    for _ in 0..50 {
        // Assignment
        let assignments: Vec<usize> = embeddings
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

        // Update
        let mut new_centroids = vec![[0f32; TOKEN_DIM]; k];
        let mut counts = vec![0usize; k];
        for (row, &ci) in embeddings.iter().zip(assignments.iter()) {
            counts[ci] += 1;
            for (d, x) in new_centroids[ci].iter_mut().zip(row.iter()) {
                *d += x;
            }
        }

        let mut converged = true;
        for (ci, count) in counts.iter().enumerate() {
            if *count == 0 {
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

/// Run k-means in parallel over all token types using rayon.
pub fn parallel_kappa_means(
    token_type_embeddings: &HashMap<u32, Vec<[f32; TOKEN_DIM]>>,
    kappa: &HashMap<u32, u32>,
) -> HashMap<u32, Vec<[f32; TOKEN_DIM]>> {
    let entries: Vec<(u32, &Vec<[f32; TOKEN_DIM]>, u32)> = token_type_embeddings
        .iter()
        .map(|(&tt, embs)| {
            let k = kappa.get(&tt).copied().unwrap_or(4);
            (tt, embs, k)
        })
        .collect();

    entries
        .par_iter()
        .map(|(token_type, embeddings, k)| {
            let seed = 0xCAFEBABE_u64 ^ *token_type as u64;
            let centroids = kmeans(embeddings, *k as usize, seed);
            (*token_type, centroids)
        })
        .collect()
}

fn dot(a: &[f32; TOKEN_DIM], b: &[f32; TOKEN_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
