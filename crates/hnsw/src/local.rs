use hnsw_rs::prelude::{DistCosine, Hnsw};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::f32::consts::PI;

/// Matches voyage-4-large output dimensions (also used for mock fallback embeddings).
const EMBEDDING_DIM: usize = 1024;

/// One step of the simulated greedy descent for visualization.
pub struct LayerHop {
    pub layer: u8,
    /// Data-map index of the node we entered this layer at.
    pub entry_idx: usize,
    /// (data_map_idx, cosine_score) — all nodes assessed at this layer, sorted desc.
    pub candidates: Vec<(usize, f32)>,
    /// Data-map index we greedily moved to.
    pub greedy_best_idx: usize,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

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
        // M=4: P(level≥1) = 1/M = 25% — reliably shows 4-5 promoted nodes in a 20-doc demo.
        // ef_construction=64 is generous for 20 docs.
        Self {
            inner: Hnsw::<'static, f32, DistCosine>::new(4, 25, 8, 64, DistCosine {}),
            doc_map: Vec::new(),
        }
    }

    pub fn insert_mock(&mut self, doc_id: u32) {
        let embedding = mock_embedding(doc_id);
        let idx = self.doc_map.len();
        self.inner.insert((&embedding, idx));
        self.doc_map.push((doc_id, embedding));
    }

    /// Insert and return per-layer neighbor lists for the new node.
    ///
    /// hnsw_rs stores each node in `get_layer_iterator` only at its **max level**.
    /// To get all-layers connections we: scan downward to find the node at its top layer,
    /// then read every layer's neighborhood from `get_neighborhood_id()`.
    ///
    /// Returns `Vec<(layer, neighbor_doc_ids)>` from layer 0 up to the node's max level.
    /// Multiple entries → node was promoted above layer 0.
    pub fn insert_traced(&mut self, doc_id: u32) -> Vec<(u8, Vec<u32>)> {
        self.insert_traced_embedding(doc_id, mock_embedding(doc_id))
    }

    /// Same as `insert_traced` but uses a supplied embedding instead of the mock.
    pub fn insert_traced_embedding(&mut self, doc_id: u32, embedding: Vec<f32>) -> Vec<(u8, Vec<u32>)> {
        let new_idx = self.doc_map.len();
        self.inner.insert((&embedding, new_idx));
        self.doc_map.push((doc_id, embedding));

        let max_level = self.inner.get_max_level_observed();

        // Phase 1: find the new node at its max layer and collect all-layer d_ids.
        // (Nodes only appear in get_layer_iterator at their top level.)
        let mut found_max: usize = 0;
        let mut all_layer_dids: Vec<Vec<usize>> = Vec::new(); // index = layer
        {
            let pi = self.inner.get_point_indexation();
            'find: for check_l in (0..=(max_level as usize)).rev() {
                for point in pi.get_layer_iterator(check_l) {
                    if point.get_origin_id() == new_idx {
                        found_max = check_l;
                        let nbhd = point.get_neighborhood_id();
                        for li in 0..=check_l {
                            let dids: Vec<usize> = nbhd
                                .get(li)
                                .map(|v| v.iter().map(|nb| nb.d_id).collect())
                                .unwrap_or_default();
                            all_layer_dids.push(dids);
                        }
                        break 'find;
                    }
                }
            }
        } // pi dropped — doc_map freely accessible

        if all_layer_dids.is_empty() {
            return vec![(0, vec![])];
        }

        // Phase 2: translate d_ids → doc_ids for all layers.
        (0..=found_max)
            .map(|l| {
                let doc_ids: Vec<u32> = all_layer_dids[l]
                    .iter()
                    .filter_map(|did| self.doc_map.get(*did).map(|(id, _)| *id))
                    .collect();
                (l as u8, doc_ids)
            })
            .collect()
    }

    /// Run the actual search then simulate greedy layer descent for visualization.
    ///
    /// Because nodes only appear in `get_layer_iterator` at their max level, we build
    /// a per-node neighborhood table upfront, then use it for the greedy walk.
    ///
    /// Returns (true ANN results, layer hops high → low).
    pub fn search_traced(&self, query_vec: &[f32], top_k: usize) -> (Vec<(u32, f32)>, Vec<LayerHop>) {
        let actual_results = self.search(query_vec, top_k);

        if self.doc_map.is_empty() {
            return (actual_results, vec![]);
        }

        let max_level = self.inner.get_max_level_observed();

        // Build: idx → (node_max_level, neighborhoods[0..=node_max_level])
        // Also record the HNSW entry point (first node at the global max level).
        let mut node_nbhds: std::collections::HashMap<usize, Vec<Vec<usize>>> =
            std::collections::HashMap::new();
        let mut entry_idx: usize = 0;
        let mut entry_found = false;
        {
            let pi = self.inner.get_point_indexation();
            for l in (0..=(max_level as usize)).rev() {
                for point in pi.get_layer_iterator(l) {
                    let idx = point.get_origin_id();
                    if !entry_found && l == max_level as usize {
                        entry_idx = idx;
                        entry_found = true;
                    }
                    let nbhd = point.get_neighborhood_id();
                    let per_layer: Vec<Vec<usize>> = (0..=l)
                        .map(|li| {
                            nbhd.get(li)
                                .map(|v| v.iter().map(|nb| nb.d_id).collect())
                                .unwrap_or_default()
                        })
                        .collect();
                    node_nbhds.insert(idx, per_layer);
                }
            }
        } // pi dropped

        // Greedy descent from the entry point, one hop per layer.
        let mut current_idx = entry_idx;
        let mut hops: Vec<LayerHop> = Vec::new();

        for l in (0..=(max_level as usize)).rev() {
            // Neighbors of current node at this layer (may be empty if node is layer-0-only
            // and we're traversing at l == 0, or if it has no edges yet).
            let neighbor_dids: Vec<usize> = node_nbhds
                .get(&current_idx)
                .and_then(|nbhds| nbhds.get(l))
                .cloned()
                .unwrap_or_default();

            // Score current node + neighbors.
            let mut candidates: Vec<(usize, f32)> = Vec::new();
            if let Some((_, emb)) = self.doc_map.get(current_idx) {
                candidates.push((current_idx, dot(query_vec, emb)));
            }
            for did in &neighbor_dids {
                if let Some((_, emb)) = self.doc_map.get(*did) {
                    candidates.push((*did, dot(query_vec, emb)));
                }
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.dedup_by_key(|c| c.0);

            let greedy_best_idx = candidates.first().map(|(i, _)| *i).unwrap_or(current_idx);

            hops.push(LayerHop {
                layer: l as u8,
                entry_idx: current_idx,
                candidates,
                greedy_best_idx,
            });

            current_idx = greedy_best_idx;
        }

        (actual_results, hops)
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

    /// Per-layer stats using cumulative node counts.
    ///
    /// hnsw_rs stores each node in `get_layer_iterator` at its max level only, so
    /// `get_layer_nb_point(l)` gives nodes whose max level == l (not all participating nodes).
    /// A node at level 4 participates in layers 0–4; cumulative counting shows this correctly
    /// and avoids confusing gaps like "Layer 0, Layer 1, Layer 4".
    ///
    /// Returns `(layer, cumulative_node_count, avg_degree_at_that_layer)`.
    pub fn layer_stats(&self) -> Vec<(u8, usize, f64)> {
        let max_level = self.inner.get_max_level_observed() as usize;

        let mut exact_count = vec![0usize; max_level + 1];
        let mut degree_sum = vec![0usize; max_level + 1];

        {
            let pi = self.inner.get_point_indexation();
            for node_level in 0..=max_level {
                for point in pi.get_layer_iterator(node_level) {
                    exact_count[node_level] += 1;
                    let nbhd = point.get_neighborhood_id();
                    // This node participates in layers 0..=node_level.
                    for l in 0..=node_level {
                        degree_sum[l] += nbhd.get(l).map(|v| v.len()).unwrap_or(0);
                    }
                }
            }
        }

        (0..=max_level)
            .filter_map(|l| {
                let cumulative: usize = exact_count[l..].iter().sum();
                if cumulative == 0 {
                    return None;
                }
                let avg_degree = degree_sum[l] as f64 / cumulative as f64;
                Some((l as u8, cumulative, avg_degree))
            })
            .collect()
    }
}

impl Default for LocalHnsw {
    fn default() -> Self {
        Self::new()
    }
}
