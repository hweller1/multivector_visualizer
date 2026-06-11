//! Large-scale "needle in a haystack" benchmark.
//!
//! The 100 real GT-bench docs (with Jina ColBERT per-token embeddings) are the only
//! potentially relevant documents.  Synthetic distractors fill the corpus to each
//! scale N.  GT labels come from cache/llm_gt.json (same as gt-bench).
//!
//! Scales: 100_000 / 1_000_000 / 10_000_000
//! Filters: None / Category / CategoryAndYear
//! Engines: HNSW (sentence-avg of Jina tokens) / PLAID / WARP / TACHIOM
//!
//! Output:
//!   Terminal table  — Recall@10 at {10%, 25%, 50%, 100%} candidates × scale
//!   plots/large_bench_recall_vs_frac.svg  — Recall@10 vs frac at N=1M
//!   plots/large_bench_recall_vs_n.svg     — Recall@10 vs N at 10% candidates

use anyhow::Result;
use hnsw_rs::prelude::*;
use plotters::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use colbert::jina;
use common::TOKEN_DIM;

// ─── constants ────────────────────────────────────────────────────────────────

const K_EVAL: usize = 10;
const N_CATEGORIES: usize = 10;
const YEAR_MIN: u16 = 2010;
const YEAR_MAX: u16 = 2024;
/// Number of synthetic token vectors per distractor doc.
const SYNTH_TOKENS: usize = 3;
/// Memory warning threshold: 2 GB
const MEM_WARN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

const SCALES: &[usize] = &[100_000, 1_000_000, 10_000_000];

/// Maps each of the 10 GT queries to one of the N_CATEGORIES categories.
/// river=0, finance=1, construction_crane=2, bird_crane=3,
/// elephant_trunk=4, car_trunk=5, physics=6, fashion=7, anatomy=8, misc=9
const QUERY_CATEGORIES: [u8; 10] = [0, 1, 1, 0, 2, 3, 4, 5, 6, 7];

const QUERY_YEAR_FLOORS: [u16; 10] = [2018, 2020, 2019, 2017, 2021, 2018, 2020, 2019, 2022, 2018];

/// The 10 GT queries (must match gtbench.rs GT_QUERIES ordering).
const GT_QUERIES: &[&str] = &[
    "river erosion along the bank",
    "open a checking account at the bank",
    "central bank interest rate monetary policy",
    "hiking trail beside the river valley",
    "construction crane lifting heavy steel beam",
    "migratory crane birds wetland habitat",
    "elephant trunk muscular grasping",
    "car trunk vehicle storage compartment",
    "speed of light optics physics experiment",
    "summer lightweight fashion dress fabric",
];

// ─── doc / filter structs ─────────────────────────────────────────────────────

/// A single document entry in the large-scale corpus.
#[derive(Clone)]
struct LargeDoc {
    id: usize,
    category: u8,
    year: u16,
    /// Per-token 128-dim embeddings.  Real docs: Jina ColBERT.  Synthetic: 3 random unit vecs.
    tokens: Vec<[f32; TOKEN_DIM]>,
    /// Sentence-average embedding for HNSW (L2-normalised avg of token rows).
    sent_emb: [f32; TOKEN_DIM],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum FilterMode {
    None,
    Category,
    CategoryAndYear,
}

impl FilterMode {
    fn label(self) -> &'static str {
        match self {
            FilterMode::None => "no-filter",
            FilterMode::Category => "cat-filter",
            FilterMode::CategoryAndYear => "cat+year",
        }
    }
}

/// Per-(engine × filter_mode) result: a vec of (candidate_frac, recall@K) points.
struct LargeBenchResult {
    engine: &'static str,
    filter: FilterMode,
    /// One entry per scale, in SCALES order.
    by_scale: Vec<Vec<(f64, f64)>>,
}

// ─── category mapping for real docs ──────────────────────────────────────────

/// Map GT_CORPUS doc IDs (0-99) to category index 0-9.
fn doc_category_for_real(corpus_id: u32) -> u8 {
    match corpus_id {
        0..=19  => 0, // river / geography
        20..=39 => 1, // finance
        40..=49 => 2, // construction crane
        50..=59 => 3, // bird crane
        60..=64 => 4, // elephant trunk
        65..=69 => 5, // car trunk
        70..=79 => 6, // physics / optics
        80..=89 => 7, // fashion / lightweight
        90..=94 => 8, // anatomy / nerve trunk
        _       => 9, // misc / distractors
    }
}

// ─── vector math ─────────────────────────────────────────────────────────────

#[inline]
fn dot128(a: &[f32; TOKEN_DIM], b: &[f32; TOKEN_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_normalize(v: &mut [f32; TOKEN_DIM]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter_mut().for_each(|x| *x /= n);
}

fn rand_unit(rng: &mut SmallRng) -> [f32; TOKEN_DIM] {
    let mut v = [0f32; TOKEN_DIM];
    for x in v.iter_mut() {
        *x = rng.gen::<f32>() * 2.0 - 1.0;
    }
    l2_normalize(&mut v);
    v
}

fn sent_avg(tokens: &[[f32; TOKEN_DIM]]) -> [f32; TOKEN_DIM] {
    let mut avg = [0f32; TOKEN_DIM];
    if tokens.is_empty() { return avg; }
    for tok in tokens {
        for (a, t) in avg.iter_mut().zip(tok.iter()) {
            *a += t;
        }
    }
    let inv = 1.0 / tokens.len() as f32;
    avg.iter_mut().for_each(|x| *x *= inv);
    l2_normalize(&mut avg);
    avg
}

fn recall_at_k(results: &[(usize, f32)], gt: &HashSet<usize>, k: usize) -> f64 {
    let n_rel = gt.len().min(k);
    if n_rel == 0 { return 0.0; }
    let found = results.iter().take(k).filter(|(id, _)| gt.contains(id)).count();
    found as f64 / n_rel as f64
}

#[allow(dead_code)]
fn maxsim(q: &[[f32; TOKEN_DIM]], d: &[[f32; TOKEN_DIM]]) -> f32 {
    q.iter()
        .map(|qi| d.iter().map(|di| dot128(qi, di)).fold(f32::NEG_INFINITY, f32::max))
        .sum()
}

// ─── synthetic doc generation ─────────────────────────────────────────────────

fn gen_synthetic_doc(id: usize, rng: &mut SmallRng) -> LargeDoc {
    let category = rng.gen_range(0..N_CATEGORIES) as u8;
    let year = YEAR_MIN + rng.gen_range(0..=(YEAR_MAX - YEAR_MIN));
    let tokens: Vec<[f32; TOKEN_DIM]> = (0..SYNTH_TOKENS).map(|_| rand_unit(rng)).collect();
    let sent_emb = sent_avg(&tokens);
    LargeDoc { id, category, year, tokens, sent_emb }
}

// ─── filtering ───────────────────────────────────────────────────────────────

/// Returns the indices into `docs` that pass the filter for this query.
fn apply_filter(docs: &[LargeDoc], query_cat: u8, year_floor: u16, filter: FilterMode) -> Vec<usize> {
    match filter {
        FilterMode::None => (0..docs.len()).collect(),
        FilterMode::Category => docs.iter().enumerate()
            .filter(|(_, d)| d.category == query_cat)
            .map(|(i, _)| i)
            .collect(),
        FilterMode::CategoryAndYear => docs.iter().enumerate()
            .filter(|(_, d)| d.category == query_cat && d.year >= year_floor)
            .map(|(i, _)| i)
            .collect(),
    }
}

// ─── k-means ─────────────────────────────────────────────────────────────────

fn kmeans_128(data: &[[f32; TOKEN_DIM]], k: usize, rng: &mut SmallRng) -> Vec<[f32; TOKEN_DIM]> {
    if data.is_empty() || k == 0 { return vec![]; }
    let k = k.min(data.len());
    let mut idxs: Vec<usize> = (0..data.len()).collect();
    for i in 0..k {
        let rem = data.len() - i;
        if rem > 1 {
            let j = i + rng.gen_range(0..rem);
            idxs.swap(i, j);
        }
    }
    let mut centers: Vec<[f32; TOKEN_DIM]> = idxs[..k].iter().map(|&i| data[i]).collect();
    for _ in 0..20 {
        let assign: Vec<usize> = data.iter().map(|d| {
            centers.iter().enumerate()
                .map(|(ci, c)| (ci, dot128(d, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci).unwrap_or(0)
        }).collect();
        let mut new_c = vec![[0f32; TOKEN_DIM]; k];
        let mut cnt = vec![0usize; k];
        for (d, &ci) in data.iter().zip(&assign) {
            for (j, &x) in d.iter().enumerate() { new_c[ci][j] += x; }
            cnt[ci] += 1;
        }
        for (c, &n) in new_c.iter_mut().zip(&cnt) {
            if n > 0 {
                let inv = 1.0 / n as f32;
                c.iter_mut().for_each(|x| *x *= inv);
                let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                c.iter_mut().for_each(|x| *x /= norm);
            }
        }
        centers = new_c;
    }
    centers
}

// ─── centroid index ───────────────────────────────────────────────────────────

struct CentIdx {
    centers: Vec<[f32; TOKEN_DIM]>,
    inv: Vec<Vec<usize>>, // centroid → doc indices
}

impl CentIdx {
    fn build(docs: &[LargeDoc], candidate_indices: &[usize], k: usize, rng: &mut SmallRng) -> Self {
        // Subsample to at most 50_000 tokens for clustering speed
        let all: Vec<([f32; TOKEN_DIM], usize)> = candidate_indices.iter()
            .flat_map(|&i| docs[i].tokens.iter().map(move |t| (*t, i)))
            .collect();
        if all.is_empty() || k == 0 { return Self { centers: vec![], inv: vec![] }; }

        let sample_n = all.len().min(50_000);
        let mut sidxs: Vec<usize> = (0..all.len()).collect();
        for i in 0..sample_n {
            let rem = all.len() - i;
            if rem > 1 {
                let j = i + rng.gen_range(0..rem);
                sidxs.swap(i, j);
            }
        }
        let sampled: Vec<[f32; TOKEN_DIM]> = sidxs[..sample_n].iter().map(|&i| all[i].0).collect();

        let centers = kmeans_128(&sampled, k, rng);
        if centers.is_empty() { return Self { centers: vec![], inv: vec![] }; }

        let mut inv: Vec<HashSet<usize>> = vec![HashSet::new(); centers.len()];
        for &(tok, doc_idx) in &all {
            let best = centers.iter().enumerate()
                .map(|(ci, c)| (ci, dot128(&tok, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci).unwrap_or(0);
            inv[best].insert(doc_idx);
        }
        Self { centers, inv: inv.into_iter().map(|s| s.into_iter().collect()).collect() }
    }

    fn probe(&self, qtoks: &[[f32; TOKEN_DIM]], n_probe: usize) -> HashSet<usize> {
        let mut cands = HashSet::new();
        if self.centers.is_empty() { return cands; }
        for qt in qtoks {
            let mut sims: Vec<(usize, f32)> = self.centers.iter().enumerate()
                .map(|(ci, c)| (ci, dot128(qt, c))).collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (ci, _) in sims.iter().take(n_probe.min(self.centers.len())) {
                self.inv[*ci].iter().for_each(|&d| { cands.insert(d); });
            }
        }
        cands
    }
}

// ─── IDF helpers (over real docs only) ───────────────────────────────────────

fn build_doc_freq_from_toks(doc_tok_ids: &[Vec<u32>]) -> HashMap<u32, usize> {
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for ids in doc_tok_ids {
        let unique: HashSet<u32> = ids.iter().copied().collect();
        for id in unique { *freq.entry(id).or_insert(0) += 1; }
    }
    freq
}

fn idf_weights_for(query_ids: &[u32], doc_freq: &HashMap<u32, usize>, n_docs: usize) -> Vec<f32> {
    let n = n_docs.max(1) as f32;
    query_ids.iter().map(|id| {
        let df = *doc_freq.get(id).unwrap_or(&0) as f32;
        (n / (1.0 + df)).ln() + 1.0
    }).collect()
}

fn maxsim_idf(q: &[[f32; TOKEN_DIM]], w: &[f32], d: &[[f32; TOKEN_DIM]]) -> f32 {
    q.iter().zip(w.iter().chain(std::iter::repeat(&1.0f32)))
        .map(|(qi, &wi)| wi * d.iter().map(|di| dot128(qi, di)).fold(f32::NEG_INFINITY, f32::max))
        .sum()
}

// ─── engine benchmarks ────────────────────────────────────────────────────────

/// HNSW over sentence-average embeddings.
/// Returns (candidate_frac, recall@K) for each ef value.
#[allow(dead_code)]
fn bench_hnsw_large(
    docs: &[LargeDoc],
    candidate_indices: &[usize],
    query_sent_embs: &[[f32; TOKEN_DIM]],
    gt_sets: &[HashSet<usize>],
) -> Vec<(f64, f64)> {
    let n_cands = candidate_indices.len();
    if n_cands == 0 { return vec![]; }

    let m_conn = 16usize.min(n_cands / 4 + 2);
    let hnsw = Hnsw::<f32, DistCosine>::new(m_conn, n_cands + 1, 8, 64, DistCosine {});
    for (pos, &doc_idx) in candidate_indices.iter().enumerate() {
        hnsw.insert((docs[doc_idx].sent_emb.as_slice(), pos));
    }

    // ef sweep — covers 10% / 25% / 50% of candidate set
    let ef_targets: Vec<usize> = vec![
        (n_cands / 10).max(K_EVAL),
        (n_cands / 4).max(K_EVAL),
        (n_cands / 2).max(K_EVAL),
        n_cands,
    ];
    let mut seen = HashSet::new();
    let mut ef_vals: Vec<usize> = Vec::new();
    for t in ef_targets {
        if seen.insert(t) { ef_vals.push(t); }
    }

    let mut pts = vec![];
    for ef in ef_vals {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qi, qe) in query_sent_embs.iter().enumerate() {
            let results = hnsw.search(qe.as_slice(), K_EVAL, ef);
            let mut scored: Vec<(usize, f32)> = results.iter()
                .map(|r| (candidate_indices[r.d_id], 1.0_f32 - r.distance))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_f += ef as f64 / n_cands as f64;
            tot_r += recall_at_k(&scored, &gt_sets[qi], K_EVAL);
        }
        let nq = query_sent_embs.len() as f64;
        pts.push((tot_f / nq, tot_r / nq));
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pts
}

/// PLAID: global k-means over candidate set, probe top-N centroids per query token.
#[allow(dead_code)]
fn bench_plaid_large(
    docs: &[LargeDoc],
    candidate_indices: &[usize],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    query_tok_ids: &[Vec<u32>],
    doc_freq: &HashMap<u32, usize>,
    n_real: usize,
    gt_sets: &[HashSet<usize>],
    rng: &mut SmallRng,
) -> Vec<(f64, f64)> {
    let n_cands = candidate_indices.len();
    if n_cands == 0 { return vec![]; }

    let k = ((n_cands as f64).sqrt() as usize).clamp(5, 200);
    let idx = CentIdx::build(docs, candidate_indices, k, rng);

    let probes = [1usize, 2, 3, 5, 8, k / 2, k];
    let mut pts = vec![];

    for &np in &probes {
        let np = np.min(idx.centers.len());
        if np == 0 { continue; }
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qi, (qtoks, qids)) in query_toks.iter().zip(query_tok_ids.iter()).enumerate() {
            let idf = idf_weights_for(qids, doc_freq, n_real);
            let cands = idx.probe(qtoks, np);
            if cands.is_empty() { continue; }
            tot_f += cands.len() as f64 / n_cands as f64;
            let mut scored: Vec<(usize, f32)> = cands.iter()
                .map(|&ci| (candidate_indices[ci], maxsim_idf(qtoks, &idf, &docs[candidate_indices[ci]].tokens)))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_r += recall_at_k(&scored, &gt_sets[qi], K_EVAL);
        }
        let nq = query_toks.len() as f64;
        let frac = tot_f / nq;
        if frac > 0.0 {
            pts.push((frac, tot_r / nq));
        }
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pts.dedup_by_key(|p| (p.0 * 1_000_000.0) as i64);
    pts
}

/// WARP: XTR token-similarity threshold.
#[allow(dead_code)]
fn bench_warp_large(
    docs: &[LargeDoc],
    candidate_indices: &[usize],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    query_tok_ids: &[Vec<u32>],
    doc_freq: &HashMap<u32, usize>,
    n_real: usize,
    gt_sets: &[HashSet<usize>],
) -> Vec<(f64, f64)> {
    let n_cands = candidate_indices.len();
    if n_cands == 0 { return vec![]; }

    let thresholds = [0.95f32, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10];
    let mut pts = vec![];

    for &t in &thresholds {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qi, (qtoks, qids)) in query_toks.iter().zip(query_tok_ids.iter()).enumerate() {
            let idf = idf_weights_for(qids, doc_freq, n_real);
            let cands: Vec<usize> = candidate_indices.iter().copied()
                .filter(|&di| {
                    let mx = docs[di].tokens.iter()
                        .flat_map(|dt| qtoks.iter().map(|qt| dot128(qt, dt)))
                        .fold(f32::NEG_INFINITY, f32::max);
                    mx > t
                })
                .collect();
            let frac = (cands.len() as f64 / n_cands as f64).max(1.0 / n_cands as f64);
            tot_f += frac;
            if cands.is_empty() { continue; }
            let mut scored: Vec<(usize, f32)> = cands.iter()
                .map(|&di| (di, maxsim_idf(qtoks, &idf, &docs[di].tokens)))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_r += recall_at_k(&scored, &gt_sets[qi], K_EVAL);
        }
        let nq = query_toks.len() as f64;
        let frac = tot_f / nq;
        if frac > 0.005 {
            pts.push((frac, tot_r / nq));
        }
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pts
}

/// TACHIOM: per-category centroid routing.
#[allow(dead_code)]
fn bench_tachiom_large(
    docs: &[LargeDoc],
    candidate_indices: &[usize],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    query_tok_ids: &[Vec<u32>],
    doc_freq: &HashMap<u32, usize>,
    n_real: usize,
    gt_sets: &[HashSet<usize>],
    rng: &mut SmallRng,
) -> Vec<(f64, f64)> {
    let n_cands = candidate_indices.len();
    if n_cands == 0 { return vec![]; }

    let k_per_cat = ((n_cands as f64).sqrt() as usize / N_CATEGORIES).clamp(3, 30);

    // Build one centroid index per category over the candidate set
    let cat_idxs: Vec<CentIdx> = (0..N_CATEGORIES).map(|cat| {
        let cat_cands: Vec<usize> = candidate_indices.iter().copied()
            .filter(|&i| docs[i].category as usize == cat)
            .collect();
        CentIdx::build(docs, &cat_cands, k_per_cat, rng)
    }).collect();

    let probes = [1usize, 2, 3, 5, 8];
    let mut pts = vec![];

    for &np in &probes {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qi, (qtoks, qids)) in query_toks.iter().zip(query_tok_ids.iter()).enumerate() {
            let idf = idf_weights_for(qids, doc_freq, n_real);
            let mut cands: HashSet<usize> = HashSet::new();
            for qt in qtoks {
                let best_cat = cat_idxs.iter().enumerate()
                    .filter(|(_, idx)| !idx.centers.is_empty())
                    .map(|(cat, idx)| {
                        let s = idx.centers.iter().map(|c| dot128(qt, c)).fold(f32::NEG_INFINITY, f32::max);
                        (cat, s)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(cat, _)| cat).unwrap_or(0);
                let n_p = np.min(cat_idxs[best_cat].centers.len().max(1));
                for d in cat_idxs[best_cat].probe(&[*qt], n_p) {
                    // d is a candidate_indices index; convert to doc id
                    cands.insert(candidate_indices[d]);
                }
            }
            tot_f += cands.len() as f64 / n_cands as f64;
            if cands.is_empty() { continue; }
            let mut scored: Vec<(usize, f32)> = cands.iter()
                .map(|&di| (di, maxsim_idf(qtoks, &idf, &docs[di].tokens)))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_r += recall_at_k(&scored, &gt_sets[qi], K_EVAL);
        }
        let nq = query_toks.len() as f64;
        pts.push((tot_f / nq, tot_r / nq));
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pts
}

// ─── run_at_scale ─────────────────────────────────────────────────────────────

/// Build corpus + run all 4 engines × 3 filter modes for one scale.
/// Returns a map (engine_name, filter_mode) → Vec<(frac, recall)> for each combination.
#[allow(clippy::too_many_arguments)]
fn run_at_scale(
    n: usize,
    real_docs: &[LargeDoc],
    gt_sets_usize: &[HashSet<usize>],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    query_tok_ids: &[Vec<u32>],
    query_sent_embs: &[[f32; TOKEN_DIM]],
    doc_freq: &HashMap<u32, usize>,
    n_real: usize,
    rng_seed: u64,
) -> HashMap<(&'static str, FilterMode), Vec<(f64, f64)>> {
    let n_synth = n.saturating_sub(real_docs.len());

    // Memory check
    let est_bytes = (n_synth as u64) * (SYNTH_TOKENS as u64) * (TOKEN_DIM as u64) * 4;
    if est_bytes > MEM_WARN_BYTES {
        eprintln!(
            "  [warn] N={n}: estimated token embedding RAM ~{:.1} GB (>{:.0} GB threshold). \
             Consider N<=1M for full storage.",
            est_bytes as f64 / 1e9,
            MEM_WARN_BYTES as f64 / 1e9
        );
    }

    let mut rng = SmallRng::seed_from_u64(rng_seed);

    // Build corpus: real docs first, then synthetic distractors
    let mut docs: Vec<LargeDoc> = real_docs.to_vec();
    // Re-ID real docs so id == position in docs vec
    for (i, d) in docs.iter_mut().enumerate() {
        d.id = i;
    }

    // At N=10M, skip full token storage for synthetics — use centroid-only approach
    let large_scale = n >= 10_000_000;

    if !large_scale {
        print!("    generating {} synthetic docs… ", n_synth);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        for i in 0..n_synth {
            let mut d = gen_synthetic_doc(real_docs.len() + i, &mut rng);
            d.id = real_docs.len() + i;
            docs.push(d);
        }
        println!("ok  (corpus size: {})", docs.len());
    } else {
        // At 10M: only store real docs + a 1M sample for centroid training.
        // For recall purposes the GT docs are always present; distractors only affect
        // centroid routing and candidate set size.
        let sample_n = 1_000_000usize.min(n_synth);
        print!("    generating {}M-doc corpus (1M sample for centroids)… ", n / 1_000_000);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        for i in 0..sample_n {
            let mut d = gen_synthetic_doc(real_docs.len() + i, &mut rng);
            d.id = real_docs.len() + i;
            docs.push(d);
        }
        println!("ok  ({} docs in RAM, {} extrapolated)", docs.len(), n);
    }

    let n_in_ram = docs.len();
    // GT sets are relative to doc position index (0..real_docs.len())
    // They were already built against position indices 0-99.

    // ── PRE-BUILD HNSW indices once (avoids O(N_queries × N_filters) builds) ──

    // Global HNSW for FilterMode::None
    print!("    building global HNSW ({} docs)… ", n_in_ram);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let global_hnsw = {
        let m_conn = 16usize.min(n_in_ram.saturating_div(4).max(2));
        let h = Hnsw::<f32, DistCosine>::new(m_conn, n_in_ram + 1, 8, 64, DistCosine {});
        for (i, doc) in docs.iter().enumerate() {
            h.insert((doc.sent_emb.as_slice(), i));
        }
        h
    };
    println!("ok");

    // Per-category HNSWs for FilterMode::Category and FilterMode::CategoryAndYear.
    // cat_hnsw_and_docs[c] = (hnsw, doc_indices_in_cat)
    print!("    building {} category HNSWs… ", N_CATEGORIES);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let cat_hnsw_and_docs: Vec<(Hnsw<f32, DistCosine>, Vec<usize>)> =
        (0..N_CATEGORIES).map(|cat| {
            let cat_docs: Vec<usize> = docs.iter().enumerate()
                .filter(|(_, d)| d.category as usize == cat)
                .map(|(i, _)| i)
                .collect();
            let n_c = cat_docs.len();
            let m_c = 16usize.min(n_c.saturating_div(4).max(2));
            let h = Hnsw::<f32, DistCosine>::new(m_c, n_c + 1, 8, 64, DistCosine {});
            for (pos, &di) in cat_docs.iter().enumerate() {
                h.insert((docs[di].sent_emb.as_slice(), pos));
            }
            (h, cat_docs)
        }).collect();
    println!("ok");

    let filters = [FilterMode::None, FilterMode::Category, FilterMode::CategoryAndYear];
    let engines = ["HNSW", "PLAID", "WARP", "TACHIOM"];

    let mut out: HashMap<(&'static str, FilterMode), Vec<(f64, f64)>> = HashMap::new();

    for &filter in &filters {
        // For large scale, adjust "total" N for frac computation to true N (not in-RAM sample)
        let effective_n = if large_scale {
            // Count how many of true N pass filter (approximate)
            match filter {
                FilterMode::None => n,
                FilterMode::Category => n / N_CATEGORIES,
                FilterMode::CategoryAndYear => {
                    // ~47% of category passes year filter on average
                    n / N_CATEGORIES * 47 / 100
                }
            }
        } else {
            n_in_ram
        };

        // Build per-query candidate index sets for filtering
        // We average across queries for the table; per-query filter is handled in bench fns
        // Here we need a *single* candidate set for the centroid index (we use the union over all queries)
        // For None filter: all docs; otherwise we pick the first query's category for the global build,
        // but each bench fn applies per-query filtering independently.
        // Because all queries have different categories, we build one centroid index
        // per (filter×query) inside each bench function; but for brevity here
        // we pass the full candidate list per query separately.

        let mut rng_local = SmallRng::seed_from_u64(rng_seed ^ 0xABCD1234 ^ filter as u64);

        // Compute per-query candidate lists
        let per_query_cands: Vec<Vec<usize>> = (0..GT_QUERIES.len()).map(|qi| {
            let cat = QUERY_CATEGORIES[qi];
            let yf = QUERY_YEAR_FLOORS[qi];
            apply_filter(&docs, cat, yf, filter)
        }).collect();

        // Build union of all candidate lists for building *one* shared centroid index
        // (for PLAID/WARP/TACHIOM we benchmark per-query separately below)
        let union_cands: Vec<usize> = {
            let mut s: HashSet<usize> = HashSet::new();
            for c in &per_query_cands { for &i in c { s.insert(i); } }
            let mut v: Vec<usize> = s.into_iter().collect();
            v.sort_unstable();
            v
        };
        let n_union = union_cands.len();

        // ── HNSW ──
        {
            // Use pre-built indices (global / per-category) to avoid O(N_q × N_f) builds.
            let mut all_pts: HashMap<i64, (f64, f64, usize)> = HashMap::new();

            for qi in 0..GT_QUERIES.len() {
                let qcat = QUERY_CATEGORIES[qi] as usize;
                let qyear = QUERY_YEAR_FLOORS[qi];

                match filter {
                    FilterMode::None => {
                        let n_c = n_in_ram;
                        let ef_targets: Vec<usize> = {
                            let mut v = vec![
                                (n_c / 10).max(K_EVAL),
                                (n_c / 4).max(K_EVAL),
                                (n_c / 2).max(K_EVAL),
                                n_c,
                            ];
                            v.dedup();
                            v
                        };
                        for ef in ef_targets {
                            let results = global_hnsw.search(
                                query_sent_embs[qi].as_slice(), K_EVAL, ef,
                            );
                            let mut scored: Vec<(usize, f32)> = results.iter()
                                .map(|r| (r.d_id, 1.0_f32 - r.distance))
                                .collect();
                            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            let frac = (ef as f64 / n_c as f64).min(1.0)
                                * (n_c as f64 / effective_n as f64).min(1.0);
                            let r = recall_at_k(&scored, &gt_sets_usize[qi], K_EVAL);
                            let key = (frac * 1_000_000.0) as i64;
                            let e = all_pts.entry(key).or_insert((frac, 0.0, 0));
                            e.1 += r;
                            e.2 += 1;
                        }
                    }
                    FilterMode::Category => {
                        let (ref h, ref cat_docs) = cat_hnsw_and_docs[qcat];
                        let n_c = cat_docs.len();
                        if n_c == 0 { continue; }
                        let ef_targets: Vec<usize> = {
                            let mut v = vec![
                                (n_c / 10).max(K_EVAL),
                                (n_c / 4).max(K_EVAL),
                                (n_c / 2).max(K_EVAL),
                                n_c,
                            ];
                            v.dedup();
                            v
                        };
                        for ef in ef_targets {
                            let results = h.search(
                                query_sent_embs[qi].as_slice(), K_EVAL, ef,
                            );
                            let mut scored: Vec<(usize, f32)> = results.iter()
                                .map(|r| (cat_docs[r.d_id], 1.0_f32 - r.distance))
                                .collect();
                            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            let frac = (ef as f64 / n_c as f64).min(1.0)
                                * (n_c as f64 / effective_n as f64).min(1.0);
                            let r = recall_at_k(&scored, &gt_sets_usize[qi], K_EVAL);
                            let key = (frac * 1_000_000.0) as i64;
                            let e = all_pts.entry(key).or_insert((frac, 0.0, 0));
                            e.1 += r;
                            e.2 += 1;
                        }
                    }
                    FilterMode::CategoryAndYear => {
                        // Pre-filter by category (cat HNSW), post-filter by year.
                        let (ref h, ref cat_docs) = cat_hnsw_and_docs[qcat];
                        let n_c = cat_docs.len();
                        if n_c == 0 { continue; }
                        let n_cat_year = cat_docs.iter()
                            .filter(|&&di| docs[di].year >= qyear)
                            .count();
                        if n_cat_year == 0 { continue; }
                        let ef_targets: Vec<usize> = {
                            let mut v = vec![
                                (n_c / 10).max(K_EVAL),
                                (n_c / 4).max(K_EVAL),
                                (n_c / 2).max(K_EVAL),
                                n_c,
                            ];
                            v.dedup();
                            v
                        };
                        for ef in ef_targets {
                            // Fetch ef results from category HNSW, then post-filter by year.
                            let results = h.search(query_sent_embs[qi].as_slice(), ef, ef);
                            let mut scored: Vec<(usize, f32)> = results.iter()
                                .filter(|r| {
                                    r.d_id < cat_docs.len()
                                        && docs[cat_docs[r.d_id]].year >= qyear
                                })
                                .map(|r| (cat_docs[r.d_id], 1.0_f32 - r.distance))
                                .collect();
                            if scored.is_empty() { continue; }
                            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            scored.truncate(K_EVAL);
                            let frac = (scored.len() as f64 / n_cat_year as f64).min(1.0)
                                * (n_cat_year as f64 / effective_n as f64).min(1.0);
                            let r = recall_at_k(&scored, &gt_sets_usize[qi], K_EVAL);
                            let key = (frac * 1_000_000.0) as i64;
                            let e = all_pts.entry(key).or_insert((frac, 0.0, 0));
                            e.1 += r;
                            e.2 += 1;
                        }
                    }
                }
            }

            let mut pts: Vec<(f64, f64)> = all_pts.values()
                .filter(|(_, _, cnt)| *cnt > 0)
                .map(|&(f, r, cnt)| (f, r / cnt as f64))
                .collect();
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            out.insert(("HNSW", filter), pts);
        }

        // ── PLAID ──
        {
            // Build one centroid index over the union candidate set for this filter
            let k = ((n_union as f64).sqrt() as usize).clamp(5, 200);
            let idx = CentIdx::build(&docs, &union_cands, k, &mut rng_local);
            let probes = [1usize, 2, 3, 5, 8, k.saturating_sub(1).max(1), k];
            let mut probe_pts: Vec<(f64, f64)> = Vec::new();
            for &np in &probes {
                let np = np.min(idx.centers.len());
                if np == 0 { continue; }
                let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
                for (qi, cands) in per_query_cands.iter().enumerate() {
                    if cands.is_empty() { continue; }
                    let idf = idf_weights_for(&query_tok_ids[qi], doc_freq, n_real);
                    let raw_cands = idx.probe(&query_toks[qi], np);
                    // Map back through union_cands to doc indices, then filter to this query's cands
                    let cand_set: HashSet<usize> = cands.iter().copied().collect();
                    let filtered_cands: Vec<usize> = raw_cands.iter()
                        .filter_map(|&ci| {
                            if ci < union_cands.len() {
                                let doc_idx = union_cands[ci];
                                if cand_set.contains(&doc_idx) { Some(doc_idx) } else { None }
                            } else { None }
                        })
                        .collect();
                    if filtered_cands.is_empty() { continue; }
                    let frac = filtered_cands.len() as f64 / effective_n as f64;
                    tot_f += frac;
                    let mut scored: Vec<(usize, f32)> = filtered_cands.iter()
                        .map(|&di| (di, maxsim_idf(&query_toks[qi], &idf, &docs[di].tokens)))
                        .collect();
                    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    tot_r += recall_at_k(&scored, &gt_sets_usize[qi], K_EVAL);
                }
                let nq = per_query_cands.iter().filter(|c| !c.is_empty()).count() as f64;
                if nq > 0.0 {
                    probe_pts.push((tot_f / nq, tot_r / nq));
                }
            }
            probe_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            probe_pts.dedup_by_key(|p| (p.0 * 1_000_000.0) as i64);
            out.insert(("PLAID", filter), probe_pts);
        }

        // ── WARP ──
        {
            let thresholds = [0.95f32, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10];
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for &t in &thresholds {
                let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
                let mut nq = 0usize;
                for (qi, cands) in per_query_cands.iter().enumerate() {
                    if cands.is_empty() { continue; }
                    nq += 1;
                    let idf = idf_weights_for(&query_tok_ids[qi], doc_freq, n_real);
                    let warp_cands: Vec<usize> = cands.iter().copied()
                        .filter(|&di| {
                            let mx = docs[di].tokens.iter()
                                .flat_map(|dt| query_toks[qi].iter().map(|qt| dot128(qt, dt)))
                                .fold(f32::NEG_INFINITY, f32::max);
                            mx > t
                        })
                        .collect();
                    let frac = (warp_cands.len() as f64 / effective_n as f64).max(1.0 / effective_n as f64);
                    tot_f += frac;
                    if warp_cands.is_empty() { continue; }
                    let mut scored: Vec<(usize, f32)> = warp_cands.iter()
                        .map(|&di| (di, maxsim_idf(&query_toks[qi], &idf, &docs[di].tokens)))
                        .collect();
                    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    tot_r += recall_at_k(&scored, &gt_sets_usize[qi], K_EVAL);
                }
                if nq > 0 {
                    let frac = tot_f / nq as f64;
                    if frac > 0.001 {
                        pts.push((frac, tot_r / nq as f64));
                    }
                }
            }
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            out.insert(("WARP", filter), pts);
        }

        // ── TACHIOM ──
        {
            let k_per_cat = ((n_union as f64).sqrt() as usize / N_CATEGORIES).clamp(3, 30);
            // Build per-category centroid indices over the union candidate set
            let cat_idxs: Vec<CentIdx> = (0..N_CATEGORIES).map(|cat| {
                let cat_cands: Vec<usize> = union_cands.iter().copied()
                    .filter(|&i| docs[i].category as usize == cat)
                    .collect();
                CentIdx::build(&docs, &cat_cands, k_per_cat, &mut rng_local)
            }).collect();

            let probes = [1usize, 2, 3, 5, 8];
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for &np in &probes {
                let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
                let mut nq = 0usize;
                for (qi, cands) in per_query_cands.iter().enumerate() {
                    if cands.is_empty() { continue; }
                    nq += 1;
                    let cand_set: HashSet<usize> = cands.iter().copied().collect();
                    let idf = idf_weights_for(&query_tok_ids[qi], doc_freq, n_real);
                    let mut raw_cands: HashSet<usize> = HashSet::new();
                    for qt in &query_toks[qi] {
                        let best_cat = cat_idxs.iter().enumerate()
                            .filter(|(_, idx)| !idx.centers.is_empty())
                            .map(|(cat, idx)| {
                                let s = idx.centers.iter().map(|c| dot128(qt, c)).fold(f32::NEG_INFINITY, f32::max);
                                (cat, s)
                            })
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .map(|(cat, _)| cat).unwrap_or(0);
                        let n_p = np.min(cat_idxs[best_cat].centers.len().max(1));
                        for local_ci in cat_idxs[best_cat].probe(&[*qt], n_p) {
                            // local_ci is an index into union_cands
                            if local_ci < union_cands.len() {
                                raw_cands.insert(union_cands[local_ci]);
                            }
                        }
                    }
                    // Only keep candidates that pass this query's filter
                    let filtered_cands: Vec<usize> = raw_cands.iter()
                        .copied()
                        .filter(|di| cand_set.contains(di))
                        .collect();
                    tot_f += filtered_cands.len() as f64 / effective_n as f64;
                    if filtered_cands.is_empty() { continue; }
                    let mut scored: Vec<(usize, f32)> = filtered_cands.iter()
                        .map(|&di| (di, maxsim_idf(&query_toks[qi], &idf, &docs[di].tokens)))
                        .collect();
                    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    tot_r += recall_at_k(&scored, &gt_sets_usize[qi], K_EVAL);
                }
                if nq > 0 {
                    pts.push((tot_f / nq as f64, tot_r / nq as f64));
                }
            }
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            out.insert(("TACHIOM", filter), pts);
        }

        let _ = (effective_n, n_union, &union_cands);
    }

    let _ = (engines, rng_seed);
    out
}

// ─── terminal summary ─────────────────────────────────────────────────────────

fn interpolate_recall(pts: &[(f64, f64)], target: f64) -> Option<f64> {
    if pts.is_empty() { return None; }
    // Find the point with frac closest to target (within 3× tolerance)
    pts.iter()
        .min_by_key(|&&(f, _)| ((f - target).abs() * 100_000.0) as i64)
        .and_then(|&(f, r)| if (f - target).abs() < target * 3.0 { Some(r) } else { None })
}

fn print_large_bench_summary(
    results: &[LargeBenchResult],
    scales: &[usize],
) {
    const CYAN: &str  = "\x1b[36m";
    const BOLD: &str  = "\x1b[1m";
    const DIM:  &str  = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    let fracs: &[(f64, &str)] = &[(0.10, "10%"), (0.25, "25%"), (0.50, "50%"), (1.00, "100%")];

    for &n in scales {
        let n_idx = SCALES.iter().position(|&s| s == n).unwrap_or(0);
        let n_label = if n >= 1_000_000 { format!("{}M", n / 1_000_000) } else { format!("{}K", n / 1_000) };

        println!();
        println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
        println!("{BOLD}  Large-Scale Needle-in-Haystack  N={n_label}  (Recall@{K_EVAL} vs LLM GT){RESET}");
        println!("{DIM}  100 real GT docs among {n_label} total · candidate fraction of full corpus{RESET}");
        println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");

        let hdr: Vec<String> = fracs.iter().map(|(_, l)| format!("{:>7}", l)).collect();
        println!();
        println!("  {:<44}  {}", "Engine / Filter", hdr.join("  "));
        println!("  {}  {}", "─".repeat(44), "─".repeat(7 * fracs.len() + 2 * (fracs.len() - 1)));

        for res in results {
            if res.by_scale.len() <= n_idx { continue; }
            let pts = &res.by_scale[n_idx];
            let cells: Vec<String> = fracs.iter().map(|(target, _)| {
                match interpolate_recall(pts, *target) {
                    Some(r) => format!("{:>7.3}", r),
                    None => "      —".to_string(),
                }
            }).collect();
            let label = format!("{} / {}", res.engine, res.filter.label());
            println!("  {:<44}  {}", label, cells.join("  "));
        }
    }
    println!();
}

// ─── plotting ─────────────────────────────────────────────────────────────────

fn to_pct_log(frac: f64) -> f64 {
    (frac.clamp(0.001, 1.0) * 100.0).log2()
}

/// plots/large_bench_recall_vs_frac.svg — Recall@10 vs candidate frac at N=1M.
fn plot_recall_vs_frac(results: &[LargeBenchResult]) -> Result<()> {
    std::fs::create_dir_all("plots")?;
    let path = "plots/large_bench_recall_vs_frac.svg";
    let root = SVGBackend::new(path, (800, 560)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find scale index for N=1M
    let scale_idx = SCALES.iter().position(|&s| s == 1_000_000).unwrap_or(1);

    let x_min = to_pct_log(0.005);
    let x_max = to_pct_log(1.00);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Recall@10 vs Candidate Fraction  (N=1M, needle-in-haystack)",
            ("sans-serif", 15).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f64..1.06f64)?;

    chart.configure_mesh()
        .x_desc("Candidate set (% of corpus, log scale)")
        .y_desc("Recall@10 vs LLM GT")
        .x_labels(7)
        .y_labels(6)
        .x_label_formatter(&|&x| {
            let v = 2f64.powf(x);
            format!("{:.1}%", v)
        })
        .light_line_style(RGBColor(220, 220, 220).stroke_width(1))
        .draw()?;

    let palette: Vec<(RGBColor, u32, &'static str)> = vec![
        (RGBColor(255, 127,  14), 2, "HNSW/none"),
        (RGBColor(255, 180,  80), 1, "HNSW/cat"),
        (RGBColor(255, 210, 140), 1, "HNSW/cat+yr"),
        (RGBColor( 33, 102, 172), 2, "PLAID/none"),
        (RGBColor( 90, 150, 210), 1, "PLAID/cat"),
        (RGBColor(150, 190, 230), 1, "PLAID/cat+yr"),
        (RGBColor(215,  48,  39), 2, "WARP/none"),
        (RGBColor(240, 110, 100), 1, "WARP/cat"),
        (RGBColor(250, 170, 160), 1, "WARP/cat+yr"),
        (RGBColor( 26, 152,  80), 3, "TACHIOM/none"),
        (RGBColor( 80, 190, 130), 1, "TACHIOM/cat"),
        (RGBColor(140, 220, 170), 1, "TACHIOM/cat+yr"),
    ];

    let order: Vec<(&'static str, FilterMode)> = vec![
        ("HNSW",    FilterMode::None),
        ("HNSW",    FilterMode::Category),
        ("HNSW",    FilterMode::CategoryAndYear),
        ("PLAID",   FilterMode::None),
        ("PLAID",   FilterMode::Category),
        ("PLAID",   FilterMode::CategoryAndYear),
        ("WARP",    FilterMode::None),
        ("WARP",    FilterMode::Category),
        ("WARP",    FilterMode::CategoryAndYear),
        ("TACHIOM", FilterMode::None),
        ("TACHIOM", FilterMode::Category),
        ("TACHIOM", FilterMode::CategoryAndYear),
    ];

    for (i, (eng, flt)) in order.iter().enumerate() {
        let res = results.iter().find(|r| r.engine == *eng && r.filter == *flt);
        let Some(res) = res else { continue };
        if res.by_scale.len() <= scale_idx { continue; }
        let pts: Vec<(f64, f64)> = res.by_scale[scale_idx].iter()
            .filter(|&&(f, _)| f >= 0.001)
            .map(|&(f, r)| (to_pct_log(f), r))
            .collect();
        if pts.is_empty() { continue; }
        let (color, sw, lbl) = palette[i % palette.len()];
        chart.draw_series(LineSeries::new(
            pts.clone(),
            ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
        ))?
        .label(lbl)
        .legend(move |(x, y)| PathElement::new(
            vec![(x, y), (x + 20, y)],
            ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
        ));
        chart.draw_series(pts.iter().map(|&(x, y)| {
            Circle::new((x, y), 4, ShapeStyle { color: color.to_rgba(), filled: true, stroke_width: 1 })
        }))?;
    }

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.88))
        .border_style(RGBColor(160, 160, 160).stroke_width(1))
        .label_font(("sans-serif", 10).into_font())
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    println!("  → {path}  (open with: open {path})");
    Ok(())
}

/// plots/large_bench_recall_vs_n.svg — Recall@10 vs N at 10% candidates.
fn plot_recall_vs_n(results: &[LargeBenchResult]) -> Result<()> {
    std::fs::create_dir_all("plots")?;
    let path = "plots/large_bench_recall_vs_n.svg";
    let root = SVGBackend::new(path, (800, 560)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_vals: Vec<f64> = SCALES.iter().map(|&n| (n as f64).log2()).collect();
    let x_min = x_vals[0] - 0.5;
    let x_max = x_vals[x_vals.len() - 1] + 0.5;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Recall@10 vs Corpus Size  (10% candidate fraction, needle-in-haystack)",
            ("sans-serif", 14).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f64..1.06f64)?;

    chart.configure_mesh()
        .x_desc("Corpus size N (log2 scale)")
        .y_desc("Recall@10 at 10% candidates")
        .x_labels(x_vals.len())
        .y_labels(6)
        .x_label_formatter(&|&x| {
            let n = 2f64.powf(x) as usize;
            if n >= 1_000_000 { format!("{}M", n / 1_000_000) }
            else { format!("{}K", n / 1_000) }
        })
        .light_line_style(RGBColor(220, 220, 220).stroke_width(1))
        .draw()?;

    let palette: Vec<(RGBColor, u32, &'static str)> = vec![
        (RGBColor(255, 127,  14), 2, "HNSW/none"),
        (RGBColor(255, 180,  80), 1, "HNSW/cat"),
        (RGBColor(255, 210, 140), 1, "HNSW/cat+yr"),
        (RGBColor( 33, 102, 172), 2, "PLAID/none"),
        (RGBColor( 90, 150, 210), 1, "PLAID/cat"),
        (RGBColor(150, 190, 230), 1, "PLAID/cat+yr"),
        (RGBColor(215,  48,  39), 2, "WARP/none"),
        (RGBColor(240, 110, 100), 1, "WARP/cat"),
        (RGBColor(250, 170, 160), 1, "WARP/cat+yr"),
        (RGBColor( 26, 152,  80), 3, "TACHIOM/none"),
        (RGBColor( 80, 190, 130), 1, "TACHIOM/cat"),
        (RGBColor(140, 220, 170), 1, "TACHIOM/cat+yr"),
    ];

    let order: Vec<(&'static str, FilterMode)> = vec![
        ("HNSW",    FilterMode::None),
        ("HNSW",    FilterMode::Category),
        ("HNSW",    FilterMode::CategoryAndYear),
        ("PLAID",   FilterMode::None),
        ("PLAID",   FilterMode::Category),
        ("PLAID",   FilterMode::CategoryAndYear),
        ("WARP",    FilterMode::None),
        ("WARP",    FilterMode::Category),
        ("WARP",    FilterMode::CategoryAndYear),
        ("TACHIOM", FilterMode::None),
        ("TACHIOM", FilterMode::Category),
        ("TACHIOM", FilterMode::CategoryAndYear),
    ];

    for (i, (eng, flt)) in order.iter().enumerate() {
        let res = results.iter().find(|r| r.engine == *eng && r.filter == *flt);
        let Some(res) = res else { continue };

        let pts: Vec<(f64, f64)> = SCALES.iter().enumerate()
            .filter_map(|(si, &n)| {
                let scale_pts = res.by_scale.get(si)?;
                let r = interpolate_recall(scale_pts, 0.10)?;
                Some(((n as f64).log2(), r))
            })
            .collect();
        if pts.is_empty() { continue; }

        let (color, sw, lbl) = palette[i % palette.len()];
        chart.draw_series(LineSeries::new(
            pts.clone(),
            ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
        ))?
        .label(lbl)
        .legend(move |(x, y)| PathElement::new(
            vec![(x, y), (x + 20, y)],
            ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
        ));
        chart.draw_series(pts.iter().map(|&(x, y)| {
            Circle::new((x, y), 5, ShapeStyle { color: color.to_rgba(), filled: true, stroke_width: 1 })
        }))?;
    }

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.88))
        .border_style(RGBColor(160, 160, 160).stroke_width(1))
        .label_font(("sans-serif", 10).into_font())
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    println!("  → {path}  (open with: open {path})");
    Ok(())
}

// ─── entry point ─────────────────────────────────────────────────────────────

pub async fn run_large_bench(_vocab_path: &Path) -> Result<()> {
    const CYAN: &str  = "\x1b[36m";
    const BOLD: &str  = "\x1b[1m";
    const DIM:  &str  = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    println!();
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!("{BOLD}  Large-Scale Needle-in-Haystack Benchmark{RESET}");
    println!("{DIM}  100 real docs (Jina ColBERT) in a growing synthetic corpus");
    println!("  Scales: 100K / 1M / 10M   Filters: none / category / cat+year");
    println!("  Engines: HNSW / PLAID / WARP / TACHIOM{RESET}");
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!();

    // ── 1. Load LLM GT labels ────────────────────────────────────────────────
    const GT_CACHE: &str = "cache/llm_gt.json";
    let gt_raw: Option<HashMap<String, Vec<u32>>> = std::fs::read_to_string(GT_CACHE).ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    let gt_map = match gt_raw {
        Some(m) => {
            println!("  [gt] loaded LLM GT labels from {GT_CACHE} ({} queries)", m.len());
            m
        }
        None => {
            eprintln!("  [warn] {GT_CACHE} not found — using heuristic category membership as GT.");
            eprintln!("         Run 'cargo run -- gt-bench' first for real LLM-judged labels.");
            // Fallback: category membership
            let mut m: HashMap<String, Vec<u32>> = HashMap::new();
            let query_cats: &[u8] = &QUERY_CATEGORIES;
            for (qi, &query) in GT_QUERIES.iter().enumerate() {
                let cat = query_cats[qi];
                let ids: Vec<u32> = (0u32..100)
                    .filter(|&id| doc_category_for_real(id) == cat)
                    .collect();
                m.insert(query.to_string(), ids);
            }
            m
        }
    };

    // GT sets using doc position indices 0-99 (real docs are always at positions 0-99)
    let gt_sets_usize: Vec<HashSet<usize>> = GT_QUERIES.iter()
        .map(|q| gt_map.get(*q).cloned().unwrap_or_default()
            .into_iter().map(|id| id as usize).collect())
        .collect();

    // ── 2. Load Jina ColBERT token embeddings ────────────────────────────────
    println!("  [jina] loading token embeddings from cache…");
    let jina_cache = jina::load_cache();
    if jina_cache.is_empty() {
        eprintln!("  [warn] Jina cache empty ({}).", jina::CACHE_PATH);
        eprintln!("         Run 'cargo run -- gt-bench' first to populate Jina token embeddings.");
    }

    // GT_CORPUS texts in order (ids 0-99)
    // We reproduce only the GT_CORPUS text list needed to look up cache entries.
    // Rather than importing GT_CORPUS from gtbench (private mod), we use the same
    // approach: look up tokens by text.  The texts below must match GT_CORPUS exactly.
    let gt_corpus_texts: &[(u32, &str)] = &[
        (0,  "The river bank was slippery after the spring flood receded."),
        (1,  "The hiking trail runs along the left bank of the Colorado River."),
        (2,  "He pitched the tent on the flat bank beside the stream."),
        (3,  "A flock of cranes migrated south along the river valley."),
        (4,  "The geological fault line runs beneath the river delta."),
        (5,  "Sediment deposits from the river formed a broad alluvial plain."),
        (6,  "The kayaker navigated the rapids near the river's rocky bank."),
        (7,  "Flooding undermined the foundation of the riverside warehouse."),
        (8,  "Heavy rains caused the river to overflow its banks overnight."),
        (9,  "Children fished off the wooden dock at the river bank."),
        (10, "Willow trees lined the gentle bank of the meandering stream."),
        (11, "The geologist studied erosion patterns along the river's edge."),
        (12, "Spring runoff carried nutrients downstream to the floodplain."),
        (13, "Sandbars formed at the river bend where the current slowed."),
        (14, "Morning fog hovered above the quiet surface of the river."),
        (15, "The bridge spanned the wide river at its narrowest crossing."),
        (16, "Nature photographers captured herons wading in the river shallows."),
        (17, "The conservation project restored native vegetation along the riparian corridor."),
        (18, "Canoeists paddled upstream against the current toward the falls."),
        (19, "The old mill wheel was powered by water drawn from the millstream."),
        (20, "She opened a savings account at the bank downtown."),
        (21, "Interest rates at the central bank rose sharply this quarter."),
        (22, "The investment bank underwrote the government bond issuance."),
        (23, "The reserve bank adjusted monetary policy after the inflation report."),
        (24, "Venture capital firms invested heavily in financial technology startups."),
        (25, "The branch manager approved a small business loan application."),
        (26, "Online banking services reduced the need for physical branches."),
        (27, "The Federal Reserve raised the benchmark interest rate by 25 basis points."),
        (28, "Mortgage refinancing applications surged as rates fell to historic lows."),
        (29, "The commercial bank syndicated a large infrastructure loan."),
        (30, "Interbank lending rates spiked during the liquidity crisis."),
        (31, "Quarterly earnings at the regional bank beat analyst expectations."),
        (32, "A fintech startup launched a mobile-first checking account service."),
        (33, "The central bank intervened in currency markets to stabilize the exchange rate."),
        (34, "Private equity firms acquired several regional banks in a consolidation wave."),
        (35, "Digital wallets disrupted traditional banking transaction models."),
        (36, "The auditors reviewed the bank's balance sheet and capital ratios."),
        (37, "Credit unions offered competitive savings account interest rates."),
        (38, "The treasury department managed foreign exchange reserves."),
        (39, "Her financial advisor recommended diversifying across asset classes."),
        (40, "The tower crane operator carefully lowered the prefabricated concrete slab."),
        (41, "Safety regulations required cranes to be inspected before each lift."),
        (42, "The harbor crane unloaded shipping containers onto the dock."),
        (43, "Engineers calculated the maximum load capacity for the crawler crane."),
        (44, "The telescoping boom crane extended to reach the skyscraper's top floors."),
        (45, "A faulty cable caused the construction crane to halt operations."),
        (46, "The crane's jib swung slowly as it repositioned the steel I-beam."),
        (47, "Mobile cranes provided flexible lifting solutions across the job site."),
        (48, "The floating crane barge lifted submerged wreckage from the harbor."),
        (49, "Hydraulic jacks stabilized the crane base on uneven ground."),
        (50, "Whooping cranes are among the rarest migratory birds in North America."),
        (51, "The crane's elegant courtship dance involves synchronized wing-spreading."),
        (52, "Sandhill cranes stopped to feed in the cornfields during their migration."),
        (53, "The crane stood motionless on one leg at the edge of the wetland."),
        (54, "Ornithologists tracked crane populations using satellite telemetry."),
        (55, "The Japanese red-crowned crane is a national symbol of longevity."),
        (56, "Crane nesting sites must be protected from human disturbance."),
        (57, "A pair of crowned cranes waded through the savanna grassland."),
        (58, "Biologists recorded crane calls to study their communication patterns."),
        (59, "The nature reserve set aside wetland habitat specifically for wintering cranes."),
        (60, "The elephant used its trunk to spray water during a bath."),
        (61, "An elephant's trunk contains over 40,000 individual muscle units."),
        (62, "The baby elephant practiced grasping fruit with its developing trunk."),
        (63, "Elephants use their trunk both for breathing and as a versatile hand."),
        (64, "A mother elephant guided her calf using gentle pressure from her trunk."),
        (65, "The mechanic found a spare tire stored in the car's trunk."),
        (66, "She loaded groceries into the trunk after shopping at the market."),
        (67, "The trunk hatch opened automatically when she approached with full hands."),
        (68, "Airport security requested that he open his trunk for inspection."),
        (69, "A GPS tracker was found hidden inside the vehicle's trunk compartment."),
        (70, "Physicists measured the speed of light through a vacuum at 299,792 km/s."),
        (71, "The physics lab measured the speed of light using an interferometer."),
        (72, "Optical fibers transmit data as pulses of infrared light."),
        (73, "The laser emitted a coherent beam of monochromatic green light."),
        (74, "A prism dispersed white light into its constituent color spectrum."),
        (75, "Quantum optics experiments explored the wave-particle duality of photons."),
        (76, "The telescope's mirror collected and focused faint starlight."),
        (77, "Astronomers measured redshift to determine how fast galaxies recede."),
        (78, "The interferometer detected changes in light path length to nanometer precision."),
        (79, "Solar panels convert incident light energy directly into electric current."),
        (80, "She wore a light cotton dress on the warm summer afternoon."),
        (81, "The designer preferred lightweight linen for summer resort collections."),
        (82, "Breathable mesh fabric kept athletes cool during outdoor competition."),
        (83, "The sheer chiffon blouse was too delicate for the cold evening."),
        (84, "Merino wool provides warmth while remaining remarkably lightweight."),
        (85, "Silk scarves have been a fashion staple across many cultures."),
        (86, "Technical fabrics in outdoor gear balance low weight with durability."),
        (87, "The collection featured pastel-colored dresses for spring."),
        (88, "High-performance running shoes use carbon fiber plates for lightness."),
        (89, "The minimalist wardrobe emphasized versatile neutral-colored pieces."),
        (90, "The surgeon operated on the nerve trunk in the patient's lower back."),
        (91, "Damage to the brachial plexus trunk caused weakness in the arm."),
        (92, "The anatomist identified the main arterial trunk supplying the organ."),
        (93, "Nerve trunk conduction velocity was measured during the EMG study."),
        (94, "The lymphatic trunk drained fluid from the thoracic region."),
        (95, "The crane operator wore a hard hat and high-visibility vest."),
        (96, "Scientists detected gravitational waves using laser light pulses."),
        (97, "The logging truck carried a full load of oak timber."),
        (98, "He packed his winter clothes into the car trunk before the road trip."),
        (99, "The paper crane origami requires twenty-five precise folds."),
    ];

    // Build real doc token list from Jina cache (fall back to empty if missing)
    let real_doc_toks: Vec<Vec<[f32; TOKEN_DIM]>> = gt_corpus_texts.iter()
        .map(|(_, text)| {
            jina_cache.get(*text)
                .map(|m| m.rows.clone())
                .unwrap_or_default()
        })
        .collect();

    // Also get token IDs for IDF — use positional 0-based IDs from the Jina cache tokens field
    // (Jina returns "[0]", "[1]", ... as token strings; we convert to indices)
    let real_doc_tok_ids: Vec<Vec<u32>> = gt_corpus_texts.iter()
        .map(|(_, text)| {
            jina_cache.get(*text)
                .map(|m| (0..m.rows.len() as u32).collect())
                .unwrap_or_default()
        })
        .collect();

    // IDF over real docs only (100 docs)
    let doc_freq = build_doc_freq_from_toks(&real_doc_tok_ids);
    let n_real = gt_corpus_texts.len();

    // ── 3. Build real LargeDoc entries ───────────────────────────────────────
    let mut rng_meta = SmallRng::seed_from_u64(0xDEAD_BEEF_1234u64);
    let real_docs: Vec<LargeDoc> = gt_corpus_texts.iter().enumerate()
        .map(|(pos, &(corpus_id, _))| {
            let tokens = real_doc_toks[pos].clone();
            let sent_emb = sent_avg(&tokens);
            let category = doc_category_for_real(corpus_id);
            let year = YEAR_MIN + rng_meta.gen_range(0..=(YEAR_MAX - YEAR_MIN));
            LargeDoc { id: pos, category, year, tokens, sent_emb }
        })
        .collect();

    // ── 4. Query token embeddings ────────────────────────────────────────────
    let query_toks: Vec<Vec<[f32; TOKEN_DIM]>> = GT_QUERIES.iter()
        .map(|q| jina_cache.get(*q).map(|m| m.rows.clone()).unwrap_or_default())
        .collect();
    let query_tok_ids: Vec<Vec<u32>> = GT_QUERIES.iter()
        .map(|q| jina_cache.get(*q)
            .map(|m| (0..m.rows.len() as u32).collect())
            .unwrap_or_default())
        .collect();
    let query_sent_embs: Vec<[f32; TOKEN_DIM]> = query_toks.iter()
        .map(|toks| sent_avg(toks))
        .collect();

    // ── 5. Run at each scale ─────────────────────────────────────────────────
    let engine_names = ["HNSW", "PLAID", "WARP", "TACHIOM"];
    let filter_modes = [FilterMode::None, FilterMode::Category, FilterMode::CategoryAndYear];

    // Initialise results
    let mut results: Vec<LargeBenchResult> = engine_names.iter()
        .flat_map(|&eng| filter_modes.iter().map(move |&flt| LargeBenchResult {
            engine: eng,
            filter: flt,
            by_scale: Vec::new(),
        }))
        .collect();

    for (scale_idx, &n) in SCALES.iter().enumerate() {
        let n_label = if n >= 1_000_000 { format!("{}M", n / 1_000_000) } else { format!("{}K", n / 1_000) };
        println!();
        println!("  {BOLD}[scale N={n_label}]{RESET}");

        let scale_map = run_at_scale(
            n,
            &real_docs,
            &gt_sets_usize,
            &query_toks,
            &query_tok_ids,
            &query_sent_embs,
            &doc_freq,
            n_real,
            0xCAFEBABE ^ (scale_idx as u64 * 0x1234567),
        );

        for res in results.iter_mut() {
            let pts = scale_map
                .get(&(res.engine, res.filter))
                .cloned()
                .unwrap_or_default();
            // Pad by_scale with empty vecs for earlier scales if needed
            while res.by_scale.len() < scale_idx {
                res.by_scale.push(vec![]);
            }
            res.by_scale.push(pts);
        }

        println!("  [scale N={n_label}] done.");
    }

    // ── 6. Print summary ─────────────────────────────────────────────────────
    print_large_bench_summary(&results, SCALES);

    // ── 7. Plots ─────────────────────────────────────────────────────────────
    println!("  {BOLD}Generating plots…{RESET}");
    plot_recall_vs_frac(&results)?;
    plot_recall_vs_n(&results)?;

    println!();
    println!("{DIM}  Token engines: Jina ColBERT v2 (128-dim) · HNSW: sentence-avg of Jina tokens");
    println!("  GT labels: LLM-judged (cache/llm_gt.json) · Synthetic: random unit vecs{RESET}");
    println!();

    Ok(())
}
