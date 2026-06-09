//! Accuracy-speed tradeoff benchmark for multivector pruning strategies.
//!
//! Generates two SVG plots:
//!   plots/tradeoff_speedrecall.svg  — Recall@10 vs speedup (1K / 10K / 100K docs)
//!   plots/recall_vs_frac.svg        — Recall@10 vs candidate fraction (10K / 100K)
//!
//! Uses a *structured* synthetic corpus: 5 topic directions, docs are biased toward
//! their topic with added noise.  This creates a realistic recall gap between:
//!   • Random sampling (lower bound)
//!   • PLAID: global k-means centroids + top-C probe per query token
//!   • WARP:  exact Xtr threshold on token-level dot products
//!   • TACHIOM: per-type centroid budgets (tail tokens = more budget = better recall
//!              for rare, discriminative queries)

use hnsw_rs::prelude::*;
use plotters::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

// ─── constants ────────────────────────────────────────────────────────────────

const DIM: usize = 128;
const TOKENS_PER_DOC: usize = 8;
const QUERY_TOKENS: usize = 10;
const N_TOPICS: usize = 5;
const K_EVAL: usize = 10;
/// Head topics (0, 1) = 30% each of corpus.  Tail topics (2, 3, 4) = 13% each.
const TOPIC_BREAKS: [f32; N_TOPICS] = [0.30, 0.60, 0.73, 0.87, 1.00];
/// Approximate topic frequencies matching TOPIC_BREAKS
const TOPIC_FREQ: [f64; N_TOPICS] = [0.30, 0.30, 0.13, 0.14, 0.13];
/// Noise levels — must satisfy noise << 1/sqrt(DIM) to keep topic signal after normalization.
/// With DIM=128, 1/sqrt(DIM) ≈ 0.088.  Using 0.20/0.06 gives same-topic dot ≈ 0.57.
const DOC_NOISE: f32 = 0.20;
const QUERY_NOISE: f32 = 0.06;

// ─── vector math ─────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32; DIM]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter_mut().for_each(|x| *x /= n);
}

fn rand_unit(rng: &mut SmallRng) -> [f32; DIM] {
    let mut v = [0f32; DIM];
    for x in v.iter_mut() {
        *x = rng.gen::<f32>() * 2.0 - 1.0;
    }
    normalize(&mut v);
    v
}

fn biased(base: &[f32; DIM], noise: f32, rng: &mut SmallRng) -> [f32; DIM] {
    let mut v = [0f32; DIM];
    for (i, x) in v.iter_mut().enumerate() {
        *x = base[i] + noise * (rng.gen::<f32>() * 2.0 - 1.0);
    }
    normalize(&mut v);
    v
}

fn maxsim(q: &[[f32; DIM]], d: &[[f32; DIM]]) -> f32 {
    q.iter()
        .map(|qi| d.iter().map(|di| dot(qi, di)).fold(f32::NEG_INFINITY, f32::max))
        .sum()
}

// ─── corpus ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Doc {
    id: usize,
    topic: usize,
    tokens: Vec<[f32; DIM]>,
}

struct Query {
    #[allow(dead_code)]
    topic: usize,
    tokens: Vec<[f32; DIM]>,
}

fn gen_corpus(n: usize, topics: &[[f32; DIM]], rng: &mut SmallRng) -> Vec<Doc> {
    (0..n)
        .map(|id| {
            let r = rng.gen::<f32>();
            let topic = TOPIC_BREAKS
                .iter()
                .position(|&b| r < b)
                .unwrap_or(N_TOPICS - 1);
            let tokens = (0..TOKENS_PER_DOC)
                .map(|_| biased(&topics[topic], DOC_NOISE, rng))
                .collect();
            Doc { id, topic, tokens }
        })
        .collect()
}

fn gen_queries(n_q: usize, topics: &[[f32; DIM]], rng: &mut SmallRng) -> Vec<Query> {
    (0..n_q)
        .map(|i| {
            let topic = i % N_TOPICS;
            let tokens = (0..QUERY_TOKENS)
                .map(|_| biased(&topics[topic], QUERY_NOISE, rng))
                .collect();
            Query { topic, tokens }
        })
        .collect()
}

// ─── oracle ───────────────────────────────────────────────────────────────────

fn oracle_topk(q: &Query, docs: &[Doc]) -> Vec<usize> {
    let mut s: Vec<(usize, f32)> = docs
        .iter()
        .map(|d| (d.id, maxsim(&q.tokens, &d.tokens)))
        .collect();
    s.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    s.iter().take(K_EVAL).map(|(id, _)| *id).collect()
}

// ─── k-means (Lloyd's, random init) ──────────────────────────────────────────

fn kmeans(data: &[[f32; DIM]], k: usize, rng: &mut SmallRng) -> Vec<[f32; DIM]> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(data.len());
    let mut idxs: Vec<usize> = (0..data.len()).collect();
    for i in 0..k {
        let remaining = data.len() - i;
        if remaining > 1 {
            let j = i + rng.gen_range(0..remaining);
            idxs.swap(i, j);
        }
    }
    let mut centers: Vec<[f32; DIM]> = idxs[..k].iter().map(|&i| data[i]).collect();

    for _ in 0..15 {
        let assign: Vec<usize> = data
            .iter()
            .map(|d| {
                centers
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, dot(d, c)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(ci, _)| ci)
                    .unwrap_or(0)
            })
            .collect();
        let mut new_c = vec![[0f32; DIM]; k];
        let mut cnt = vec![0usize; k];
        for (d, &ci) in data.iter().zip(&assign) {
            for (j, &x) in d.iter().enumerate() {
                new_c[ci][j] += x;
            }
            cnt[ci] += 1;
        }
        for (c, &n) in new_c.iter_mut().zip(&cnt) {
            if n > 0 {
                let inv = 1.0 / n as f32;
                c.iter_mut().for_each(|x| *x *= inv);
                normalize(c);
            }
        }
        centers = new_c;
    }
    centers
}

// ─── centroid index ───────────────────────────────────────────────────────────

struct CentroidIdx {
    centers: Vec<[f32; DIM]>,
    inv: Vec<Vec<usize>>, // centroid_id → [doc_id]
}

impl CentroidIdx {
    fn build(docs: &[Doc], topic_filter: Option<&dyn Fn(usize) -> bool>, k: usize, rng: &mut SmallRng) -> Self {
        let filtered: Vec<&Doc> = docs
            .iter()
            .filter(|d| topic_filter.map_or(true, |f| f(d.topic)))
            .collect();
        if filtered.is_empty() {
            return CentroidIdx { centers: vec![], inv: vec![] };
        }
        // Collect all (token, doc_id) pairs; subsample for clustering
        let all: Vec<([f32; DIM], usize)> = filtered
            .iter()
            .flat_map(|d| d.tokens.iter().map(|t| (*t, d.id)).collect::<Vec<_>>())
            .collect();
        let sample_n = all.len().min(20_000);
        let mut sidxs: Vec<usize> = (0..all.len()).collect();
        for i in 0..sample_n {
            let rem = all.len() - i;
            if rem > 1 {
                let j = i + rng.gen_range(0..rem);
                sidxs.swap(i, j);
            }
        }
        let sampled: Vec<[f32; DIM]> = sidxs[..sample_n].iter().map(|&i| all[i].0).collect();
        let k = k.min(sampled.len());
        let centers = kmeans(&sampled, k, rng);
        if centers.is_empty() {
            return CentroidIdx { centers: vec![], inv: vec![] };
        }
        // Build inverted index over all tokens
        let mut inv: Vec<HashSet<usize>> = vec![HashSet::new(); centers.len()];
        for &(tok, doc_id) in &all {
            let best = centers
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, dot(&tok, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            inv[best].insert(doc_id);
        }
        let inv = inv.into_iter().map(|s| s.into_iter().collect()).collect();
        CentroidIdx { centers, inv }
    }

    fn probe(&self, qtoks: &[[f32; DIM]], n_probe: usize) -> HashSet<usize> {
        let mut cands = HashSet::new();
        if self.centers.is_empty() {
            return cands;
        }
        for qt in qtoks {
            let mut sims: Vec<(usize, f32)> = self
                .centers
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, dot(qt, c)))
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (ci, _) in sims.iter().take(n_probe) {
                self.inv[*ci].iter().for_each(|&d| {
                    cands.insert(d);
                });
            }
        }
        cands
    }

    fn n_centers(&self) -> usize {
        self.centers.len()
    }
}

// ─── engine simulations ───────────────────────────────────────────────────────

fn warp_candidates(qtoks: &[[f32; DIM]], docs: &[Doc], thresh: f32) -> HashSet<usize> {
    docs.iter()
        .filter_map(|d| {
            let max = d
                .tokens
                .iter()
                .flat_map(|dt| qtoks.iter().map(|qt| dot(qt, dt)))
                .fold(f32::NEG_INFINITY, f32::max);
            if max > thresh { Some(d.id) } else { None }
        })
        .collect()
}

fn score_candidates(qtoks: &[[f32; DIM]], docs: &[Doc], cands: &HashSet<usize>) -> Vec<(usize, f32)> {
    let mut s: Vec<(usize, f32)> = docs
        .iter()
        .filter(|d| cands.contains(&d.id))
        .map(|d| (d.id, maxsim(qtoks, &d.tokens)))
        .collect();
    s.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    s
}

fn recall_k(oracle: &[usize], scored: &[(usize, f32)]) -> f64 {
    let ret: HashSet<usize> = scored.iter().take(K_EVAL).map(|(id, _)| *id).collect();
    oracle.iter().filter(|id| ret.contains(id)).count() as f64 / K_EVAL as f64
}

// ─── output types ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Pt {
    frac: f64,   // fraction of corpus in candidate set
    recall: f64, // Recall@K averaged over queries
}

#[derive(Clone)]
struct Curve {
    name: &'static str,
    short: &'static str,
    color: RGBColor,
    stroke: u32,
    pts: Vec<Pt>,
}

// ─── per-N benchmark ──────────────────────────────────────────────────────────

fn bench_at_n(n_docs: usize, rng: &mut SmallRng) -> Vec<Curve> {
    let n_q = if n_docs >= 100_000 { 15 } else if n_docs >= 10_000 { 25 } else { 40 };
    let n_rep = if n_docs >= 100_000 { 1 } else if n_docs >= 10_000 { 2 } else { 4 };

    print!("  N={n_docs:>7}  building corpus…");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let topics: Vec<[f32; DIM]> = (0..N_TOPICS).map(|_| rand_unit(rng)).collect();
    let docs = gen_corpus(n_docs, &topics, rng);
    let queries = gen_queries(n_q, &topics, rng);

    print!(" oracle…");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t_oracle = Instant::now();
    let oracle: Vec<Vec<usize>> = queries.iter().map(|q| oracle_topk(q, &docs)).collect();
    let oracle_ms = t_oracle.elapsed().as_secs_f64() * 1000.0 / n_q as f64;
    println!(" {oracle_ms:.1}ms/query");

    // ── Random baseline ──────────────────────────────────────────────────────
    let mut rand_pts = vec![];
    for &p in &[0.01f64, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00] {
        let rec = queries
            .iter()
            .zip(&oracle)
            .map(|(q, o)| {
                let mut c: HashSet<usize> =
                    (0..n_docs).filter(|_| rng.gen::<f64>() < p).collect();
                if c.is_empty() {
                    c.insert(0);
                }
                recall_k(o, &score_candidates(&q.tokens, &docs, &c))
            })
            .sum::<f64>()
            / n_q as f64;
        rand_pts.push(Pt { frac: p, recall: rec });
    }

    // ── PLAID ─────────────────────────────────────────────────────────────────
    let k_global = (n_docs as f64).sqrt() as usize;
    let k_global = k_global.clamp(10, 400);
    print!("  N={n_docs:>7}  PLAID k={k_global}…");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let plaid = CentroidIdx::build(&docs, None, k_global, rng);
    let k_plaid = plaid.n_centers();
    println!(" {k_plaid} centers");
    let probes = [1, 2, 3, 5, 8, 13, 21, 34, k_plaid];
    let mut plaid_pts = vec![];
    for &np in &probes {
        if np > k_plaid { break; }
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for _ in 0..n_rep {
            for (q, o) in queries.iter().zip(&oracle) {
                let c = plaid.probe(&q.tokens, np);
                tot_f += c.len() as f64 / n_docs as f64;
                let c = if c.is_empty() { std::iter::once(0).collect() } else { c };
                tot_r += recall_k(o, &score_candidates(&q.tokens, &docs, &c));
            }
        }
        let n = (n_rep * n_q) as f64;
        plaid_pts.push(Pt { frac: tot_f / n, recall: tot_r / n });
    }

    // ── WARP ──────────────────────────────────────────────────────────────────
    // With DOC_NOISE=0.20 and DIM=128: same-topic E[dot] ≈ 0.57, max-of-80 ≈ 0.69.
    // Different-topic max-of-80 ≈ 0.12.  Sweep thresholds from 0.70 (near zero candidates)
    // down to 0.15 (captures noise floor) to trace the full recall curve.
    print!("  N={n_docs:>7}  WARP sweep…");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let thresholds = [
        0.72f32, 0.69, 0.66, 0.63, 0.60, 0.57, 0.54, 0.51, 0.48, 0.45,
        0.42, 0.38, 0.34, 0.29, 0.24, 0.18, 0.13,
    ];
    let mut warp_pts = vec![];
    for &t in &thresholds {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (q, o) in queries.iter().zip(&oracle) {
            let c = warp_candidates(&q.tokens, &docs, t);
            tot_f += (c.len() as f64 / n_docs as f64).max(1.0 / n_docs as f64);
            let c = if c.is_empty() { std::iter::once(0).collect() } else { c };
            tot_r += recall_k(o, &score_candidates(&q.tokens, &docs, &c));
        }
        let frac = tot_f / n_q as f64;
        if frac > 0.001 {
            warp_pts.push(Pt { frac, recall: tot_r / n_q as f64 });
        }
    }
    warp_pts.sort_by(|a, b| a.frac.partial_cmp(&b.frac).unwrap());
    println!(" {} points", warp_pts.len());

    // ── TACHIOM ───────────────────────────────────────────────────────────────
    // TACHIOM's key insight: build a SEPARATE centroid index per token type,
    // then allocate budget ∝ 1/frequency (rare token types = more centroids
    // per doc = better recall for discriminative queries).
    //
    // We model this by building one CentroidIdx per topic and routing each
    // query token to its best-matching per-topic index.  Since tail topics
    // (2,3,4) have fewer docs, they get proportionally MORE centroids per doc
    // (budget ∝ 1/freq).  At query time each query token routes to exactly its
    // own topic → zero contamination from other topics, unlike PLAID's global index.
    let total_inv_freq: f64 = TOPIC_FREQ.iter().map(|f| 1.0 / f).sum();
    let per_topic_k: Vec<usize> = TOPIC_FREQ
        .iter()
        .map(|&f| ((k_global as f64 / total_inv_freq / f) as usize).max(3))
        .collect();
    print!("  N={n_docs:>7}  TACHIOM per-topic…");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let topic_idxs: Vec<CentroidIdx> = (0..N_TOPICS)
        .map(|t| {
            let k = per_topic_k[t];
            CentroidIdx::build(&docs, Some(&|dt: usize| dt == t), k, rng)
        })
        .collect();
    let k_per: Vec<usize> = topic_idxs.iter().map(|idx| idx.n_centers()).collect();
    println!(" k={k_per:?}");

    let mut tac_pts = vec![];
    for &np in &[1usize, 2, 3, 5, 8, 13, 21, 34] {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for _ in 0..n_rep {
            for (q, o) in queries.iter().zip(&oracle) {
                // Route each query token to its nearest per-topic index, probe it.
                let mut cands: HashSet<usize> = HashSet::new();
                for qt in &q.tokens {
                    // Find which topic centroid is nearest to this query token
                    let best_t = topic_idxs
                        .iter()
                        .enumerate()
                        .filter(|(_, idx)| !idx.centers.is_empty())
                        .map(|(t, idx)| {
                            let s = idx.centers.iter().map(|c| dot(qt, c)).fold(f32::NEG_INFINITY, f32::max);
                            (t, s)
                        })
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(t, _)| t)
                        .unwrap_or(0);
                    // Probe that topic's index — all results are same-type docs
                    let n = np.min(k_per[best_t].max(1));
                    for d in topic_idxs[best_t].probe(&[*qt], n) {
                        cands.insert(d);
                    }
                }
                tot_f += cands.len() as f64 / n_docs as f64;
                let cands = if cands.is_empty() { std::iter::once(0).collect() } else { cands };
                tot_r += recall_k(o, &score_candidates(&q.tokens, &docs, &cands));
            }
        }
        let n = (n_rep * n_q) as f64;
        tac_pts.push(Pt { frac: tot_f / n, recall: tot_r / n });
    }

    // ── HNSW — sentence avg vector ────────────────────────────────────────────
    // Single-vector ANN: each doc is represented as the L2-normalized mean of
    // its token embeddings.  HNSW is competitive on well-separated topics but
    // has a recall ceiling vs ColBERT oracle because MaxSim uses per-token
    // interactions that a sentence average loses.
    print!("  N={n_docs:>7}  HNSW sentence avg…");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let sent_vecs: Vec<Vec<f32>> = docs.iter().map(|d| {
        let mut avg = vec![0f32; DIM];
        for tok in &d.tokens { for (a, &t) in avg.iter_mut().zip(tok.iter()) { *a += t; } }
        let n = d.tokens.len() as f32;
        avg.iter_mut().for_each(|x| *x /= n);
        let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        avg.iter_mut().for_each(|x| *x /= norm);
        avg
    }).collect();
    let query_svecs: Vec<Vec<f32>> = queries.iter().map(|q| {
        let mut avg = vec![0f32; DIM];
        for tok in &q.tokens { for (a, &t) in avg.iter_mut().zip(tok.iter()) { *a += t; } }
        let n = q.tokens.len() as f32;
        avg.iter_mut().for_each(|x| *x /= n);
        let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        avg.iter_mut().for_each(|x| *x /= norm);
        avg
    }).collect();

    let m_conn = 8usize.min(n_docs / 4 + 2);
    let hnsw_idx = Hnsw::<f32, DistCosine>::new(m_conn, n_docs + 1, 8, 64, DistCosine {});
    for (i, sv) in sent_vecs.iter().enumerate() {
        hnsw_idx.insert((sv.as_slice(), i));
    }

    // Sweep to ~50% of corpus to prove the plateau is flat (not a numCandidates issue).
    let ef_vals: Vec<usize> = [4usize, 8, 12, 16, 24, 32, 64, 128, 256, 512,
                                1000, 2000, 5000]
        .iter().copied().filter(|&ef| ef <= n_docs && ef >= K_EVAL).collect();
    let mut hnsw_pts = vec![];
    for ef in ef_vals {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for _ in 0..n_rep {
            for ((q, o), qsv) in queries.iter().zip(&oracle).zip(&query_svecs) {
                let results = hnsw_idx.search(qsv.as_slice(), ef, ef);
                let cands: HashSet<usize> = results.iter().map(|r| r.d_id).collect();
                tot_f += cands.len() as f64 / n_docs as f64;
                let cands = if cands.is_empty() { std::iter::once(0).collect() } else { cands };
                tot_r += recall_k(o, &score_candidates(&q.tokens, &docs, &cands));
            }
        }
        let n = (n_rep * n_q) as f64;
        hnsw_pts.push(Pt { frac: tot_f / n, recall: tot_r / n });
    }
    hnsw_pts.sort_by(|a, b| a.frac.partial_cmp(&b.frac).unwrap());
    println!(" {} ef-points", hnsw_pts.len());

    vec![
        Curve { name: "Random (lower bound)", short: "Random", color: RGBColor(140, 140, 140), stroke: 1, pts: rand_pts },
        Curve { name: "HNSW — sentence avg vector", short: "HNSW", color: RGBColor(255, 127, 14), stroke: 2, pts: hnsw_pts },
        Curve { name: "PLAID — global k-means centroids", short: "PLAID", color: RGBColor(33, 102, 172), stroke: 2, pts: plaid_pts },
        Curve { name: "WARP — Xtr token similarity threshold", short: "WARP", color: RGBColor(215, 48, 39), stroke: 2, pts: warp_pts },
        Curve { name: "TACHIOM — per-type centroid budgets", short: "TACHIOM", color: RGBColor(26, 152, 80), stroke: 3, pts: tac_pts },
    ]
}

// ─── plotting helpers ─────────────────────────────────────────────────────────

/// Map frac → log2(1/frac) for the speedup axis.
fn to_speedup_log(frac: f64) -> f64 {
    (1.0 / frac.clamp(0.001, 1.0)).log2()
}

/// Map frac → log2(frac * 100) for the "% of corpus" axis.
fn to_pct_log(frac: f64) -> f64 {
    (frac.clamp(0.001, 1.0) * 100.0).log2()
}

fn draw_panel_speedrecall(
    panel: &DrawingArea<SVGBackend, plotters::coord::Shift>,
    curves: &[Curve],
    caption: &str,
    show_y_label: bool,
) -> anyhow::Result<()> {
    let x_max = 7.0f64; // log2(128x) ≈ 7
    let mut chart = ChartBuilder::on(panel)
        .caption(caption, ("sans-serif", 15).into_font())
        .margin(10)
        .x_label_area_size(44)
        .y_label_area_size(if show_y_label { 54 } else { 10 })
        .build_cartesian_2d(0f64..x_max, 0f64..1.06f64)?;

    chart
        .configure_mesh()
        .x_desc("Speedup over ColBERT full scan")
        .y_desc(if show_y_label { "Recall@10" } else { "" })
        .x_labels(7)
        .y_labels(6)
        .x_label_formatter(&|&x| {
            let v = 2f64.powf(x);
            if x < 0.05 {
                "1×".into()
            } else if v >= 10.0 {
                format!("{:.0}×", v)
            } else {
                format!("{:.1}×", v)
            }
        })
        .light_line_style(RGBColor(220, 220, 220).stroke_width(1))
        .draw()?;

    for curve in curves {
        let pts: Vec<(f64, f64)> = curve
            .pts
            .iter()
            .map(|p| (to_speedup_log(p.frac), p.recall))
            .collect();
        if pts.is_empty() {
            continue;
        }
        let color = curve.color;
        let sw = curve.stroke;
        chart
            .draw_series(LineSeries::new(
                pts.clone(),
                ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
            ))?
            .label(curve.name)
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 22, y)],
                    ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
                )
            });
        chart.draw_series(pts.iter().map(|&(x, y)| {
            Circle::new((x, y), 4, ShapeStyle { color: color.to_rgba(), filled: true, stroke_width: 1 })
        }))?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.88))
        .border_style(RGBColor(160, 160, 160).stroke_width(1))
        .label_font(("sans-serif", 11).into_font())
        .position(SeriesLabelPosition::LowerRight)
        .draw()?;

    Ok(())
}

fn draw_panel_recall_frac(
    panel: &DrawingArea<SVGBackend, plotters::coord::Shift>,
    curves: &[Curve],
    caption: &str,
    show_y_label: bool,
) -> anyhow::Result<()> {
    let x_min = to_pct_log(0.005); // ~0.5%
    let x_max = to_pct_log(1.00);  // 100%

    let mut chart = ChartBuilder::on(panel)
        .caption(caption, ("sans-serif", 15).into_font())
        .margin(10)
        .x_label_area_size(44)
        .y_label_area_size(if show_y_label { 54 } else { 10 })
        .build_cartesian_2d(x_min..x_max, 0f64..1.06f64)?;

    chart
        .configure_mesh()
        .x_desc("Candidate set (% of corpus, log scale)")
        .y_desc(if show_y_label { "Recall@10" } else { "" })
        .x_labels(7)
        .y_labels(6)
        .x_label_formatter(&|&x| {
            let v = 2f64.powf(x);
            if v < 2.0 {
                format!("{:.1}%", v)
            } else {
                format!("{:.0}%", v)
            }
        })
        .light_line_style(RGBColor(220, 220, 220).stroke_width(1))
        .draw()?;

    for curve in curves {
        let pts: Vec<(f64, f64)> = curve
            .pts
            .iter()
            .filter(|p| p.frac >= 0.003)
            .map(|p| (to_pct_log(p.frac), p.recall))
            .collect();
        if pts.is_empty() {
            continue;
        }
        let color = curve.color;
        let sw = curve.stroke;
        chart
            .draw_series(LineSeries::new(
                pts.clone(),
                ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
            ))?
            .label(curve.short)
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 22, y)],
                    ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
                )
            });
        chart.draw_series(pts.iter().map(|&(x, y)| {
            Circle::new((x, y), 4, ShapeStyle { color: color.to_rgba(), filled: true, stroke_width: 1 })
        }))?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.88))
        .border_style(RGBColor(160, 160, 160).stroke_width(1))
        .label_font(("sans-serif", 11).into_font())
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    Ok(())
}

// ─── plot: speed-recall (3 panels: 1K / 10K / 100K) ──────────────────────────

fn plot_speedrecall(results: &[(&str, Vec<Curve>)], path: &str) -> anyhow::Result<()> {
    let n = results.len() as u32;
    let root = SVGBackend::new(path, (440 * n + 10, 540)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, results.len()));
    for ((label, curves), panel) in results.iter().zip(&panels) {
        let first = label == &results[0].0;
        draw_panel_speedrecall(panel, curves, &format!("N = {label}"), first)?;
    }
    root.present()?;
    Ok(())
}

// ─── plot: recall vs candidate fraction (2 panels: 10K / 100K) ───────────────

fn plot_recall_vs_frac(results: &[(&str, Vec<Curve>)], path: &str) -> anyhow::Result<()> {
    let n = results.len() as u32;
    let root = SVGBackend::new(path, (480 * n + 10, 540)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, results.len()));
    for ((label, curves), panel) in results.iter().zip(&panels) {
        let first = label == &results[0].0;
        draw_panel_recall_frac(panel, curves, &format!("N = {label}"), first)?;
    }
    root.present()?;
    Ok(())
}

// ─── terminal summary ─────────────────────────────────────────────────────────

fn nearest_recall(curve: &Curve, target_frac: f64) -> String {
    let p = curve.pts.iter().min_by_key(|p| {
        let diff = (p.frac - target_frac).abs();
        (diff * 100_000.0) as i64
    });
    match p {
        Some(p) if (p.frac - target_frac).abs() < target_frac * 2.5 => {
            format!("{:.3}", p.recall)
        }
        _ => "  — ".into(),
    }
}

fn print_summary(results: &[(&str, Vec<Curve>)]) {
    const CYAN: &str = "\x1b[36m";
    const BOLD: &str = "\x1b[1m";
    const RESET: &str = "\x1b[0m";
    const GREEN: &str = "\x1b[32m";
    const DIM_C: &str = "\x1b[2m";

    println!();
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!("{BOLD}  Recall@10 at fixed candidate fractions  (higher = better){RESET}");
    println!("{DIM_C}  Each cell: how much of oracle top-10 is recovered when scoring that % of docs.{RESET}");
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    let fracs: &[(f64, &str)] = &[
        (0.01, "1%"),
        (0.05, "5%"),
        (0.10, "10%"),
        (0.20, "20%"),
        (0.50, "50%"),
    ];
    for (n_label, curves) in results {
        println!();
        println!("  {BOLD}N = {n_label}{RESET}");
        let hdr: Vec<String> = fracs.iter().map(|(_, l)| format!("{:>7}", l)).collect();
        println!("  {:<32}  {}", "Engine", hdr.join("  "));
        println!("  {}  {}", "─".repeat(32), "─".repeat(7 * fracs.len() + 2 * (fracs.len() - 1)));
        for curve in curves {
            let cells: Vec<String> = fracs
                .iter()
                .map(|(f, _)| format!("{:>7}", nearest_recall(curve, *f)))
                .collect();
            let name = if curve.short == "TACHIOM" {
                format!("{GREEN}{BOLD}{:<32}{RESET}", curve.name)
            } else {
                format!("{:<32}", curve.name)
            };
            println!("  {name}  {}", cells.join("  "));
        }
    }
    println!();
    println!("{DIM_C}  WARP advantage: exact token-level similarities, no centroid approximation.");
    println!("  TACHIOM advantage: rare (tail) token types get proportionally more centroid budget");
    println!("    → better recall for specific/rare queries at the same candidate fraction.{RESET}");
    println!();
}

// ─── entry point ─────────────────────────────────────────────────────────────

pub fn run_tradeoff() -> anyhow::Result<()> {
    const CYAN: &str = "\x1b[36m";
    const BOLD: &str = "\x1b[1m";
    const DIM_C: &str = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    println!();
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!("{BOLD}  Accuracy-Speed Tradeoff Benchmark{RESET}");
    println!("{DIM_C}  Structured synthetic corpus: 5 topics (2 head, 3 tail), {DIM}-dim tokens");
    println!("  Engines: Random / HNSW / PLAID / WARP / TACHIOM  —  Recall@{K_EVAL}, 3 scales{RESET}");
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!();

    let mut rng = SmallRng::seed_from_u64(0xCAFE_BABE_u64);
    let sizes: &[(usize, &str)] = &[(1_000, "1K"), (10_000, "10K"), (100_000, "100K")];
    let results: Vec<(&str, Vec<Curve>)> = sizes
        .iter()
        .map(|&(n, lbl)| (lbl, bench_at_n(n, &mut rng)))
        .collect();

    print_summary(&results);

    std::fs::create_dir_all("plots")?;

    let p1 = "plots/tradeoff_speedrecall.svg";
    plot_speedrecall(&results, p1)?;
    println!("  {BOLD}→ {p1}{RESET}  (Recall@10 vs Speedup, 3 panels)");

    let sub: Vec<(&str, Vec<Curve>)> = results
        .iter()
        .filter(|(l, _)| *l == "10K" || *l == "100K")
        .map(|(l, c)| (*l, c.clone()))
        .collect();
    let p2 = "plots/recall_vs_frac.svg";
    plot_recall_vs_frac(&sub, p2)?;
    println!("  {BOLD}→ {p2}{RESET}  (Recall@10 vs candidate fraction %, 2 panels)");

    println!();
    println!("{DIM_C}  Open in any browser.  Import into LaTeX: \\includegraphics[width=\\linewidth]{{{p1}}}{RESET}");
    println!();

    Ok(())
}
