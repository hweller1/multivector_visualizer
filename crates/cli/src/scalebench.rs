/// Synthetic scalability benchmark.
///
/// Generates random 128-dim unit-vector token matrices in memory, then measures:
///   - ColBERT full scan: MaxSim over every document (O(N))
///   - Pruned scan at p%: MaxSim over p% of docs (simulating PLAID/WARP candidate selection)
///
/// Real 1M-doc ColBERT requires ~4 GB RAM and minutes to build; instead we measure
/// at 1K / 10K / 100K and extrapolate linearly (both workloads are strictly O(N)).

use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

const DIM: usize = 128;
const TOKENS_PER_DOC: usize = 8;
const N_QUERY_TOKENS: usize = 10;
const N_TIMED: usize = 10; // averaged query count per cell
const N_WARMUP: usize = 3;

// ─── flat index ───────────────────────────────────────────────────────────────

/// Layout: data[doc_id * TOKENS_PER_DOC * DIM .. +TOKENS_PER_DOC*DIM]
struct FlatIndex {
    data: Vec<f32>,
    n_docs: usize,
}

impl FlatIndex {
    fn build(n_docs: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut data = vec![0f32; n_docs * TOKENS_PER_DOC * DIM];
        let stride = TOKENS_PER_DOC * DIM;
        for d in 0..n_docs {
            let base = d * stride;
            for t in 0..TOKENS_PER_DOC {
                let s = base + t * DIM;
                let mut norm_sq = 0f32;
                for i in 0..DIM {
                    let v: f32 = rng.gen::<f32>() * 2.0 - 1.0;
                    data[s + i] = v;
                    norm_sq += v * v;
                }
                let scale = 1.0 / norm_sq.sqrt().max(1e-8);
                for i in 0..DIM {
                    data[s + i] *= scale;
                }
            }
        }
        Self { data, n_docs }
    }

    #[inline]
    fn maxsim_one(&self, doc_id: usize, query: &[f32]) -> f32 {
        let base = doc_id * TOKENS_PER_DOC * DIM;
        let mut total = 0f32;
        for qt in 0..N_QUERY_TOKENS {
            let q = &query[qt * DIM..(qt + 1) * DIM];
            let mut best = f32::NEG_INFINITY;
            for dt in 0..TOKENS_PER_DOC {
                let s = base + dt * DIM;
                let dot: f32 = (0..DIM).map(|i| unsafe {
                    *self.data.get_unchecked(s + i) * q.get_unchecked(i)
                }).sum();
                if dot > best { best = dot; }
            }
            total += best;
        }
        total
    }

    fn full_scan(&self, query: &[f32]) -> f32 {
        (0..self.n_docs).map(|i| self.maxsim_one(i, query)).sum()
    }

    fn pruned_scan(&self, query: &[f32], n_candidates: usize) -> f32 {
        (0..n_candidates).map(|i| self.maxsim_one(i, query)).sum()
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn gen_query(seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut q = vec![0f32; N_QUERY_TOKENS * DIM];
    for t in 0..N_QUERY_TOKENS {
        let s = t * DIM;
        let mut norm_sq = 0f32;
        for i in 0..DIM {
            let v: f32 = rng.gen::<f32>() * 2.0 - 1.0;
            q[s + i] = v;
            norm_sq += v * v;
        }
        let scale = 1.0 / norm_sq.sqrt().max(1e-8);
        for i in 0..DIM { q[s + i] *= scale; }
    }
    q
}

fn median_ms<F: Fn() -> f32>(f: &F, n_warmup: usize, n_timed: usize) -> f64 {
    for _ in 0..n_warmup { black_box(f()); }
    let mut times: Vec<f64> = (0..n_timed).map(|_| {
        let t = Instant::now();
        black_box(f());
        t.elapsed().as_secs_f64() * 1000.0
    }).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[n_timed / 2]
}

fn fmt_ms(ms: f64) -> String {
    if ms < 0.1 { format!("{:.0}µs", ms * 1000.0) }
    else if ms < 10.0 { format!("{:.2}ms", ms) }
    else if ms < 1000.0 { format!("{:.0}ms", ms) }
    else { format!("{:.1}s", ms / 1000.0) }
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.0}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.0}K", n as f64 / 1_000.0) }
    else { n.to_string() }
}

// ─── main entry point ─────────────────────────────────────────────────────────

pub fn run_scalebench() {
    const CYAN: &str = "\x1b[36m";
    const BOLD: &str = "\x1b[1m";
    const DIM_C: &str = "\x1b[2m";
    const RESET: &str = "\x1b[0m";
    const GREEN: &str = "\x1b[32m";
    const YELLOW: &str = "\x1b[33m";

    let sizes: &[usize] = &[1_000, 10_000, 100_000];
    let prune_pcts: &[usize] = &[20, 5, 1];

    println!();
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!("{BOLD}  Synthetic Scalability Benchmark{RESET}");
    println!("{DIM_C}  {TOKENS_PER_DOC} tokens/doc  ×  {DIM}-dim per token  ×  {N_QUERY_TOKENS} query tokens");
    println!("  all vectors are random 128-dim unit vectors (no real text)");
    println!("  {N_TIMED} queries median per cell — ColBERT vs pruned MaxSim{RESET}");
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!();

    // Header
    println!("  {:<10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "N docs",
        "ColBERT",
        "20% prune",
        "5% prune",
        "1% prune",
        "speedup(5%)",
    );
    println!("  {}{}{}{}{}{}{}", "─".repeat(10), "  ", "─".repeat(10), "  ",
        "─".repeat(10), "  ", "─".repeat(42));

    let mut last: Option<[f64; 4]> = None; // [colbert, 20%, 5%, 1%]

    for &n in sizes {
        let mb = (n * TOKENS_PER_DOC * DIM * 4) as f64 / 1_048_576.0;
        print!("  {:<10}  {DIM_C}building index ({mb:.0} MB)…{RESET}", fmt_n(n));
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let index = FlatIndex::build(n, 0xDEAD_BEEF_u64);
        let query = gen_query(0x1234_5678_u64);

        let colbert_ms = median_ms(&|| index.full_scan(&query), N_WARMUP, N_TIMED);
        let pruned: Vec<f64> = prune_pcts.iter().map(|&p| {
            let n_cands = (n * p / 100).max(1);
            median_ms(&|| index.pruned_scan(&query, n_cands), N_WARMUP, N_TIMED)
        }).collect();

        let speedup_5 = colbert_ms / pruned[1];

        last = Some([colbert_ms, pruned[0], pruned[1], pruned[2]]);

        print!("\r");
        println!("  {:<10}  {:>10}  {DIM_C}{:>10}{RESET}  {GREEN}{:>10}{RESET}  {GREEN}{:>10}{RESET}  {:>9.0}x",
            fmt_n(n),
            fmt_ms(colbert_ms),
            fmt_ms(pruned[0]),
            fmt_ms(pruned[1]),
            fmt_ms(pruned[2]),
            speedup_5,
        );
    }

    // Extrapolate 1M (strictly linear since both are O(N))
    if let Some(base) = last {
        let factor = 1_000_000f64 / 100_000f64;
        let ext: Vec<f64> = base.iter().map(|&v| v * factor).collect();
        let speedup_5 = ext[0] / ext[2];
        println!();
        println!("  {:<10}  {:>10}  {DIM_C}{:>10}{RESET}  {YELLOW}{:>10}{RESET}  {YELLOW}{:>10}{RESET}  {:>9.0}x  {DIM_C}← extrapolated (O(N)){RESET}",
            "1M",
            fmt_ms(ext[0]),
            fmt_ms(ext[1]),
            fmt_ms(ext[2]),
            fmt_ms(ext[3]),
            speedup_5,
        );
    }

    println!();
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!();
    println!("  {BOLD}ColBERT (full scan){RESET}  — scores ALL N docs via MaxSim. O(N × D × Q).");
    println!("  {BOLD}PLAID / WARP (pruned){RESET} — centroid ANN or Xtr threshold prunes to p% of docs.");
    println!("    Prune overhead is O(K_centroids) — constant w.r.t. N — so total is O(N × p).");
    println!("    Same final MaxSim quality on surviving candidates.");
    println!();

    if let Some(base) = last {
        let ext_5pct = base[2] * 10.0; // extrapolated 1M at 5%
        let ext_full = base[0] * 10.0;
        println!("  At 1M docs:  ColBERT ≈ {BOLD}{}{RESET}/query — too slow for production.",
            fmt_ms(ext_full));
        println!("               PLAID (5% prune) ≈ {GREEN}{BOLD}{}{RESET}/query — production-viable.",
            fmt_ms(ext_5pct));
        println!();
        println!("  {DIM_C}Note: HNSW (single-vector ANN) would be <1ms at any N, but loses");
        println!("  per-token disambiguation — see `cargo run -- demo hnsw` vs `demo colbert`.{RESET}");
        println!("  {DIM_C}Memory: 1M docs × {TOKENS_PER_DOC} tokens × {DIM} dims × 4B = {:.1} GB — not run directly.{RESET}",
            1_000_000f64 * TOKENS_PER_DOC as f64 * DIM as f64 * 4.0 / 1e9);
    }
    println!();
}
