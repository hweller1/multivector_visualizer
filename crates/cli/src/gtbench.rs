//! Ground-truth benchmark using LLM-as-judge relevance labels.
//!
//! Corpus:  100 short documents spanning 8 ambiguous-word categories
//!          (river bank, financial bank, construction crane, bird crane,
//!           elephant trunk, car trunk, physics light, fashion lightweight).
//! Engines: HNSW (Voyage sentence embeddings), ColBERT, PLAID, WARP, TACHIOM
//!          (all multivector engines use WordPiece + RandomProjection tokens).
//! GT:      Claude Haiku judges which docs are relevant to each query.
//!          Labels cached in cache/llm_gt.json.
//! Plots:   plots/gt_recall.svg

use anyhow::{anyhow, Result};
use common::{RandomProjection, WordPieceTokenizer, TOKEN_DIM};
use hnsw_rs::prelude::*;
#[allow(unused_imports)]
use std::f32;
use plotters::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ─── corpus ───────────────────────────────────────────────────────────────────

const GT_CORPUS: &[(u32, &str)] = &[
    // ── river / geography (0–19) ─────────────────────────────────────────────
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
    // ── financial / banking (20–39) ───────────────────────────────────────────
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
    // ── construction crane (40–49) ────────────────────────────────────────────
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
    // ── bird crane (50–59) ────────────────────────────────────────────────────
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
    // ── elephant trunk (60–64) ────────────────────────────────────────────────
    (60, "The elephant used its trunk to spray water during a bath."),
    (61, "An elephant's trunk contains over 40,000 individual muscle units."),
    (62, "The baby elephant practiced grasping fruit with its developing trunk."),
    (63, "Elephants use their trunk both for breathing and as a versatile hand."),
    (64, "A mother elephant guided her calf using gentle pressure from her trunk."),
    // ── car trunk (65–69) ─────────────────────────────────────────────────────
    (65, "The mechanic found a spare tire stored in the car's trunk."),
    (66, "She loaded groceries into the trunk after shopping at the market."),
    (67, "The trunk hatch opened automatically when she approached with full hands."),
    (68, "Airport security requested that he open his trunk for inspection."),
    (69, "A GPS tracker was found hidden inside the vehicle's trunk compartment."),
    // ── physics / optics / light (70–79) ──────────────────────────────────────
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
    // ── fashion / lightweight fabric (80–89) ──────────────────────────────────
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
    // ── anatomy / nerve trunk (90–94) ─────────────────────────────────────────
    (90, "The surgeon operated on the nerve trunk in the patient's lower back."),
    (91, "Damage to the brachial plexus trunk caused weakness in the arm."),
    (92, "The anatomist identified the main arterial trunk supplying the organ."),
    (93, "Nerve trunk conduction velocity was measured during the EMG study."),
    (94, "The lymphatic trunk drained fluid from the thoracic region."),
    // ── mixed / distractors (95–99) ───────────────────────────────────────────
    (95, "The crane operator wore a hard hat and high-visibility vest."),
    (96, "Scientists detected gravitational waves using laser light pulses."),
    (97, "The logging truck carried a full load of oak timber."),
    (98, "He packed his winter clothes into the car trunk before the road trip."),
    (99, "The paper crane origami requires twenty-five precise folds."),
];

const GT_QUERIES: &[&str] = &[
    "river erosion along the bank",                       // 0 → river/geo
    "open a checking account at the bank",                // 1 → finance
    "central bank interest rate monetary policy",         // 2 → finance
    "hiking trail beside the river valley",               // 3 → river/geo
    "construction crane lifting heavy steel beam",        // 4 → construction
    "migratory crane birds wetland habitat",              // 5 → bird crane
    "elephant trunk muscular grasping",                   // 6 → elephant
    "car trunk vehicle storage compartment",              // 7 → car trunk
    "speed of light optics physics experiment",           // 8 → physics
    "summer lightweight fashion dress fabric",            // 9 → fashion
];

const K_EVAL: usize = 10;

// ─── LLM judge ───────────────────────────────────────────────────────────────

struct AnthropicClient {
    api_key: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    fn from_env() -> Option<Self> {
        let _ = dotenvy::dotenv();
        let key = std::env::var("ANTHROPIC_API_KEY").ok()?;
        let key = key.trim().to_string();
        if key.is_empty() { return None; }
        Some(Self { api_key: key, http: reqwest::Client::new() })
    }

    async fn judge_query(&self, query: &str, docs: &[(u32, &str)]) -> Result<Vec<u32>> {
        let doc_list: String = docs.iter()
            .map(|(id, text)| format!("Doc {id}: \"{text}\""))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are a relevance assessor for an IR benchmark.\n\n\
            Query: \"{query}\"\n\n\
            Below are {n} documents. Output ONLY a JSON array of the document IDs \
            that are relevant to the query (i.e. they address the same information need).\n\n\
            {doc_list}\n\n\
            Output ONLY the JSON array, no explanation. Example: [0, 5, 14]",
            n = docs.len()
        );

        let body = serde_json::json!({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        });

        let resp = self.http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?
            .json::<Value>()
            .await?;

        let text = resp["content"][0]["text"].as_str()
            .ok_or_else(|| anyhow!("unexpected Anthropic response: {resp}"))?;

        // Strip markdown fences if present
        let cleaned = text.trim().trim_start_matches("```json").trim_start_matches("```")
            .trim_end_matches("```").trim();

        let ids: Vec<u32> = serde_json::from_str(cleaned)
            .map_err(|e| anyhow!("LLM response not valid JSON array: {e}\nGot: {cleaned}"))?;

        Ok(ids)
    }
}

// ─── Voyage embeddings (sentence-level, for HNSW) ────────────────────────────

struct VoyageEmbed {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl VoyageEmbed {
    fn from_env() -> Option<Self> {
        let _ = dotenvy::dotenv();
        let key = std::env::var("VOYAGE_API_KEY").ok()?.trim().to_string();
        if key.is_empty() { return None; }
        let model = std::env::var("VOYAGE_MODEL").unwrap_or_else(|_| "voyage-4-large".into());
        Some(Self { api_key: key, model, http: reqwest::Client::new() })
    }

    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let resp = self.http
            .post("https://ai.mongodb.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .json::<Value>()
            .await?;

        let data = resp["data"].as_array()
            .ok_or_else(|| anyhow!("Voyage API unexpected response: {resp}"))?;

        data.iter().map(|item| {
            item["embedding"].as_array()
                .ok_or_else(|| anyhow!("missing embedding field"))
                .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
        }).collect()
    }
}

// ─── cache helpers ────────────────────────────────────────────────────────────

fn load_gt_cache(path: &str) -> Option<HashMap<String, Vec<u32>>> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn save_gt_cache(path: &str, labels: &HashMap<String, Vec<u32>>) -> Result<()> {
    std::fs::create_dir_all("cache")?;
    std::fs::write(path, serde_json::to_string_pretty(labels)?)?;
    Ok(())
}

fn load_emb_cache(path: &str) -> Option<HashMap<u32, Vec<f32>>> {
    let raw = std::fs::read_to_string(path).ok()?;
    let map: HashMap<String, Vec<f32>> = serde_json::from_str(&raw).ok()?;
    Some(map.into_iter().filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v))).collect())
}

fn save_emb_cache(path: &str, embs: &HashMap<u32, Vec<f32>>) -> Result<()> {
    std::fs::create_dir_all("cache")?;
    let str_keyed: HashMap<String, &Vec<f32>> = embs.iter().map(|(k, v)| (k.to_string(), v)).collect();
    std::fs::write(path, serde_json::to_string(&str_keyed)?)?;
    Ok(())
}

// ─── token embeddings ─────────────────────────────────────────────────────────

fn embed_text_tokens(
    text: &str,
    wt: &WordPieceTokenizer,
    proj: &mut RandomProjection,
) -> Vec<[f32; TOKEN_DIM]> {
    match wt.inner.encode(text, false) {
        Ok(enc) => {
            let ids = enc.get_ids();
            let tokens: Vec<String> = enc.get_tokens().to_vec();
            proj.embed(ids, &tokens).rows
        }
        Err(_) => vec![],
    }
}

// ─── vector math ──────────────────────────────────────────────────────────────

fn dot128(a: &[f32; TOKEN_DIM], b: &[f32; TOKEN_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn maxsim(q: &[[f32; TOKEN_DIM]], d: &[[f32; TOKEN_DIM]]) -> f32 {
    q.iter()
        .map(|qi| d.iter().map(|di| dot128(qi, di)).fold(f32::NEG_INFINITY, f32::max))
        .sum()
}

fn recall_at_k(results: &[(u32, f32)], gt: &HashSet<u32>, k: usize) -> f64 {
    let n_rel = gt.len().min(k);
    if n_rel == 0 { return 0.0; }
    let found = results.iter().take(k).filter(|(id, _)| gt.contains(id)).count();
    found as f64 / n_rel as f64
}

// ─── k-means + centroid index ─────────────────────────────────────────────────

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

struct CentIdx {
    centers: Vec<[f32; TOKEN_DIM]>,
    inv: Vec<Vec<u32>>,
}

impl CentIdx {
    fn build(doc_toks: &[(u32, Vec<[f32; TOKEN_DIM]>)], k: usize, rng: &mut SmallRng) -> Self {
        let all: Vec<([f32; TOKEN_DIM], u32)> = doc_toks.iter()
            .flat_map(|(id, toks)| toks.iter().map(|t| (*t, *id)).collect::<Vec<_>>())
            .collect();
        if all.is_empty() || k == 0 { return Self { centers: vec![], inv: vec![] }; }
        let toks_only: Vec<[f32; TOKEN_DIM]> = all.iter().map(|(t, _)| *t).collect();
        let centers = kmeans_128(&toks_only, k, rng);
        if centers.is_empty() { return Self { centers: vec![], inv: vec![] }; }
        let mut inv: Vec<HashSet<u32>> = vec![HashSet::new(); centers.len()];
        for &(tok, doc_id) in &all {
            let best = centers.iter().enumerate()
                .map(|(ci, c)| (ci, dot128(&tok, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci).unwrap_or(0);
            inv[best].insert(doc_id);
        }
        Self { centers, inv: inv.into_iter().map(|s| s.into_iter().collect()).collect() }
    }

    fn probe(&self, qtoks: &[[f32; TOKEN_DIM]], n_probe: usize) -> HashSet<u32> {
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

// ─── per-engine benchmarks ────────────────────────────────────────────────────

struct EngineResult {
    name: &'static str,
    short: &'static str,
    color: RGBColor,
    stroke: u32,
    pts: Vec<(f64, f64)>, // (candidate_frac, recall@K)
}

fn bench_hnsw(
    doc_embs: &[(u32, Vec<f32>)],
    query_embs: &[Vec<f32>],
    gt: &[HashSet<u32>],
) -> EngineResult {
    let n = doc_embs.len();
    let m_conn = 16usize.min(n / 4 + 2);
    let hnsw = Hnsw::<f32, DistCosine>::new(m_conn, n + 1, 8, 64, DistCosine {});
    for (idx, (_, emb)) in doc_embs.iter().enumerate() {
        hnsw.insert((emb.as_slice(), idx));
    }
    let idx_to_id: Vec<u32> = doc_embs.iter().map(|(id, _)| *id).collect();

    let ef_vals: Vec<usize> = [5usize, 10, 15, 20, 30, 50, 75, 100]
        .iter().copied().filter(|&ef| ef <= n).collect();

    let mut pts = vec![];
    for ef in ef_vals {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qe, gt_set) in query_embs.iter().zip(gt) {
            let results = hnsw.search(qe.as_slice(), ef, ef);
            // DistCosine returns 1 - cosine_similarity, so higher cosine = lower distance.
            let mut scored: Vec<(u32, f32)> = results.iter()
                .map(|r| (idx_to_id[r.d_id], 1.0_f32 - r.distance))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_f += scored.len() as f64 / n as f64;
            tot_r += recall_at_k(&scored, gt_set, K_EVAL);
        }
        let nq = query_embs.len() as f64;
        pts.push((tot_f / nq, tot_r / nq));
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    EngineResult {
        name: "HNSW — Voyage sentence embeddings",
        short: "HNSW",
        color: RGBColor(255, 127, 14),
        stroke: 2,
        pts,
    }
}

fn bench_colbert(
    doc_toks: &[(u32, Vec<[f32; TOKEN_DIM]>)],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    gt: &[HashSet<u32>],
) -> EngineResult {
    let n = doc_toks.len();
    let mut pts = vec![];
    let mut tot_r = 0.0f64;
    for (qtoks, gt_set) in query_toks.iter().zip(gt) {
        let mut scored: Vec<(u32, f32)> = doc_toks.iter()
            .map(|(id, dtoks)| (*id, maxsim(qtoks, dtoks)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        tot_r += recall_at_k(&scored, gt_set, K_EVAL);
    }
    let rec = tot_r / query_toks.len() as f64;
    pts.push((1.0, rec));
    EngineResult {
        name: "ColBERT — full scan MaxSim",
        short: "ColBERT",
        color: RGBColor(128, 0, 128),
        stroke: 2,
        pts,
    }
}

fn bench_plaid(
    doc_toks: &[(u32, Vec<[f32; TOKEN_DIM]>)],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    gt: &[HashSet<u32>],
    rng: &mut SmallRng,
) -> EngineResult {
    let n = doc_toks.len();
    let k = ((n as f64).sqrt() as usize).clamp(5, 50);
    let idx = CentIdx::build(doc_toks, k, rng);
    let probes = [1, 2, 3, 5, 8, k];
    let mut pts = vec![];
    for &np in &probes {
        if np > idx.centers.len() { break; }
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qtoks, gt_set) in query_toks.iter().zip(gt) {
            let cands = idx.probe(qtoks, np);
            tot_f += cands.len() as f64 / n as f64;
            let cands = if cands.is_empty() { std::iter::once(0u32).collect() } else { cands };
            let mut scored: Vec<(u32, f32)> = doc_toks.iter()
                .filter(|(id, _)| cands.contains(id))
                .map(|(id, dtoks)| (*id, maxsim(qtoks, dtoks)))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_r += recall_at_k(&scored, gt_set, K_EVAL);
        }
        let nq = query_toks.len() as f64;
        pts.push((tot_f / nq, tot_r / nq));
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    EngineResult {
        name: "PLAID — global k-means centroids",
        short: "PLAID",
        color: RGBColor(33, 102, 172),
        stroke: 2,
        pts,
    }
}

fn bench_warp(
    doc_toks: &[(u32, Vec<[f32; TOKEN_DIM]>)],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    gt: &[HashSet<u32>],
) -> EngineResult {
    let n = doc_toks.len();
    let thresholds = [0.95f32, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20];
    let mut pts = vec![];
    for &t in &thresholds {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qtoks, gt_set) in query_toks.iter().zip(gt) {
            let cands: HashSet<u32> = doc_toks.iter()
                .filter_map(|(id, dtoks)| {
                    let mx = dtoks.iter()
                        .flat_map(|dt| qtoks.iter().map(|qt| dot128(qt, dt)))
                        .fold(f32::NEG_INFINITY, f32::max);
                    if mx > t { Some(*id) } else { None }
                }).collect();
            let frac = (cands.len() as f64 / n as f64).max(1.0 / n as f64);
            tot_f += frac;
            let cands = if cands.is_empty() { std::iter::once(0u32).collect() } else { cands };
            let mut scored: Vec<(u32, f32)> = doc_toks.iter()
                .filter(|(id, _)| cands.contains(id))
                .map(|(id, dtoks)| (*id, maxsim(qtoks, dtoks)))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_r += recall_at_k(&scored, gt_set, K_EVAL);
        }
        let nq = query_toks.len() as f64;
        let frac = tot_f / nq;
        if frac > 0.005 {
            pts.push((frac, tot_r / nq));
        }
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    EngineResult {
        name: "WARP — Xtr token similarity threshold",
        short: "WARP",
        color: RGBColor(215, 48, 39),
        stroke: 2,
        pts,
    }
}

fn bench_tachiom(
    doc_toks: &[(u32, Vec<[f32; TOKEN_DIM]>)],
    query_toks: &[Vec<[f32; TOKEN_DIM]>],
    gt: &[HashSet<u32>],
    rng: &mut SmallRng,
    n_categories: usize,
    doc_categories: &HashMap<u32, usize>,
) -> EngineResult {
    let n = doc_toks.len();
    let k_per_cat = ((n as f64).sqrt() as usize / n_categories).clamp(3, 20);
    // Build one centroid index per category
    let cat_idxs: Vec<CentIdx> = (0..n_categories).map(|cat| {
        let cat_docs: Vec<(u32, Vec<[f32; TOKEN_DIM]>)> = doc_toks.iter()
            .filter(|(id, _)| doc_categories.get(id).copied().unwrap_or(0) == cat)
            .cloned().collect();
        CentIdx::build(&cat_docs, k_per_cat, rng)
    }).collect();

    let probes = [1, 2, 3, 5, 8];
    let mut pts = vec![];
    for &np in &probes {
        let (mut tot_f, mut tot_r) = (0.0f64, 0.0f64);
        for (qtoks, gt_set) in query_toks.iter().zip(gt) {
            let mut cands: HashSet<u32> = HashSet::new();
            for qt in qtoks {
                // Route to the category whose centroid is nearest this query token
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
                    cands.insert(d);
                }
            }
            tot_f += cands.len() as f64 / n as f64;
            let cands = if cands.is_empty() { std::iter::once(0u32).collect() } else { cands };
            let mut scored: Vec<(u32, f32)> = doc_toks.iter()
                .filter(|(id, _)| cands.contains(id))
                .map(|(id, dtoks)| (*id, maxsim(qtoks, dtoks)))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            tot_r += recall_at_k(&scored, gt_set, K_EVAL);
        }
        let nq = query_toks.len() as f64;
        pts.push((tot_f / nq, tot_r / nq));
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    EngineResult {
        name: "TACHIOM — per-type centroid budgets",
        short: "TACHIOM",
        color: RGBColor(26, 152, 80),
        stroke: 3,
        pts,
    }
}

// ─── plotting ─────────────────────────────────────────────────────────────────

fn to_pct_log(frac: f64) -> f64 {
    (frac.clamp(0.001, 1.0) * 100.0).log2()
}

fn plot_gt_recall(results: &[EngineResult], path: &str, title: &str) -> Result<()> {
    let root = SVGBackend::new(path, (720, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = to_pct_log(0.03);
    let x_max = to_pct_log(1.00);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 16).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f64..1.06f64)?;

    chart.configure_mesh()
        .x_desc("Candidate set (% of corpus, log scale)")
        .y_desc("Recall@10 vs LLM ground truth")
        .x_labels(7)
        .y_labels(6)
        .x_label_formatter(&|&x| {
            let v = 2f64.powf(x);
            if v < 2.0 { format!("{:.0}%", v) } else { format!("{:.0}%", v) }
        })
        .light_line_style(RGBColor(220, 220, 220).stroke_width(1))
        .draw()?;

    for engine in results {
        let pts: Vec<(f64, f64)> = engine.pts.iter()
            .filter(|&&(f, _)| f >= 0.02)
            .map(|&(f, r)| (to_pct_log(f), r))
            .collect();
        if pts.is_empty() { continue; }
        let color = engine.color;
        let sw = engine.stroke;
        chart.draw_series(LineSeries::new(
            pts.clone(),
            ShapeStyle { color: color.to_rgba(), filled: false, stroke_width: sw },
        ))?
        .label(engine.name)
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
        .label_font(("sans-serif", 12).into_font())
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    Ok(())
}

// ─── doc category map ─────────────────────────────────────────────────────────

fn doc_category_map() -> HashMap<u32, usize> {
    let mut m = HashMap::new();
    for id in 0..=19u32  { m.insert(id, 0); } // river/geo
    for id in 20..=39u32 { m.insert(id, 1); } // finance
    for id in 40..=49u32 { m.insert(id, 2); } // construction crane
    for id in 50..=59u32 { m.insert(id, 3); } // bird crane
    for id in 60..=64u32 { m.insert(id, 4); } // elephant trunk
    for id in 65..=69u32 { m.insert(id, 5); } // car trunk
    for id in 70..=79u32 { m.insert(id, 6); } // physics
    for id in 80..=89u32 { m.insert(id, 7); } // fashion
    for id in 90..=94u32 { m.insert(id, 8); } // anatomy
    for id in 95..=99u32 { m.insert(id, 9); } // misc
    m
}

// ─── terminal summary ─────────────────────────────────────────────────────────

fn print_gt_summary(query_labels: &[&str], results: &[EngineResult], gt: &[HashSet<u32>]) {
    const CYAN: &str  = "\x1b[36m";
    const BOLD: &str  = "\x1b[1m";
    const DIM:  &str  = "\x1b[2m";
    const RESET: &str = "\x1b[0m";
    const GREEN: &str = "\x1b[32m";

    println!();
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!("{BOLD}  Ground-Truth Recall@{K_EVAL}  (LLM-judged labels, 100-doc corpus){RESET}");
    println!("{DIM}  Each cell shows Recall@{K_EVAL} at that approximate candidate fraction.{RESET}");
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");

    let fracs: &[(f64, &str)] = &[(0.10, "10%"), (0.25, "25%"), (0.50, "50%"), (1.00, "100%")];
    let hdr: Vec<String> = fracs.iter().map(|(_, l)| format!("{:>7}", l)).collect();
    println!();
    println!("  {:<38}  {}", "Engine", hdr.join("  "));
    println!("  {}  {}", "─".repeat(38), "─".repeat(7 * fracs.len() + 2 * (fracs.len() - 1)));

    for engine in results {
        let cells: Vec<String> = fracs.iter().map(|(target, _)| {
            let nearest = engine.pts.iter().min_by_key(|&&(f, _)| {
                ((f - target).abs() * 100_000.0) as i64
            });
            match nearest {
                Some(&(f, r)) if (f - target).abs() < target * 3.0 => format!("{:>7.3}", r),
                _ => "      —".to_string(),
            }
        }).collect();
        let name_str = if engine.short == "HNSW" {
            format!("{GREEN}{BOLD}{:<38}{RESET}", engine.name)
        } else {
            format!("{:<38}", engine.name)
        };
        println!("  {name_str}  {}", cells.join("  "));
    }

    println!();
    println!("{DIM}  LLM-judged GT: relevant doc counts per query:{RESET}");
    for (q, gt_set) in query_labels.iter().zip(gt) {
        println!("{DIM}    \"{q}\" → {} relevant docs{RESET}", gt_set.len());
    }
    println!();
}

// ─── entry point ─────────────────────────────────────────────────────────────

pub async fn run_gtbench(vocab_path: &Path) -> Result<()> {
    const CYAN: &str  = "\x1b[36m";
    const BOLD: &str  = "\x1b[1m";
    const DIM:  &str  = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    println!();
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!("{BOLD}  Ground-Truth Benchmark (LLM-as-Judge){RESET}");
    println!("{DIM}  100 real-text docs · 10 queries · All 5 engines");
    println!("  GT labels: Claude Haiku relevance judgments (cached in cache/llm_gt.json)");
    println!("  HNSW: Voyage sentence embeddings · Others: WordPiece + RandomProjection{RESET}");
    println!("{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}");
    println!();

    // ── 1. LLM ground truth labels ───────────────────────────────────────────
    const GT_CACHE: &str = "cache/llm_gt.json";
    let mut gt_map: HashMap<String, Vec<u32>> = load_gt_cache(GT_CACHE).unwrap_or_default();
    let missing_queries: Vec<&str> = GT_QUERIES.iter().copied()
        .filter(|q| !gt_map.contains_key(*q))
        .collect();

    if !missing_queries.is_empty() {
        match AnthropicClient::from_env() {
            Some(client) => {
                println!("  [llm] judging {} queries via Claude Haiku…", missing_queries.len());
                for query in &missing_queries {
                    print!("    {query:50} … ");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                    let ids = client.judge_query(query, GT_CORPUS).await?;
                    println!("{} relevant docs", ids.len());
                    gt_map.insert(query.to_string(), ids);
                }
                save_gt_cache(GT_CACHE, &gt_map)?;
                println!("  [llm] cached GT labels → {GT_CACHE}");
            }
            None => {
                eprintln!("  [warn] ANTHROPIC_API_KEY not set — using heuristic topic labels");
                eprintln!("         Set ANTHROPIC_API_KEY in .env for LLM-judged ground truth.");
                // Fallback: use category membership as relevance
                let cat_map = doc_category_map();
                let query_cats: &[usize] = &[0, 1, 1, 0, 2, 3, 4, 5, 6, 7];
                for (qi, query) in GT_QUERIES.iter().enumerate() {
                    if gt_map.contains_key(*query) { continue; }
                    let relevant_cat = query_cats[qi];
                    let ids: Vec<u32> = GT_CORPUS.iter()
                        .filter(|(id, _)| cat_map.get(id).copied().unwrap_or(99) == relevant_cat)
                        .map(|(id, _)| *id)
                        .collect();
                    gt_map.insert(query.to_string(), ids);
                }
            }
        }
    } else {
        println!("  [llm] loaded GT labels from cache ({} queries)", gt_map.len());
    }

    let gt_sets: Vec<HashSet<u32>> = GT_QUERIES.iter()
        .map(|q| gt_map.get(*q).cloned().unwrap_or_default().into_iter().collect())
        .collect();

    // ── 2. Voyage sentence embeddings for HNSW ───────────────────────────────
    const EMB_CACHE: &str = "cache/gt_voyage_embeddings.json";
    let mut emb_map: HashMap<u32, Vec<f32>> = load_emb_cache(EMB_CACHE).unwrap_or_default();
    let missing_ids: Vec<(u32, &str)> = GT_CORPUS.iter()
        .filter(|(id, _)| !emb_map.contains_key(id))
        .map(|(id, text)| (*id, *text))
        .collect();

    if !missing_ids.is_empty() {
        match VoyageEmbed::from_env() {
            Some(client) => {
                print!("  [voyage] embedding {} docs via {}… ", missing_ids.len(), client.model);
                let _ = std::io::Write::flush(&mut std::io::stdout());
                let texts: Vec<&str> = missing_ids.iter().map(|(_, t)| *t).collect();
                let embs = client.embed_texts(&texts).await?;
                for ((id, _), emb) in missing_ids.iter().zip(embs.into_iter()) {
                    emb_map.insert(*id, emb);
                }
                save_emb_cache(EMB_CACHE, &emb_map)?;
                println!("cached → {EMB_CACHE}");
            }
            None => {
                eprintln!("  [warn] VOYAGE_API_KEY not set — using mock sentence embeddings for HNSW.");
                eprintln!("         HNSW results will be random (not semantic). Set key for real comparison.");
                let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
                for (id, _) in &missing_ids {
                    let mut v: Vec<f32> = (0..1024).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                    v.iter_mut().for_each(|x| *x /= norm);
                    emb_map.insert(*id, v);
                }
            }
        }
    } else {
        println!("  [voyage] loaded {} sentence embeddings from cache", emb_map.len());
    }

    // Also embed queries for HNSW
    const QEMB_CACHE: &str = "cache/gt_voyage_query_embeddings.json";
    let mut qemb_map: HashMap<u32, Vec<f32>> = load_emb_cache(QEMB_CACHE).unwrap_or_default();
    let missing_qids: Vec<(usize, &&str)> = GT_QUERIES.iter().enumerate()
        .filter(|(i, _)| !qemb_map.contains_key(&(*i as u32)))
        .collect();

    if !missing_qids.is_empty() {
        match VoyageEmbed::from_env() {
            Some(client) => {
                print!("  [voyage] embedding {} queries… ", missing_qids.len());
                let _ = std::io::Write::flush(&mut std::io::stdout());
                let texts: Vec<&str> = missing_qids.iter().map(|(_, q)| **q).collect();
                let embs = client.embed_texts(&texts).await?;
                for ((i, _), emb) in missing_qids.iter().zip(embs.into_iter()) {
                    qemb_map.insert(*i as u32, emb);
                }
                save_emb_cache(QEMB_CACHE, &qemb_map)?;
                println!("ok");
            }
            None => {
                let mut rng = SmallRng::seed_from_u64(0xFACEFEED);
                for (i, _) in &missing_qids {
                    let mut v: Vec<f32> = (0..1024).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                    v.iter_mut().for_each(|x| *x /= norm);
                    qemb_map.insert(*i as u32, v);
                }
            }
        }
    }

    let doc_embs: Vec<(u32, Vec<f32>)> = GT_CORPUS.iter()
        .filter_map(|(id, _)| emb_map.get(id).map(|e| (*id, e.clone())))
        .collect();
    let query_embs: Vec<Vec<f32>> = GT_QUERIES.iter().enumerate()
        .filter_map(|(i, _)| qemb_map.get(&(i as u32)).cloned())
        .collect();

    // ── 3. Token embeddings for multivector engines ──────────────────────────
    print!("  [tokens] tokenizing + random-projecting {} docs… ", GT_CORPUS.len());
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let tokenizer = WordPieceTokenizer::from_vocab(vocab_path)?;
    let mut proj = RandomProjection::new(0x0123456789ABCDEFu64);
    let doc_toks: Vec<(u32, Vec<[f32; TOKEN_DIM]>)> = GT_CORPUS.iter()
        .map(|(id, text)| (*id, embed_text_tokens(text, &tokenizer, &mut proj)))
        .collect();
    let query_toks: Vec<Vec<[f32; TOKEN_DIM]>> = GT_QUERIES.iter()
        .map(|q| embed_text_tokens(q, &tokenizer, &mut proj))
        .collect();
    println!("ok");

    let doc_cats = doc_category_map();

    // ── 4. Run all 5 engines ─────────────────────────────────────────────────
    println!("  [bench] running engines…");
    let mut rng = SmallRng::seed_from_u64(0xCAFEBABE);

    print!("    HNSW…       ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let hnsw_res = bench_hnsw(&doc_embs, &query_embs, &gt_sets);
    println!("{} points", hnsw_res.pts.len());

    print!("    ColBERT…    ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let colbert_res = bench_colbert(&doc_toks, &query_toks, &gt_sets);
    println!("{} points", colbert_res.pts.len());

    print!("    PLAID…      ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let plaid_res = bench_plaid(&doc_toks, &query_toks, &gt_sets, &mut rng);
    println!("{} points", plaid_res.pts.len());

    print!("    WARP…       ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let warp_res = bench_warp(&doc_toks, &query_toks, &gt_sets);
    println!("{} points", warp_res.pts.len());

    print!("    TACHIOM…    ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let tac_res = bench_tachiom(&doc_toks, &query_toks, &gt_sets, &mut rng, 10, &doc_cats);
    println!("{} points", tac_res.pts.len());

    let all_results = vec![hnsw_res, colbert_res, plaid_res, warp_res, tac_res];

    // ── 5. Print summary ─────────────────────────────────────────────────────
    print_gt_summary(GT_QUERIES, &all_results, &gt_sets);

    // ── 6. Plot ──────────────────────────────────────────────────────────────
    std::fs::create_dir_all("plots")?;
    let path = "plots/gt_recall.svg";
    plot_gt_recall(
        &all_results,
        path,
        &format!("Recall@{K_EVAL} vs LLM Ground Truth  (N=100 real-text docs)"),
    )?;
    println!("  {BOLD}→ {path}{RESET}  (open with: open {path})");
    println!("{DIM}  LLM judge: Claude Haiku · HNSW: Voyage sentence embeddings");
    println!("  Multivector engines: WordPiece + RandomProjection tokens{RESET}");
    println!();

    Ok(())
}
