use common::{corpus::SHARED_CORPUS, Engine, TOKEN_DIM};
use tachiom::{
    tac::{
        budget::{BudgetReconciler, EPSILON, THETA},
        clustering::kmeans,
        damping::{damped_weight, DampedScorer},
        tail::{TailClass, TailHandler, MU, TAU},
    },
    TachiomEngine,
};
use std::path::PathBuf;
use std::collections::HashMap;

fn vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("vocab/wordpiece_vocab.txt")
}

// ─── TailHandler ─────────────────────────────────────────────────────────────

#[test]
fn tail_handler_classifies_tail() {
    let mut handler = TailHandler::new();
    let token_id = 42u32;
    // Add fewer than MU occurrences → Tail
    for _ in 0..(MU - 1) {
        handler.update(token_id);
    }
    assert_eq!(
        handler.classify(token_id),
        TailClass::Tail,
        "freq={} < MU={MU} must be Tail",
        MU - 1
    );
}

#[test]
fn tail_handler_classifies_normal() {
    let mut handler = TailHandler::new();
    let token_id = 7u32;
    for _ in 0..MU {
        handler.update(token_id);
    }
    assert_eq!(
        handler.classify(token_id),
        TailClass::Normal,
        "freq=MU={MU} must be Normal"
    );
}

#[test]
fn tail_handler_classifies_heavy() {
    let mut handler = TailHandler::new();
    let token_id = 99u32;
    for _ in 0..TAU {
        handler.update(token_id);
    }
    assert_eq!(
        handler.classify(token_id),
        TailClass::Heavy,
        "freq=TAU={TAU} must be Heavy"
    );
}

#[test]
fn tail_handler_unknown_token_is_tail() {
    let handler = TailHandler::new();
    // Token with zero occurrences → Tail (freq < MU)
    assert_eq!(handler.classify(12345), TailClass::Tail, "unknown token must be Tail");
}

#[test]
fn tail_handler_boundary_conditions() {
    let mut handler = TailHandler::new();
    let t = 55u32;
    // freq = TAU - 1 → Normal
    for _ in 0..(TAU - 1) {
        handler.update(t);
    }
    assert_eq!(handler.classify(t), TailClass::Normal);
    // One more → Heavy
    handler.update(t);
    assert_eq!(handler.classify(t), TailClass::Heavy);
}

// ─── DampedScorer ────────────────────────────────────────────────────────────

#[test]
fn damped_weight_is_positive_for_positive_inputs() {
    assert!(damped_weight(1.0, 10) > 0.0, "damped_weight must be positive");
    assert!(damped_weight(0.5, 4) > 0.0);
}

#[test]
fn damped_weight_formula_correct() {
    // w = sqrt(n) * variance
    let v = 2.0f32;
    let n = 9u32;
    let expected = 9f32.sqrt() * v; // 3.0 * 2.0 = 6.0
    let got = damped_weight(v, n);
    assert!(
        (got - expected).abs() < 1e-4,
        "damped_weight({v}, {n}) must be {expected}, got {got}"
    );
}

#[test]
fn damped_weight_zero_variance() {
    // If all embeddings are identical, variance=0 → weight=0
    assert_eq!(damped_weight(0.0, 100), 0.0);
}

#[test]
fn damped_scorer_computes_weights() {
    let mut embeddings: HashMap<u32, Vec<[f32; TOKEN_DIM]>> = HashMap::new();
    // Token type 1: two identical embeddings → variance=0
    let e1 = {
        let mut a = [0.0f32; TOKEN_DIM];
        a[0] = 1.0;
        a
    };
    embeddings.insert(1, vec![e1, e1]);
    // Token type 2: two opposite embeddings → high variance
    let e2a = { let mut a = [0.0f32; TOKEN_DIM]; a[1] = 1.0; a };
    let e2b = { let mut a = [0.0f32; TOKEN_DIM]; a[1] = -1.0; a };
    embeddings.insert(2, vec![e2a, e2b]);

    let scorer = DampedScorer::compute(&embeddings);
    let w1 = scorer.weights.get(&1).copied().unwrap_or(0.0);
    let w2 = scorer.weights.get(&2).copied().unwrap_or(0.0);
    // High-variance token type 2 must have higher weight than zero-variance type 1
    assert!(
        w2 > w1,
        "high-variance type (w2={w2}) must outweigh zero-variance type (w1={w1})"
    );
}

// ─── BudgetReconciler ────────────────────────────────────────────────────────

#[test]
fn budget_kappa_always_in_epsilon_theta() {
    let reconciler = BudgetReconciler::new(200);
    let mut weights: HashMap<u32, f32> = HashMap::new();
    weights.insert(0, 1.0);
    weights.insert(1, 5.0);
    weights.insert(2, 0.5);
    weights.insert(3, 10.0);

    let kappa = reconciler.allocate(&weights);
    for (&token_type, &k) in &kappa {
        assert!(
            k >= EPSILON && k <= THETA,
            "token_type={token_type}: kappa_j={k} must be in [{EPSILON}, {THETA}]"
        );
    }
}

#[test]
fn budget_all_weights_equal_produces_uniform_allocation() {
    let reconciler = BudgetReconciler::new(100);
    let mut weights: HashMap<u32, f32> = HashMap::new();
    for i in 0..4u32 {
        weights.insert(i, 1.0); // equal weights
    }
    let kappa = reconciler.allocate(&weights);
    // All kappa_j should be equal (or within 1 due to integer rounding)
    let values: Vec<u32> = kappa.values().copied().collect();
    let min = values.iter().min().copied().unwrap_or(0);
    let max = values.iter().max().copied().unwrap_or(0);
    assert!(
        max - min <= 1,
        "equal weights must produce near-uniform kappa, got min={min} max={max}"
    );
}

#[test]
fn budget_epsilon_theta_constants() {
    assert_eq!(EPSILON, 4, "EPSILON must be 4");
    assert_eq!(THETA, 39, "THETA must be 39");
}

// ─── KMeans ──────────────────────────────────────────────────────────────────

#[test]
fn kmeans_produces_correct_count() {
    let data: Vec<[f32; TOKEN_DIM]> = (0..10)
        .map(|i| {
            let mut row = [0.0f32; TOKEN_DIM];
            row[i % TOKEN_DIM] = 1.0;
            row
        })
        .collect();
    let centroids = kmeans(&data, 3, 42);
    assert_eq!(centroids.len(), 3, "kmeans must produce exactly k centroids");
}

#[test]
fn kmeans_clamps_k_to_n_points() {
    let data: Vec<[f32; TOKEN_DIM]> = (0..3)
        .map(|i| {
            let mut row = [0.0f32; TOKEN_DIM];
            row[i] = 1.0;
            row
        })
        .collect();
    // k > n: should clamp to n without panicking
    let centroids = kmeans(&data, 100, 42);
    assert!(
        centroids.len() <= 3,
        "kmeans must clamp k to n; produced {} centroids for 3 points",
        centroids.len()
    );
}

#[test]
fn kmeans_single_point_returns_that_point() {
    let mut data = [[0.0f32; TOKEN_DIM]; 1];
    data[0][0] = 1.0;
    let centroids = kmeans(&data, 1, 42);
    assert_eq!(centroids.len(), 1);
    assert!(
        (centroids[0][0] - 1.0).abs() < 1e-3,
        "single-point kmeans must return that point as centroid"
    );
}

// ─── TachiomEngine (Engine trait) ────────────────────────────────────────────

#[tokio::test]
async fn tachiom_engine_indexes_and_queries() {
    let mut eng = TachiomEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS.iter().take(5) {
        eng.index(*doc_id, text).await.unwrap();
    }
    let (results, _log) = eng.query("bank interest rate", 3).await.unwrap();
    assert!(!results.is_empty(), "TACHIOM must return results after indexing");
}

#[tokio::test]
async fn tachiom_engine_pq_inspect() {
    let mut eng = TachiomEngine::new(&vocab_path()).unwrap();
    eng.index(0, "test").await.unwrap();
    let out = eng.inspect(Some("pq")).await.unwrap();
    assert!(
        out.contains("Level") || out.contains("level") || out.contains("PQ"),
        "inspect(pq) must describe the 3-level PQ layout, got: {out}"
    );
}

#[tokio::test]
async fn tachiom_engine_centroids_inspect() {
    let mut eng = TachiomEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS.iter().take(5) {
        eng.index(*doc_id, text).await.unwrap();
    }
    let out = eng.inspect(Some("centroids")).await.unwrap();
    assert!(!out.is_empty(), "inspect(centroids) must return non-empty string");
}

#[tokio::test]
async fn tachiom_engine_trace_events() {
    let mut eng = TachiomEngine::new(&vocab_path()).unwrap();
    let log = eng.index(0, "river bank").await.unwrap();
    assert!(!log.events.is_empty(), "TACHIOM index must emit trace events");
}

#[tokio::test]
async fn tachiom_engine_all_docs_indexable() {
    let mut eng = TachiomEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS {
        eng.index(*doc_id, text).await.expect(&format!(
            "indexing doc {doc_id} must not fail"
        ));
    }
    let (results, _) = eng.query("bank interest rate", 5).await.unwrap();
    assert!(!results.is_empty(), "TACHIOM query must return results after indexing all docs");
}

#[test]
fn tachiom_verify_harness_passes() {
    let mut eng = TachiomEngine::new(&vocab_path()).unwrap();
    eng.verify().expect("TACHIOM verify harness must pass");
}
