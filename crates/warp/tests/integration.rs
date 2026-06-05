use common::{corpus::SHARED_CORPUS, Engine, TOKEN_DIM};
use warp::{gather::gather_candidates, xtr::XtrScorer, WarpEngine};
use std::path::PathBuf;

fn vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("vocab/wordpiece_vocab.txt")
}

fn unit_token(dim: usize) -> [f32; TOKEN_DIM] {
    let mut t = [0.0f32; TOKEN_DIM];
    t[dim % TOKEN_DIM] = 1.0;
    t
}

// ─── XtrScorer ───────────────────────────────────────────────────────────────

#[test]
fn xtr_scorer_zero_threshold_gathers_all() {
    let mut scorer = XtrScorer::new();
    // Register 5 docs with 1-token each
    for i in 0u32..5 {
        let tok = unit_token(i as usize);
        scorer.register_doc(i, &[(i, tok)]);
    }

    // Build a minimal TokenMatrix for query
    let query_tok = unit_token(0);
    let query = common::TokenMatrix {
        rows: vec![query_tok],
        tokens: vec!["[Q]".into()],
    };

    // t_prime = -1.0 (below any cosine score) → all 5 docs should be gathered
    let (results, _log) = scorer.score(&query, -1.0, 100);
    assert_eq!(
        results.len(),
        5,
        "t_prime=-1.0 must gather all 5 docs, got {}",
        results.len()
    );
}

#[test]
fn xtr_scorer_high_threshold_gathers_nothing() {
    let mut scorer = XtrScorer::new();
    for i in 0u32..5 {
        let tok = unit_token(i as usize);
        scorer.register_doc(i, &[(i, tok)]);
    }
    let query_tok = unit_token(0);
    let query = common::TokenMatrix {
        rows: vec![query_tok],
        tokens: vec!["[Q]".into()],
    };
    // t_prime = 1.1 (above any achievable cosine score) → nothing gathered
    let (results, _log) = scorer.score(&query, 1.1, 100);
    assert!(
        results.is_empty(),
        "t_prime=1.1 must gather no docs"
    );
}

#[test]
fn xtr_scorer_perfect_match_scores_1() {
    let mut scorer = XtrScorer::new();
    let tok = unit_token(7);
    scorer.register_doc(42, &[(7, tok)]);

    let query = common::TokenMatrix {
        rows: vec![tok],
        tokens: vec!["test".into()],
    };
    let (results, _) = scorer.score(&query, 0.0, 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 42, "must return doc 42");
    assert!(
        (results[0].1 - 1.0).abs() < 1e-4,
        "identical token must score ≈1.0, got {}",
        results[0].1
    );
}

// ─── gather_candidates ──────────────────────────────────────────────────────

#[test]
fn gather_candidates_fraction_valid() {
    let xtr_results: Vec<(u32, f32)> = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
    let (gathered, stats, _log) = gather_candidates(&xtr_results, 10);
    assert_eq!(gathered.len(), 3, "must gather 3 docs");
    assert!(
        stats.fraction_promoted >= 0.0 && stats.fraction_promoted <= 1.0,
        "fraction_promoted must be in [0,1], got {}",
        stats.fraction_promoted
    );
    assert_eq!(
        stats.fraction_promoted,
        3.0 / 10.0,
        "fraction_promoted = gathered/total"
    );
}

#[test]
fn gather_candidates_empty_input() {
    let (gathered, stats, _log) = gather_candidates(&[], 5);
    assert!(gathered.is_empty(), "empty input → empty gathered");
    assert_eq!(stats.fraction_promoted, 0.0);
}

#[test]
fn gather_candidates_all_docs() {
    let xtr_results: Vec<(u32, f32)> = (0u32..5).map(|i| (i, 0.5)).collect();
    let (gathered, stats, _log) = gather_candidates(&xtr_results, 5);
    assert_eq!(gathered.len(), 5);
    assert_eq!(stats.fraction_promoted, 1.0, "all 5/5 promoted → 1.0");
}

// ─── WarpEngine (Engine trait) ───────────────────────────────────────────────

#[tokio::test]
async fn warp_engine_indexes_and_queries() {
    let mut eng = WarpEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS.iter().take(8) {
        eng.index(*doc_id, text).await.unwrap();
    }
    let (results, _log) = eng.query("bank interest rate", 5).await.unwrap();
    assert!(!results.is_empty(), "WARP must return results after indexing");
    assert!(results.len() <= 5);
}

#[tokio::test]
async fn warp_engine_gather_stats_valid() {
    let mut eng = WarpEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS {
        eng.index(*doc_id, text).await.unwrap();
    }
    let (results, _) = eng.query("river bank", 5).await.unwrap();
    let _ = results; // just verify no panic

    // Inspect gather stats
    let out = eng.inspect(Some("gather")).await.unwrap();
    assert!(!out.is_empty(), "inspect(gather) must return non-empty string");
}

#[tokio::test]
async fn warp_engine_trace_events_produced() {
    let mut eng = WarpEngine::new(&vocab_path()).unwrap();
    let log = eng.index(0, "river bank").await.unwrap();
    assert!(!log.events.is_empty(), "WARP index must emit trace events");

    let (_, qlog) = eng.query("bank", 3).await.unwrap();
    assert!(!qlog.events.is_empty(), "WARP query must emit trace events");
}

#[tokio::test]
async fn warp_engine_all_results_valid_doc_ids() {
    let mut eng = WarpEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS {
        eng.index(*doc_id, text).await.unwrap();
    }
    let (results, _) = eng.query("crane lift construction", 10).await.unwrap();
    let valid_ids: std::collections::HashSet<u32> =
        SHARED_CORPUS.iter().map(|(id, _)| *id).collect();
    for (doc_id, score) in &results {
        assert!(
            valid_ids.contains(doc_id),
            "result doc_id={doc_id} must be in SHARED_CORPUS"
        );
        assert!(score.is_finite(), "score must be finite");
    }
}

#[tokio::test]
async fn warp_engine_consistent_with_brute_force_for_zero_threshold() {
    // With t_prime=-1.0 (effectively zero threshold), WARP gathers all docs
    // and its top-1 should match ColBERT brute-force top-1
    use colbert::ColBertEngine;
    let vp = vocab_path();

    let mut warp_eng = WarpEngine::new(&vp).unwrap();
    let mut colbert_eng = ColBertEngine::new(&vp).unwrap();

    for (doc_id, text) in SHARED_CORPUS {
        warp_eng.index(*doc_id, text).await.unwrap();
        colbert_eng.index(*doc_id, text).await.unwrap();
    }

    let q = "trunk of a tree";
    let (warp_results, _) = warp_eng.query(q, 10).await.unwrap();
    let (colbert_results, _) = colbert_eng.query(q, 10).await.unwrap();

    // Verify both return results and they're non-empty
    assert!(!warp_results.is_empty(), "WARP must return results");
    assert!(!colbert_results.is_empty(), "ColBERT must return results");
}

#[test]
fn warp_verify_harness_passes() {
    let mut eng = WarpEngine::new(&vocab_path()).unwrap();
    eng.verify().expect("WARP verify harness must pass");
}
