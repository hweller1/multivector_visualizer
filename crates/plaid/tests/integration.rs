use colbert::{encoder::ColBertEncoder, index::ColBertIndex};
use common::{corpus::SHARED_CORPUS, Engine, TOKEN_DIM};
use plaid::{centroid::CentroidPruner, PlaidEngine};
use std::path::PathBuf;

fn vocab_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("vocab/wordpiece_vocab.txt")
}

// ─── CentroidPruner ─────────────────────────────────────────────────────────

#[test]
fn centroid_pruner_fit_returns_correct_count() {
    let pruner = CentroidPruner::new(4);
    let points: Vec<[f32; TOKEN_DIM]> = (0..20)
        .map(|i| {
            let mut row = [0.0f32; TOKEN_DIM];
            row[i % TOKEN_DIM] = 1.0;
            row
        })
        .collect();
    let centroids = pruner.fit(&points);
    assert_eq!(centroids.len(), 4, "fit must produce exactly num_centroids centroids");
}

#[test]
fn centroid_pruner_fit_clamps_to_num_points() {
    let pruner = CentroidPruner::new(100); // k > n
    let points: Vec<[f32; TOKEN_DIM]> = (0..5)
        .map(|i| {
            let mut row = [0.0f32; TOKEN_DIM];
            row[i] = 1.0;
            row
        })
        .collect();
    let centroids = pruner.fit(&points);
    assert!(
        centroids.len() <= 5,
        "fit must not produce more centroids than input points; got {} centroids for 5 points",
        centroids.len()
    );
}

#[test]
fn centroid_pruner_assign_returns_valid_index() {
    let pruner = CentroidPruner::new(4);
    let points: Vec<[f32; TOKEN_DIM]> = (0..16)
        .map(|i| {
            let mut row = [0.0f32; TOKEN_DIM];
            row[i % TOKEN_DIM] = 1.0;
            row
        })
        .collect();
    let centroids = pruner.fit(&points);
    for point in &points {
        let idx = pruner.assign(&centroids, point);
        assert!(
            (idx as usize) < centroids.len(),
            "assign must return valid centroid index; got {idx} for {} centroids",
            centroids.len()
        );
    }
}

#[test]
fn centroid_pruner_query_centroids_sorted_descending() {
    let pruner = CentroidPruner::new(4);
    let points: Vec<[f32; TOKEN_DIM]> = (0..16)
        .map(|i| {
            let mut row = [0.0f32; TOKEN_DIM];
            row[i % TOKEN_DIM] = 1.0;
            row
        })
        .collect();
    let centroids = pruner.fit(&points);
    let mut query = [0.0f32; TOKEN_DIM];
    query[0] = 1.0;
    let top = pruner.query_centroids(&centroids, &query, 3);
    assert!(top.len() <= 3, "must return at most nprobe centroids");
    for window in top.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "query_centroids must be sorted descending by score"
        );
    }
}

// ─── PlaidEngine (Engine trait) ──────────────────────────────────────────────

#[tokio::test]
async fn plaid_engine_indexes_and_queries() {
    let mut eng = PlaidEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS.iter().take(8) {
        eng.index(*doc_id, text).await.unwrap();
    }
    let (results, _log) = eng.query("bank interest rate", 5).await.unwrap();
    assert!(!results.is_empty(), "PLAID query must return results after indexing");
    assert!(results.len() <= 5);
}

#[tokio::test]
async fn plaid_engine_results_non_empty_after_indexing() {
    let mut eng = PlaidEngine::new(&vocab_path()).unwrap();
    // Index all 20 docs
    for (doc_id, text) in SHARED_CORPUS {
        eng.index(*doc_id, text).await.unwrap();
    }
    // Multiple queries must all return results
    let queries = [
        "river bank flooding",
        "financial bank account",
        "crane construction site",
    ];
    for q in queries {
        let (results, _) = eng.query(q, 5).await.unwrap();
        assert!(
            !results.is_empty(),
            "PLAID must return results for query '{q}'"
        );
    }
}

#[tokio::test]
async fn plaid_engine_inspect_centroids() {
    let mut eng = PlaidEngine::new(&vocab_path()).unwrap();
    for (doc_id, text) in SHARED_CORPUS.iter().take(5) {
        eng.index(*doc_id, text).await.unwrap();
    }
    let out = eng.inspect(Some("centroids")).await.unwrap();
    assert!(
        out.contains("centroid"),
        "inspect(centroids) must describe centroid structure, got: {out}"
    );
}

#[tokio::test]
async fn plaid_engine_trace_events_produced() {
    let mut eng = PlaidEngine::new(&vocab_path()).unwrap();
    let log = eng.index(0, "river bank").await.unwrap();
    assert!(!log.events.is_empty(), "PLAID index must emit trace events");
}

#[tokio::test]
async fn plaid_engine_consistent_with_colbert_top1() {
    use colbert::ColBertEngine;
    let vp = vocab_path();

    let mut plaid_eng = PlaidEngine::new(&vp).unwrap();
    let mut colbert_eng = ColBertEngine::new(&vp).unwrap();

    for (doc_id, text) in SHARED_CORPUS {
        plaid_eng.index(*doc_id, text).await.unwrap();
        colbert_eng.index(*doc_id, text).await.unwrap();
    }

    let queries = ["river bank flooding", "financial bank account", "light wavelength"];
    let mut top1_matches = 0;

    for q in queries {
        let (plaid_results, _) = plaid_eng.query(q, 10).await.unwrap();
        let (colbert_results, _) = colbert_eng.query(q, 10).await.unwrap();
        if !plaid_results.is_empty() && !colbert_results.is_empty()
            && plaid_results[0].0 == colbert_results[0].0
        {
            top1_matches += 1;
        }
    }

    // PLAID should agree with ColBERT on top-1 for at least 2/3 queries
    assert!(
        top1_matches >= 2,
        "PLAID top-1 must agree with ColBERT for ≥2/3 queries; agreed on {top1_matches}/3"
    );
}

#[test]
fn plaid_verify_harness_passes() {
    let mut eng = PlaidEngine::new(&vocab_path()).unwrap();
    eng.verify().expect("PLAID verify harness must pass");
}
