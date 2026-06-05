use hnsw::{mock_embedding, HnswEngine, LocalHnsw};
use common::corpus::SHARED_CORPUS;
use common::Engine;

// ─── mock_embedding ─────────────────────────────────────────────────────────

#[test]
fn mock_embedding_is_deterministic() {
    let a = mock_embedding(42);
    let b = mock_embedding(42);
    assert_eq!(a, b, "mock_embedding must be deterministic for the same doc_id");
}

#[test]
fn mock_embedding_differs_across_doc_ids() {
    let a = mock_embedding(0);
    let b = mock_embedding(1);
    let c = mock_embedding(99);
    assert_ne!(a, b, "doc 0 and doc 1 must have different embeddings");
    assert_ne!(a, c, "doc 0 and doc 99 must have different embeddings");
    assert_ne!(b, c);
}

#[test]
fn mock_embedding_is_unit_length() {
    for doc_id in [0u32, 1, 7, 19, 100, 65535] {
        let emb = mock_embedding(doc_id);
        assert_eq!(emb.len(), 1536, "embedding must be 1536-dim");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "doc_id={doc_id}: L2 norm={norm} must be ≈1.0"
        );
    }
}

// ─── LocalHnsw ───────────────────────────────────────────────────────────────

#[test]
fn local_hnsw_insert_and_search_self() {
    let mut hnsw = LocalHnsw::new();
    hnsw.insert_mock(42);
    let query = mock_embedding(42);
    let results = hnsw.search(&query, 1);
    assert_eq!(results.len(), 1, "should return 1 result");
    assert_eq!(results[0].0, 42, "top-1 must be the indexed doc itself");
    assert!(
        results[0].1 > 0.99,
        "self-similarity must be near 1.0, got {}",
        results[0].1
    );
}

#[test]
fn local_hnsw_recall_all_docs() {
    let mut hnsw = LocalHnsw::new();
    for (doc_id, _) in SHARED_CORPUS {
        hnsw.insert_mock(*doc_id);
    }
    let mut hits = 0;
    for (doc_id, _) in SHARED_CORPUS {
        let query = mock_embedding(*doc_id);
        let results = hnsw.search(&query, 1);
        if !results.is_empty() && results[0].0 == *doc_id {
            hits += 1;
        }
    }
    assert_eq!(
        hits,
        SHARED_CORPUS.len(),
        "recall@1 must be 1.0 for all 20 docs using their own mock embeddings"
    );
}

#[test]
fn local_hnsw_len_tracks_insertions() {
    let mut hnsw = LocalHnsw::new();
    assert_eq!(hnsw.len(), 0);
    hnsw.insert_mock(0);
    assert_eq!(hnsw.len(), 1);
    hnsw.insert_mock(1);
    hnsw.insert_mock(2);
    assert_eq!(hnsw.len(), 3);
}

#[test]
fn local_hnsw_insert_raw_roundtrip() {
    let mut hnsw = LocalHnsw::new();
    let emb = mock_embedding(77);
    hnsw.insert_raw(77, emb.clone());
    let results = hnsw.search(&emb, 1);
    assert_eq!(results[0].0, 77, "insert_raw doc must be retrievable");
}

// ─── HnswEngine (local mode) ─────────────────────────────────────────────────

#[tokio::test]
async fn hnsw_engine_index_and_query() {
    let mut eng = HnswEngine::new_local();
    for (doc_id, text) in SHARED_CORPUS.iter().take(5) {
        eng.index(*doc_id, text).await.expect("index must succeed");
    }
    let (results, _log) = eng.query("some query text", 3).await.expect("query must succeed");
    // Local mode returns results based on text hash; just verify non-empty and valid
    assert!(!results.is_empty(), "query must return at least one result");
    for (doc_id, score) in &results {
        assert!(
            SHARED_CORPUS.iter().any(|(id, _)| id == doc_id),
            "returned doc_id={doc_id} must be in SHARED_CORPUS"
        );
        assert!(score.is_finite(), "score must be finite");
    }
}

#[tokio::test]
async fn hnsw_engine_inspect_returns_string() {
    let mut eng = HnswEngine::new_local();
    eng.index(0, "test").await.unwrap();
    let s = eng.inspect(None).await.unwrap();
    assert!(!s.is_empty(), "inspect(None) must return non-empty string");
    let s2 = eng.inspect(Some("layers")).await.unwrap();
    assert!(!s2.is_empty());
    let s3 = eng.inspect(Some("graph")).await.unwrap();
    assert!(!s3.is_empty());
}

#[tokio::test]
async fn hnsw_engine_trace_log_has_events() {
    let mut eng = HnswEngine::new_local();
    let log = eng.index(5, "hello world").await.unwrap();
    assert!(!log.events.is_empty(), "indexing must produce at least one trace event");
    let (_, log2) = eng.query("hello world", 1).await.unwrap();
    assert!(!log2.events.is_empty(), "query must produce at least one trace event");
}

#[test]
fn hnsw_verify_harness_passes() {
    let mut eng = HnswEngine::new_local();
    eng.verify().expect("HNSW verify harness must pass");
}
