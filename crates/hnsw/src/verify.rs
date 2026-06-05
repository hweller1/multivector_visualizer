use crate::engine::HnswEngine;
use crate::local::mock_embedding;
use anyhow::Result;
use common::corpus::SHARED_CORPUS;

/// Verify LocalHnsw only — no Atlas, no network calls.
///
/// For each indexed doc, queries with its own mock_embedding and asserts it
/// returns as the top-1 result. Mock embeddings are unique by doc_id
/// construction, so recall@1=1.0 is guaranteed.
pub fn run(engine: &mut HnswEngine) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Only test local mode — Atlas requires live credentials
    assert!(
        engine.atlas.is_none(),
        "verify harness must run in local mode (no Atlas)"
    );

    // Index all 20 corpus docs with insert_mock
    for (doc_id, _text) in SHARED_CORPUS {
        engine.local.insert_mock(*doc_id);
    }

    let total = SHARED_CORPUS.len();
    let mut recall_1_hits = 0u32;

    // For each doc, query with its own mock_embedding → must find itself as top-1
    for (doc_id, _) in SHARED_CORPUS {
        let query_vec = mock_embedding(*doc_id);
        let results = engine.local.search(&query_vec, 10);

        assert!(
            !results.is_empty(),
            "HNSW search returned no results for doc_id={doc_id}"
        );

        if results[0].0 == *doc_id {
            recall_1_hits += 1;
        }
    }

    assert_eq!(
        recall_1_hits as usize,
        total,
        "HNSW LocalHnsw recall@1 = {}/{} — expected 1.0 (mock embeddings are unique per doc_id)",
        recall_1_hits,
        total
    );

    // Also verify recall@10 ≥ 0.9
    let mut recall_10_hits = 0u32;
    for (doc_id, _) in SHARED_CORPUS {
        let query_vec = mock_embedding(*doc_id);
        let results = engine.local.search(&query_vec, 10);
        if results.iter().take(10).any(|(id, _)| *id == *doc_id) {
            recall_10_hits += 1;
        }
    }
    assert!(
        recall_10_hits as f64 / total as f64 >= 0.9,
        "HNSW recall@10 below 0.9: {}/{}",
        recall_10_hits,
        total
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::HnswEngine;

    #[test]
    fn verify_engine() {
        let mut engine = HnswEngine::new_local();
        run(&mut engine).expect("HNSW verify failed");
    }
}
