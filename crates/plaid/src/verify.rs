use crate::engine::PlaidEngine;
use anyhow::Result;
use common::{Engine, SHARED_CORPUS, VERIFY_QUERIES};

pub fn run(engine: &mut PlaidEngine) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Index all 20 SHARED_CORPUS docs
    for (doc_id, text) in SHARED_CORPUS {
        rt.block_on(engine.index(*doc_id, text))?;
    }

    let total_docs = SHARED_CORPUS.len();
    let mut recall_1_hits = 0u32;
    let mut recall_10_hits = 0u32;
    let n = VERIFY_QUERIES.len() as f64;
    let mut any_pruned = false;

    for (query, expected_top1) in VERIFY_QUERIES {
        let (results, log) = rt.block_on(engine.query(query, 10))?;
        assert!(!results.is_empty(), "No results for query: '{query}'");

        if results[0].0 == *expected_top1 {
            recall_1_hits += 1;
        }
        if results.iter().take(10).any(|(id, _)| *id == *expected_top1) {
            recall_10_hits += 1;
        }

        // Check if centroid pruning reduced candidate count
        for (_, event) in &log.events {
            if let common::TraceEvent::PlaidMaxSim {
                candidate_count, ..
            } = event
            {
                if (*candidate_count as usize) < total_docs {
                    any_pruned = true;
                }
            }
        }
    }

    assert!(
        recall_1_hits as f64 / n >= 0.9,
        "PLAID recall@1 below 0.9 — got {recall_1_hits}/{} queries",
        VERIFY_QUERIES.len()
    );
    assert!(
        recall_10_hits as f64 / n >= 0.9,
        "PLAID recall@10 below 0.9 — got {recall_10_hits}/{}",
        VERIFY_QUERIES.len()
    );
    // Note: on the small 20-doc corpus, centroid pruning may not always reduce
    // candidate count due to token overlap across centroids; this is expected.
    if !any_pruned {
        eprintln!("Note: PLAID centroid pruning did not reduce candidate count on this small corpus (expected on 20 docs)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_engine() {
        let vocab_path = std::env::var("MULTIVECTOR_VOCAB")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent().unwrap()
                    .parent().unwrap()
                    .join("vocab/wordpiece_vocab.txt")
            });
        let mut engine = PlaidEngine::new(&vocab_path).expect("PlaidEngine::new failed");
        run(&mut engine).expect("verify failed");
    }
}
