use crate::engine::TachiomEngine;
use crate::tac::budget::{EPSILON, THETA};
use anyhow::Result;
use common::{Engine, SHARED_CORPUS, VERIFY_QUERIES};

pub fn run(engine: &mut TachiomEngine) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Index all 20 SHARED_CORPUS docs
    for (doc_id, text) in SHARED_CORPUS {
        rt.block_on(engine.index(*doc_id, text))?;
    }

    let mut recall_1_hits = 0u32;
    let mut recall_10_hits = 0u32;
    let n = VERIFY_QUERIES.len() as f64;

    for (query, expected_top1) in VERIFY_QUERIES {
        let (results, _log) = rt.block_on(engine.query(query, 10))?;
        assert!(!results.is_empty(), "No results for query: '{query}'");

        if results[0].0 == *expected_top1 {
            recall_1_hits += 1;
        }
        if results.iter().take(10).any(|(id, _)| *id == *expected_top1) {
            recall_10_hits += 1;
        }
    }

    assert!(
        recall_1_hits as f64 / n >= 0.9,
        "TACHIOM recall@1 below 0.9 — got {recall_1_hits}/{} queries",
        VERIFY_QUERIES.len()
    );
    assert!(
        recall_10_hits as f64 / n >= 0.9,
        "TACHIOM recall@10 below 0.9 — got {recall_10_hits}/{}",
        VERIFY_QUERIES.len()
    );

    // For token types with enough embeddings, assert κⱼ is in [EPSILON, THETA]
    for (token_type, centroids) in &engine.centroids_by_type {
        let kj = centroids.len() as u32;
        // Skip token types where the corpus is too small to produce EPSILON centroids
        let emb_count = engine
            .token_type_embeddings
            .get(token_type)
            .map(|v| v.len())
            .unwrap_or(0);
        if emb_count < EPSILON as usize {
            continue; // Can't form EPSILON clusters from fewer points
        }
        assert!(
            (EPSILON..=THETA).contains(&kj),
            "Token type {token_type} (embs={emb_count}) has κⱼ={kj} outside [{EPSILON}, {THETA}]"
        );
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
                std::env::current_dir()
                    .unwrap()
                    .join("vocab/wordpiece_vocab.txt")
            });
        let mut engine = TachiomEngine::new(&vocab_path).expect("TachiomEngine::new failed");
        run(&mut engine).expect("verify failed");
    }
}
