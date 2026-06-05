use anyhow::Result;
use common::{Engine, SHARED_CORPUS, VERIFY_QUERIES};
use crate::engine::ColBertEngine;

pub fn run(engine: &mut ColBertEngine) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Index all 20 SHARED_CORPUS docs
    for (doc_id, text) in SHARED_CORPUS {
        rt.block_on(engine.index(*doc_id, text))?;
    }

    // Run all 20 VERIFY_QUERIES; check recall@1 and recall@10
    let mut recall_1_hits = 0u32;
    let mut recall_10_hits = 0u32;
    let n = VERIFY_QUERIES.len() as f64;

    for (query, expected_top1) in VERIFY_QUERIES {
        let (results, _) = rt.block_on(engine.query(query, 10))?;
        assert!(
            !results.is_empty(),
            "No results for query: '{query}'"
        );
        if results[0].0 == *expected_top1 {
            recall_1_hits += 1;
        }
        if results.iter().take(10).any(|(id, _)| *id == *expected_top1) {
            recall_10_hits += 1;
        }
    }

    assert_eq!(
        recall_1_hits as f64 / n,
        1.0,
        "recall@1 not 1.0 — got {recall_1_hits}/{} queries",
        VERIFY_QUERIES.len()
    );
    assert!(
        recall_10_hits as f64 / n >= 0.9,
        "recall@10 below 0.9 — got {recall_10_hits}/{}",
        VERIFY_QUERIES.len()
    );

    // Determinism check: second engine, first query must match
    let vocab_path = std::env::var("MULTIVECTOR_VOCAB")
        .map(|s| std::path::PathBuf::from(s))
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap()
                .join("vocab/wordpiece_vocab.txt")
        });
    let mut engine2 = ColBertEngine::new(&vocab_path)?;
    for (doc_id, text) in SHARED_CORPUS {
        rt.block_on(engine2.index(*doc_id, text))?;
    }
    let (r1, _) = rt.block_on(engine.query(VERIFY_QUERIES[0].0, 10))?;
    let (r2, _) = rt.block_on(engine2.query(VERIFY_QUERIES[0].0, 10))?;
    assert_eq!(r1, r2, "non-deterministic output across identical runs");

    // p99 latency < 100ms
    let mut latencies_ms: Vec<f64> = Vec::new();
    for (query, _) in VERIFY_QUERIES {
        let t0 = std::time::Instant::now();
        let _ = rt.block_on(engine.query(query, 10))?;
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99_idx = ((0.99 * latencies_ms.len() as f64) as usize).min(latencies_ms.len() - 1);
    let p99 = latencies_ms[p99_idx];
    assert!(p99 < 100.0, "p99 latency {p99:.1}ms exceeds 100ms limit");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_engine() {
        let vocab_path = std::env::var("MULTIVECTOR_VOCAB")
            .map(|s| std::path::PathBuf::from(s))
            .unwrap_or_else(|_| {
                std::env::current_dir()
                    .unwrap()
                    .join("vocab/wordpiece_vocab.txt")
            });
        let mut engine = ColBertEngine::new(&vocab_path)
            .expect("ColBertEngine::new failed");
        run(&mut engine).expect("verify failed");
    }
}
