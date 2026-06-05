use colbert::ColBertEngine;
use common::Engine;
use hnsw::HnswEngine;
use plaid::PlaidEngine;
use tachiom::TachiomEngine;
use warp::WarpEngine;

pub async fn run_compare() -> anyhow::Result<()> {
    let vocab_path = std::env::var("MULTIVECTOR_VOCAB")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap()
                .join("vocab/wordpiece_vocab.txt")
        });

    let query = "bank interest rate";

    println!("Indexing all 20 corpus docs into each engine...");

    // Index all engines
    let mut colbert = ColBertEngine::new(&vocab_path)?;
    let mut plaid = PlaidEngine::new(&vocab_path)?;
    let mut warp_engine = WarpEngine::new(&vocab_path)?;
    let mut tachiom = TachiomEngine::new(&vocab_path)?;
    let mut hnsw_engine = HnswEngine::new_local();

    for (doc_id, text) in common::SHARED_CORPUS {
        colbert.index(*doc_id, text).await?;
        plaid.index(*doc_id, text).await?;
        warp_engine.index(*doc_id, text).await?;
        tachiom.index(*doc_id, text).await?;
        hnsw_engine.index(*doc_id, text).await?;
    }

    println!("\nQuery: \"{query}\"");
    println!();

    // Print header
    println!(
        "{:<12} | {:<20} | {:<30} | Top-1 Result",
        "Engine", "index structure", "Pipeline Stages"
    );
    println!("{}", "-".repeat(85));

    // ColBERT
    let (colbert_results, _) = colbert.query(query, 5).await?;
    let colbert_top1 = colbert_results
        .first()
        .map(|(id, s)| format!("doc={id} ({s:.3})"))
        .unwrap_or_default();
    println!(
        "{:<12} | {:<20} | {:<30} | {}",
        "colbert", "flat token list", "Tokenize→Embed→MaxSim", colbert_top1
    );

    // PLAID
    let (plaid_results, _) = plaid.query(query, 5).await?;
    let plaid_top1 = plaid_results
        .first()
        .map(|(id, s)| format!("doc={id} ({s:.3})"))
        .unwrap_or_default();
    println!(
        "{:<12} | {:<20} | {:<30} | {}",
        "plaid", "centroid inverted idx", "CentANN→Expand→MaxSim", plaid_top1
    );

    // WARP
    let (warp_results, _) = warp_engine.query(query, 5).await?;
    let warp_top1 = warp_results
        .first()
        .map(|(id, s)| format!("doc={id} ({s:.3})"))
        .unwrap_or_default();
    println!(
        "{:<12} | {:<20} | {:<30} | {}",
        "warp", "xtr token registry", "XtrScore→Gather→Refine", warp_top1
    );

    // TACHIOM
    let (tachiom_results, _) = tachiom.query(query, 5).await?;
    let tachiom_top1 = tachiom_results
        .first()
        .map(|(id, s)| format!("doc={id} ({s:.3})"))
        .unwrap_or_default();
    println!(
        "{:<12} | {:<20} | {:<30} | {}",
        "tachiom", "per-type centroids", "TAC→CentANN→MaxSim", tachiom_top1
    );

    // HNSW
    let (hnsw_results, _) = hnsw_engine.query(query, 5).await?;
    let hnsw_top1 = hnsw_results
        .first()
        .map(|(id, s)| format!("doc={id} ({s:.3})"))
        .unwrap_or_default();
    println!(
        "{:<12} | {:<20} | {:<30} | {}",
        "hnsw", "HNSW graph", "GreedyANN (single-vec)", hnsw_top1
    );

    println!();
    println!("Memory per token:");
    println!("  colbert:  128 × 4 bytes = 512 bytes/token (dense f32)");
    println!("  plaid:    512 + centroid overhead");
    println!("  warp:     512 + xtr vocab registry");
    println!("  tachiom:  512 + per-type centroid sets (budget-allocated)");
    println!("  hnsw:     1536 × 4 bytes = 6144 bytes/doc (single-vector)");

    // Write compare.json
    std::fs::create_dir_all("output")?;
    let compare_data = serde_json::json!({
        "query": query,
        "results": {
            "hnsw": hnsw_results,
            "colbert": colbert_results,
            "plaid": plaid_results,
            "warp": warp_results,
            "tachiom": tachiom_results,
        }
    });
    std::fs::write(
        "output/compare.json",
        serde_json::to_string_pretty(&compare_data)?,
    )?;
    println!("\nResults written to output/compare.json");

    Ok(())
}
