mod commands;
mod compare;
mod demo;
mod gtbench;
mod largebench;
mod msmarco;
mod repl;
mod scalebench;
mod tradeoffbench;

use clap::Parser;
use colbert::{ColBertEngine, JinaColBertClient};
use commands::{BenchTarget, Cli, EngineCmd, TopCommand};
use common::{corpus::SHARED_CORPUS, SuggestionMode, VizRepl};
use hnsw::HnswEngine;
use plaid::PlaidEngine;
use tachiom::TachiomEngine;
use warp::WarpEngine;

/// Pre-fetch Jina ColBERT token embeddings for all shared corpus texts so
/// that demo/repl engines automatically use learned embeddings via the disk cache.
async fn warm_jina_for_corpus() {
    let Some(client) = JinaColBertClient::from_env() else { return };
    let texts: Vec<&str> = SHARED_CORPUS.iter().map(|(_, t)| *t).collect();
    if let Err(e) = client.refresh_cache(&texts).await {
        eprintln!("  [jina] cache warm failed: {e}");
    }
}

fn vocab_path() -> std::path::PathBuf {
    std::env::var("MULTIVECTOR_VOCAB")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap()
                .join("vocab/wordpiece_vocab.txt")
        })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        TopCommand::Demo { name, dry_run } => {
            warm_jina_for_corpus().await;
            demo::run_demo(&name, dry_run, cli.trace_json).await?;
        }
        TopCommand::Repl { engine } => {
            warm_jina_for_corpus().await;
            let vp = vocab_path();
            match engine {
                EngineCmd::Colbert => {
                    let mut eng = ColBertEngine::new(&vp)?;
                    let viz = VizRepl {
                        engine_name: "colbert",
                        suggestions: SuggestionMode::Sequence {
                            suggestions: vec![
                                "index 0".into(),
                                "index 1".into(),
                                "inspect tokens 0".into(),
                                "query river erosion along the bank".into(),
                                "query open a checking account at the bank".into(),
                                "trace maxsim".into(),
                                "inspect tokens 1".into(),
                            ],
                            index: 0,
                        },
                        trace_path: cli.trace_json,
                    };
                    repl::run_repl(&mut eng, viz).await?;
                }
                EngineCmd::Plaid => {
                    let mut eng = PlaidEngine::new(&vp)?;
                    let viz = VizRepl {
                        engine_name: "plaid",
                        suggestions: SuggestionMode::Sequence {
                            suggestions: vec![
                                "index 0".into(),
                                "index 1".into(),
                                "inspect centroids".into(),
                                "query river erosion along the bank".into(),
                            ],
                            index: 0,
                        },
                        trace_path: cli.trace_json,
                    };
                    repl::run_repl(&mut eng, viz).await?;
                }
                EngineCmd::Warp => {
                    let mut eng = WarpEngine::new(&vp)?;
                    let viz = VizRepl {
                        engine_name: "warp",
                        suggestions: SuggestionMode::Sequence {
                            suggestions: vec![
                                "index 0".into(),
                                "index 1".into(),
                                "inspect gather".into(),
                                "query river erosion along the bank".into(),
                            ],
                            index: 0,
                        },
                        trace_path: cli.trace_json,
                    };
                    repl::run_repl(&mut eng, viz).await?;
                }
                EngineCmd::Tachiom => {
                    let mut eng = TachiomEngine::new(&vp)?;
                    let viz = VizRepl {
                        engine_name: "tachiom",
                        suggestions: SuggestionMode::Sequence {
                            suggestions: vec![
                                "index 0".into(),
                                "index 1".into(),
                                "inspect centroids".into(),
                                "query river erosion along the bank".into(),
                            ],
                            index: 0,
                        },
                        trace_path: cli.trace_json,
                    };
                    repl::run_repl(&mut eng, viz).await?;
                }
                EngineCmd::Hnsw => {
                    let mut eng = HnswEngine::new_local();
                    let viz = VizRepl {
                        engine_name: "hnsw",
                        suggestions: SuggestionMode::Sequence {
                            suggestions: vec![
                                "index 0".into(),
                                "index 1".into(),
                                "inspect layers".into(),
                                "query river erosion along the bank".into(),
                                "query open a checking account at the bank".into(),
                                "inspect graph".into(),
                            ],
                            index: 0,
                        },
                        trace_path: cli.trace_json,
                    };
                    repl::run_repl(&mut eng, viz).await?;
                }
            }
        }
        TopCommand::Scale => {
            scalebench::run_scalebench();
        }
        TopCommand::Tradeoff => {
            tradeoffbench::run_tradeoff()?;
        }
        TopCommand::GtBench => {
            let vp = vocab_path();
            gtbench::run_gtbench(&vp).await?;
        }
        TopCommand::LargeBench => {
            let vp = vocab_path();
            largebench::run_large_bench(&vp).await?;
        }
        TopCommand::EmbedMsMarco { collection, out_dir, jina_only, voyage_only, limit } => {
            msmarco::run_embed_msmarco(&collection, &out_dir, jina_only, voyage_only, limit).await?;
        }
        TopCommand::Bench { target } => {
            let runner = bench::BenchRunner::new();
            match target {
                BenchTarget::All => {
                    runner.run_all().await?;
                }
                BenchTarget::CheckSift => {
                    runner.check_sift().await?;
                }
                _ => {
                    println!("Individual engine benchmarks not yet implemented. Use 'bench all'.");
                }
            }
        }
    }
    Ok(())
}
