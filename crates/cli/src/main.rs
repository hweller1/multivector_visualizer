mod commands;
mod compare;
mod demo;
mod repl;
mod scalebench;
mod tradeoffbench;

use clap::Parser;
use colbert::ColBertEngine;
use commands::{BenchTarget, Cli, EngineCmd, TopCommand};
use common::{SuggestionMode, VizRepl};
use hnsw::HnswEngine;
use plaid::PlaidEngine;
use tachiom::TachiomEngine;
use warp::WarpEngine;

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
            demo::run_demo(&name, dry_run, cli.trace_json).await?;
        }
        TopCommand::Repl { engine } => {
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
