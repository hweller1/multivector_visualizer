mod commands;
mod repl;
mod demo;
mod compare;

use clap::Parser;
use commands::{Cli, EngineCmd, TopCommand};
use common::{SuggestionMode, VizRepl};
use colbert::ColBertEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        TopCommand::Demo { name, dry_run } => {
            demo::run_demo(&name, dry_run, cli.trace_json).await?;
        }
        TopCommand::Repl { engine } => {
            match engine {
                EngineCmd::Colbert => {
                    let vocab_path = std::env::var("MULTIVECTOR_VOCAB")
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| {
                            std::env::current_dir()
                                .unwrap()
                                .join("vocab/wordpiece_vocab.txt")
                        });
                    let mut engine = ColBertEngine::new(&vocab_path)?;
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
                    repl::run_repl(&mut engine, viz).await?;
                }
                _ => {
                    println!("repl for this engine not yet implemented");
                }
            }
        }
        TopCommand::Bench { .. } => {
            println!("bench not yet implemented");
        }
    }
    Ok(())
}
