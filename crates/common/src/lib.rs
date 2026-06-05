pub mod trace;
pub mod bench_types;
pub mod corpus;
pub mod token;
pub mod engine;
pub mod viz;
pub mod scenario;

pub use trace::{TraceEvent, TraceLog, TailClass, TachiomTimings, JsonTracer};
pub use engine::Engine;
pub use token::{TOKEN_DIM, TokenMatrix, Tokenizer, WordPieceTokenizer, RandomProjection};
pub use viz::{VizRepl, VizGuard, SuggestionMode};
pub use scenario::{ScenarioRunner, ScenarioFile, StepDef, CorpusDef};
pub use bench_types::{BenchResult, BuildStats, PlotData};
pub use corpus::{SHARED_CORPUS, VERIFY_QUERIES};
