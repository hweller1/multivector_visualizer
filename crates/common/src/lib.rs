pub mod bench_types;
pub mod corpus;
pub mod engine;
pub mod scenario;
pub mod token;
pub mod trace;
pub mod viz;
pub mod visualize;

pub use bench_types::{BenchResult, BuildStats, PlotData};
pub use corpus::{SHARED_CORPUS, VERIFY_QUERIES};
pub use engine::Engine;
pub use scenario::{CorpusDef, ScenarioFile, ScenarioRunner, StepDef};
pub use token::{RandomProjection, TokenMatrix, Tokenizer, WordPieceTokenizer, TOKEN_DIM};
pub use trace::{JsonTracer, OpTiming, TachiomTimings, TailClass, TraceEvent, TraceLog};
pub use viz::{SuggestionMode, VizGuard, VizRepl};
pub use visualize::{render_trace, viz_delay_ms};
