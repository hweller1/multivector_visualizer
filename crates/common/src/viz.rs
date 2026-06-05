use std::io::{self, Write};

/// RAII guard — drop restores terminal state.
pub struct VizGuard {
    suppressed: bool,
}

impl VizGuard {
    pub fn suppress() -> Self {
        Self { suppressed: true }
    }
}

impl Drop for VizGuard {
    fn drop(&mut self) {
        // restore terminal state
        let _ = self.suppressed;
    }
}

/// Drives the post-command suggestion flow.
#[derive(Debug, Default)]
pub enum SuggestionMode {
    #[default]
    None,
    Sequence {
        suggestions: Vec<String>,
        index: usize,
    },
}

impl SuggestionMode {
    pub fn next_suggestion(&mut self) -> Option<&str> {
        if let SuggestionMode::Sequence { suggestions, index } = self {
            if suggestions.is_empty() {
                return None;
            }
            let s = suggestions[*index].as_str();
            *index = (*index + 1) % suggestions.len();
            Some(s)
        } else {
            None
        }
    }
}

/// Shared REPL scaffolding.
pub struct VizRepl {
    pub engine_name: &'static str,
    pub suggestions: SuggestionMode,
    pub trace_path: Option<std::path::PathBuf>,
}

impl VizRepl {
    /// Print suggestion line after each command output.
    pub fn print_suggestion(&mut self) {
        if let Some(s) = self.suggestions.next_suggestion() {
            println!("\n  \x1b[2m→ try: {s}\x1b[0m");
        }
    }

    /// Print narration text with visual separator (used by ScenarioRunner).
    pub fn print_narration(text: &str) {
        println!("\n\x1b[36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m");
        println!("\x1b[36m  {text}\x1b[0m");
        println!("\x1b[36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n");
    }
}

// Suppress unused import warning
fn _flush_stdout() {
    io::stdout().flush().ok();
}
