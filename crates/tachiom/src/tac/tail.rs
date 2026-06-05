use common::trace::TraceEvent;
use std::collections::HashMap;

pub const MU: u32 = 128;
pub const TAU: u32 = 256;

#[derive(Debug, Clone, PartialEq)]
pub enum TailClass {
    /// freq < MU
    Tail,
    /// MU <= freq < TAU
    Normal,
    /// freq >= TAU
    Heavy,
}

impl From<TailClass> for common::trace::TailClass {
    fn from(tc: TailClass) -> Self {
        match tc {
            TailClass::Tail => common::trace::TailClass::Tail,
            TailClass::Normal => common::trace::TailClass::Normal,
            TailClass::Heavy => common::trace::TailClass::Heavy,
        }
    }
}

pub struct TailHandler {
    pub freq: HashMap<u32, u32>,
}

impl TailHandler {
    pub fn new() -> Self {
        Self {
            freq: HashMap::new(),
        }
    }

    pub fn update(&mut self, token_id: u32) {
        *self.freq.entry(token_id).or_insert(0) += 1;
    }

    pub fn classify(&self, token_type_id: u32) -> TailClass {
        let f = self.freq.get(&token_type_id).copied().unwrap_or(0);
        if f >= TAU {
            TailClass::Heavy
        } else if f >= MU {
            TailClass::Normal
        } else {
            TailClass::Tail
        }
    }

    pub fn trace_all(&self) -> Vec<TraceEvent> {
        self.freq
            .iter()
            .map(|(token_id, &count)| {
                let class = self.classify(*token_id);
                TraceEvent::TailHandle {
                    token_type: token_id.to_string(),
                    count,
                    classification: class.into(),
                }
            })
            .collect()
    }
}

impl Default for TailHandler {
    fn default() -> Self {
        Self::new()
    }
}
