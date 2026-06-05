use common::token::TOKEN_DIM;

pub struct PQLevel {
    pub dimensions: u32,
    pub num_subquantizers: u32,
    pub code_bits: u8,
}

pub struct HierarchicalPQ {
    pub levels: [PQLevel; 3],
}

impl HierarchicalPQ {
    pub fn new() -> Self {
        Self {
            levels: [
                PQLevel {
                    dimensions: 128,
                    num_subquantizers: 8,
                    code_bits: 1,
                },
                PQLevel {
                    dimensions: 64,
                    num_subquantizers: 4,
                    code_bits: 1,
                },
                PQLevel {
                    dimensions: 32,
                    num_subquantizers: 2,
                    code_bits: 1,
                },
            ],
        }
    }

    /// Format 3-level table: Level | Dims | Subq | CodeBits
    pub fn inspect(&self) -> String {
        let mut out = String::from("Level | Dims | Subq | CodeBits\n");
        out.push_str("------+------+------+---------\n");
        for (i, level) in self.levels.iter().enumerate() {
            out.push_str(&format!(
                "  {:2}  | {:4} | {:4} | {:8}\n",
                i + 1,
                level.dimensions,
                level.num_subquantizers,
                level.code_bits
            ));
        }
        out
    }

    /// Training is a no-op for this demo implementation.
    pub fn train(&mut self, _data: &[[f32; TOKEN_DIM]]) {
        // No-op: PQ training not implemented in the demo
    }

    /// Encode a vector into bytes (trivial: just take first N dims as bytes).
    pub fn encode(&self, vec: &[f32; TOKEN_DIM]) -> Vec<u8> {
        // Simple demo encoding: quantize each f32 to u8
        vec.iter().map(|&x| ((x + 1.0) * 127.5) as u8).collect()
    }
}

impl Default for HierarchicalPQ {
    fn default() -> Self {
        Self::new()
    }
}
