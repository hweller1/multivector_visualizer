use common::trace::TraceEvent;
use std::collections::HashMap;

pub const EPSILON: u32 = 4;
pub const THETA: u32 = 39;

pub struct BudgetReconciler {
    pub total_budget: u32,
}

impl BudgetReconciler {
    pub fn new(total_budget: u32) -> Self {
        Self { total_budget }
    }

    /// Allocate budget proportionally, clamped to [EPSILON, THETA].
    /// Returns HashMap<token_type, raw_kappa> and emits BudgetBound events.
    pub fn allocate(&self, weights: &HashMap<u32, f32>) -> HashMap<u32, u32> {
        let sum_w: f32 = weights.values().sum();
        if sum_w <= 0.0 {
            return weights.keys().map(|&k| (k, EPSILON)).collect();
        }

        let mut result = HashMap::new();
        for (&token_type, &w) in weights {
            let raw = self.total_budget as f32 * w / sum_w;
            let clamped = (raw.round() as u32).clamp(EPSILON, THETA);
            result.insert(token_type, clamped);
        }
        result
    }

    /// Emit BudgetBound trace events.
    pub fn trace_allocate(&self, weights: &HashMap<u32, f32>) -> Vec<TraceEvent> {
        let sum_w: f32 = weights.values().sum();
        weights
            .iter()
            .map(|(&token_type, &w)| {
                let raw_kappa = if sum_w > 0.0 {
                    self.total_budget as f32 * w / sum_w
                } else {
                    EPSILON as f32
                };
                let floored = (raw_kappa.floor() as u32).max(EPSILON);
                let ceiled = (raw_kappa.ceil() as u32).min(THETA);
                let final_kappa = (raw_kappa.round() as u32).clamp(EPSILON, THETA);
                TraceEvent::BudgetBound {
                    token_type: token_type.to_string(),
                    raw_kappa,
                    floored,
                    ceiled,
                    final_kappa,
                }
            })
            .collect()
    }

    /// Redistribute leftover budget to types below THETA, by weight rank.
    pub fn reconcile(
        &self,
        raw: &mut HashMap<u32, u32>,
        weights: &HashMap<u32, f32>,
    ) -> Vec<TraceEvent> {
        let allocated: u32 = raw.values().sum();
        let budget = self.total_budget;

        let mut redistributed = 0u32;
        if allocated < budget {
            let mut leftover = budget - allocated;
            // Sort by weight descending, give to those below THETA
            let mut by_weight: Vec<(u32, f32)> = weights.iter().map(|(&k, &w)| (k, w)).collect();
            by_weight.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (token_type, _) in &by_weight {
                if leftover == 0 {
                    break;
                }
                let current = raw.get(token_type).copied().unwrap_or(EPSILON);
                if current < THETA {
                    let can_add = THETA - current;
                    let add = can_add.min(leftover);
                    *raw.entry(*token_type).or_insert(EPSILON) += add;
                    leftover -= add;
                    redistributed += add;
                }
            }
        }

        vec![TraceEvent::BudgetReconcile {
            total_budget: budget,
            allocated,
            redistributed,
        }]
    }
}
