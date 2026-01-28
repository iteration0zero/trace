use std::default::Default;

#[derive(Debug, Clone)]
pub struct CounterfactualConfig {
    pub max_iterations: usize,
    // Hard cap on reduction steps per evaluation (no scaling).
    pub max_eval_steps: usize,
    pub max_eval_nodes: usize,
    pub max_global_nodes: usize,
    pub max_ted_cells: usize,
    pub edit_threshold_base: f64,
    pub variance_weight: f64,
    pub output_collision_penalty: f64,
    pub max_edits_per_iter: usize,
    pub max_edits_per_path: usize,
    pub max_replacement_nodes: usize,
    pub max_edit_depth: usize,
    pub max_candidate_evals: usize,
    pub max_candidates_per_iter: usize,
    pub memo_budget_bytes: usize,
    pub verbose: bool,
    pub debug: bool,
}

impl Default for CounterfactualConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_eval_steps: 5000,
            max_eval_nodes: 100_000,
            max_global_nodes: 2_000_000,
            max_ted_cells: 50_000_000,
            edit_threshold_base: 0.5,
            variance_weight: 0.5,
            output_collision_penalty: 100.0,
            max_edits_per_iter: 10,
            max_edits_per_path: 1000,
            max_replacement_nodes: 1024,
            max_edit_depth: 5,
            max_candidate_evals: 100_000,
            max_candidates_per_iter: 5_000,
            memo_budget_bytes: 512 * 1024 * 1024,
            verbose: true,
            debug: false,
        }
    }
}
