//! # Sensitivity - Structural Gradient Computation
//!
//! This module implements "blame assignment" for gradient-guided program synthesis.
//! It computes sensitivity signals that guide structural mutations.

use crate::arena::NodeId;
use crate::trace::Trace;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Default)]
pub struct SubGradients {
    pub numeric: f64,
    pub expansion: f64,
    pub pruning: f64,
}

pub struct SensitivityMap {
    pub gradients: HashMap<NodeId, SubGradients>,
}

impl SensitivityMap {
    pub fn new() -> Self {
        Self { gradients: HashMap::new() }
    }
    
    pub fn add(&mut self, node: NodeId, val: f64) {
        // Default accumulation logic: add to numeric (for floats)
        let entry = self.gradients.entry(node).or_default();
        entry.numeric += val;
    }
    
    pub fn add_struct(&mut self, node: NodeId, expansion: f64, pruning: f64) {
        let entry = self.gradients.entry(node).or_default();
        entry.expansion += expansion;
        entry.pruning += pruning;
    }
    
    pub fn get(&self, node: NodeId) -> SubGradients {
        *self.gradients.get(&node).unwrap_or(&SubGradients::default())
    }
}

/// Compute Structural Sensitivity from Trace and Loss.
/// Implements "Differential Triage" signals.
pub fn compute_sensitivity(trace: &Trace, loss: f64) -> SensitivityMap {
    let mut map = SensitivityMap::new();
    
    // Decay factor for backpropagation
    let decay = 0.9;
    let mut current_grad = loss;
    
    // Iterate backward
    for step in trace.steps.iter().rev() {
        let subject = step.subject_id;
        
        // Router Sensitivity:
        match step.branch {
            crate::trace::Branch::Leaf => {
                // If we are at Leaf and have error, likely "Missing Computation".
                // Signal: Expansion (Leaf -> Stem)
                map.add_struct(subject, current_grad, 0.0);
            },
            crate::trace::Branch::Stem => {
                // Determine direction based on local heuristics?
                // For now, assign mixed signal. Paper heuristic:
                // "Wrong Branch" -> Swapping (handled by mutation search logic usually)
                // Here we signal that this structure is hot.
                // Assuming Growth is usually needed for complex tasks:
                map.add_struct(subject, current_grad * 0.5, current_grad * 0.5);
            },
            crate::trace::Branch::Fork => {
                // If Fork is wrong, maybe we over-complicated output?
                // Signal: Pruning (Fork -> Stem)
                map.add_struct(subject, 0.0, current_grad);
            }
        }
        
        // Also accrue numeric gradient for direct parameter tuning if node is Float
        map.add(subject, current_grad);
        
        // Decay
        current_grad *= decay;
    }
    
    map
}
