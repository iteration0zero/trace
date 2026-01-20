//! # Trace - Execution Path Recording
//!
//! This module provides data structures for recording Triage execution paths.
//! Traces enable structural sensitivity analysis for gradient-guided synthesis.

use crate::arena::NodeId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Branch {
    Leaf,
    Stem,
    Fork,
}

#[derive(Debug, Clone)]
pub struct TraceStep {
    pub node_id: NodeId, 
    pub subject_id: NodeId,
    pub branch: Branch,
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub steps: Vec<TraceStep>,
}

impl Trace {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }
    
    pub fn record(&mut self, subject: NodeId, branch: Branch) {
        self.steps.push(TraceStep {
            node_id: NodeId(0), // Placeholder
            subject_id: subject,
            branch,
        });
    }
}
