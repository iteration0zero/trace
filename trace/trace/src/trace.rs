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

use std::collections::HashMap;

/// Identifies which reduction rule fired
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleId {
    /// 44yz → y (K-like: discard z)
    K,
    /// 4(4x)yz → xz(yz) (S-like: duplicate z)
    S,
    /// 4(4wx)y4 → w (Triage: leaf case)
    TriageLeaf,
    /// 4(4wx)y(4u) → xu (Triage: stem case)
    TriageStem,
    /// 4(4wx)y(4uv) → yuv (Triage: fork case)
    TriageFork,
    /// Application (creates structure, no "computation")
    App,
    /// Primitive operation (Add, Mul, etc.)
    Prim,
}

/// A single event in the execution trace
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Which rule fired
    pub rule: RuleId,
    /// The redex (function position) node
    pub redex: NodeId,
    /// The argument(s) involved
    pub args: Vec<NodeId>,
    /// The result node
    pub result: NodeId,
    /// Step number in the execution
    pub step: usize,
}

/// Execution trace: a sequence of reduction events
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    /// Ordered list of events
    pub events: Vec<TraceEvent>,
    /// Final result node
    pub result: Option<NodeId>,
    /// Blame map: node -> accumulated blame score
    pub blame: HashMap<NodeId, f64>,
}

impl ExecutionTrace {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record(&mut self, rule: RuleId, redex: NodeId, args: Vec<NodeId>, result: NodeId) {
        let step = self.events.len();
        self.events.push(TraceEvent {
            rule,
            redex,
            args,
            result,
            step,
        });
    }
    
    pub fn set_result(&mut self, result: NodeId) {
        self.result = Some(result);
    }
    
    /// Initialize blame on a set of output nodes
    pub fn seed_blame(&mut self, nodes: &[(NodeId, f64)]) {
        for (node, blame) in nodes {
            *self.blame.entry(*node).or_insert(0.0) += blame;
        }
    }
    
    /// Propagate blame backwards through the trace
    /// 
    /// Returns a map of source node -> total blame
    pub fn backpropagate(&mut self) -> HashMap<NodeId, f64> {
        let mut source_blame: HashMap<NodeId, f64> = HashMap::new();
        
        // Walk events in reverse order
        for event in self.events.iter().rev() {
            // Get blame on the result of this event
            let result_blame = self.blame.get(&event.result).copied().unwrap_or(0.0);
            
            if result_blame == 0.0 {
                continue;
            }
            
            match event.rule {
                RuleId::K => {
                    // Focus blame on the combinator structure (redex) only.
                    *self.blame.entry(event.redex).or_insert(0.0) += result_blame;
                    *source_blame.entry(event.redex).or_insert(0.0) += result_blame;
                }
                RuleId::S => {
                    // Focus blame on the combinator structure (redex) only.
                    *self.blame.entry(event.redex).or_insert(0.0) += result_blame;
                    *source_blame.entry(event.redex).or_insert(0.0) += result_blame;
                }
                RuleId::TriageLeaf => {
                    // Focus blame on the combinator structure (redex) only.
                    *self.blame.entry(event.redex).or_insert(0.0) += result_blame;
                    *source_blame.entry(event.redex).or_insert(0.0) += result_blame;
                }
                RuleId::TriageStem => {
                    // Focus blame on the combinator structure (redex) only.
                    *self.blame.entry(event.redex).or_insert(0.0) += result_blame;
                    *source_blame.entry(event.redex).or_insert(0.0) += result_blame;
                }
                RuleId::TriageFork => {
                    // Focus blame on the combinator structure (redex) only.
                    *self.blame.entry(event.redex).or_insert(0.0) += result_blame;
                    *source_blame.entry(event.redex).or_insert(0.0) += result_blame;
                }
                RuleId::App | RuleId::Prim => {
                    // Blame only the redex (program structure), avoid pushing into inputs
                    *self.blame.entry(event.redex).or_insert(0.0) += result_blame;
                    *source_blame.entry(event.redex).or_insert(0.0) += result_blame;
                }
            }
        }
        // Any remaining blame that never appeared as a result in the trace
        // should still be attributed to its node (static structure).
        for (node, b) in self.blame.iter() {
            if *b > 0.0 {
                *source_blame.entry(*node).or_insert(0.0) += *b;
            }
        }

        source_blame
    }
    
    /// Get the top-k nodes by blame
    pub fn top_blamed_nodes(&self, k: usize) -> Vec<(NodeId, f64)> {
        let mut sorted: Vec<_> = self.blame.iter().map(|(&n, &b)| (n, b)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }
}
