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
                    // K discards the third arg (z). Blame goes to y (first arg).
                    // 44yz → y: args[0] = y, args[1] = z
                    if !event.args.is_empty() {
                        *self.blame.entry(event.args[0]).or_insert(0.0) += result_blame;
                        *source_blame.entry(event.args[0]).or_insert(0.0) += result_blame;
                    }
                    // z gets zero blame (discarded)
                }
                RuleId::S => {
                    // S duplicates z: xz(yz)
                    // Both x and y get blame, z gets blame from both uses
                    // args[0] = x, args[1] = y, args[2] = z
                    if event.args.len() >= 3 {
                        let split = result_blame / 2.0;
                        *self.blame.entry(event.args[0]).or_insert(0.0) += split;
                        *self.blame.entry(event.args[1]).or_insert(0.0) += split;
                        // z gets full blame (used twice)
                        *self.blame.entry(event.args[2]).or_insert(0.0) += result_blame;
                        *source_blame.entry(event.args[0]).or_insert(0.0) += split;
                        *source_blame.entry(event.args[1]).or_insert(0.0) += split;
                        *source_blame.entry(event.args[2]).or_insert(0.0) += result_blame;
                    }
                }
                RuleId::TriageLeaf => {
                    // 4(4wx)y4 → w: only w contributed
                    // args[0] = w
                    if !event.args.is_empty() {
                        *self.blame.entry(event.args[0]).or_insert(0.0) += result_blame;
                        *source_blame.entry(event.args[0]).or_insert(0.0) += result_blame;
                    }
                }
                RuleId::TriageStem => {
                    // 4(4wx)y(4u) → xu: x and u contribute
                    // args[0] = x, args[1] = u
                    if event.args.len() >= 2 {
                        let split = result_blame / 2.0;
                        *self.blame.entry(event.args[0]).or_insert(0.0) += split;
                        *self.blame.entry(event.args[1]).or_insert(0.0) += split;
                        *source_blame.entry(event.args[0]).or_insert(0.0) += split;
                        *source_blame.entry(event.args[1]).or_insert(0.0) += split;
                    }
                }
                RuleId::TriageFork => {
                    // 4(4wx)y(4uv) → yuv: y, u, v contribute
                    // args[0] = y, args[1] = u, args[2] = v
                    if event.args.len() >= 3 {
                        let split = result_blame / 3.0;
                        for i in 0..3 {
                            *self.blame.entry(event.args[i]).or_insert(0.0) += split;
                            *source_blame.entry(event.args[i]).or_insert(0.0) += split;
                        }
                    }
                }
                RuleId::App | RuleId::Prim => {
                    // Pass blame to all args equally
                    if !event.args.is_empty() {
                        let split = result_blame / event.args.len() as f64;
                        for arg in &event.args {
                            *self.blame.entry(*arg).or_insert(0.0) += split;
                            *source_blame.entry(*arg).or_insert(0.0) += split;
                        }
                    }
                }
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
