use crate::arena::{Primitive, NodeId, Graph, Node};
use std::collections::HashMap;

/// Logits for [Leaf, Stem, Fork, Lib0, Lib1, ...]
pub type Logits = [f64; 3];  // Base logits (kept for backprop compatibility)
pub type Probs = [f64; 3];

/// Index in the RunGraph (Dynamic Graph)
pub type RunId = usize;

/// Learnable Skeleton Node
#[derive(Debug, Clone)]
pub struct ParamNode {
    pub logits: Logits,           // Base logits [Leaf, Stem, Fork]
    pub lib_logits: Vec<f64>,     // Additional logits for library functions
    pub index: usize,             // Original index in Skeleton
    // Pointers to children in Skeleton
    pub stem_child: usize, 
    pub fork_left: usize,
    pub fork_right: usize,
}

impl ParamNode {
    pub fn new(index: usize, stem_child: usize, fork_left: usize, fork_right: usize) -> Self {
        Self {
            logits: [0.0, 0.0, 0.0],
            lib_logits: Vec::new(),
            index,
            stem_child,
            fork_left,
            fork_right
        }
    }
    
    pub fn new_with_library(index: usize, stem_child: usize, fork_left: usize, fork_right: usize, lib_size: usize) -> Self {
        Self {
            logits: [0.0, 0.0, 0.0],
            lib_logits: vec![0.0; lib_size],
            index,
            stem_child,
            fork_left,
            fork_right
        }
    }
    
    /// Get total number of choices (3 base + library size)
    pub fn num_choices(&self) -> usize {
        3 + self.lib_logits.len()
    }
    
    /// Get all logits as a single vector
    pub fn all_logits(&self) -> Vec<f64> {
        let mut all = self.logits.to_vec();
        all.extend(&self.lib_logits);
        all
    }
}

/// SoftGraph container for the Skeleton
#[derive(Debug, Clone)]
pub struct SoftGraph {
    pub nodes: Vec<ParamNode>,
    pub root: usize,
    pub library: Vec<NodeId>,  // Library of learned programs
}

impl SoftGraph {
    pub fn new_complete(depth: usize) -> Self {
        Self::new_complete_with_library(depth, Vec::new())
    }
    
    pub fn new_complete_with_library(depth: usize, library: Vec<NodeId>) -> Self {
        let lib_size = library.len();
        let mut nodes = Vec::new();
        let root = Self::build_rec_with_lib(&mut nodes, depth, lib_size);
        Self { nodes, root, library }
    }
    
    fn build_rec_with_lib(nodes: &mut Vec<ParamNode>, depth: usize, lib_size: usize) -> usize {
        let idx = nodes.len();
        nodes.push(ParamNode::new_with_library(idx, 0, 0, 0, lib_size));
        
        if depth > 0 {
             let s = Self::build_rec_with_lib(nodes, depth - 1, lib_size);
             let l = Self::build_rec_with_lib(nodes, depth - 1, lib_size);
             let r = Self::build_rec_with_lib(nodes, depth - 1, lib_size);
             let node = &mut nodes[idx];
             node.stem_child = s;
             node.fork_left = l;
             node.fork_right = r;
        } else {
             let node = &mut nodes[idx];
             node.stem_child = idx;
             node.fork_left = idx;
             node.fork_right = idx;
        }
        idx
    }
    
    /// Compute self-similarity score for the skeleton
    /// Measures similarity between left and right children of Fork nodes
    /// Returns a value in [0, 1] where 1 = perfectly self-similar
    pub fn compute_self_similarity(&self) -> f64 {
        let mut total_sim = 0.0;
        let mut count = 0;
        
        for node in &self.nodes {
            // For each node, compute similarity between its Fork children
            let left_idx = node.fork_left;
            let right_idx = node.fork_right;
            
            // Skip if pointing to self (leaf node)
            if left_idx == node.index || right_idx == node.index {
                continue;
            }
            
            // Compare probability distributions of left and right children
            let left_node = &self.nodes[left_idx];
            let right_node = &self.nodes[right_idx];
            
            let left_logits = left_node.all_logits();
            let right_logits = right_node.all_logits();
            
            // Note: We need a dynamic softmax that handles Vec<f64>
            // For now, we manually compute it or add a helper
            let left_probs = softmax_vec(&left_logits);
            let right_probs = softmax_vec(&right_logits);
            
            // Cosine similarity between probability vectors
            // Truncate to min length if they differ (should be same structure though)
            let len = std::cmp::min(left_probs.len(), right_probs.len());
            
            let dot: f64 = left_probs.iter().zip(right_probs.iter()).take(len)
                .map(|(a, b)| a * b)
                .sum();
            let mag_l: f64 = left_probs.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mag_r: f64 = right_probs.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            let sim = if mag_l > 1e-9 && mag_r > 1e-9 {
                dot / (mag_l * mag_r)
            } else {
                0.0
            };
            
            total_sim += sim;
            count += 1;
        }
        
        if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        }
    }
}

/// Dynamic Runtime Node
#[derive(Debug, Clone)]
pub enum RunNode {
    /// A generic skeleton node (superposition)
    Skeleton(usize), 
    /// Concrete dynamic structures
    Leaf,
    Stem(RunId),
    Fork(RunId, RunId),
    /// Primitive Operator
    Prim(Primitive),
    /// Float Value (Pass-through)
    Float(f64),
    /// Weighted Sum (Superposition produced during runtime)
    Superposition(Vec<(RunId, f64)>),
}

/// Operation Trace for Backprop
#[derive(Debug, Clone)]
pub enum Op {
    /// Weighted Sum: result = sum(weights[i] * inputs[i])
    WeightedSum {
        weights: Vec<f64>,
        inputs: Vec<RunId>,
        // Index of the Skeleton Node whose logits generated these weights
        skeleton_idx: Option<usize>,
        // Per-branch scores for gradient computation (filled in by compare_targets)
        branch_scores: Vec<f64>,
    },
    /// Construction: result = Stem(input)
    ConstructStem(RunId),
    /// Construction: result = Fork(l, r)
    ConstructFork(RunId, RunId),
    /// Stochastic Choice: result = input (Passed through from Skeleton choice)
    Choice {
        skeleton_idx: usize,
        choice_idx: usize,
        input: RunId,
    },
    /// Source (Constant or Parameter Node)
    Source,
}



/// The Execution Graph (Dynamic)
/// Contains all nodes created during soft execution.
pub struct RunGraph<'a> {
    pub nodes: Vec<RunNode>,
    pub ops: Vec<Op>,
    pub skeleton: &'a Vec<ParamNode>,
    pub library: Vec<RunId>,  // Pre-lifted Component Roots
}

impl<'a> RunGraph<'a> {
    pub fn new(skeleton: &'a Vec<ParamNode>) -> Self {
        Self {
            nodes: Vec::new(),
            ops: Vec::new(),
            skeleton,
            library: Vec::new(),
        }
    }
    
    pub fn new_with_library(skeleton: &'a Vec<ParamNode>, library_refs: Vec<NodeId>, g: &Graph) -> Self {
        let mut rg = Self {
            nodes: Vec::new(),
            ops: Vec::new(),
            skeleton,
            library: Vec::new(),
        };
        
        // Lift library items
        for node_id in library_refs {
             let run_id = lift_target(g, node_id, &mut rg);
             rg.library.push(run_id);
        }
        rg
    }
    
    pub fn add_node(&mut self, node: RunNode, op: Op) -> RunId {
        let id = self.nodes.len();
        self.nodes.push(node);
        self.ops.push(op);
        id
    }
    
    pub fn add_skeleton(&mut self, idx: usize) -> RunId {
        self.add_node(RunNode::Skeleton(idx), Op::Source)
    }
}

pub type SoftValue = RunId;

/// Resolve a node to expose its structure (for pattern matching)
/// Unlike eval_soft, this expands Skeleton nodes even without args
pub fn resolve_soft(g: &mut RunGraph, subject: SoftValue, depth: usize) -> SoftValue {
    if depth > 12 { return subject; }
    
    let node = g.nodes[subject].clone();
    
    match node {
        RunNode::Skeleton(skel_idx) => {
            let all_logits = g.skeleton[skel_idx].all_logits();
            let probs = softmax_vec(&all_logits);
            
            let mut comps = Vec::new();
            let mut inputs = Vec::new();
            let mut weights = Vec::new();
            
            // 0: Leaf
            if probs[0] > 1e-6 {
                let l_id = g.add_node(RunNode::Leaf, Op::Source);
                comps.push((l_id, probs[0])); inputs.push(l_id); weights.push(probs[0]);
            }
            
            // 1: Stem
            if probs[1] > 1e-6 {
                let s_child_idx = g.skeleton[skel_idx].stem_child;
                let s_child = g.add_skeleton(s_child_idx);
                let s_id = g.add_node(RunNode::Stem(s_child), Op::ConstructStem(s_child));
                comps.push((s_id, probs[1])); inputs.push(s_id); weights.push(probs[1]);
            }
            
            // 2: Fork
            if probs[2] > 1e-6 {
                let f_l_idx = g.skeleton[skel_idx].fork_left;
                let f_r_idx = g.skeleton[skel_idx].fork_right;
                let f_l = g.add_skeleton(f_l_idx);
                let f_r = g.add_skeleton(f_r_idx);
                let f_id = g.add_node(RunNode::Fork(f_l, f_r), Op::ConstructFork(f_l, f_r));
                comps.push((f_id, probs[2])); inputs.push(f_id); weights.push(probs[2]);
            }
            
            // 3..N: Library
            for k in 0..g.skeleton[skel_idx].lib_logits.len() {
                let p = probs[3 + k];
                if p > 1e-6 {
                    if k < g.library.len() {
                        let lib_run_id = g.library[k];
                        // Library items are pre-lifted roots. We use them directly.
                        // But wait, we need Op::Choice for backprop to work?
                        // Or Op::WeightedSum inputs?
                        // Yes, just add to WeightedSum inputs.
                        // Note: We don't add a new node for library item, we reference existing run_id.
                        comps.push((lib_run_id, p)); inputs.push(lib_run_id); weights.push(p);
                    }
                }
            }
            
            g.add_node(RunNode::Superposition(comps), Op::WeightedSum {
                weights,
                inputs,
                skeleton_idx: Some(skel_idx),
                branch_scores: vec![], // Will be filled by compare_targets
            })
        },
        // Everything else is already resolved
        _ => subject
    }
}

/// Forward Pass - Monte Carlo Sampling (Soft)
pub fn eval_soft(g: &mut RunGraph, subject: SoftValue, args: Vec<SoftValue>, depth: usize) -> SoftValue {
    if depth > 12 { return subject; } // Depth limit
    if args.is_empty() { return subject; }
    
    let node = g.nodes[subject].clone(); 
    
    match node {

        RunNode::Skeleton(skel_idx) => {
            let all_logits = g.skeleton[skel_idx].all_logits();
            let probs = softmax_vec(&all_logits); 
            
            // Expand branches and apply args
            let mut comps = Vec::new();
            let mut inputs = Vec::new();
            let mut weights = Vec::new();
            
            // 0: Leaf
            if probs[0] > 1e-6 {
                let l_id = g.add_node(RunNode::Leaf, Op::Source);
                let res = eval_soft(g, l_id, args.clone(), depth + 1);
                comps.push((res, probs[0])); inputs.push(res); weights.push(probs[0]);
            }
            
            // 1: Stem
            if probs[1] > 1e-6 {
                let s_child_idx = g.skeleton[skel_idx].stem_child;
                let s_child = g.add_skeleton(s_child_idx);
                let s_id = g.add_node(RunNode::Stem(s_child), Op::ConstructStem(s_child));
                let res = eval_soft(g, s_id, args.clone(), depth + 1);
                comps.push((res, probs[1])); inputs.push(res); weights.push(probs[1]);
            }
            
            // 2: Fork
            if probs[2] > 1e-6 {
                let f_l_idx = g.skeleton[skel_idx].fork_left;
                let f_r_idx = g.skeleton[skel_idx].fork_right;
                let f_l = g.add_skeleton(f_l_idx);
                let f_r = g.add_skeleton(f_r_idx);
                let f_id = g.add_node(RunNode::Fork(f_l, f_r), Op::ConstructFork(f_l, f_r));
                let res = eval_soft(g, f_id, args.clone(), depth + 1);
                comps.push((res, probs[2])); inputs.push(res); weights.push(probs[2]);
            }
            
            // 3..N: Library
            for k in 0..g.skeleton[skel_idx].lib_logits.len() {
                let p = probs[3 + k];
                if p > 1e-6 {
                    if k < g.library.len() {
                        let lib_run_id = g.library[k];
                        let res = eval_soft(g, lib_run_id, args.clone(), depth + 1);
                        comps.push((res, p)); inputs.push(res); weights.push(p);
                    }
                }
            }
            
            g.add_node(RunNode::Superposition(comps), Op::WeightedSum {
                weights,
                inputs,
                skeleton_idx: Some(skel_idx),
                branch_scores: vec![],  // Filled in by compare_targets
            })
        },
        RunNode::Leaf => {
            let stem = g.add_node(RunNode::Stem(args[0]), Op::ConstructStem(args[0]));
            eval_soft(g, stem, args[1..].to_vec(), depth)
        },
        RunNode::Stem(s) => {
            let fork = g.add_node(RunNode::Fork(s, args[0]), Op::ConstructFork(s, args[0]));
            eval_soft(g, fork, args[1..].to_vec(), depth)
        },
        RunNode::Fork(l, r) => {
            if args.is_empty() { return subject; }
            let z = args[0];
            
            // Resolve L to determine reduction rule - must expand even without args
            let l_val = resolve_soft(g, l, depth + 1);
            let l_node = g.nodes[l_val].clone();
            
            match l_node {
                RunNode::Leaf => {
                     // Rule 3a: Fork(Leaf, x) z -> x
                     // result is r
                     let res = r;
                     if args.len() > 1 { eval_soft(g, res, args[1..].to_vec(), depth) } else { res }
                },
                RunNode::Stem(u) => {
                     // Rule 3b: Fork(Stem(u), x) z -> x z (u z)
                     let r_z = eval_soft(g, r, vec![z], depth + 1);
                     let u_z = eval_soft(g, u, vec![z], depth + 1);
                     let res = eval_soft(g, r_z, vec![u_z], depth + 1);
                     if args.len() > 1 { eval_soft(g, res, args[1..].to_vec(), depth) } else { res }
                },
                RunNode::Fork(u, v) => {
                     // Rule 3c: Match
                     let z_val = eval_soft(g, z, vec![], depth + 1);
                     let z_node = g.nodes[z_val].clone();
                     // Dispatch Match on z's type
                     match z_node {
                         RunNode::Leaf => {
                             let res = u;
                             if args.len() > 1 { eval_soft(g, res, args[1..].to_vec(), depth) } else { res }
                         },
                         RunNode::Stem(s) => {
                             let res = eval_soft(g, v, vec![s], depth + 1);
                             if args.len() > 1 { eval_soft(g, res, args[1..].to_vec(), depth) } else { res }
                         },
                         RunNode::Fork(a, b) => {
                             let r_a = eval_soft(g, r, vec![a], depth + 1);
                             let res = eval_soft(g, r_a, vec![b], depth + 1);
                             if args.len() > 1 { eval_soft(g, res, args[1..].to_vec(), depth) } else { res }
                         },
                         RunNode::Superposition(comps) => {
                             // Match distributed over z
                             let mut new_comps = Vec::new();
                             let mut new_inputs = Vec::new();
                             let mut new_weights = Vec::new();
                             
                             // Check if subject (the node we are matching on) has a skeleton_idx
                             let mut inherited_skel_idx = None;
                             if let Op::WeightedSum { skeleton_idx: Some(idx), .. } = &g.ops[subject] {
                                 inherited_skel_idx = Some(*idx);
                             }

                             for (id, w) in comps {
                                 let sub_res = match g.nodes[id].clone() {
                                     RunNode::Leaf => u,
                                     RunNode::Stem(s) => eval_soft(g, v, vec![s], depth + 1),
                                     RunNode::Fork(a, b) => {
                                         let r_a = eval_soft(g, r, vec![a], depth + 1);
                                         eval_soft(g, r_a, vec![b], depth + 1)
                                     },
                                     _ => subject 
                                 };
                                 new_comps.push((sub_res, w));
                                 new_inputs.push(sub_res);
                                 new_weights.push(w);
                             }
                             let res = g.add_node(RunNode::Superposition(new_comps), Op::WeightedSum { 
                                 weights: new_weights, 
                                 inputs: new_inputs, 
                                 skeleton_idx: inherited_skel_idx, // Propagate!
                                 branch_scores: vec![] 
                             });
                             if args.len() > 1 { eval_soft(g, res, args[1..].to_vec(), depth) } else { res }
                         },
                         _ => subject
                     }
                },
                RunNode::Superposition(comps) => {
                    // L is Superposition. Distribute!
                    let mut new_comps = Vec::new();
                    let mut new_inputs = Vec::new();
                    let mut new_weights = Vec::new();
                    
                    // Check if subject has skeleton_idx
                    let mut inherited_skel_idx = None;
                    if let Op::WeightedSum { skeleton_idx: Some(idx), .. } = &g.ops[subject] {
                        inherited_skel_idx = Some(*idx);
                    }
                    
                    for (id, w) in comps {
                         let fork = g.add_node(RunNode::Fork(id, r), Op::ConstructFork(id, r));
                         let res = eval_soft(g, fork, args.clone(), depth + 1);
                         new_comps.push((res, w));
                         new_inputs.push(res);
                         new_weights.push(w);
                    }
                    g.add_node(RunNode::Superposition(new_comps), Op::WeightedSum { 
                        weights: new_weights, 
                        inputs: new_inputs, 
                        skeleton_idx: inherited_skel_idx, // Propagate!
                        branch_scores: vec![] 
                    })
                },
                _ => subject
            }
        },
        RunNode::Superposition(components) => {
            // Distribute application over superposition components
            if components.is_empty() { return subject; }
            
            let mut new_comps = Vec::new();
            let mut new_inputs = Vec::new();
            let mut new_weights = Vec::new();
            
            // Check if subject has skeleton_idx
            let mut inherited_skel_idx = None;
            if let Op::WeightedSum { skeleton_idx: Some(idx), .. } = &g.ops[subject] {
                inherited_skel_idx = Some(*idx);
            }
            
            for (id, w) in components {
                if w < 1e-4 { continue; } // Pruning
                let res = eval_soft(g, id, args.clone(), depth + 1);
                new_comps.push((res, w));
                new_inputs.push(res);
                new_weights.push(w);
            }
             g.add_node(RunNode::Superposition(new_comps), Op::WeightedSum { 
                 weights: new_weights, 
                 inputs: new_inputs, 
                 skeleton_idx: inherited_skel_idx, // Propagate!
                 branch_scores: vec![] 
             })
        },
        RunNode::Prim(p) => {
            if args.is_empty() { return subject; }
            apply_prim(g, subject, p, &args, depth)
        },
        RunNode::Float(_) => subject, 
    }
}


/// Helper to convert Target Term (Graph) to Soft RunGraph Target (RunId)
pub fn lift_target(g: &Graph, node: NodeId, rg: &mut RunGraph) -> RunId {
    let n = g.get(node).clone();
    match n {
        Node::Leaf => rg.add_node(RunNode::Leaf, Op::Source),
        Node::Stem(c) => {
            let c_id = lift_target(g, c, rg);
            rg.add_node(RunNode::Stem(c_id), Op::ConstructStem(c_id))
        },
        Node::Fork(l, r) => {
            let l_id = lift_target(g, l, rg);
            let r_id = lift_target(g, r, rg);
            rg.add_node(RunNode::Fork(l_id, r_id), Op::ConstructFork(l_id, r_id))
        },
        Node::Prim(p) => rg.add_node(RunNode::Prim(p), Op::Source),
        Node::Float(f) => rg.add_node(RunNode::Float(f), Op::Source),
        Node::App{func, args} => {
            let f = lift_target(g, func, rg);
            let _a: Vec<usize> = args.iter().map(|&x| lift_target(g, x, rg)).collect();
            f 
        },
        _ => rg.add_node(RunNode::Leaf, Op::Source),
    }
}

/// Simple pseudo-random between 0 and 1
pub fn rand_simple() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        if SEED % 1000 == 0 {
            if let Ok(d) = SystemTime::now().duration_since(UNIX_EPOCH) {
                SEED ^= d.as_nanos() as u64;
            }
        }
        ((SEED >> 16) & 0x7fff) as f64 / 32768.0
    }
}

pub fn softmax(logits: &[f64; 3]) -> [f64; 3] {
    let max = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    [exps[0]/sum, exps[1]/sum, exps[2]/sum]
}

fn apply_prim(g: &mut RunGraph, _subject: RunId, p: Primitive, args: &[RunId], depth: usize) -> RunId {
    match p {
        Primitive::Match => {
            // T hL hS hF X
            if args.len() < 4 { return g.add_node(RunNode::Leaf, Op::Source); } 
            let hl = args[0];
            let hs = args[1];
            let hf = args[2];
            let target = args[3];
            
            let res = triage_soft(g, target, hl, hs, hf, depth);
            if args.len() > 4 {
                eval_soft(g, res, args[4..].to_vec(), depth)
            } else {
                res
            }
        },
        _ => g.add_node(RunNode::Leaf, Op::Source)
    }
}

// Triage handler
fn triage_soft(g: &mut RunGraph, subject: RunId, hl: RunId, hs: RunId, hf: RunId, depth: usize) -> RunId {
    let node = g.nodes[subject].clone();
    match node {
        RunNode::Leaf => eval_soft(g, hl, vec![], depth + 1),
        RunNode::Stem(child) => eval_soft(g, hs, vec![child], depth + 1),
        RunNode::Fork(l, r) => eval_soft(g, hf, vec![l, r], depth + 1),
        RunNode::Superposition(comps) => {
             // Distribute triage
             if comps.is_empty() { return subject; }
             
             let mut new_comps = Vec::new();
             let mut new_inputs = Vec::new();
             let mut new_weights = Vec::new();
             
             for (id, w) in comps {
                 if w < 1e-4 { continue; }
                 let res = triage_soft(g, id, hl, hs, hf, depth);
                 new_comps.push((res, w));
                 new_inputs.push(res);
                 new_weights.push(w);
             }
             g.add_node(RunNode::Superposition(new_comps), Op::WeightedSum { weights: new_weights, inputs: new_inputs, skeleton_idx: None, branch_scores: vec![] })
        },
        _ => subject
    }
}

/// Backpropagation with correct soft differentiation
/// 
/// For skeleton WeightedSum nodes, the gradient is computed using the per-branch
/// scores from comparison. The gradient w.r.t. logit z_j is:
/// dL/dz_j = Σ_i s_i * dp_i/dz_j where s_i is the branch score
pub fn backprop(g: &RunGraph, loss_map: HashMap<RunId, f64>, branch_scores_map: &HashMap<usize, Vec<f64>>) -> Vec<(Logits, Vec<f64>)> {
    let mut adjoints: Vec<f64> = vec![0.0; g.nodes.len()];
    // Gradients for (base_logits, lib_logits)
    let mut param_grads: Vec<(Logits, Vec<f64>)> = g.skeleton.iter().map(|n| ([0.0; 3], vec![0.0; n.lib_logits.len()])).collect();
    
    // Inject loss (gradient of Score/Utility)
    for (id, val) in loss_map {
        if id < adjoints.len() {
            adjoints[id] += val;
        }
    }
    
    // Propagate adjoints backward through the graph
    for (id, op) in g.ops.iter().enumerate().rev() {
        let adj = adjoints[id];
        if adj.abs() < 1e-9 { continue; }
        
        match op {
            Op::Choice { input, .. } => {
                adjoints[*input] += adj;
            },
            Op::WeightedSum { weights, inputs, .. } => {
                // Propagate weighted adjoint to children
                for (i, &inp_id) in inputs.iter().enumerate() {
                    adjoints[inp_id] += adj * weights[i];
                }
            },
            Op::ConstructStem(inp) => {
                adjoints[*inp] += adj;
            },
            Op::ConstructFork(l, r) => {
                adjoints[*l] += adj;
                adjoints[*r] += adj;
            },
            Op::Source => {}
        }
    }
    
    // Compute parameter gradients for skeleton nodes using branch_scores from comparison
    for skel_idx in 0..g.skeleton.len() {
        if let Some(scores) = branch_scores_map.get(&skel_idx) {
            let node = &g.skeleton[skel_idx];
            let all_logits = node.all_logits();
            let num_choices = all_logits.len();
            
            if scores.len() == num_choices {
                let probs = softmax_vec(&all_logits);
                
                // Gradient: dL/dz_j = Σ_i scores[i] * dp_i/dz_j
                // where dp_i/dz_j = p_i * (δ_ij - p_j)
                for j in 0..num_choices {
                    let mut grad_z = 0.0;
                    for i in 0..num_choices {
                        let delta = if i == j { 1.0 } else { 0.0 };
                        let dp_i_dz_j = probs[i] * (delta - probs[j]);
                        grad_z += scores[i] * dp_i_dz_j;
                    }
                    
                    // Assign to correct bucket
                    if j < 3 {
                        param_grads[skel_idx].0[j] += grad_z;
                    } else {
                        let lib_idx = j - 3;
                        if lib_idx < param_grads[skel_idx].1.len() {
                            param_grads[skel_idx].1[lib_idx] += grad_z;
                        }
                    }
                }
            }
        }
    }
    
    param_grads
}

/// Compute Softmax for dynamic vector
pub fn softmax_vec(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum_exp = 0.0;
    
    for l in logits {
        let e = (l - max_logit).exp();
        exps.push(e);
        sum_exp += e;
    }
    
    for e in &mut exps {
        *e /= sum_exp;
    }
    exps
}
