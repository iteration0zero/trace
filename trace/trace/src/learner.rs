use crate::arena::{Graph, Node, NodeId};
use crate::soft::{SoftGraph, RunGraph, eval_soft, backprop, RunNode, RunId, Logits, softmax, softmax_vec};
use crate::types::Type;
use crate::inference::InferenceEngine;
use std::collections::HashMap;

/// Configuration for MCID Learner
pub struct LearnerConfig {
    pub epochs: usize,
    pub lr: f64,
    pub lambda: f64,
    pub skeleton_depth: usize,
    pub samples_per_step: usize, // Number of Monte Carlo samples per training step
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            epochs: 500,
            lr: 0.1,
            lambda: 0.01,
            skeleton_depth: 10,
            samples_per_step: 5, // Average over 5 samples for variance reduction
        }
    }
}




/// Hardens a SoftGraph into a discrete Graph
fn harden(sg: &SoftGraph, g: &mut Graph) -> NodeId {
    harden_rec(sg, sg.root, g, 0)
}

fn harden_rec(sg: &SoftGraph, idx: usize, g: &mut Graph, depth: usize) -> NodeId {
    // Prevent stack overflow with depth limit
    if depth > 20 {
        return g.add(Node::Leaf);
    }
    
    let node = &sg.nodes[idx];
    let probs = softmax(&node.logits); 
    
    // ArgMax
    let mut max_p = -1.0;
    let mut choice = 0;
    for i in 0..3 {
        if probs[i] > max_p {
            max_p = probs[i];
            choice = i;
        }
    }
    
    match choice {
        0 => g.add(Node::Leaf),
        1 => {
            let child = harden_rec(sg, node.stem_child, g, depth + 1);
            g.add(Node::Stem(child))
        },
        2 => {
            let l = harden_rec(sg, node.fork_left, g, depth + 1);
            let r = harden_rec(sg, node.fork_right, g, depth + 1);
            g.add(Node::Fork(l, r))
        },
        _ => g.add(Node::Leaf)
    }
}

/// Main Learner Entry Point - Learn from a target function
pub fn learn(
    g: &mut Graph, 
    target_func: NodeId, 
    target_type: Option<Type>, 
    config: LearnerConfig,
    library: Vec<NodeId>
) -> Option<NodeId> {
    println!("MCID Learner Started (Depth {}, Epochs {})", config.skeleton_depth, config.epochs);
    
    // Generate Training Data by applying target_func to simple inputs
    let mut train_data: Vec<(NodeId, NodeId)> = Vec::new();
    
    let l_node = g.add(Node::Leaf);
    let stem_child = g.add(Node::Leaf);
    let stem_node = g.add(Node::Stem(stem_child));
    let f_l = g.add(Node::Leaf);
    let f_r = g.add(Node::Leaf);
    let fork_node = g.add(Node::Fork(f_l, f_r));
    
    let float_val = g.add(Node::Float(42.0));
    
    let inputs = vec![l_node, stem_node, fork_node, float_val];
    
    use crate::engine::{EvalContext, reduce};
    for &inp in &inputs {
        let app = g.add(Node::App { func: target_func, args: smallvec::smallvec![inp] });
        let mut ctx = EvalContext::default();
        ctx.step_limit = 200; // Allow enough steps for target reduction
        let res = reduce(g, app, &mut ctx);
        train_data.push((inp, res));
        println!("  Example: input -> output");
    }
    
    learn_from_examples_with_library(g, train_data, target_type, config, library)
}

/// Learn from explicit input-output examples
/// Returns the best learned program (for curriculum learning)
pub fn learn_from_examples(
    g: &mut Graph, 
    train_data: Vec<(NodeId, NodeId)>, 
    target_type: Option<Type>, 
    config: LearnerConfig
) -> Option<NodeId> {
    learn_from_examples_with_library(g, train_data, target_type, config, Vec::new())
}

/// Learn from explicit input-output examples with library support
/// Returns the best learned program (for curriculum learning)
pub fn learn_from_examples_with_library(
    g: &mut Graph, 
    train_data: Vec<(NodeId, NodeId)>, 
    target_type: Option<Type>, 
    config: LearnerConfig,
    library: Vec<NodeId>
) -> Option<NodeId> {
    println!("Learning from {} examples (Depth {}, Epochs {}, Library: {})", 
             train_data.len(), config.skeleton_depth, config.epochs, library.len());
    
    // Initialize Soft Skeleton with library
    let mut soft_graph = SoftGraph::new_complete_with_library(config.skeleton_depth, library.clone());
    
    // Initialize Type Constraints
    let mut constraints: Vec<Type> = vec![Type::Var(0); soft_graph.nodes.len()];
    if let Some(tt) = target_type {
        constraints[soft_graph.root] = tt;
        // Propagate Constraints
        for i in 0..soft_graph.nodes.len() {
            let (s, l, r) = InferenceEngine::propagate_type_constraints(&constraints[i]);
            let s_idx = soft_graph.nodes[i].stem_child;
            let l_idx = soft_graph.nodes[i].fork_left;
            let r_idx = soft_graph.nodes[i].fork_right;
            
            if s_idx > i { constraints[s_idx] = s; }
            if l_idx > i { constraints[l_idx] = l; }
            if r_idx > i { constraints[r_idx] = r; }
        }
    }
    
    // Adam optimizer state
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    // Dynamic size per node: 3 + library.len()
    let mut m: Vec<Vec<f64>> = soft_graph.nodes.iter().map(|n| vec![0.0; n.num_choices()]).collect(); // First moment
    let mut v: Vec<Vec<f64>> = soft_graph.nodes.iter().map(|n| vec![0.0; n.num_choices()]).collect(); // Second moment
    
    // Training Loop - Soft Differentiable Execution with Adam
    println!("Starting training loop...");
    
    for epoch in 0..config.epochs {
        let mut total_loss = 0.0;
        let mut grad_accum: Vec<Vec<f64>> = soft_graph.nodes.iter().map(|n| vec![0.0; n.num_choices()]).collect();
        
        for (_idx, (inp, expected)) in train_data.iter().enumerate() {
            // Soft execution with superpositions
            for _sample in 0..config.samples_per_step {
                let mut rg = RunGraph::new_with_library(&soft_graph.nodes, library.clone(), g);
                let inp_soft = crate::soft::lift_target(g, *inp, &mut rg);
                let root_soft = rg.add_skeleton(soft_graph.root);
                let out_soft = eval_soft(&mut rg, root_soft, vec![inp_soft], 0);
                
                let (score, loss_map, branch_scores) = compare_targets(g, *expected, &rg, out_soft);
                total_loss -= score / (config.samples_per_step as f64); 
                
                let sample_grads = backprop(&rg, loss_map, &branch_scores);
                for (i, (base_grad, lib_grad)) in sample_grads.iter().enumerate() {
                    let divisor = config.samples_per_step as f64;
                    
                    // Accumulate Base Gradients
                    for j in 0..3 {
                        grad_accum[i][j] += base_grad[j] / divisor;
                    }
                    
                    // Accumulate Library Gradients
                    for k in 0..lib_grad.len() {
                        if 3 + k < grad_accum[i].len() {
                             grad_accum[i][3 + k] += lib_grad[k] / divisor;
                        }
                    }
                }
            }
        }
        
        // Add Self-Similarity Regularization
        let sim_score = soft_graph.compute_self_similarity();
        let sim_lambda = 0.1; // Weight for similarity prior
        total_loss -= sim_score * sim_lambda;
        
        // Adam update
        let t = (epoch + 1) as f64;
        for (i, node) in soft_graph.nodes.iter_mut().enumerate() {
            let mask = InferenceEngine::get_structural_mask(&constraints[i]);
            let num_base = 3;
            let total_choices = node.num_choices();
            
            // Update All Logits (Base + Library)
            for j in 0..total_choices {
                let g_t = grad_accum[i][j] + (if j == 2 { sim_score * 0.01 } else { 0.0 });
                
                // Update biased first moment estimate
                m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g_t;
                // Update biased second raw moment estimate
                v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * g_t * g_t;
                
                // Compute bias-corrected first moment estimate
                let m_hat = m[i][j] / (1.0 - beta1.powf(t));
                // Compute bias-corrected second raw moment estimate
                let v_hat = v[i][j] / (1.0 - beta2.powf(t));
                
                let step = config.lr * m_hat / (v_hat.sqrt() + eps);
                
                // Update parameters
                if j < 3 {
                    node.logits[j] += step;
                    // Apply type constraints mask (Base only)
                    if mask[j] == f64::NEG_INFINITY {
                        node.logits[j] = f64::NEG_INFINITY;
                    }
                } else {
                    let lib_idx = j - 3;
                    if lib_idx < node.lib_logits.len() {
                        node.lib_logits[lib_idx] += step;
                    }
                }
            }
        }
        
        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            // Compute gradient magnitude
            let grad_mag: f64 = grad_accum.iter()
                .flat_map(|g| g.iter())
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            
            print!("\rEpoch {}: Loss {:.4} GradMag {:.6}    ", epoch, total_loss, grad_mag);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    
    println!(); // Newline after progress
    
    // Evaluate hardened
    use crate::engine::unparse;
    let result = harden(&soft_graph, g);
    
    let passed = evaluate_program(g, result, &train_data);
    if passed {
        println!("Learner converged!");
    } else {
        println!("Learner finished (did not perfectly converge).");
    }

    println!("Learned Program: {}", unparse(g, result));
    Some(result)
}

/// Sample a discrete program from the skeleton and track choices for gradient computation
/// Returns (program_node, vec of (skeleton_idx, choice) pairs)
fn sample_with_choices(sg: &SoftGraph, g: &mut Graph) -> (NodeId, Vec<(usize, usize)>) {
    let mut choices = Vec::new();
    let program = sample_with_choices_rec(sg, sg.root, g, &mut choices, 0);
    (program, choices)
}

fn sample_with_choices_rec(
    sg: &SoftGraph, 
    idx: usize, 
    g: &mut Graph, 
    choices: &mut Vec<(usize, usize)>,
    depth: usize
) -> NodeId {
    if depth > 20 {
        return g.add(Node::Leaf);
    }
    
    let node = &sg.nodes[idx];
    let probs = softmax(&node.logits);
    
    let r: f64 = rand::random();
    let choice = if r < probs[0] { 0 }
                 else if r < probs[0] + probs[1] { 1 }
                 else { 2 };
    
    // Record the choice for this skeleton node
    choices.push((idx, choice));
    
    match choice {
        0 => g.add(Node::Leaf),
        1 => {
            let child = sample_with_choices_rec(sg, node.stem_child, g, choices, depth + 1);
            g.add(Node::Stem(child))
        },
        _ => {
            let l = sample_with_choices_rec(sg, node.fork_left, g, choices, depth + 1);
            let r = sample_with_choices_rec(sg, node.fork_right, g, choices, depth + 1);
            g.add(Node::Fork(l, r))
        },
    }
}

/// Compare two nodes for structural equality (after resolution)
fn nodes_equal(g: &Graph, a: NodeId, b: NodeId) -> bool {
    let a = g.resolve(a);
    let b = g.resolve(b);
    
    if a == b { return true; }
    
    match (g.get(a), g.get(b)) {
        (Node::Leaf, Node::Leaf) => true,
        (Node::Stem(ac), Node::Stem(bc)) => nodes_equal(g, *ac, *bc),
        (Node::Fork(al, ar), Node::Fork(bl, br)) => {
            nodes_equal(g, *al, *bl) && nodes_equal(g, *ar, *br)
        },
        (Node::Float(fa), Node::Float(fb)) => (fa - fb).abs() < 1e-9,
        (Node::Prim(pa), Node::Prim(pb)) => pa == pb,
        _ => false,
    }
}

fn print_distribution_and_scores(sg: &SoftGraph, g: &mut Graph, examples: &Vec<(NodeId, NodeId)>) -> Option<NodeId> {
    let mut stats: HashMap<NodeId, (usize, bool)> = HashMap::new(); // NodeId -> (count, passed_all)
    let n_samples = 10;
    
    for _ in 0..n_samples {
        let node = sample_structure(sg, sg.root, g);
        let entry = stats.entry(node).or_insert((0, false));
        if entry.0 == 0 {
            // First time seeing this program, evaluate it
            entry.1 = evaluate_program(g, node, examples);
        }
        entry.0 += 1;
    }
    
    let mut sorted: Vec<(NodeId, usize, bool)> = stats.into_iter()
        .map(|(id, (count, passed))| (id, count, passed))
        .collect();
    
    // Sort by: Passed (desc), Count (desc)
    sorted.sort_by(|a, b| {
        if a.2 != b.2 {
            b.2.cmp(&a.2) // Passed first
        } else {
            b.1.cmp(&a.1) // Then frequent
        }
    });
    
    println!("Top Learned Programs (Distribution over {} samples):", n_samples);
    use crate::engine::unparse;
    for (node, count, passed) in sorted.iter().take(5) {
        let pct = (*count as f64 / n_samples as f64) * 100.0;
        let status = if *passed { "PASS" } else { "FAIL" };
        println!("  {:.1}% [{}] : {}", pct, status, unparse(g, *node));
    }
    
    sorted.first().map(|x| x.0)
}

fn evaluate_program(g: &mut Graph, program: NodeId, examples: &Vec<(NodeId, NodeId)>) -> bool {
    use crate::engine::{reduce, EvalContext};
    use crate::arena::Node;
    
    for (i, (inp, target)) in examples.iter().enumerate() {
        // Construct App(program, inp) manually to avoid altering graph permanently??
        // We can just add it. Graph is append-only.
        let app = g.add(Node::App { func: program, args: smallvec::smallvec![*inp] });
        let mut ctx = EvalContext::default();
        ctx.step_limit = 100; // Restricted execution for reporting samples to prevent hangs
        
        let res = reduce(g, app, &mut ctx);
        
        // Strict comparison using nodes_equal (handles epsilon for floats)
        if !nodes_equal(g, res, *target) {
            return false;
        }
    }
    true
}

fn sample_structure(sg: &SoftGraph, idx: usize, g: &mut Graph) -> NodeId {
    sample_structure_rec(sg, idx, g, 0)
}

fn sample_structure_rec(sg: &SoftGraph, idx: usize, g: &mut Graph, depth: usize) -> NodeId {
    // Prevent stack overflow with depth limit
    if depth > 10 { // Reduced from 20 to prevent massive tree generation hang
        return g.add(Node::Leaf);
    }
    
    let node = &sg.nodes[idx];
    let all_logits = node.all_logits();
    let probs = softmax_vec(&all_logits); 
    
    let r: f64 = rand::random();
    let mut cum_sum = 0.0;
    let mut choice = probs.len() - 1; // Default to last if precision error
    for (i, &p) in probs.iter().enumerate() {
        cum_sum += p;
        if r < cum_sum {
            choice = i;
            break;
        }
    }
    
    match choice {
        0 => g.add(Node::Leaf),
        1 => {
            let child = sample_structure_rec(sg, node.stem_child, g, depth + 1);
            g.add(Node::Stem(child))
        },
        2 => {
            let l = sample_structure_rec(sg, node.fork_left, g, depth + 1);
            let r = sample_structure_rec(sg, node.fork_right, g, depth + 1);
            g.add(Node::Fork(l, r))
        },
        k => {
            // Library item
            let lib_idx = k - 3;
            if lib_idx < sg.library.len() {
                sg.library[lib_idx]
            } else {
                g.add(Node::Leaf) // Should not happen
            }
        }
    }
}

/// Compare output to target and compute per-branch scores for gradient computation
/// 
/// Returns: (score, loss_map, branch_scores_map)
/// - score: the overall comparison score
/// - loss_map: maps RunIds to partial credit for backprop adjoint injection
/// - branch_scores_map: maps skeleton_idx to Vec of per-branch scores for gradient
fn compare_targets(g: &Graph, target: NodeId, rg: &RunGraph, out: RunId) -> (f64, HashMap<RunId, f64>, HashMap<usize, Vec<f64>>) {
    let mut map = HashMap::new();
    let mut branch_scores: HashMap<usize, Vec<f64>> = HashMap::new();
    let score = compare_rec(g, target, rg, out, &mut map, &mut branch_scores, 0);
    (score, map, branch_scores)
}

/// Recursive comparison - only gives credit for exact structural matches
fn compare_rec(
    g: &Graph, 
    target: NodeId, 
    rg: &RunGraph, 
    out: RunId, 
    map: &mut HashMap<RunId, f64>,
    branch_scores: &mut HashMap<usize, Vec<f64>>,  // keyed by skeleton_idx
    depth: usize
) -> f64 {
    if depth > 10 { return 0.0; }
    
    let t_node = g.get(target);
    let r_node = &rg.nodes[out];
    
    match (t_node, r_node) {
        // Exact structural matches
        (Node::Leaf, RunNode::Leaf) => {
            *map.entry(out).or_default() += 1.0;
            1.0
        },
        (Node::Stem(tc), RunNode::Stem(rc)) => {
            let child_score = compare_rec(g, *tc, rg, *rc, map, branch_scores, depth+1);
            if child_score > 0.0 {
                *map.entry(out).or_default() += 1.0;
                1.0 + child_score
            } else {
                -1.0
            }
        },
        (Node::Fork(tl, tr), RunNode::Fork(rl, rr)) => {
            let left_score = compare_rec(g, *tl, rg, *rl, map, branch_scores, depth+1);
            let right_score = compare_rec(g, *tr, rg, *rr, map, branch_scores, depth+1);
            if left_score > 0.0 && right_score > 0.0 {
                *map.entry(out).or_default() += 1.0;
                1.0 + left_score + right_score
            } else {
                -1.0
            }
        },
        
        // Float match
        (Node::Float(a), RunNode::Float(b)) => {
            if (a - b).abs() < 1e-6 {
                *map.entry(out).or_default() += 2.0; 
                2.0
            } else {
                -1.0
            }
        },
        
        // Primitive match
        (Node::Prim(a), RunNode::Prim(b)) => {
            if a == b {
                *map.entry(out).or_default() += 2.0;
                2.0
            } else {
                -1.0
            }
        },
        
        // Superposition: compute per-branch scores for gradient
        (_, RunNode::Superposition(comps)) => {
            let mut sum_score = 0.0;
            let mut scores_vec = Vec::with_capacity(comps.len());
            
            for (id, w) in comps {
                let s = if *w > 1e-4 {
                    compare_rec(g, target, rg, *id, map, branch_scores, depth + 1)
                } else {
                    0.0
                };
                scores_vec.push(s);
                sum_score += w * s;
            }
            
            // Store per-branch scores keyed by skeleton_idx if available
            // At depth 0, if skeleton_idx is None, use root skeleton index 0
            let skel_idx_opt = if let crate::soft::Op::WeightedSum { skeleton_idx, .. } = &rg.ops[out] {
                *skeleton_idx
            } else {
                None
            };
            
            let skel_idx = skel_idx_opt.or_else(|| if depth == 0 && scores_vec.len() == 3 { Some(0) } else { None });
            
            if let Some(idx) = skel_idx {
                // Merge scores - if skeleton already has scores, average them
                branch_scores.entry(idx)
                    .and_modify(|existing| {
                        // Pad if needed
                        while existing.len() < scores_vec.len() { existing.push(0.0); }
                        for (i, s) in scores_vec.iter().enumerate() {
                            if i < existing.len() { existing[i] += s; }
                        }
                    })
                    .or_insert(scores_vec.clone());
            }
            sum_score
        },
        
        // All other cases are mismatches - NO reward shaping
        _ => -1.0 
    }
}
