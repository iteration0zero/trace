//! IGTC: Intensional Gradient Tree Calculus
//!
//! Main synthesis engine implementing gradient-based program synthesis
//! over a superposition of discrete Triage Calculus programs.

use crate::arena::{Graph, Node, NodeId};
use crate::engine::{reduce, unparse, EvalContext};
use crate::learner::loss::tree_edit_distance;
use crate::trace::ExecutionTrace;
use rand::Rng;
use std::collections::HashMap;
use smallvec::SmallVec;

/// Configuration for IGTC synthesis
#[derive(Debug, Clone)]
pub struct IgtcConfig {
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Learning rate for weight updates
    pub learning_rate: f64,
    /// Maximum support set size
    pub max_support_size: usize,
    /// Minimum weight before pruning
    pub min_weight: f64,
    /// Maximum evaluation steps per program
    pub max_eval_steps: usize,
    /// Number of neighbor edits per iteration
    pub num_edits: usize,
    /// Print detailed progress
    pub verbose: bool,
    /// Log frequency (iterations)
    pub log_every: usize,
    /// Print best candidate output vs expected each log
    pub debug_best_io: bool,
    /// Keep newly created candidates for N prune cycles
    pub keep_new_iters: usize,
    /// Allow using curriculum/library nodes as direct replacement edits
    pub use_library_edits: bool,
}

impl Default for IgtcConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            learning_rate: 0.01,        // Very slow learning rate
            max_support_size: 2000,       // Default cap; override per learn-from if needed
            min_weight: 0.0001,         // Very slow pruning (0.01%)
            max_eval_steps: 250,        // Base multiplier; actual budget scales with term size
            num_edits: 100,              // More edits per iteration
            verbose: false,
            log_every: 10,
            debug_best_io: false,
            keep_new_iters: 10,
            use_library_edits: true,
        }
    }
}

/// A candidate program with its weight
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The program tree
    pub program: NodeId,
    /// Current weight (0, 1]
    pub weight: f64,
    /// Cached loss (updated each iteration)
    pub loss: Option<f64>,
    /// Blame map from last evaluation
    pub blame: Option<HashMap<NodeId, f64>>,
    /// Age in iterations (used to protect new candidates from pruning)
    pub age: usize,
}

impl Candidate {
    pub fn new(program: NodeId) -> Self {
        Self {
            program,
            weight: 1.0,
            loss: None,
            blame: None,
            age: 0,
        }
    }
    
    pub fn with_weight(program: NodeId, weight: f64) -> Self {
        Self {
            program,
            weight,
            loss: None,
            blame: None,
            age: 0,
        }
    }
}

/// The IGTC synthesizer
pub struct IgtcSynthesizer {
    /// Configuration
    pub config: IgtcConfig,
    /// Support set of candidate programs
    pub support: Vec<Candidate>,
    /// Training examples: (input, expected_output)
    pub examples: Vec<(NodeId, NodeId)>,
    /// Best program found so far
    pub best: Option<(NodeId, f64)>,
    /// Allowed replacement programs (from curriculum library)
    pub allowed_replacements: Vec<NodeId>,
}

impl IgtcSynthesizer {
    pub fn new(config: IgtcConfig) -> Self {
        Self {
            config,
            support: Vec::new(),
            examples: Vec::new(),
            best: None,
            allowed_replacements: Vec::new(),
        }
    }

    /// Restrict edit replacements to these programs (curriculum/library)
    pub fn set_allowed_replacements(&mut self, replacements: Vec<NodeId>) {
        self.allowed_replacements = replacements;
    }
    
    /// Add a training example
    pub fn add_example(&mut self, input: NodeId, output: NodeId) {
        self.examples.push((input, output));
    }
    
    /// Initialize the support set with seed programs
    pub fn seed(&mut self, programs: Vec<NodeId>) {
        // Start with equal weight, but higher base to avoid immediate pruning
        let initial_weight = 1.0;  // Each starts at 1.0, normalize later
        self.support = programs
            .into_iter()
            .map(|p| Candidate::with_weight(p, initial_weight))
            .collect();
    }
    
    /// Run the synthesis loop
    pub fn synthesize(&mut self, g: &mut Graph) -> Option<NodeId> {
        for iteration in 0..self.config.max_iterations {
            if iteration > 0 {
                self.increment_ages();
            }
            let do_log = self.config.verbose && (self.config.log_every > 0) && (iteration % self.config.log_every == 0);
            if do_log {
                println!("IGTC Iteration {}:", iteration);
                println!("  Phase 1/5: evaluate");
            }
            // Phase 1: Evaluate all candidates
            self.evaluate_all(g);
            
            // Check for convergence
            if let Some((best_prog, best_loss)) = &self.best {
                if *best_loss == 0.0 {
                    println!("IGTC: Found perfect solution at iteration {}", iteration);
                    return Some(*best_prog);
                }
            }
            
            if do_log {
                let best_loss = self.best.as_ref().map(|b| b.1).unwrap_or(f64::INFINITY);
                let avg_loss = self.support.iter().filter_map(|c| c.loss).sum::<f64>()
                    / (self.support.len().max(1) as f64);
                println!("  Eval summary: support={}, best_loss={:.4}, avg_loss={:.4}", 
                         self.support.len(), best_loss, avg_loss);
                if self.config.debug_best_io {
                    if let Some(best) = self.support.iter().filter(|c| c.loss == Some(best_loss)).next() {
                        println!("  Best program (NF): {}", unparse(g, best.program));
                        let mut total = 0.0;
                        for (idx, (inp, expected)) in self.examples.iter().enumerate() {
                            let (loss, actual_str, expected_str) = eval_candidate_loss_and_outputs(
                                g,
                                best.program,
                                *inp,
                                *expected,
                                self.config.max_eval_steps,
                            );
                            total += loss;
                            let input_str = unparse(g, *inp);
                            println!("  Example {} loss={:.4}", idx, loss);
                            println!("    Input:        {}", input_str);
                            println!("    Actual (NF):   {}", actual_str);
                            println!("    Expected (NF): {}", expected_str);
                        }
                        println!("  Sum loss (debug): {:.4}", total);
                        // Print top blamed program paths for this best candidate
                        let path_map = collect_paths_map(g, best.program);
                        let mut blamed: Vec<(Vec<u8>, f64)> = Vec::new();
                        if let Some(blame) = &best.blame {
                            for (&node, &b) in blame.iter() {
                                if let Some(paths) = path_map.get(&node) {
                                    let split = b / paths.len().max(1) as f64;
                                    for path in paths {
                                        blamed.push((path.clone(), split));
                                    }
                                }
                            }
                            blamed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            for (i, (path, b)) in blamed.iter().take(5).enumerate() {
                                if let Some(node) = node_at_path(g, best.program, path) {
                                    let node_str = unparse(g, node);
                                    println!("  Blame {}: {:.4} at path {:?} -> {}", i, b, path, node_str);
                                }
                            }
                        }
                    }
                }
                println!("  Phase 2/5: update weights");
            }
            // Phase 2: Update weights (exponentiated gradient)
            self.update_weights();
            
            if do_log {
                println!("  Phase 3/5: prune");
            }
            // Phase 3: Prune low-weight candidates
            self.prune();
            
            if do_log {
                println!("  Phase 4/5: expand support");
            }
            // Phase 4: Expand support with edits guided by blame
            self.expand_support(g);
            
            if do_log {
                println!("  Phase 5/5: normalize");
            }
            // Normalize weights
            self.normalize_weights();
            // Enforce hard support cap after expansion
            self.cap_support();
            if do_log {
                let best_loss = self.best.as_ref().map(|b| b.1).unwrap_or(f64::INFINITY);
                println!(
                    "  Summary: support={}, cap={}, best_loss={:.4}",
                    self.support.len(),
                    self.config.max_support_size,
                    best_loss
                );
            }
            
            if !self.config.verbose && iteration % 10 == 0 {
                let best_loss = self.best.as_ref().map(|b| b.1).unwrap_or(f64::INFINITY);
                println!("IGTC Iteration {}: Support={}, BestLoss={:.4}", 
                         iteration, self.support.len(), best_loss);
            }
        }
        
        self.best.map(|(p, _)| p)
    }
    
    /// Evaluate all candidates on all examples
    fn evaluate_all(&mut self, g: &mut Graph) {
        for candidate in &mut self.support {
            let mut total_loss = 0.0;
            let mut combined_blame: HashMap<NodeId, f64> = HashMap::new();
            
            for (input, expected) in &self.examples {
                let (loss, blame_map) = evaluate_candidate(
                    g,
                    candidate.program,
                    *input,
                    *expected,
                    self.config.max_eval_steps,
                );
                
                total_loss += loss;
                
                if loss > 0.0 {
                    for (node, b) in blame_map {
                        *combined_blame.entry(node).or_insert(0.0) += b;
                    }
                }
            }

            if combined_blame.is_empty() && total_loss > 0.0 {
                // Fallback: blame the whole program if we couldn't attribute structure.
                *combined_blame.entry(candidate.program).or_insert(0.0) += total_loss;
            }
            
            candidate.loss = Some(total_loss);
            candidate.blame = Some(combined_blame);
            
            // Track best
            if self.best.is_none() || total_loss < self.best.as_ref().unwrap().1 {
                self.best = Some((candidate.program, total_loss));
            }
        }
    }
    
    /// Update weights using exponentiated gradient
    fn update_weights(&mut self) {
        let lr = self.config.learning_rate;
        
        for candidate in &mut self.support {
            if let Some(loss) = candidate.loss {
                // Exponentiated gradient: w *= exp(-lr * loss)
                candidate.weight *= (-lr * loss).exp();
            }
        }
    }
    
    /// Normalize weights to sum to 1
    fn normalize_weights(&mut self) {
        let total: f64 = self.support.iter().map(|c| c.weight).sum();
        if total > 0.0 {
            for candidate in &mut self.support {
                candidate.weight /= total;
            }
        }
    }
    
    /// Prune low-weight candidates
    fn prune(&mut self) {
        // Pre-compute the best loss to identify which candidate to keep
        let best_loss = self.support.iter()
            .filter_map(|c| c.loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Prune low-weight candidates, but keep at least one with best loss
        let mut kept_best = false;
        let min_weight = self.config.min_weight;
        let keep_new_iters = self.config.keep_new_iters;
        self.support.retain(|c| {
            if c.age <= keep_new_iters {
                return true;
            }
            if c.weight >= min_weight {
                return true;
            }
            // Keep the best candidate even if low weight
            if !kept_best && c.loss == best_loss {
                kept_best = true;
                return true;
            }
            false
        });
        
        // Cap support size
        if self.support.len() > self.config.max_support_size {
            self.support.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
            self.support.truncate(self.config.max_support_size);
        }
    }

    fn cap_support(&mut self) {
        if self.support.len() > self.config.max_support_size {
            let keep_new_iters = self.config.keep_new_iters;
            let mut young: Vec<Candidate> = Vec::new();
            let mut rest: Vec<Candidate> = Vec::new();
            for c in self.support.drain(..) {
                if c.age <= keep_new_iters {
                    young.push(c);
                } else {
                    rest.push(c);
                }
            }

            if young.len() >= self.config.max_support_size {
                young.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
                young.truncate(self.config.max_support_size);
                self.support = young;
                return;
            }

            rest.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
            let needed = self.config.max_support_size - young.len();
            rest.truncate(needed);
            young.extend(rest);
            self.support = young;
        }
    }

    fn increment_ages(&mut self) {
        for c in &mut self.support {
            c.age = c.age.saturating_add(1);
        }
    }
    
    /// Expand support by generating edits guided by blame
    fn expand_support(&mut self, g: &mut Graph) {
        let mut new_candidates = Vec::new();
        let mut seen: std::collections::HashSet<NodeId> = self.support.iter().map(|c| c.program).collect();
        
        // For each candidate with reasonable weight, generate edits at blamed locations
        for candidate in &self.support {
            // Only skip if weight is extremely low (below 10% of min_weight)
            if candidate.weight < self.config.min_weight * 0.1 {
                continue;
            }
            
            if let Some(blame) = &candidate.blame {
                // Map blamed nodes to concrete paths in the program
                let path_map = collect_paths_map(g, candidate.program);
                let mut blamed_paths: Vec<(Vec<u8>, f64)> = Vec::new();
                
                for (&node, &b) in blame.iter() {
                    if let Some(paths) = path_map.get(&node) {
                        let split = b / paths.len().max(1) as f64;
                        for path in paths {
                            blamed_paths.push((path.clone(), split));
                        }
                    }
                }
                
                // Sort by blame magnitude
                blamed_paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Use blame magnitude to choose edit targets (and probabilistically go deeper)
                let total_blame: f64 = blamed_paths.iter().map(|(_, b)| *b).sum();
                let mut rng = rand::thread_rng();
                let mut edit_targets: Vec<Vec<u8>> = Vec::new();
                let mut seen_targets: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
                let max_ancestor_depth = 2;

                for (path, b) in blamed_paths.iter() {
                    if edit_targets.len() >= self.config.num_edits {
                        break;
                    }
                    // Always include the base path
                    if seen_targets.insert(path.clone()) {
                        edit_targets.push(path.clone());
                    }

                    // Also include a few ancestors so we can change structure higher up
                    if !path.is_empty() {
                        let mut anc = path.clone();
                        for _ in 0..max_ancestor_depth {
                            if anc.is_empty() { break; }
                            anc.pop();
                            if seen_targets.insert(anc.clone()) {
                                edit_targets.push(anc.clone());
                                if edit_targets.len() >= self.config.num_edits {
                                    break;
                                }
                            }
                        }
                    }


                    if total_blame > 0.0 {
                        let share = (b / total_blame).clamp(0.0, 1.0);
                        // Add child edits with probability proportional to blame share
                        if edit_targets.len() < self.config.num_edits {
                            if let Some(node) = node_at_path(g, candidate.program, path) {
                                match g.get(g.resolve(node)) {
                                    Node::Stem(_) => {
                                        if rng.gen::<f64>() < share {
                                            let mut child = path.clone();
                                            child.push(0);
                                            if seen_targets.insert(child.clone()) {
                                                edit_targets.push(child);
                                            }
                                        }
                                    }
                                    Node::Fork(_, _) => {
                                        if rng.gen::<f64>() < share {
                                            let mut left = path.clone();
                                            left.push(0);
                                            if seen_targets.insert(left.clone()) {
                                                edit_targets.push(left);
                                            }
                                        }
                                        if edit_targets.len() < self.config.num_edits && rng.gen::<f64>() < share {
                                            let mut right = path.clone();
                                            right.push(1);
                                            if seen_targets.insert(right.clone()) {
                                                edit_targets.push(right);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                // Always allow editing the root when there's blame, to enable global structural changes
                if !blamed_paths.is_empty() && edit_targets.len() < self.config.num_edits {
                    if seen_targets.insert(Vec::new()) {
                        edit_targets.push(Vec::new());
                    }
                }

                // Generate edits for chosen targets
                for path in edit_targets {
                    let edits = self.generate_edits(g, candidate.program, &path);
                    for edited in edits {
                        if seen.insert(edited) {
                            new_candidates.push(Candidate::with_weight(
                                edited,
                                candidate.weight * 0.5, // New candidates start at half weight
                            ));
                        }
                    }
                }
            }
        }
        
        self.support.extend(new_candidates);
    }
    
    /// Generate structural edits at a specific node
fn generate_edits(&self, g: &mut Graph, program: NodeId, path: &[u8]) -> Vec<NodeId> {
    let mut edits = Vec::new();
        
        let target = match node_at_path(g, program, path) {
            Some(t) => t,
            None => return edits,
        };
        let target_node = g.get(g.resolve(target)).clone();
        let leaf = g.add(Node::Leaf);
        let fork_leaf_leaf = g.add(Node::Fork(leaf, leaf));
        let mut replacements: Vec<NodeId> = Vec::new();
        
        match target_node {
            Node::Leaf => {
                // Simple growth: leaf -> stem(leaf) or fork(leaf,leaf)
                replacements.push(g.add(Node::Stem(leaf)));
                replacements.push(fork_leaf_leaf);
            }
            Node::Stem(child) => {
                // Simple edits: stem -> fork(leaf, leaf) or fork(child, leaf) / fork(leaf, child)
                replacements.push(fork_leaf_leaf);
                replacements.push(g.add(Node::Fork(child, leaf)));
                replacements.push(g.add(Node::Fork(leaf, child)));
                // Allow duplication of the child (enables self-application patterns)
                replacements.push(g.add(Node::Fork(child, child)));
            }
            Node::Fork(left, right) => {
                // Simple prune: fork -> leaf
                replacements.push(leaf);
                // Simple focus: fork -> stem(left/right)
                replacements.push(g.add(Node::Stem(left)));
                replacements.push(g.add(Node::Stem(right)));
                // Swap branches
                replacements.push(g.add(Node::Fork(right, left)));
                // Allow duplicating one branch
                replacements.push(g.add(Node::Fork(left, left)));
                replacements.push(g.add(Node::Fork(right, right)));
            }
            _ => {}
        }

        // Generic wrappers: allow growing around the existing subtree
        replacements.push(g.add(Node::Stem(target)));
        replacements.push(g.add(Node::Fork(target, leaf)));
        replacements.push(g.add(Node::Fork(leaf, target)));
        replacements.push(g.add(Node::Fork(target, target)));

        // Also allow replacements from curriculum/library only
        for &rep in &self.allowed_replacements {
            replacements.push(rep);
        }
        
        for replacement in replacements {
            if replacement == target { continue; }
            let edited = replace_at_path(g, program, path, replacement);
            edits.push(edited);
        }
        
        edits
    }
}

fn evaluate_candidate(
    g: &Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
) -> (f64, HashMap<NodeId, f64>) {
    // Build an isolated evaluation graph so reductions cannot mutate shared nodes.
    let mut eval_g = Graph::new_uninterned();

    let mut prog_memo = HashMap::new();
    let mut eval_to_orig: HashMap<NodeId, NodeId> = HashMap::new();
    let prog_eval = clone_subtree(g, &mut eval_g, program, &mut prog_memo, Some(&mut eval_to_orig));

    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);

    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let applied = eval_g.add(Node::App {
        func: prog_eval,
        args: smallvec::smallvec![input_eval],
    });

    let mut ctx = EvalContext::default();
    let size = estimate_size(&eval_g, applied);
    let mut budget = max_steps.saturating_mul(size.max(1));
    if budget > 200_000 {
        budget = 200_000;
    }
    ctx.step_limit = budget;
    let mut trace = ExecutionTrace::new();
    ctx.igtc_trace = Some(&mut trace);

    let actual = reduce(&mut eval_g, applied, &mut ctx);

    // Normalize expected output before computing TED to avoid penalizing
    // equivalent programs that differ only by reduction.
    let mut expected_ctx = EvalContext::default();
    let expected_size = estimate_size(&eval_g, expected_eval);
    let mut expected_budget = max_steps.saturating_mul(expected_size.max(1));
    if expected_budget > 200_000 {
        expected_budget = 200_000;
    }
    expected_ctx.step_limit = expected_budget;
    let expected_nf = reduce(&mut eval_g, expected_eval, &mut expected_ctx);

    let loss = tree_edit_distance(&eval_g, actual, expected_nf) as f64;

    if loss == 0.0 {
        return (0.0, HashMap::new());
    }

    let mut seeds: Vec<(NodeId, f64)> = Vec::new();
    seed_structural_blame(&eval_g, actual, expected_nf, loss, &mut seeds);
    if seeds.is_empty() {
        seeds.push((actual, loss));
    }
    trace.seed_blame(&seeds);
    let blame_eval = trace.backpropagate();

    // Map blame back to original program nodes using the clone map
    let mut blame_orig: HashMap<NodeId, f64> = HashMap::new();
    for (eval_node, b) in blame_eval {
        if let Some(orig_node) = eval_to_orig.get(&eval_node) {
            *blame_orig.entry(*orig_node).or_insert(0.0) += b;
        }
    }

    (loss, blame_orig)
}

fn eval_candidate_loss_and_outputs(
    g: &Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
) -> (f64, String, String) {
    let mut eval_g = Graph::new_uninterned();

    let mut prog_memo = HashMap::new();
    let prog_eval = clone_subtree(g, &mut eval_g, program, &mut prog_memo, None);

    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);

    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let applied = eval_g.add(Node::App {
        func: prog_eval,
        args: smallvec::smallvec![input_eval],
    });

    let mut ctx = EvalContext::default();
    let size = estimate_size(&eval_g, applied);
    let mut budget = max_steps.saturating_mul(size.max(1));
    if budget > 200_000 {
        budget = 200_000;
    }
    ctx.step_limit = budget;
    let actual = reduce(&mut eval_g, applied, &mut ctx);

    let mut expected_ctx = EvalContext::default();
    let expected_size = estimate_size(&eval_g, expected_eval);
    let mut expected_budget = max_steps.saturating_mul(expected_size.max(1));
    if expected_budget > 200_000 {
        expected_budget = 200_000;
    }
    expected_ctx.step_limit = expected_budget;
    let expected_nf = reduce(&mut eval_g, expected_eval, &mut expected_ctx);

    let loss = tree_edit_distance(&eval_g, actual, expected_nf) as f64;
    (loss, unparse(&eval_g, actual), unparse(&eval_g, expected_nf))
}

fn clone_subtree(
    g_src: &Graph,
    g_dst: &mut Graph,
    id: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
    eval_to_orig: Option<&mut HashMap<NodeId, NodeId>>,
) -> NodeId {
    let resolved = g_src.resolve(id);
    if let Some(&cached) = memo.get(&resolved) {
        return cached;
    }

    let mut eval_to_orig = eval_to_orig;
    let node = g_src.get(resolved).clone();
    let new_id = match node {
        Node::Leaf => g_dst.add(Node::Leaf),
        Node::Stem(inner) => {
            let c = clone_subtree(g_src, g_dst, inner, memo, eval_to_orig.as_deref_mut());
            g_dst.add(Node::Stem(c))
        }
        Node::Fork(l, r) => {
            let nl = clone_subtree(g_src, g_dst, l, memo, eval_to_orig.as_deref_mut());
            let nr = clone_subtree(g_src, g_dst, r, memo, eval_to_orig.as_deref_mut());
            g_dst.add(Node::Fork(nl, nr))
        }
        Node::Prim(p) => g_dst.add(Node::Prim(p)),
        Node::Float(f) => g_dst.add(Node::Float(f)),
        Node::Handle(h) => g_dst.add(Node::Handle(h)),
        Node::Ind(inner) => {
            clone_subtree(g_src, g_dst, inner, memo, eval_to_orig.as_deref_mut())
        }
        Node::App { func, args } => {
            let nf = clone_subtree(g_src, g_dst, func, memo, eval_to_orig.as_deref_mut());
            let mut new_args: SmallVec<[NodeId; 2]> = SmallVec::new();
            for arg in args {
                let na = clone_subtree(g_src, g_dst, arg, memo, eval_to_orig.as_deref_mut());
                new_args.push(na);
            }
            g_dst.add(Node::App { func: nf, args: new_args })
        }
    };

    memo.insert(resolved, new_id);
    if let Some(map) = eval_to_orig.as_deref_mut() {
        map.insert(new_id, resolved);
    }

    new_id
}

fn estimate_size(g: &Graph, root: NodeId) -> usize {
    let mut memo = HashMap::new();
    estimate_size_rec(g, root, &mut memo)
}

fn estimate_size_rec(g: &Graph, id: NodeId, memo: &mut HashMap<NodeId, usize>) -> usize {
    let resolved = g.resolve(id);
    if let Some(&cached) = memo.get(&resolved) {
        return cached;
    }
    let size = match g.get(resolved) {
        Node::Leaf | Node::Prim(_) | Node::Float(_) | Node::Handle(_) => 1,
        Node::Stem(inner) => 1 + estimate_size_rec(g, *inner, memo),
        Node::Fork(l, r) => 1 + estimate_size_rec(g, *l, memo) + estimate_size_rec(g, *r, memo),
        Node::App { func, args } => {
            let mut total = 1 + estimate_size_rec(g, *func, memo);
            for arg in args {
                total += estimate_size_rec(g, *arg, memo);
            }
            total
        }
        Node::Ind(inner) => estimate_size_rec(g, *inner, memo),
    };
    memo.insert(resolved, size);
    size
}

fn seed_structural_blame(
    g: &Graph,
    actual: NodeId,
    expected: NodeId,
    weight: f64,
    out: &mut Vec<(NodeId, f64)>,
) {
    let a = g.resolve(actual);
    let e = g.resolve(expected);
    let an = g.get(a).clone();
    let en = g.get(e).clone();

    // Mismatch at root: blame the actual node
    if shape_label(&an) != shape_label(&en) {
        out.push((a, weight));
        return;
    }

    match (&an, &en) {
        (Node::Stem(ia), Node::Stem(ie)) => {
            seed_structural_blame(g, *ia, *ie, weight, out);
        }
        (Node::Fork(la, ra), Node::Fork(le, re)) => {
            let half = weight * 0.5;
            seed_structural_blame(g, *la, *le, half, out);
            seed_structural_blame(g, *ra, *re, half, out);
        }
        (Node::Ind(ia), _) => seed_structural_blame(g, *ia, expected, weight, out),
        (_, Node::Ind(ie)) => seed_structural_blame(g, actual, *ie, weight, out),
        // Leaf/Prim/Float/Handle/App with same shape label: still check exact label
        _ => {
            let al = exact_label(&an);
            let el = exact_label(&en);
            if al != el {
                out.push((a, weight));
            } else {
                // Exact labels match; no blame to assign at this node.
                // (Do not stop early for matching shapes; handled above by recursion.)
            }
        }
    }
}

fn shape_label(n: &Node) -> u8 {
    match n {
        Node::Leaf => 0,
        Node::Stem(_) => 1,
        Node::Fork(_, _) => 2,
        Node::Prim(_) | Node::Float(_) | Node::Handle(_) => 3,
        Node::App { .. } => 4,
        Node::Ind(_) => 5,
    }
}

fn exact_label(n: &Node) -> u64 {
    match n {
        Node::Leaf => 0,
        Node::Stem(_) => 1,
        Node::Fork(_, _) => 2,
        Node::Float(f) => 0x10_0000_0000_0000u64 | f.to_bits(),
        Node::Prim(p) => 0x20_0000_0000_0000u64 | prim_label_local(*p),
        Node::Handle(h) => 0x30_0000_0000_0000u64 | (*h as u64),
        Node::App { .. } => 0x40_0000_0000_0000u64,
        Node::Ind(_) => 0x50_0000_0000_0000u64,
    }
}

fn prim_label_local(p: crate::arena::Primitive) -> u64 {
    use crate::arena::Primitive::*;
    match p {
        Add => 1,
        Sub => 2,
        Mul => 3,
        Div => 4,
        Eq => 5,
        Gt => 6,
        Lt => 7,
        If => 8,
        S => 9,
        K => 10,
        I => 11,
        First => 12,
        Rest => 13,
        Trace => 14,
        TagInt => 20,
        TagFloat => 21,
        TagStr => 22,
        TagChar => 23,
        TypeOf => 30,
        Any => 31,
        Match => 32,
        Mod => 33,
    }
}

fn collect_paths_map(g: &Graph, root: NodeId) -> HashMap<NodeId, Vec<Vec<u8>>> {
    let mut map: HashMap<NodeId, Vec<Vec<u8>>> = HashMap::new();
    let mut path = Vec::new();
    collect_paths_rec(g, root, &mut path, &mut map);
    map
}

fn collect_paths_rec(
    g: &Graph,
    id: NodeId,
    path: &mut Vec<u8>,
    map: &mut HashMap<NodeId, Vec<Vec<u8>>>,
) {
    let resolved = g.resolve(id);
    map.entry(resolved).or_default().push(path.clone());
    match g.get(resolved) {
        Node::Stem(inner) => {
            path.push(0);
            collect_paths_rec(g, *inner, path, map);
            path.pop();
        }
        Node::Fork(l, r) => {
            path.push(0);
            collect_paths_rec(g, *l, path, map);
            path.pop();
            path.push(1);
            collect_paths_rec(g, *r, path, map);
            path.pop();
        }
        Node::App { func, args } => {
            path.push(0);
            collect_paths_rec(g, *func, path, map);
            path.pop();
            for (idx, arg) in args.iter().enumerate() {
                if idx >= 250 { break; }
                path.push((idx as u8) + 1);
                collect_paths_rec(g, *arg, path, map);
                path.pop();
            }
        }
        Node::Ind(inner) => collect_paths_rec(g, *inner, path, map),
        _ => {}
    }
}

fn node_at_path(g: &Graph, root: NodeId, path: &[u8]) -> Option<NodeId> {
    let mut curr = g.resolve(root);
    for &dir in path {
        match g.get(curr) {
            Node::Stem(inner) => {
                if dir != 0 { return None; }
                curr = g.resolve(*inner);
            }
            Node::Fork(l, r) => {
                curr = if dir == 0 { g.resolve(*l) } else { g.resolve(*r) };
            }
            Node::App { func, args } => {
                if dir == 0 {
                    curr = g.resolve(*func);
                } else {
                    let idx = (dir - 1) as usize;
                    if idx >= args.len() { return None; }
                    curr = g.resolve(args[idx]);
                }
            }
            Node::Ind(inner) => {
                curr = g.resolve(*inner);
            }
            _ => return None,
        }
    }
    Some(curr)
}

fn replace_at_path(g: &mut Graph, root: NodeId, path: &[u8], replacement: NodeId) -> NodeId {
    if path.is_empty() {
        return replacement;
    }

    let resolved = g.resolve(root);
    match g.get(resolved).clone() {
        Node::Stem(inner) => {
            if path[0] != 0 {
                resolved
            } else {
                let new_inner = replace_at_path(g, inner, &path[1..], replacement);
                g.add(Node::Stem(new_inner))
            }
        }
        Node::Fork(l, r) => {
            if path[0] == 0 {
                let new_l = replace_at_path(g, l, &path[1..], replacement);
                g.add(Node::Fork(new_l, r))
            } else {
                let new_r = replace_at_path(g, r, &path[1..], replacement);
                g.add(Node::Fork(l, new_r))
            }
        }
        Node::App { func, args } => {
            if path[0] == 0 {
                let new_func = replace_at_path(g, func, &path[1..], replacement);
                g.add(Node::App { func: new_func, args })
            } else {
                let idx = (path[0] - 1) as usize;
                if idx >= args.len() { return resolved; }
                let mut new_args = args.clone();
                let updated = replace_at_path(g, args[idx], &path[1..], replacement);
                new_args[idx] = updated;
                g.add(Node::App { func, args: new_args })
            }
        }
        Node::Ind(inner) => replace_at_path(g, inner, path, replacement),
        _ => resolved,
    }
}

/// Convenience function for CLI integration
pub fn synthesize(
    g: &mut Graph,
    examples: Vec<(NodeId, NodeId)>,
    config: IgtcConfig,
) -> Option<NodeId> {
    synthesize_with_seeds(g, examples, config, &[])
}

/// Synthesize with an external seed library (curriculum / reuse)
pub fn synthesize_with_seeds(
    g: &mut Graph,
    examples: Vec<(NodeId, NodeId)>,
    config: IgtcConfig,
    extra_seeds: &[NodeId],
) -> Option<NodeId> {
    let mut synth = IgtcSynthesizer::new(config);
    
    for (input, output) in examples {
        synth.add_example(input, output);
    }
    
    // Build seed set using ONLY pure tree structures (Leaf/Stem/Fork)
    // No primitives, no Apps - just raw Triage Calculus values
    let mut seeds = Vec::new();
    
    // Level 0: Leaf
    let leaf = g.add(Node::Leaf);  // n
    seeds.push(leaf);
    
    // Level 1: Stem(Leaf)
    let stem1 = g.add(Node::Stem(leaf));  // (n n)
    seeds.push(stem1);
    
    // Level 2: Fork(Leaf, Leaf) = K combinator (returns first of 2 args)
    let k = g.add(Node::Fork(leaf, leaf));  // (n n n)
    seeds.push(k);
    
    // Level 2: Stem(Stem(Leaf))
    let stem2 = g.add(Node::Stem(stem1));  // (n (n n))
    seeds.push(stem2);
    
    // K' patterns - combinators that return second argument
    // K' = Fork(Leaf, Fork(Leaf, Leaf)) - in K-rule terms
    let k_prime = g.add(Node::Fork(leaf, k));  // (n (n n n) n)
    seeds.push(k_prime);
    
    // Level 3: Various combinations
    let fork_stem_leaf = g.add(Node::Fork(stem1, leaf));  // ((n n) n n)
    seeds.push(fork_stem_leaf);
    
    let fork_leaf_stem = g.add(Node::Fork(leaf, stem1));  // (n (n n) n)
    seeds.push(fork_leaf_stem);
    
    let stem_k = g.add(Node::Stem(k));  // (n (n n n))
    seeds.push(stem_k);
    
    // Level 4: Deeper structures for triage
    let fork_k_leaf = g.add(Node::Fork(k, leaf));  // ((n n n) n n)
    seeds.push(fork_k_leaf);
    
    let fork_leaf_k = g.add(Node::Fork(leaf, k));  // (n (n n n) n)  
    seeds.push(fork_leaf_k);
    
    let fork_stem_stem = g.add(Node::Fork(stem1, stem1));  // ((n n) (n n) n)
    seeds.push(fork_stem_stem);
    
    let fork_k_k = g.add(Node::Fork(k, k));  // ((n n n) (n n n) n)
    seeds.push(fork_k_k);
    
    // CRITICAL: Triage patterns for first/rest
    // For Fork(a,b) argument: triage{w,x,y} Fork(a,b) → y a b
    
    // first = triage{_, _, K} where K a b = a
    // Fork(Fork(w,x), K) = triage with fork-handler = K
    let triage_first = g.add(Node::Fork(k, k));  // triage{n,n,K}
    seeds.push(triage_first);
    
    // rest = triage{_, _, K'} where K' a b = b  
    // We need Fork(Fork(_,_), second) where second returns 2nd arg
    // second = Fork(Leaf, K) but that gives K when applied to first arg
    // Actually: Fork(Fork(Leaf,Leaf), Fork(Leaf, K)) for triage
    let triage_rest_inner = g.add(Node::Fork(k, k_prime));  // K' gets second
    seeds.push(triage_rest_inner);
    
    let triage_rest = g.add(Node::Fork(k, fork_leaf_k));  // Another attempt
    seeds.push(triage_rest);
    
    // More K' variations (getting second element)
    let k_prime2 = g.add(Node::Fork(k, stem1));  // Use stem for different behavior
    seeds.push(k_prime2);
    
    // Add stems of key patterns
    let stem_k_prime = g.add(Node::Stem(k_prime));
    seeds.push(stem_k_prime);
    
    let stem_fork_k_k = g.add(Node::Stem(fork_k_k));
    seeds.push(stem_fork_k_k);
    
    // Deep triage nesting
    let deep_triage1 = g.add(Node::Fork(fork_k_leaf, k));
    seeds.push(deep_triage1);
    
    let deep_triage2 = g.add(Node::Fork(fork_k_leaf, k_prime));
    seeds.push(deep_triage2);
    
    let deep_triage3 = g.add(Node::Fork(fork_leaf_k, leaf));
    seeds.push(deep_triage3);
    
    let deep_triage4 = g.add(Node::Fork(fork_leaf_k, k));
    seeds.push(deep_triage4);
    
    // Fork variations with K and K'
    let fork_k_kprime = g.add(Node::Fork(k, k_prime));
    seeds.push(fork_k_kprime);
    
    let fork_kprime_k = g.add(Node::Fork(k_prime, k));
    seeds.push(fork_kprime_k);
    
    let fork_kprime_leaf = g.add(Node::Fork(k_prime, leaf));
    seeds.push(fork_kprime_leaf);
    
    let fork_leaf_kprime = g.add(Node::Fork(leaf, k_prime));
    seeds.push(fork_leaf_kprime);
    
    // Stem variations
    let stem3 = g.add(Node::Stem(stem2));
    seeds.push(stem3);
    
    let stem_kprime = g.add(Node::Stem(k_prime));
    seeds.push(stem_kprime);
    
    // Add external seeds (dedup) only if library edits are enabled
    if synth.config.use_library_edits {
        for &s in extra_seeds {
            seeds.push(s);
        }
    }
    // Dedup
    let mut seen = std::collections::HashSet::new();
    seeds.retain(|id| seen.insert(*id));
    
    synth.seed(seeds);
    if synth.config.use_library_edits {
        synth.set_allowed_replacements(extra_seeds.to_vec());
    } else {
        synth.set_allowed_replacements(Vec::new());
    }
    
    synth.synthesize(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_igtc_basic() {
        let mut g = Graph::new();
        
        // Try to synthesize identity-like: input leaf, output leaf
        let input = g.add(Node::Leaf);
        let output = g.add(Node::Leaf);
        
        let config = IgtcConfig {
            max_iterations: 10,
            ..Default::default()
        };
        
        let result = synthesize(&mut g, vec![(input, output)], config);
        assert!(result.is_some());
    }
}
