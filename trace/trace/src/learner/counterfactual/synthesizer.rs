use crate::arena::{Graph, Node, NodeId};
use crate::engine::{set_reduce_progress_ms_override, tree_hash};
use crate::learner::counterfactual::cache::{
    candidate_entry_size, estimate_size, CandidateKey, CandidateStats, MemoCaches,
};
use crate::learner::counterfactual::config::CounterfactualConfig;
use crate::learner::counterfactual::diagnostics::*;
use crate::learner::counterfactual::evaluator::{
    evaluate_edit_on_examples_cached, evaluate_with_traces, program_stats_cached,
};
use crate::learner::counterfactual::utils::{
    collect_paths_map, mean, node_at_path, replace_at_path, unparse_limited, variance,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct EditPlan {
    pub path: Vec<u8>,
    pub replacement: NodeId,
    pub replacement_hash: u64,
    pub improvement: f64,
    pub depth: usize,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct EditKey {
    pub path: Vec<u8>,
    pub replacement_hash: u64,
}

#[derive(Clone, Copy)]
pub struct Replacement {
    pub node: NodeId,
    pub hash: u64,
}

fn hash_examples(g: &Graph, examples: &[(NodeId, NodeId)]) -> u64 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    for (input, expected) in examples {
        let input_hash = tree_hash(g, *input);
        let expected_hash = tree_hash(g, *expected);
        h.write_u64(input_hash);
        h.write_u64(expected_hash);
    }
    h.finish()
}

fn make_replacement(g: &Graph, node: NodeId) -> Replacement {
    Replacement {
        node,
        hash: tree_hash(g, node),
    }
}

fn threshold_for_blame(base: f64, blame: f64, total_blame: f64) -> f64 {
    let denom = if total_blame > 0.0 { total_blame } else { 1.0 };
    let prop = (blame / denom).clamp(0.0, 1.0);
    let decay = 7.0;
    let t = base * (-(decay * prop)).exp();
    if t < 1e-6 {
        0.0
    } else {
        t
    }
}

fn has_conflict(existing: &HashSet<Vec<u8>>, candidate: &[u8]) -> bool {
    for path in existing {
        if is_prefix(path, candidate) || is_prefix(candidate, path) {
            return true;
        }
    }
    false
}

fn is_prefix(a: &[u8], b: &[u8]) -> bool {
    if a.len() > b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

pub fn synthesize(
    g: &mut Graph,
    examples: Vec<(NodeId, NodeId)>,
    config: CounterfactualConfig,
) -> Option<NodeId> {
    if examples.is_empty() {
        return None;
    }

    let examples_hash = hash_examples(g, &examples);
    let caches = MemoCaches::new(config.memo_budget_bytes, config.max_replacement_nodes);
    set_debug_oom(config.debug);
    if config.debug {
        set_reduce_progress_ms_override(None);
    } else {
        set_reduce_progress_ms_override(Some(0));
    }
    let eval_budget = 512 * 1024 * 1024;
    set_eval_budget_bytes(eval_budget);
    let seed = select_seed(g, &examples, examples_hash, &config, &caches);
    let seed_candidates = {
        let leaf = g.add(Node::Leaf);
        vec![leaf]
    };
    let mut seed_cursor = seed_candidates
        .iter()
        .position(|&s| s == seed)
        .unwrap_or(0);
    let mut current = seed;
    let mut best_program = seed;
    let mut best_score = f64::INFINITY;
    let mut best_mean = f64::INFINITY;
    let mut best_var = f64::INFINITY;
    let mut history: Vec<(NodeId, u64, EditKey)> = Vec::new();
    let mut banned_by_program: HashMap<u64, HashSet<EditKey>> = HashMap::new();
    let mut banned_paths_by_program: HashMap<u64, HashSet<Vec<u8>>> = HashMap::new();
    if config.verbose {
        println!("Counterfactual learner: starting seed {}", seed.0);
    }

    for iter in 0..config.max_iterations {
        reset_peak();
        reset_cap_hit();
        reset_iter_counters();
        set_debug_context(Some(current), None, None);
        let current_hash = tree_hash(g, current);
        let iter_start = Instant::now();
        let (losses, active_paths) = evaluate_with_traces(
            g,
            current,
            &examples,
            config.max_eval_steps,
            config.max_eval_nodes,
            config.max_ted_cells,
            config.output_collision_penalty,
            Some(&caches),
        );
        let eval_time = iter_start.elapsed();
        if clear_cap_hit_if_allowed() {
            if let Some((parent, parent_hash, last_edit)) = history.pop() {
                banned_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.clone());
                banned_paths_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.path.clone());
                banned_paths_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.path.clone());
                if config.verbose || config.debug {
                    println!(
                        "  Cap hit; backtracking and banning edit {} -> {}",
                        path_to_string(&last_edit.path),
                        last_edit.replacement_hash
                    );
                }
                current = parent;
                report_peak(iter);
                continue;
            } else {
                if config.verbose || config.debug {
                    println!("  Cap hit; stopping (no parent).");
                }
                report_peak(iter);
                break;
            }
        }
        let base_mean = mean(&losses);
        let base_var = variance(&losses);
        let base_score = base_mean + config.variance_weight * base_var;

        if base_score < best_score {
            best_score = base_score;
            best_mean = base_mean;
            best_var = base_var;
            best_program = current;
            if config.verbose {
                println!(
                    "  Best so far: mean_loss={:.4} var={:.4} score={:.4}",
                    best_mean, best_var, best_score
                );
            }
        }

        if config.verbose {
            println!(
                "Iter {}: mean_loss={:.4} var={:.4}",
                iter, base_mean, base_var
            );
        }
        if config.verbose {
            println!("  Program: {}", unparse_limited(g, current));
            let total_events: usize = active_paths.iter().map(|v| v.len()).sum();
            println!("  Trace events total: {}", total_events);
            for (idx, loss) in losses.iter().enumerate() {
                let events = active_paths.get(idx).map(|v| v.len()).unwrap_or(0);
                println!("  Example {}: loss={:.4} events={}", idx, loss, events);
            }
            println!("  Eval time: {:?}\n", eval_time);
        }

        if base_mean <= 1e-9 {
            return Some(current);
        }

        let banned = banned_by_program.get(&current_hash);
        let banned_paths = banned_paths_by_program.get(&current_hash);
        reset_cap_hit();
        let edit = candidate_edits(
            g,
            current,
            &examples,
            &losses,
            &active_paths,
            base_mean,
            base_var,
            &config,
            config.verbose,
            banned,
            banned_paths,
            examples_hash,
            &caches,
        );
        if config.debug {
            let (paths_scanned, candidate_evals) = snapshot_iter_counters();
            println!(
                "  Iteration stats: examples={} paths_scanned={} candidate_evals={}",
                examples.len(),
                paths_scanned,
                candidate_evals
            );
        }

        if clear_cap_hit_if_allowed() {
            if let Some((parent, parent_hash, last_edit)) = history.pop() {
                banned_by_program
                    .entry(parent_hash)
                    .or_insert_with(HashSet::new)
                    .insert(last_edit.clone());
                if config.verbose || config.debug {
                    println!(
                        "  Cap hit during edit search; backtracking and banning edit {} -> {}",
                        path_to_string(&last_edit.path),
                        last_edit.replacement_hash
                    );
                }
                current = parent;
                report_peak(iter);
                continue;
            } else {
                if config.verbose || config.debug {
                    println!("  Cap hit during edit search; stopping (no parent).");
                }
                report_peak(iter);
                break;
            }
        }

        let edit = match edit {
            Some(edit) => edit,
            None => {
                if let Some((parent, parent_hash, last_edit)) = history.pop() {
                    banned_by_program
                        .entry(parent_hash)
                        .or_insert_with(HashSet::new)
                        .insert(last_edit.clone());
                    banned_paths_by_program
                        .entry(parent_hash)
                        .or_insert_with(HashSet::new)
                        .insert(last_edit.path.clone());
                    if config.verbose || config.debug {
                        println!(
                            "  No edits (max depth reached); backtracking and banning edit {} -> {}",
                            path_to_string(&last_edit.path),
                            last_edit.replacement_hash
                        );
                    }
                    current = parent;
                    report_peak(iter);
                    continue;
                } else {
                    let mut restart = None;
                    for _ in 0..seed_candidates.len() {
                        seed_cursor = (seed_cursor + 1) % seed_candidates.len();
                        let cand = seed_candidates[seed_cursor];
                        if cand != current {
                            restart = Some(cand);
                            break;
                        }
                    }
                    if let Some(cand) = restart {
                        if config.verbose || config.debug {
                            println!("  No edits found; restarting from seed {}", cand.0);
                        }
                        current = cand;
                        report_peak(iter);
                        continue;
                    }
                    if config.verbose || config.debug {
                        println!("  No edits found; stopping.");
                    }
                    report_peak(iter);
                    break;
                }
            }
        };

        if config.verbose {
            println!(
                "  Chosen edit:\n    path {} depth={} improvement={:.6} replacement={}",
                path_to_string(&edit.path),
                edit.depth,
                edit.improvement,
                unparse_limited(g, edit.replacement)
            );
        }

        set_debug_context(Some(current), Some(&edit.path), Some(edit.depth));
        if crate::learner::counterfactual::evaluator::check_global_limit(g, &config, "apply_replacement") {
            return Some(best_program);
        }
        let child = replace_at_path(g, current, &edit.path, edit.replacement);
        if crate::learner::counterfactual::evaluator::check_global_limit(g, &config, "apply_edit") {
            return Some(best_program);
        }
        let child_stats = program_stats_cached(
            g,
            child,
            &examples,
            config.max_eval_steps,
            config.max_eval_nodes,
            config.max_ted_cells,
            config.output_collision_penalty,
            &caches,
        );
        if clear_cap_hit_if_allowed() {
            if config.verbose || config.debug {
                println!(
                    "  Cap hit during child eval; banning edit {} -> {}",
                    path_to_string(&edit.path),
                    edit.replacement_hash
                );
            }
            banned_by_program
                .entry(current_hash)
                .or_insert_with(HashSet::new)
                .insert(EditKey {
                    path: edit.path.clone(),
                    replacement_hash: edit.replacement_hash,
                });
            report_peak(iter);
            continue;
        }
        let child_score = child_stats.mean + config.variance_weight * child_stats.var;
        if child_stats.mean > base_mean && child_stats.var > base_var || child_score > base_score {
            if config.verbose || config.debug {
                println!(
                    "  Rejected edit (worse): score {:.4}->{:.4} mean {:.4}->{:.4} var {:.4}->{:.4}",
                    base_score,
                    child_score,
                    base_mean,
                    child_stats.mean,
                    base_var,
                    child_stats.var
                );
            }
            banned_by_program
                .entry(current_hash)
                .or_insert_with(HashSet::new)
                .insert(EditKey {
                    path: edit.path.clone(),
                    replacement_hash: edit.replacement_hash,
                });
            report_peak(iter);
            continue;
        }
        history.push((
            current,
            current_hash,
            EditKey {
                path: edit.path.clone(),
                replacement_hash: edit.replacement_hash,
            },
        ));
        current = child;
        report_peak(iter);
    }

    if config.debug {
        println!(
            "  Final best: mean_loss={:.4} var={:.4} score={:.4}",
            best_mean, best_var, best_score
        );
        println!("  Best program: {}", unparse_limited(g, best_program));
    }
    Some(best_program)
}

fn select_best_edit(edits: Vec<EditPlan>) -> Option<EditPlan> {
    edits.into_iter().max_by(|a, b| {
        a.improvement
            .partial_cmp(&b.improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn candidate_edits(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    losses: &[f64],
    active_paths: &[Vec<Vec<u8>>],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    log_search: bool,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
    examples_hash: u64,
    caches: &MemoCaches,
) -> Option<EditPlan> {
    set_debug_context(Some(program), None, None);
    if clear_cap_hit_if_allowed() {
        return None;
    }
    let total_loss: f64 = losses.iter().sum();
    let path_map = collect_paths_map(g, program);
    if config.max_eval_nodes > 0 && path_map.len() > config.max_eval_nodes {
        log_path_cap_hit("collect_paths_map", path_map.len(), config.max_eval_nodes);
        return None;
    }
    let mut all_program_paths: HashSet<Vec<u8>> = HashSet::new();
    for paths in path_map.values() {
        for p in paths {
            all_program_paths.insert(p.clone());
        }
    }
    if config.max_eval_nodes > 0 && all_program_paths.len() > config.max_eval_nodes {
        log_path_cap_hit(
            "collect_program_paths",
            all_program_paths.len(),
            config.max_eval_nodes,
        );
        return None;
    }
    let mut path_blame: HashMap<Vec<u8>, f64> = HashMap::new();
    for (idx, paths) in active_paths.iter().enumerate() {
        let loss = losses[idx];
        if loss <= 0.0 {
            continue;
        }
        let events = paths.len();
        if events == 0 {
            continue;
        }
        let per_event = loss / (events as f64);
        for path in paths {
            *path_blame.entry(path.clone()).or_insert(0.0) += per_event;
        }
    }

    if path_blame.is_empty() {
        if total_loss > 0.0 {
            eprintln!(
                "ERROR: zero blame with nonzero loss; trace_events={}",
                active_paths.iter().map(|v| v.len()).sum::<usize>()
            );
            panic!("Zero blame with nonzero loss");
        }
        return None;
    }

    let mut all_paths: HashSet<Vec<u8>> = HashSet::new();
    for path in path_blame.keys() {
        let mut p = path.clone();
        loop {
            all_paths.insert(p.clone());
            if p.is_empty() {
                break;
            }
            p.pop();
        }
    }

    let mut ordered_paths: Vec<Vec<u8>> = all_paths.into_iter().collect();
    ordered_paths.sort_by(|a, b| b.len().cmp(&a.len()));

    if log_search {
        println!("  Candidate program: {}", unparse_limited(g, program));
        let total_events: usize = active_paths.iter().map(|v| v.len()).sum();
        println!("  Trace events total: {}", total_events);
        for (idx, loss) in losses.iter().enumerate() {
            let events = active_paths.get(idx).map(|v| v.len()).unwrap_or(0);
            println!("  Example {}: loss={:.4} events={}", idx, loss, events);
        }
        println!("  mean_loss={:.4} var={:.4}", base_mean, base_var);
        let mut blame_list: Vec<(Vec<u8>, f64)> = path_blame
            .iter()
            .map(|(p, b)| (p.clone(), *b))
            .collect();
        blame_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  Blame map (path -> blame):");
        for (p, b) in blame_list {
            println!("    {} -> {:.6}", path_to_string(&p), b);
        }
    }

    let mut all_program_paths: Vec<Vec<u8>> = all_program_paths.into_iter().collect();
    all_program_paths.sort_by(|a, b| b.len().cmp(&a.len()));

    let base_path_blame = path_blame.clone();
    let total_blame: f64 = base_path_blame.values().sum();
    let max_depth = config.max_edit_depth.max(1);
    for depth in 1..=max_depth {
        if log_search {
            println!("  Candidate edit depth {}", depth);
        }

        let local = collect_local_edits_with_propagation(
            g,
            program,
            &ordered_paths,
            &base_path_blame,
            total_blame,
            examples,
            base_mean,
            base_var,
            config,
            depth,
            log_search,
            banned,
            banned_paths,
            examples_hash,
            caches,
        );
        if clear_cap_hit_if_allowed() {
            return None;
        }
        if !local.is_empty() {
            return select_best_edit(local);
        }

        if log_search {
            println!(
                "  No passing local edits; skipping non-local scan at depth {}",
                depth
            );
        }
    }

    None
}

fn select_seed(
    g: &mut Graph,
    examples: &[(NodeId, NodeId)],
    _examples_hash: u64,
    config: &CounterfactualConfig,
    caches: &MemoCaches,
) -> NodeId {
    let leaf = g.add(Node::Leaf);
    // Always start from Leaf.
    let _ = (examples, config, caches);
    leaf
}

fn candidate_stats(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    examples_hash: u64,
    caches: &MemoCaches,
    _log_timing: bool,
) -> CandidateStats {
    let log_timing = false;
    let replacement_hash = tree_hash(g, replacement);
    let key = CandidateKey {
        program_hash: tree_hash(g, program),
        path: path.to_vec(),
        replacement_hash,
        examples_hash,
        max_steps,
        max_nodes,
        max_ted_cells,
    };
    if let Some(stats) = caches.get_candidate(&key) {
        if log_timing {
            println!(
                "    Candidate cached path={} repl_nodes={} repl_hash={}",
                path_to_string(path),
                estimate_size(g, replacement),
                replacement_hash
            );
        }
        return stats;
    }
    if log_timing {
        println!(
            "    Candidate start path={} repl_nodes={} repl_hash={} evals={}",
            path_to_string(path),
            estimate_size(g, replacement),
            replacement_hash,
            examples.len()
        );
    }
    let timing = if log_timing {
        Some(Arc::new(Mutex::new(TimingAcc::default())))
    } else {
        None
    };
    if let Some(acc) = timing.as_ref() {
        set_timing_acc(Some(acc.clone()));
    }
    let wall_start = Instant::now();
    let losses = evaluate_edit_on_examples_cached(
        g,
        program,
        path,
        replacement,
        examples,
        max_steps,
        max_nodes,
        max_ted_cells,
        output_collision_penalty,
        caches,
    );
    let wall_elapsed = wall_start.elapsed();
    if timing.is_some() {
        set_timing_acc(None);
        let acc = timing.unwrap();
        let acc = acc.lock().unwrap();
        println!(
            "    Candidate timing path={} repl_nodes={} repl_hash={} evals={} wall={:?} cpu_total={:?} reduce_actual={:?} reduce_expected={:?} ted={:?} cache_hits(actual={}, expected={})",
            path_to_string(path),
            estimate_size(g, replacement),
            replacement_hash,
            acc.evals,
            wall_elapsed,
            acc.total,
            acc.reduce_actual,
            acc.reduce_expected,
            acc.ted,
            acc.reduce_actual_hits,
            acc.reduce_expected_hits
        );
    }
    let mean_loss = mean(&losses);
    let var_loss = variance(&losses);
    let stats = CandidateStats {
        mean: mean_loss,
        var: var_loss,
    };
    caches.insert_candidate(key.clone(), stats, candidate_entry_size(&key));
    stats
}

fn best_edit_for_path_with_depth(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    examples: &[(NodeId, NodeId)],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    depth: usize,
    max_edits: usize,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
    examples_hash: u64,
    caches: &MemoCaches,
) -> (Option<EditPlan>, usize) {
    if let Some(banned_paths) = banned_paths {
        if banned_paths.contains(path) {
            return (None, 0);
        }
    }
    let target = match node_at_path(g, program, path) {
        Some(t) => t,
        None => return (None, 0),
    };
    bump_iter_paths_scanned(1);
    set_debug_context(Some(program), Some(path), Some(depth));
    if config.max_eval_nodes > 0 && estimate_size(g, target) > config.max_eval_nodes {
        log_repr_cap_hit(
            "best_edit_target",
            estimate_size(g, target),
            config.max_eval_nodes,
        );
        return (None, 0);
    }
    let mut best: Option<EditPlan> = None;
    let mut evaluated = 0usize;
    let cap = config.max_candidate_evals.max(1);
    let max_edits = max_edits.max(1);
    let mut cap_seen = false;
    let replacements = generate_edits_with_depth(
        g,
        program,
        path,
        depth,
        max_edits,
        config.max_replacement_nodes,
    );
    let generated = replacements.len();
    for rep in replacements {
        if evaluated >= cap {
            break;
        }
        if let Some(banned) = banned {
            let key = EditKey {
                path: path.to_vec(),
                replacement_hash: rep.hash,
            };
            if banned.contains(&key) {
                continue;
            }
        }
        evaluated += 1;
        bump_iter_candidate_evals(1);
        reset_cap_hit();
        let stats = candidate_stats(
            g,
            program,
            path,
            rep.node,
            examples,
            config.max_eval_steps,
            config.max_eval_nodes,
            config.max_ted_cells,
            config.output_collision_penalty,
            examples_hash,
            caches,
            config.debug,
        );
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                cap_seen = true;
                break;
            }
            continue;
        }
        let improvement =
            (base_mean - stats.mean) + config.variance_weight * (base_var - stats.var);
        let plan = EditPlan {
            path: path.to_vec(),
            replacement: rep.node,
            replacement_hash: rep.hash,
            improvement,
            depth,
        };
        match &best {
            Some(existing) if existing.improvement >= plan.improvement => {}
            _ => best = Some(plan),
        }
    }
    if cap_seen {
        return (None, generated);
    }
    if evaluated >= cap && config.debug {
        println!(
            "  Candidate cap hit at path {} depth {}: {} -> {}",
            path_to_string(path),
            depth,
            evaluated,
            cap
        );
    }
    (best, generated)
}

fn collect_local_edits_with_propagation(
    g: &mut Graph,
    program: NodeId,
    ordered_paths: &[Vec<u8>],
    base_path_blame: &HashMap<Vec<u8>, f64>,
    total_blame: f64,
    examples: &[(NodeId, NodeId)],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    depth: usize,
    log_search: bool,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
    examples_hash: u64,
    caches: &MemoCaches,
) -> Vec<EditPlan> {
    let mut path_blame = base_path_blame.clone();
    let mut edits: Vec<EditPlan> = Vec::new();
    let mut best_positive: Option<EditPlan> = None;
    let mut edited_paths: HashSet<Vec<u8>> = HashSet::new();
    let mut remaining = config.max_candidate_evals.max(1);

    for path in ordered_paths {
        if remaining == 0 {
            if log_search {
                println!("  Local edit budget exhausted; stopping local scan.");
            }
            break;
        }
        if let Some(banned_paths) = banned_paths {
            if banned_paths.contains(path) {
                continue;
            }
        }
        let blame = path_blame.get(path).copied().unwrap_or(0.0);
        if blame <= 0.0 {
            continue;
        }
        if has_conflict(&edited_paths, path) {
            continue;
        }
        let threshold = threshold_for_blame(config.edit_threshold_base, blame, total_blame);
        let path_budget = remaining.min(config.max_edits_per_path.max(1));
        let (best, generated) = best_edit_for_path_with_depth(
            g,
            program,
            path,
            examples,
            base_mean,
            base_var,
            config,
            depth,
            path_budget,
            banned,
            banned_paths,
            examples_hash,
            caches,
        );
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                return Vec::new();
            }
            continue;
        }
        remaining = remaining.saturating_sub(generated);
        if log_search {
            let (best_improve, best_rep) = match &best {
                Some(plan) => (plan.improvement, unparse_limited(g, plan.replacement)),
                None => (0.0, "<none>".to_string()),
            };
            println!(
                "  Path {} blame={:.4} threshold={:.6} best_improve={:.6} best={}",
                path_to_string(path),
                blame,
                threshold,
                best_improve,
                best_rep
            );
        }
        if let Some(plan) = best.as_ref() {
            if plan.improvement > 0.0
                && best_positive
                    .as_ref()
                    .map(|p| plan.improvement > p.improvement)
                    .unwrap_or(true)
            {
                best_positive = Some(plan.clone());
            }
        }
        if let Some(plan) = best {
            if plan.improvement > threshold {
                edited_paths.insert(path.clone());
                edits.push(plan);
                continue;
            }
        }

        if path.is_empty() {
            if let Some(plan) = propagate_downward(
                g,
                program,
                path.clone(),
                blame,
                total_blame,
                examples,
                base_mean,
                base_var,
                config,
                &mut edited_paths,
                examples_hash,
                caches,
                banned,
                banned_paths,
            ) {
                edits.push(plan);
            }
            if cap_hit() {
                if clear_cap_hit_if_allowed() {
                    return Vec::new();
                }
                continue;
            }
        } else {
            let mut parent = path.clone();
            parent.pop();
            *path_blame.entry(parent).or_insert(0.0) += blame;
        }
    }

    if edits.is_empty() {
        if let Some(plan) = best_positive {
            edits.push(plan);
        }
    }
    edits
}

fn propagate_downward(
    g: &mut Graph,
    program: NodeId,
    path: Vec<u8>,
    blame: f64,
    total_blame: f64,
    examples: &[(NodeId, NodeId)],
    base_mean: f64,
    base_var: f64,
    config: &CounterfactualConfig,
    edited_paths: &mut HashSet<Vec<u8>>,
    examples_hash: u64,
    caches: &MemoCaches,
    banned: Option<&HashSet<EditKey>>,
    banned_paths: Option<&HashSet<Vec<u8>>>,
) -> Option<EditPlan> {
    #[derive(Clone)]
    struct WorkItem {
        path: Vec<u8>,
        blame: f64,
    }

    let mut stack: Vec<WorkItem> = Vec::new();
    let mut seen: HashSet<(NodeId, Vec<u8>)> = HashSet::new();
    stack.push(WorkItem { path, blame });

    while let Some(item) = stack.pop() {
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                return None;
            }
            continue;
        }
        if let Some(banned_paths) = banned_paths {
            if banned_paths.contains(&item.path) {
                continue;
            }
        }
        if item.blame <= 0.0 || has_conflict(edited_paths, &item.path) {
            continue;
        }
        let threshold = threshold_for_blame(config.edit_threshold_base, item.blame, total_blame);
        let (best, _) = best_edit_for_path_with_depth(
            g,
            program,
            &item.path,
            examples,
            base_mean,
            base_var,
            config,
            1,
            config.max_edits_per_path.max(1),
            banned,
            banned_paths,
            examples_hash,
            caches,
        );
        if cap_hit() {
            if clear_cap_hit_if_allowed() {
                return None;
            }
            continue;
        }
        if let Some(plan) = best {
            if plan.improvement > threshold {
                edited_paths.insert(item.path.clone());
                return Some(plan);
            }
        }

        let Some(node) = node_at_path(g, program, &item.path) else { continue; };
        if !seen.insert((node, item.path.clone())) {
            continue;
        }
        match g.get(node).clone() {
            Node::Stem(_) => {
                let mut child = item.path.clone();
                child.push(0);
                stack.push(WorkItem {
                    path: child,
                    blame: item.blame * 0.5,
                });
            }
            Node::Fork(_, _) => {
                let mut right = item.path.clone();
                right.push(1);
                let mut left = item.path.clone();
                left.push(0);
                stack.push(WorkItem {
                    path: right,
                    blame: item.blame * 0.5,
                });
                stack.push(WorkItem {
                    path: left,
                    blame: item.blame * 0.5,
                });
            }
            Node::App { func: _, args } => {
                let mut func_path = item.path.clone();
                func_path.push(0);
                for (idx, _arg) in args.iter().enumerate().take(4).rev() {
                    let mut arg_path = item.path.clone();
                    arg_path.push((idx as u8) + 1);
                    if node_at_path(g, program, &arg_path).is_some() {
                        stack.push(WorkItem {
                            path: arg_path,
                            blame: item.blame * 0.5,
                        });
                    }
                }
                if node_at_path(g, program, &func_path).is_some() {
                    stack.push(WorkItem {
                        path: func_path,
                        blame: item.blame * 0.5,
                    });
                }
            }
            Node::Ind(inner) => {
                if inner != node {
                    stack.push(WorkItem {
                        path: item.path,
                        blame: item.blame * 0.5,
                    });
                }
            }
            _ => {}
        }
    }

    None
}

fn generate_edits(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    max_nodes: usize,
) -> Vec<Replacement> {
    let target = match node_at_path(g, program, path) {
        Some(t) => t,
        None => return Vec::new(),
    };
    generate_replacements_for_node(g, target, max_nodes)
}

fn generate_edits_with_depth(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    depth: usize,
    max_edits: usize,
    max_nodes: usize,
) -> Vec<Replacement> {
    let max_edits = max_edits.max(1);
    if depth <= 1 {
        let mut edits = generate_edits(g, program, path, max_nodes);
        if edits.len() > max_edits {
            edits.truncate(max_edits);
        }
        return edits;
    }
    let mut seen: HashSet<u64> = HashSet::new();
    let mut all: Vec<Replacement> = Vec::new();
    let mut frontier: Vec<Replacement> = Vec::new();
    let base = generate_edits(g, program, path, max_nodes);
    for rep in base {
        if seen.insert(rep.hash) {
            frontier.push(rep.clone());
            all.push(rep);
            if all.len() >= max_edits {
                return all;
            }
        }
    }

    for _ in 2..=depth {
        let mut next_frontier: Vec<Replacement> = Vec::new();
        'frontier: for node in &frontier {
            for rep in generate_replacements_for_node(g, node.node, max_nodes) {
                if seen.insert(rep.hash) {
                    next_frontier.push(rep.clone());
                    all.push(rep);
                    if all.len() >= max_edits {
                        return all;
                    }
                }
            }

            let mut child_paths: HashSet<Vec<u8>> = HashSet::new();
            let path_map = collect_paths_map(g, node.node);
            for paths in path_map.values() {
                for p in paths {
                    if p.is_empty() {
                        continue;
                    }
                    child_paths.insert(p.clone());
                }
            }

            for path in child_paths {
                let edits = generate_edits(g, node.node, &path, max_nodes);
                for rep in edits {
                    let new_root = replace_at_path(g, node.node, &path, rep.node);
                    if max_nodes > 0 && estimate_size(g, new_root) > max_nodes {
                        continue;
                    }
                    let new_rep = make_replacement(g, new_root);
                    if seen.insert(new_rep.hash) {
                        next_frontier.push(new_rep.clone());
                        all.push(new_rep);
                        if all.len() >= max_edits {
                            return all;
                        }
                    }
                }
                if all.len() >= max_edits {
                    break 'frontier;
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    all
}

fn generate_replacements_for_node(
    g: &mut Graph,
    target: NodeId,
    max_nodes: usize,
) -> Vec<Replacement> {
    let mut edits: Vec<Replacement> = Vec::new();
    let leaf = g.add(Node::Leaf);
    let fork_leaf_leaf = g.add(Node::Fork(leaf, leaf));

    fn push_if_small(g: &Graph, node: NodeId, max_nodes: usize, out: &mut Vec<Replacement>) {
        if max_nodes == 0 || estimate_size(g, node) <= max_nodes {
            out.push(make_replacement(g, node));
        }
    }

    let target = g.resolve(target);
    match g.get(target) {
        Node::Leaf => {
            let stem = g.add(Node::Stem(leaf));
            push_if_small(g, stem, max_nodes, &mut edits);
            push_if_small(g, fork_leaf_leaf, max_nodes, &mut edits);
        }
        Node::Stem(inner) => {
            let child = g.resolve(*inner);
            let fork_cl = g.add(Node::Fork(child, leaf));
            let fork_lc = g.add(Node::Fork(leaf, child));
            let fork_cc = g.add(Node::Fork(child, child));
            push_if_small(g, fork_leaf_leaf, max_nodes, &mut edits);
            push_if_small(g, fork_cl, max_nodes, &mut edits);
            push_if_small(g, fork_lc, max_nodes, &mut edits);
            push_if_small(g, fork_cc, max_nodes, &mut edits);
        }
        Node::Fork(left, right) => {
            let l = g.resolve(*left);
            let r = g.resolve(*right);
            let stem_l = g.add(Node::Stem(l));
            let stem_r = g.add(Node::Stem(r));
            let fork_rl = g.add(Node::Fork(r, l));
            let fork_ll = g.add(Node::Fork(l, l));
            let fork_rr = g.add(Node::Fork(r, r));
            push_if_small(g, leaf, max_nodes, &mut edits);
            push_if_small(g, stem_l, max_nodes, &mut edits);
            push_if_small(g, stem_r, max_nodes, &mut edits);
            push_if_small(g, fork_rl, max_nodes, &mut edits);
            push_if_small(g, fork_ll, max_nodes, &mut edits);
            push_if_small(g, fork_rr, max_nodes, &mut edits);
        }
        _ => {}
    }

    edits
}
