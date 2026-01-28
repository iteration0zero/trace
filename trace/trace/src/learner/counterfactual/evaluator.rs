use crate::arena::{Graph, Node, NodeId};
use crate::engine::{reduce, tree_hash, EvalContext};
use crate::learner::loss::tree_edit_distance_capped;
use crate::trace::{ExecutionTrace, RuleId};
use crate::learner::counterfactual::cache::{
    estimate_size, loss_entry_size, CandidateStats, LossEntry, LossKey, LossSample, MemoCaches,
};
use crate::learner::counterfactual::config::CounterfactualConfig;
use crate::learner::counterfactual::diagnostics::*;
use crate::learner::counterfactual::utils::{
    clone_subtree, collect_paths_map, replace_at_path,
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

pub fn check_global_limit(g: &Graph, config: &CounterfactualConfig, phase: &'static str) -> bool {
    if config.max_global_nodes == 0 {
        return false;
    }
    let nodes = g.nodes.len();
    if nodes <= config.max_global_nodes {
        return false;
    }
    mark_cap_hit();
    if config.debug {
        let ctx = get_debug_context();
        let path_str = ctx
            .path
            .as_ref()
            .map(|p| path_to_string(p))
            .unwrap_or_else(|| "<none>".to_string());
        let program_str = ctx
            .program
            .map(|p| p.0.to_string())
            .unwrap_or_else(|| "<none>".to_string());
        let depth_str = ctx
            .depth
            .map(|d| d.to_string())
            .unwrap_or_else(|| "<none>".to_string());
        eprintln!(
            "GLOBAL_NODE_LIMIT_HIT phase={} nodes={} max_nodes={} program={} path={} depth={}",
            phase, nodes, config.max_global_nodes, program_str, path_str, depth_str
        );
    }
    true
}

pub fn preflight_eval_limit(
    g: &Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_nodes: usize,
    max_ted_cells: usize,
    phase: &'static str,
) -> Option<(usize, usize, usize)> {
    let _ = (max_nodes, max_ted_cells, phase);
    let prog_size = estimate_size(g, program);
    let input_size = estimate_size(g, input);
    let expected_size = estimate_size(g, expected);
    Some((prog_size, input_size, expected_size))
}

pub fn preflight_eval_limit_with_replacement(
    g: &Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    input: NodeId,
    expected: NodeId,
    max_nodes: usize,
    max_ted_cells: usize,
    phase: &'static str,
) -> bool {
    let _ = (
        g,
        program,
        path,
        replacement,
        input,
        expected,
        max_nodes,
        max_ted_cells,
        phase,
    );
    true
}

pub fn evaluate_with_traces(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: Option<&MemoCaches>,
) -> (Vec<f64>, Vec<Vec<Vec<u8>>>) {
    let len = examples.len();
    let mut samples = Vec::with_capacity(len);
    let mut active_paths = Vec::with_capacity(len);
    for (idx, (input, expected)) in examples.iter().enumerate() {
        let (sample, active) = eval_with_trace(
            g,
            program,
            *input,
            *expected,
            max_steps,
            max_nodes,
            max_ted_cells,
            caches,
            idx,
        );
        samples.push(sample);
        active_paths.push(active);
    }
    let mut losses: Vec<f64> = samples.iter().map(|s| s.loss).collect();
    apply_output_collision_penalty(&mut losses, &samples, output_collision_penalty);
    (losses, active_paths)
}

fn apply_output_collision_penalty(losses: &mut [f64], samples: &[LossSample], penalty: f64) {
    if penalty <= 0.0 {
        return;
    }
    let mut expected_by_actual: HashMap<u64, HashSet<u64>> = HashMap::new();
    for sample in samples {
        expected_by_actual
            .entry(sample.actual_hash)
            .or_insert_with(HashSet::new)
            .insert(sample.expected_hash);
    }
    let mut penalty_by_actual: HashMap<u64, f64> = HashMap::new();
    for (actual_hash, expected_set) in expected_by_actual {
        if expected_set.len() > 1 {
            let bump = penalty * (expected_set.len() as f64 - 1.0);
            penalty_by_actual.insert(actual_hash, bump);
        }
    }
    for (idx, sample) in samples.iter().enumerate() {
        if let Some(bump) = penalty_by_actual.get(&sample.actual_hash) {
            losses[idx] += *bump;
        }
    }
}

pub fn eval_with_trace(
    g: &mut Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    _caches: Option<&MemoCaches>,
    example_idx: usize,
) -> (LossSample, Vec<Vec<u8>>) {
    let program_resolved = g.resolve(program);
    let _ = preflight_eval_limit(
        g,
        program_resolved,
        input,
        expected,
        max_nodes,
        max_ted_cells,
        "trace_preflight",
    );

    // Build an isolated evaluation graph so reductions cannot mutate shared nodes.
    let mut eval_g = Graph::new_uninterned();
    let mut prog_memo = HashMap::new();
    let mut eval_to_orig: HashMap<NodeId, NodeId> = HashMap::new();
    let program_eval = clone_subtree(
        g,
        &mut eval_g,
        program_resolved,
        &mut prog_memo,
        Some(&mut eval_to_orig),
    );
    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);
    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let base_nodes = eval_g.nodes.len();
    let applied = eval_g.add(Node::App {
        func: program_eval,
        args: smallvec::smallvec![input_eval],
    });

    let size = estimate_size(&eval_g, applied);
    let budget = scaled_eval_budget(max_steps, size);
    let mut trace = ExecutionTrace::new();
    record_initial_canonicalization(&mut eval_g, &mut trace, program_eval, input_eval, applied);
    let mut actual = applied;
    let mut steps = 0usize;
    let mut step_limit = budget;
    let mut limit_hit = false;
    let mut node_limit_hit = false;
    let mut step_limit_hit = false;
    let mut ctx = EvalContext::default();
    ctx.step_limit = budget;
    ctx.node_limit = max_nodes;
    ctx.base_nodes = base_nodes;
    ctx.depth_limit = 1000;
    ctx.exec_trace = Some(&mut trace);
    ctx.redex_cache = None;
    actual = reduce(&mut eval_g, applied, &mut ctx);
    steps = ctx.steps;
    step_limit = ctx.step_limit;
    node_limit_hit = ctx.node_limit_hit;
    step_limit_hit = ctx.step_limit_hit;
    limit_hit = node_limit_hit || step_limit_hit;
    if limit_hit {
        let prog_size = estimate_size(&eval_g, program_eval);
        let input_size = estimate_size(&eval_g, input_eval);
        let expected_size = estimate_size(&eval_g, expected_eval);
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        eprintln!(
            "CAP_HIT trace_reduce_actual example program={} nodes={} max_nodes={} steps={} step_limit={} size={} prog_size={} input_size={} expected_size={} node_limit_hit={} step_limit_hit={}",
            program_resolved.0,
            eval_nodes,
            max_nodes,
            steps,
            step_limit,
            size,
            prog_size,
            input_size,
            expected_size,
            node_limit_hit,
            step_limit_hit
        );
        if node_limit_hit {
            log_node_limit_hit(
                "trace_reduce_actual",
                eval_nodes,
                max_nodes,
                steps,
                step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if step_limit_hit {
            log_step_limit_hit(
                "trace_reduce_actual",
                eval_nodes,
                max_nodes,
                steps,
                step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
    }
    if trace.events.is_empty() {
        // No trace info (e.g., pre-eval cap); continue without blame.
    }

    let expected_nf = eval_g.resolve(expected_eval);
    let actual_size = estimate_size(&eval_g, actual);
    let expected_size = estimate_size(&eval_g, expected_nf);
    let loss = if max_ted_cells > 0
        && (actual_size as u128) * (expected_size as u128) > max_ted_cells as u128
    {
        eprintln!(
            "CAP_HIT ted_cap example program={} actual_size={} expected_size={} cells={} cap={}",
            program_resolved.0,
            actual_size,
            expected_size,
            (actual_size as u128) * (expected_size as u128),
            max_ted_cells
        );
        log_ted_cap_hit(actual_size, expected_size, max_ted_cells);
        (actual_size + expected_size) as f64
    } else {
        match tree_edit_distance_capped(&eval_g, actual, expected_nf, max_ted_cells) {
            Ok(d) => d as f64,
            Err((a, b)) => {
                eprintln!(
                    "CAP_HIT ted_cap example program={} actual_size={} expected_size={} cells={} cap={}",
                    program_resolved.0,
                    a,
                    b,
                    (a as u128) * (b as u128),
                    max_ted_cells
                );
                log_ted_cap_hit(a, b, max_ted_cells);
                (a + b) as f64
            }
        }
    };
    if example_idx == 0 {
        eprintln!(
            "TED_TRACE idx={} actual_size={} expected_size={} loss={}",
            example_idx, actual_size, expected_size, loss
        );
    }

    let mut origin_path_map: HashMap<NodeId, Vec<u8>> = HashMap::new();
    for (node, paths) in collect_paths_map(g, program_resolved) {
        if let Some(best) = paths.into_iter().min_by_key(|p| p.len()) {
            origin_path_map.insert(node, best);
        }
    }
    let mut eval_to_path: HashMap<NodeId, Vec<u8>> = HashMap::new();
    for (eval_node, orig_node) in eval_to_orig {
        if let Some(path) = origin_path_map.get(&orig_node) {
            eval_to_path.insert(eval_node, path.clone());
        }
    }
    let mut active_paths: Vec<Vec<u8>> = Vec::new();
    for event in &trace.events {
        let active_eval = leftmost_head(&eval_g, event.redex);
        let path = map_active_to_path(&eval_g, active_eval, &origin_path_map, &eval_to_path);
        eval_to_path.entry(event.result).or_insert(path.clone());
        active_paths.push(path);
    }

    let prog_size = estimate_size(&eval_g, program_eval);
    let input_size = estimate_size(&eval_g, input_eval);
    let expected_size = estimate_size(&eval_g, expected_eval);
    let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
    update_peak(
        "trace_done",
        eval_nodes,
        steps,
        step_limit,
        size,
        prog_size,
        input_size,
        expected_size,
        get_debug_context(),
    );

    let actual_hash = tree_hash(&eval_g, actual);
    let expected_hash = tree_hash(&eval_g, expected_nf);
    (
        LossSample {
            loss,
            actual_hash,
            expected_hash,
        },
        active_paths,
    )
}

fn leftmost_head(g: &Graph, root: NodeId) -> NodeId {
    let mut curr = g.resolve(root);
    loop {
        match g.get(curr) {
            Node::App { func, .. } => {
                curr = g.resolve(*func);
            }
            _ => return curr,
        }
    }
}

fn map_active_to_path(
    eval_g: &Graph,
    active_eval: NodeId,
    origin_map: &HashMap<NodeId, Vec<u8>>,
    eval_to_path: &HashMap<NodeId, Vec<u8>>,
) -> Vec<u8> {
    if let Some(path) = origin_map
        .get(&active_eval)
        .or_else(|| eval_to_path.get(&active_eval))
    {
        return path.clone();
    }

    // Search within the active subtree for any mapped descendant.
    let mut stack = Vec::new();
    let mut seen = HashSet::new();
    stack.push(active_eval);
    let mut visited = 0usize;
    while let Some(node) = stack.pop() {
        let resolved = eval_g.resolve(node);
        if !seen.insert(resolved) {
            continue;
        }
        visited += 1;
        if visited > 512 {
            break;
        }
        if let Some(path) = origin_map
            .get(&resolved)
            .or_else(|| eval_to_path.get(&resolved))
        {
            return path.clone();
        }
        match eval_g.get(resolved) {
            Node::Stem(inner) => stack.push(*inner),
            Node::Fork(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Node::App { func, args } => {
                stack.push(*func);
                for arg in args {
                    stack.push(*arg);
                }
            }
            Node::Ind(inner) => stack.push(*inner),
            _ => {}
        }
    }

    Vec::new()
}

fn record_initial_canonicalization(
    g: &mut Graph,
    trace: &mut ExecutionTrace,
    func: NodeId,
    arg: NodeId,
    applied: NodeId,
) {
    if matches!(g.get(g.resolve(applied)), Node::App { .. }) {
        return;
    }
    let redex = g.resolve(func);
    trace.record(RuleId::App, redex, vec![arg], applied);
}

pub fn evaluate_program_on_examples_cached(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: &MemoCaches,
) -> Vec<f64> {
    let len = examples.len();
    let log_examples = false;
    let mut samples = Vec::with_capacity(len);
    for (idx, (input, expected)) in examples.iter().enumerate() {
        let ex_start = Instant::now();
        if log_examples {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            println!(
                "EXAMPLE_BEGIN phase=prog_example idx={} prog_size={} input_size={} expected_size={}",
                idx, prog_size, input_size, expected_size
            );
        }
        let loss = eval_loss_cached(
            g,
            program,
            *input,
            *expected,
            max_steps,
            max_nodes,
            max_ted_cells,
            caches,
            "prog_example",
            idx,
        );
        let ex_elapsed = ex_start.elapsed();
        if log_examples && ex_elapsed.as_millis() >= SLOW_EVAL_MS {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            log_slow_example(
                "prog_example",
                idx,
                ex_elapsed,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if log_examples {
            println!(
                "EXAMPLE_DONE phase=prog_example idx={} elapsed={:?}",
                idx, ex_elapsed
            );
        }
        samples.push(loss);
    }
    let mut losses: Vec<f64> = samples.iter().map(|s| s.loss).collect();
    apply_output_collision_penalty(&mut losses, &samples, output_collision_penalty);
    losses
}

pub fn evaluate_edit_on_examples_cached(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: &MemoCaches,
) -> Vec<f64> {
    let len = examples.len();
    let log_examples = false;
    let mut samples = Vec::with_capacity(len);
    for (idx, (input, expected)) in examples.iter().enumerate() {
        let ex_start = Instant::now();
        if log_examples {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            println!(
                "EXAMPLE_BEGIN phase=edit_example idx={} prog_size={} input_size={} expected_size={}",
                idx, prog_size, input_size, expected_size
            );
        }
        let loss = eval_loss_edit_uncached(
            g,
            program,
            path,
            replacement,
            *input,
            *expected,
            max_steps,
            max_nodes,
            max_ted_cells,
            Some(caches),
            "edit_example",
            idx,
        );
        let ex_elapsed = ex_start.elapsed();
        if log_examples && ex_elapsed.as_millis() >= SLOW_EVAL_MS {
            let prog_size = estimate_size(g, program);
            let input_size = estimate_size(g, *input);
            let expected_size = estimate_size(g, *expected);
            log_slow_example(
                "edit_example",
                idx,
                ex_elapsed,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if log_examples {
            println!(
                "EXAMPLE_DONE phase=edit_example idx={} elapsed={:?}",
                idx, ex_elapsed
            );
        }
        samples.push(loss);
    }
    let mut losses: Vec<f64> = samples.iter().map(|s| s.loss).collect();
    apply_output_collision_penalty(&mut losses, &samples, output_collision_penalty);
    losses
}

pub fn eval_loss_cached(
    g: &mut Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    caches: &MemoCaches,
    phase: &'static str,
    example_idx: usize,
) -> LossSample {
    let key = LossKey {
        program_hash: tree_hash(g, program),
        input_hash: tree_hash(g, input),
        expected_hash: tree_hash(g, expected),
        max_steps,
        max_nodes,
        max_ted_cells,
    };
    if let Some(v) = caches.get_loss(&key) {
        return LossSample {
            loss: v.loss,
            actual_hash: v.actual_hash,
            expected_hash: key.expected_hash,
        };
    }
    let sample = eval_loss_uncached(
        g,
        program,
        input,
        expected,
        max_steps,
        max_nodes,
        max_ted_cells,
        Some(caches),
        phase,
        example_idx,
    );
    caches.insert_loss(
        key.clone(),
        LossEntry {
            loss: sample.loss,
            actual_hash: sample.actual_hash,
        },
        loss_entry_size(&key),
    );
    sample
}

pub fn eval_loss_uncached(
    g: &mut Graph,
    program: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    _caches: Option<&MemoCaches>,
    phase: &'static str,
    example_idx: usize,
) -> LossSample {
    let _ = preflight_eval_limit(
        g,
        program,
        input,
        expected,
        max_nodes,
        max_ted_cells,
        "loss_preflight",
    );

    // Build an isolated evaluation graph so reductions cannot mutate shared nodes.
    let mut eval_g = Graph::new_uninterned();
    let mut prog_memo = HashMap::new();
    let prog_eval = clone_subtree(g, &mut eval_g, program, &mut prog_memo, None);
    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);
    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let prog_size = estimate_size(&eval_g, prog_eval);
    let input_size = estimate_size(&eval_g, input_eval);
    let expected_size = estimate_size(&eval_g, expected_eval);

    let total_start = Instant::now();
    let mut reduce_actual_time = Duration::ZERO;
    let reduce_expected_time = Duration::ZERO;
    let mut ted_time = Duration::ZERO;

    let base_nodes = eval_g.nodes.len();
    let applied = eval_g.add(Node::App {
        func: prog_eval,
        args: smallvec::smallvec![input_eval],
    });
    let size = estimate_size(&eval_g, applied);
    let budget = scaled_eval_budget(max_steps, size);
    if max_nodes > 0 && size > max_nodes {
        log_node_limit_hit(
            "loss_pre_eval",
            size,
            max_nodes,
            0,
            budget,
            size,
            prog_size,
            input_size,
            expected_size,
        );
    }

    let mut limit_hit = false;
    let mut actual_steps = 0usize;
    let mut actual_step_limit = budget;
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "begin",
        prog_size,
        input_size,
        expected_size,
        program,
        None,
        None,
        None,
    );
    let mut ctx = EvalContext::default();
    ctx.step_limit = budget;
    ctx.node_limit = max_nodes;
    ctx.base_nodes = base_nodes;
    ctx.depth_limit = 1000;
    ctx.redex_cache = None;
    let reduce_start = Instant::now();
    let actual = reduce(&mut eval_g, applied, &mut ctx);
    let reduce_elapsed = reduce_start.elapsed();
    reduce_actual_time = reduce_actual_time.saturating_add(reduce_elapsed);
    actual_steps = ctx.steps;
    actual_step_limit = ctx.step_limit;
    if reduce_elapsed.as_millis() >= SLOW_EVAL_MS {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        log_slow_eval(
            "loss_reduce_actual",
            reduce_elapsed,
            eval_nodes,
            ctx.steps,
            ctx.step_limit,
            prog_size,
            input_size,
            expected_size,
        );
    }
    if ctx.node_limit_hit || ctx.step_limit_hit {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        if ctx.node_limit_hit {
            log_node_limit_hit(
                "loss_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if ctx.step_limit_hit {
            log_step_limit_hit(
                "loss_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        limit_hit = true;
    }
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "end",
        prog_size,
        input_size,
        expected_size,
        program,
        None,
        None,
        Some(reduce_actual_time),
    );
    log_stage_slow(
        phase,
        example_idx,
        "reduce_actual",
        reduce_actual_time,
        prog_size,
        input_size,
        expected_size,
        program,
        None,
        None,
    );
    if limit_hit {
        // Keep going with partial result.
    }

    let expected_nf = eval_g.resolve(expected_eval);
    let ted_actual = estimate_size(&eval_g, actual);
    let ted_expected = estimate_size(&eval_g, expected_nf);
    log_example_stage(
        phase,
        example_idx,
        "ted",
        "begin",
        prog_size,
        ted_actual,
        ted_expected,
        program,
        None,
        None,
        None,
    );
    let dist = if max_ted_cells > 0
        && (ted_actual as u128) * (ted_expected as u128) > max_ted_cells as u128
    {
        log_ted_cap_hit(ted_actual, ted_expected, max_ted_cells);
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            None,
            None,
            Some(Duration::ZERO),
        );
        ted_actual.saturating_add(ted_expected)
    } else {
        let ted_start = Instant::now();
        let mut ted_capped = false;
        let d = match tree_edit_distance_capped(&eval_g, actual, expected_nf, max_ted_cells) {
            Ok(d) => d,
            Err((a, b)) => {
                ted_capped = true;
                log_ted_cap_hit(a, b, max_ted_cells);
                a.saturating_add(b)
            }
        };
        let ted_elapsed = ted_start.elapsed();
        ted_time = ted_time.saturating_add(ted_elapsed);
        if !ted_capped && ted_elapsed.as_millis() >= SLOW_EVAL_MS {
            let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
            let steps = actual_steps;
            let step_limit = actual_step_limit;
            log_slow_eval(
                "loss_ted",
                ted_elapsed,
                eval_nodes,
                steps,
                step_limit,
                prog_size,
                input_size,
                expected_size,
            );
        }
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            None,
            None,
            Some(ted_elapsed),
        );
        log_stage_slow(
            phase,
            example_idx,
            "ted",
            ted_elapsed,
            prog_size,
            ted_actual,
            ted_expected,
            program,
            None,
            None,
        );
        d
    };
    let total_elapsed = total_start.elapsed();
    with_timing_acc(|acc| {
        let mut acc = acc.lock().unwrap();
        acc.total = acc.total.saturating_add(total_elapsed);
        acc.reduce_actual = acc.reduce_actual.saturating_add(reduce_actual_time);
        acc.reduce_expected = acc.reduce_expected.saturating_add(reduce_expected_time);
        acc.ted = acc.ted.saturating_add(ted_time);
        acc.evals = acc.evals.saturating_add(1);
    });
    let steps = actual_steps;
    let step_limit = actual_step_limit;
    let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
    update_peak(
        "loss_done",
        eval_nodes,
        steps,
        step_limit,
        size,
        prog_size,
        input_size,
        expected_size,
        get_debug_context(),
    );
    let actual_hash = tree_hash(&eval_g, actual);
    let expected_hash = tree_hash(&eval_g, expected_nf);
    LossSample {
        loss: dist as f64,
        actual_hash,
        expected_hash,
    }
}

pub fn eval_loss_edit_uncached(
    g: &mut Graph,
    program: NodeId,
    path: &[u8],
    replacement: NodeId,
    input: NodeId,
    expected: NodeId,
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    _caches: Option<&MemoCaches>,
    phase: &'static str,
    example_idx: usize,
) -> LossSample {
    let _ = preflight_eval_limit_with_replacement(
        g,
        program,
        path,
        replacement,
        input,
        expected,
        max_nodes,
        max_ted_cells,
        "edit_preflight",
    );

    // Build an isolated evaluation graph so reductions cannot mutate shared nodes.
    let mut eval_g = Graph::new_uninterned();
    let mut prog_memo = HashMap::new();
    let prog_eval_src = clone_subtree(g, &mut eval_g, program, &mut prog_memo, None);
    let mut repl_memo = HashMap::new();
    let repl_eval = clone_subtree(g, &mut eval_g, replacement, &mut repl_memo, None);
    let mut input_memo = HashMap::new();
    let input_eval = clone_subtree(g, &mut eval_g, input, &mut input_memo, None);
    let mut expected_memo = HashMap::new();
    let expected_eval = clone_subtree(g, &mut eval_g, expected, &mut expected_memo, None);

    let input_size = estimate_size(&eval_g, input_eval);
    let expected_size = estimate_size(&eval_g, expected_eval);

    let total_start = Instant::now();
    let mut reduce_actual_time = Duration::ZERO;
    let reduce_expected_time = Duration::ZERO;
    let mut ted_time = Duration::ZERO;

    let base_nodes = eval_g.nodes.len();
    let prog_eval = replace_at_path(&mut eval_g, prog_eval_src, path, repl_eval);
    let prog_size = estimate_size(&eval_g, prog_eval);
    let applied = eval_g.add(Node::App {
        func: prog_eval,
        args: smallvec::smallvec![input_eval],
    });

    let size = estimate_size(&eval_g, applied);
    let budget = scaled_eval_budget(max_steps, size);
    if max_nodes > 0 && size > max_nodes {
        log_node_limit_hit(
            "edit_pre_eval",
            size,
            max_nodes,
            0,
            budget,
            size,
            prog_size,
            input_size,
            expected_size,
        );
    }

    let mut limit_hit = false;
    let mut actual_steps = 0usize;
    let mut actual_step_limit = budget;
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "begin",
        prog_size,
        input_size,
        expected_size,
        program,
        Some(path),
        None,
        None,
    );
    let mut ctx = EvalContext::default();
    ctx.step_limit = budget;
    ctx.node_limit = max_nodes;
    ctx.base_nodes = base_nodes;
    ctx.depth_limit = 1000;
    ctx.redex_cache = None;
    let reduce_start = Instant::now();
    let actual = reduce(&mut eval_g, applied, &mut ctx);
    let reduce_elapsed = reduce_start.elapsed();
    reduce_actual_time = reduce_actual_time.saturating_add(reduce_elapsed);
    actual_steps = ctx.steps;
    actual_step_limit = ctx.step_limit;
    if reduce_elapsed.as_millis() >= SLOW_EVAL_MS {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        log_slow_eval(
            "edit_reduce_actual",
            reduce_elapsed,
            eval_nodes,
            ctx.steps,
            ctx.step_limit,
            prog_size,
            input_size,
            expected_size,
        );
    }
    if ctx.node_limit_hit || ctx.step_limit_hit {
        let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
        if ctx.node_limit_hit {
            log_node_limit_hit(
                "edit_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        if ctx.step_limit_hit {
            log_step_limit_hit(
                "edit_reduce_actual",
                eval_nodes,
                max_nodes,
                ctx.steps,
                ctx.step_limit,
                size,
                prog_size,
                input_size,
                expected_size,
            );
        }
        limit_hit = true;
    }
    log_example_stage(
        phase,
        example_idx,
        "reduce_actual",
        "end",
        prog_size,
        input_size,
        expected_size,
        program,
        Some(path),
        None,
        Some(reduce_actual_time),
    );
    log_stage_slow(
        phase,
        example_idx,
        "reduce_actual",
        reduce_actual_time,
        prog_size,
        input_size,
        expected_size,
        program,
        Some(path),
        None,
    );
    if limit_hit {
        // Keep going with the partial reduction result.
    }

    let expected_nf = eval_g.resolve(expected_eval);
    let ted_actual = estimate_size(&eval_g, actual);
    let ted_expected = estimate_size(&eval_g, expected_nf);
    log_example_stage(
        phase,
        example_idx,
        "ted",
        "begin",
        prog_size,
        ted_actual,
        ted_expected,
        program,
        Some(path),
        None,
        None,
    );
    let dist = if max_ted_cells > 0
        && (ted_actual as u128) * (ted_expected as u128) > max_ted_cells as u128
    {
        log_ted_cap_hit(ted_actual, ted_expected, max_ted_cells);
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            Some(path),
            None,
            Some(Duration::ZERO),
        );
        ted_actual.saturating_add(ted_expected)
    } else {
        let ted_start = Instant::now();
        let mut ted_capped = false;
        let d = match tree_edit_distance_capped(&eval_g, actual, expected_nf, max_ted_cells) {
            Ok(d) => d,
            Err((a, b)) => {
                ted_capped = true;
                log_ted_cap_hit(a, b, max_ted_cells);
                a.saturating_add(b)
            }
        };
        let ted_elapsed = ted_start.elapsed();
        ted_time = ted_time.saturating_add(ted_elapsed);
        if !ted_capped && ted_elapsed.as_millis() >= SLOW_EVAL_MS {
            let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
            let steps = actual_steps;
            let step_limit = actual_step_limit;
            log_slow_eval(
                "edit_ted",
                ted_elapsed,
                eval_nodes,
                steps,
                step_limit,
                prog_size,
                input_size,
                expected_size,
            );
        }
        log_example_stage(
            phase,
            example_idx,
            "ted",
            "end",
            prog_size,
            ted_actual,
            ted_expected,
            program,
            Some(path),
            None,
            Some(ted_elapsed),
        );
        log_stage_slow(
            phase,
            example_idx,
            "ted",
            ted_elapsed,
            prog_size,
            ted_actual,
            ted_expected,
            program,
            Some(path),
            None,
        );
        d
    };
    let total_elapsed = total_start.elapsed();
    with_timing_acc(|acc| {
        let mut acc = acc.lock().unwrap();
        acc.total = acc.total.saturating_add(total_elapsed);
        acc.reduce_actual = acc.reduce_actual.saturating_add(reduce_actual_time);
        acc.reduce_expected = acc.reduce_expected.saturating_add(reduce_expected_time);
        acc.ted = acc.ted.saturating_add(ted_time);
        acc.evals = acc.evals.saturating_add(1);
    });
    let steps = actual_steps;
    let step_limit = actual_step_limit;
    let eval_nodes = eval_g.nodes.len().saturating_sub(base_nodes);
    update_peak(
        "edit_done",
        eval_nodes,
        steps,
        step_limit,
        size,
        prog_size,
        input_size,
        expected_size,
        get_debug_context(),
    );

    let actual_hash = tree_hash(&eval_g, actual);
    let expected_hash = tree_hash(&eval_g, expected_nf);
    LossSample {
        loss: dist as f64,
        actual_hash,
        expected_hash,
    }
}

pub fn program_stats_cached(
    g: &mut Graph,
    program: NodeId,
    examples: &[(NodeId, NodeId)],
    max_steps: usize,
    max_nodes: usize,
    max_ted_cells: usize,
    output_collision_penalty: f64,
    caches: &MemoCaches,
) -> CandidateStats {
    let losses = evaluate_program_on_examples_cached(
        g,
        program,
        examples,
        max_steps,
        max_nodes,
        max_ted_cells,
        output_collision_penalty,
        caches,
    );
    use crate::learner::counterfactual::utils::{mean, variance};
    CandidateStats {
        mean: mean(&losses),
        var: variance(&losses),
    }
}
