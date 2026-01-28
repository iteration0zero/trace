use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::trace::{RuleId, Branch};
use crate::engine::types::{EvalContext, WhnfCont, ApplyState, mk_seq, collect_top_spine, REDUCE_PROGRESS_MS_OVERRIDE, REDUCE_DEBUG_LEVEL_OVERRIDE};
use crate::engine::primitives::{apply_primitive, decode_prim_tag_tree, prim_need_whnf_indices, primitive_min_args, apply_primitive_whnf};
use crate::engine::unparse::{debug_unparse, debug_args, node_kind};

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;
use smallvec::SmallVec;

fn reduce_progress_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    let override_ms = REDUCE_PROGRESS_MS_OVERRIDE.load(Ordering::Relaxed);
    if override_ms != u64::MAX {
        return override_ms;
    }
    *MS.get_or_init(|| {
        std::env::var("TRACE_REDUCE_PROGRESS_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(3000)
    })
}

pub fn set_reduce_progress_ms_override(ms: Option<u64>) {
    let val = ms.unwrap_or(u64::MAX);
    REDUCE_PROGRESS_MS_OVERRIDE.store(val, Ordering::Relaxed);
}

fn reduce_debug_level() -> u64 {
    static LEVEL: OnceLock<u64> = OnceLock::new();
    let override_level = REDUCE_DEBUG_LEVEL_OVERRIDE.load(Ordering::Relaxed);
    if override_level != u64::MAX {
        return override_level;
    }
    *LEVEL.get_or_init(|| {
        std::env::var("TRACE_REDUCE_DEBUG")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    })
}

pub fn set_reduce_debug_level_override(level: Option<u64>) {
    let val = level.unwrap_or(u64::MAX);
    REDUCE_DEBUG_LEVEL_OVERRIDE.store(val, Ordering::Relaxed);
}

fn reduce_debug(level: u64) -> bool {
    reduce_debug_level() >= level
}

pub fn reduce(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> NodeId {
    let mut curr = id;
    let start = std::time::Instant::now();
    let mut last_progress = start;
    let progress_ms = reduce_progress_ms();
    ctx.step_limit_hit = false;
    if reduce_debug(1) {
        eprintln!(
            "REDUCE_BEGIN root={} form={}",
            id.0,
            debug_unparse(g, id)
        );
    }
    while ctx.steps < ctx.step_limit {
        if progress_ms > 0 && last_progress.elapsed().as_millis() >= progress_ms as u128 {
            eprintln!(
                "REDUCE_PROGRESS elapsed_ms={} steps={} nodes={} node_limit_hit={} step_limit={} node_limit={} root={}",
                start.elapsed().as_millis(),
                ctx.steps,
                g.nodes.len(),
                ctx.node_limit_hit,
                ctx.step_limit,
                ctx.node_limit,
                id.0
            );
            last_progress = std::time::Instant::now();
        }
        if ctx.node_limit > 0
            && g.nodes.len().saturating_sub(ctx.base_nodes) > ctx.node_limit
        {
            ctx.node_limit_hit = true;
            break;
        }
        curr = g.resolve(curr);
        if reduce_debug(2) {
            eprintln!(
                "REDUCE_LOOP step={} root={} kind={}",
                ctx.steps,
                curr.0,
                node_kind(g, curr)
            );
        }
        if let Some(next) = reduce_step(g, curr, ctx) {
            if reduce_debug(2) {
                eprintln!(
                    "REDUCE_STEP step={} from={} to={}",
                    ctx.steps,
                    curr.0,
                    next.0
                );
            }
            curr = next;
        } else {
            if reduce_debug(1) {
                eprintln!(
                    "REDUCE_NORMAL_FORM steps={} root={} form={}",
                    ctx.steps,
                    curr.0,
                    debug_unparse(g, curr)
                );
            }
            return curr;
        }
        ctx.steps += 1;
        if ctx.node_limit > 0
            && g.nodes.len().saturating_sub(ctx.base_nodes) > ctx.node_limit
        {
            ctx.node_limit_hit = true;
            break;
        }
    }
    if ctx.steps >= ctx.step_limit {
        ctx.step_limit_hit = true;
    }
    if let Some(trace) = &mut ctx.exec_trace {
        trace.set_result(curr);
    }
    if reduce_debug(1) {
        eprintln!(
            "REDUCE_END steps={} step_limit_hit={} node_limit_hit={} root={} form={}",
            ctx.steps,
            ctx.step_limit_hit,
            ctx.node_limit_hit,
            curr.0,
            debug_unparse(g, curr)
        );
    }
    curr
}

pub fn reduce_step(g: &mut Graph, root: NodeId, ctx: &mut EvalContext) -> Option<NodeId> {
    let mut hash_memo: HashMap<NodeId, u64> = HashMap::new();
    let resolved = g.resolve(root);
    let (head, args) = collect_top_spine(g, resolved);
    if reduce_debug(2) {
        eprintln!(
            "REDUCE_SPINE root={} head={} head_kind={} args_len={} args=[{}]",
            resolved.0,
            head.0,
            node_kind(g, head),
            args.len(),
            debug_args(g, &args)
        );
    }
    if args.is_empty() {
        return None;
    }
    if let Some(result) = reduce_top_no_app(g, resolved, head, &args, ctx, &mut hash_memo) {
        let res = g.resolve(result);
        if res == resolved {
            return None;
        }
        return Some(res);
    }
    None
}

fn reduce_top_no_app(
    g: &mut Graph,
    root: NodeId,
    head: NodeId,
    args: &SmallVec<[NodeId; 2]>,
    ctx: &mut EvalContext,
    _hash_memo: &mut HashMap<NodeId, u64>,
) -> Option<NodeId> {
    if args.is_empty() {
        return None;
    }
    let mut curr_head = g.resolve(head);
    let _idx = 0usize;
    let mut changed = false;
    let mut pending: SmallVec<[NodeId; 2]> = args.clone();

    loop {
        if pending.is_empty() {
            break;
        }
        curr_head = g.resolve(curr_head);
        // Primitives encoded as tagged trees should intercept before normal fork handling.
        if let Some(p) = decode_prim_tag_tree(g, curr_head) {
            if let Some(res) = apply_primitive(g, p, &pending, ctx) {
                if let Some(trace) = &mut ctx.exec_trace {
                    trace.record(RuleId::Prim, root, pending.to_vec(), res);
                }
                curr_head = res;
                changed = true;
                pending.clear();
                break;
            }
        }

        let head_node = g.get(curr_head).clone();
        match head_node {
            Node::Ind(inner) => {
                curr_head = g.resolve(inner);
                changed = true;
            }
            Node::App { .. } => {
                let (h, extra) = collect_top_spine(g, curr_head);
                if !extra.is_empty() {
                    let mut merged: SmallVec<[NodeId; 2]> = extra;
                    merged.extend(pending.into_iter());
                    pending = merged;
                    curr_head = h;
                    changed = true;
                } else {
                    break;
                }
            }
            Node::Leaf => {
                let arg = ensure_tree(g, pending[0], ctx, "Leaf-arg");
                pending.remove(0);
                let new_node = g.add(Node::Stem(arg));
                if let Some(trace) = &mut ctx.exec_trace {
                    trace.record(RuleId::App, curr_head, vec![arg], new_node);
                }
                curr_head = new_node;
                changed = true;
                continue;
            }
            Node::Stem(x) => {
                let arg = ensure_tree(g, pending[0], ctx, "Stem-arg");
                pending.remove(0);
                let new_node = g.add(Node::Fork(x, arg));
                if let Some(trace) = &mut ctx.exec_trace {
                    trace.record(RuleId::App, curr_head, vec![arg], new_node);
                }
                curr_head = new_node;
                changed = true;
                continue;
            }
            Node::Fork(p, q) => {
                let arg = ensure_tree(g, pending[0], ctx, "Fork-arg");
                pending.remove(0);
                if let Some(res) = triage_reduce(g, root, p, q, arg, ctx) {
                    curr_head = res;
                    changed = true;
                    continue;
                } else {
                    return None;
                }
            }
            Node::Prim(p) => {
                let rem: SmallVec<[NodeId; 2]> = pending.clone();
                if let Some(res) = apply_primitive(g, p, &rem, ctx) {
                    curr_head = res;
                    changed = true;
                    pending.clear();
                    break;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    if !changed {
        return None;
    }
    if !pending.is_empty() {
        return Some(mk_seq(g, curr_head, pending));
    }
    Some(curr_head)
}

fn ensure_tree(g: &mut Graph, id: NodeId, _ctx: &mut EvalContext, _context: &str) -> NodeId {
    g.resolve(id)
}

fn triage_reduce(
    g: &mut Graph,
    redex_id: NodeId,
    p: NodeId,
    q: NodeId,
    r: NodeId,
    ctx: &mut EvalContext,
) -> Option<NodeId> {
    if reduce_debug(2) {
        eprintln!(
            "TRIAGE_BEGIN redex={} p_kind={} q_kind={} r_kind={}",
            redex_id.0,
            node_kind(g, p),
            node_kind(g, q),
            node_kind(g, r)
        );
    }
    let p_resolved = g.resolve(p);
    let p_whnf = match g.get(p_resolved) {
        Node::App { .. } => reduce_whnf_with_ctx(g, p_resolved, ctx),
        _ => p_resolved,
    };
    let p_node = g.get(p_whnf).clone();
    // Triage requires z to be in WHNF (Leaf/Stem/Fork). Reduce z only at the top if needed.
    let r_resolved = g.resolve(r);
    let r_whnf = match g.get(r_resolved) {
        Node::App { .. } => reduce_whnf_with_ctx(g, r_resolved, ctx),
        _ => r_resolved,
    };
    if reduce_debug(2) {
        eprintln!(
            "TRIAGE_R_WHNF redex={} r={} r_kind={}",
            redex_id.0,
            r_whnf.0,
            node_kind(g, r_whnf)
        );
    }
    let (res, rule_id, captured_args) = match p_node {
        // Rule 1: △△ y z -> y
        Node::Leaf => {
            if reduce_debug(2) {
                eprintln!("TRIAGE_RULE K redex={} result={}", redex_id.0, q.0);
            }
            (Some(q), Some(RuleId::K), vec![q, r]) // y=q, z=r
        }
        // Rule 2: △(△x) y z -> x z (y z)
        Node::Stem(x) => {
            let xz = mk_seq(g, x, smallvec::smallvec![r]);
            let yz = mk_seq(g, q, smallvec::smallvec![r]);
            if reduce_debug(2) {
                eprintln!(
                    "TRIAGE_RULE S redex={} x={} y={} z={}",
                    redex_id.0,
                    x.0,
                    q.0,
                    r.0
                );
            }
            (
                Some(mk_seq(g, xz, smallvec::smallvec![yz])),
                Some(RuleId::S),
                vec![x, q, r] // x, y=q, z=r
            )
        }
        // Rules 3-5: triage on z when p is a fork
        Node::Fork(w, x) => {
            match g.get(r_whnf).clone() {
                Node::Leaf => {
                    if let Some(trace) = &mut ctx.trace {
                        trace.record(r, Branch::Leaf);
                    }
                    if reduce_debug(2) {
                        eprintln!("TRIAGE_RULE Leaf redex={} result={}", redex_id.0, w.0);
                    }
                    (Some(w), Some(RuleId::TriageLeaf), vec![w]) // w=w
                }
                Node::Stem(u) => {
                    if let Some(trace) = &mut ctx.trace {
                        trace.record(r, Branch::Stem);
                    }
                    if reduce_debug(2) {
                        eprintln!(
                            "TRIAGE_RULE Stem redex={} result_app_of={} arg={}",
                            redex_id.0,
                            x.0,
                            u.0
                        );
                    }
                    (
                        Some(mk_seq(g, x, smallvec::smallvec![u])),
                        Some(RuleId::TriageStem),
                        vec![x, u]
                    )
                }
                Node::Fork(u, v) => {
                    if let Some(trace) = &mut ctx.trace {
                        trace.record(r, Branch::Fork);
                    }
                    if reduce_debug(2) {
                        eprintln!(
                            "TRIAGE_RULE Fork redex={} result_app_of={} args=[{},{}]",
                            redex_id.0,
                            q.0,
                            u.0,
                            v.0
                        );
                    }
                    (
                        Some(mk_seq(g, q, smallvec::smallvec![u, v])),
                        Some(RuleId::TriageFork),
                        vec![q, u, v] // y=q, u, v
                    )
                }
                _ => (None, None, vec![]),
            }
        }
        _ => (None, None, vec![]),
    };

    if let Some(result) = res {
        if let (Some(trace), Some(rule)) = (&mut ctx.exec_trace, rule_id) {
            trace.record(rule, redex_id, captured_args, result);
        }
        Some(result)
    } else {
        None
    }
}

pub fn reduce_whnf(g: &mut Graph, id: NodeId) -> NodeId {
    let mut ctx = EvalContext::default();
    reduce_whnf_with_ctx(g, id, &mut ctx)
}

pub fn reduce_whnf_with_ctx(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> NodeId {
    reduce_whnf_depth_with_ctx(g, id, 0, ctx)
}

pub fn reduce_whnf_depth_with_ctx(
    g: &mut Graph,
    id: NodeId,
    depth: usize,
    ctx: &mut EvalContext,
) -> NodeId {
    let mut curr = id;
    if depth > ctx.depth_limit {
        return curr;
    }

    let mut stack: Vec<WhnfCont> = Vec::new();

    'reduce: loop {
        if ctx.steps >= ctx.step_limit {
            ctx.step_limit_hit = true;
            break;
        }
        if ctx.node_limit > 0
            && g.nodes.len().saturating_sub(ctx.base_nodes) > ctx.node_limit
        {
            ctx.node_limit_hit = true;
            break;
        }

        curr = g.resolve(curr);
        match g.get(curr).clone() {
            Node::App { .. } => {
                let (head, args) = collect_top_spine(g, curr);
                if args.is_empty() {
                    curr = head;
                    continue;
                }
                let state = ApplyState {
                    redex: curr,
                    head,
                    pending: args,
                    changed: false,
                    skip_tag: false,
                };
                stack.push(WhnfCont::Apply(state));
                curr = head;
                continue;
            }
            Node::Ind(inner) => {
                curr = inner;
                continue;
            }
            _ => {}
        }

        let cont = match stack.pop() {
            None => return curr,
            Some(cont) => cont,
        };

        match cont {
            WhnfCont::Apply(mut state) => {
                state.head = curr;
                let mut abort_no_change = false;
                loop {
                    if state.pending.is_empty() {
                        break;
                    }
                    let resolved = g.resolve(state.head);
                    if resolved != state.head {
                        state.head = resolved;
                        state.changed = true;
                    }
                    if !state.skip_tag {
                        if let Some(p) = decode_prim_tag_tree(g, state.head) {
                            let min = primitive_min_args(p).unwrap_or(0);
                            if state.pending.len() < min {
                                state.skip_tag = true;
                            } else {
                                let need = prim_need_whnf_indices(p);
                                if !need.is_empty() {
                                    let reduced = vec![None; need.len()];
                                    state.skip_tag = true;
                                    let idx = need[0];
                                    let arg = state.pending[idx];
                                    stack.push(WhnfCont::Prim {
                                        state,
                                        prim: p,
                                        need_idxs: need,
                                        pos: 0,
                                        reduced,
                                        strict: false,
                                    });
                                    curr = arg;
                                    continue 'reduce;
                                }
                                if let Some(res) = apply_primitive_whnf(g, p, &state.pending) {
                                    if let Some(trace) = &mut ctx.exec_trace {
                                        trace.record(RuleId::Prim, state.redex, state.pending.to_vec(), res);
                                    }
                                    state.head = res;
                                    state.pending.clear();
                                    state.changed = true;
                                    continue;
                                } else {
                                    state.skip_tag = true;
                                }
                            }
                        }
                    }

                    let head_node = g.get(state.head).clone();
                    match head_node {
                        Node::App { .. } => {
                            let (h, extra) = collect_top_spine(g, state.head);
                            if !extra.is_empty() {
                                let mut merged = extra;
                                merged.extend(state.pending.drain(..));
                                state.pending = merged;
                                state.head = h;
                                state.changed = true;
                                continue;
                            }
                            break;
                        }
                        Node::Leaf => {
                            let arg = state.pending.remove(0);
                            let new_node = g.add(Node::Stem(arg));
                            if let Some(trace) = &mut ctx.exec_trace {
                                trace.record(RuleId::App, state.head, vec![arg], new_node);
                            }
                            state.head = new_node;
                            state.changed = true;
                            continue;
                        }
                        Node::Stem(x) => {
                            let arg = state.pending.remove(0);
                            let new_node = g.add(Node::Fork(x, arg));
                            if let Some(trace) = &mut ctx.exec_trace {
                                trace.record(RuleId::App, state.head, vec![arg], new_node);
                            }
                            state.head = new_node;
                            state.changed = true;
                            continue;
                        }
                        Node::Fork(p, q) => {
                            let arg = state.pending.remove(0);
                            let p_res = g.resolve(p);
                            if matches!(g.get(p_res), Node::App { .. }) {
                                stack.push(WhnfCont::TriageP {
                                    state,
                                    q,
                                    r: arg,
                                });
                                curr = p_res;
                                continue 'reduce;
                            }
                            let r_res = g.resolve(arg);
                            if matches!(g.get(r_res), Node::App { .. }) {
                                stack.push(WhnfCont::TriageR {
                                    state,
                                    p_whnf: p_res,
                                    q,
                                    r_original: arg,
                                });
                                curr = r_res;
                                continue 'reduce;
                            }
                            if let Some(res) = triage_apply(g, state.redex, p_res, q, arg, r_res, ctx) {
                                state.head = res;
                                state.changed = true;
                                continue;
                            } else {
                                abort_no_change = true;
                                break;
                            }
                        }
                        Node::Prim(p) => {
                            let min = primitive_min_args(p).unwrap_or(0);
                            if state.pending.len() < min {
                                break;
                            }
                            let need = prim_need_whnf_indices(p);
                            if !need.is_empty() {
                                let reduced = vec![None; need.len()];
                                let idx = need[0];
                                let arg = state.pending[idx];
                                stack.push(WhnfCont::Prim {
                                    state,
                                    prim: p,
                                    need_idxs: need,
                                    pos: 0,
                                    reduced,
                                    strict: true,
                                });
                                curr = arg;
                                continue 'reduce;
                            }
                            if let Some(res) = apply_primitive_whnf(g, p, &state.pending) {
                                if let Some(trace) = &mut ctx.exec_trace {
                                    trace.record(RuleId::Prim, state.redex, state.pending.to_vec(), res);
                                }
                                state.head = res;
                                state.pending.clear();
                                state.changed = true;
                                continue;
                            } else {
                                break;
                            }
                        }
                        _ => break,
                    }
                }

                if abort_no_change || !state.changed {
                    curr = state.redex;
                    if stack.is_empty() {
                        return curr;
                    }
                    continue;
                }

                let res = if state.pending.is_empty() {
                    state.head
                } else {
                    mk_seq(g, state.head, state.pending)
                };
                curr = res;
                ctx.steps = ctx.steps.saturating_add(1);
                continue;
            }
            WhnfCont::TriageP { mut state, q, r } => {
                let p_whnf = curr;
                let r_res = g.resolve(r);
                if matches!(g.get(r_res), Node::App { .. }) {
                    stack.push(WhnfCont::TriageR {
                        state,
                        p_whnf,
                        q,
                        r_original: r,
                    });
                    curr = r_res;
                    continue;
                }
                if let Some(res) = triage_apply(g, state.redex, p_whnf, q, r, r_res, ctx) {
                    state.head = res;
                    state.changed = true;
                    stack.push(WhnfCont::Apply(state));
                    curr = res;
                    continue;
                }
                curr = state.redex;
                if stack.is_empty() {
                    return curr;
                }
                continue;
            }
            WhnfCont::TriageR {
                mut state,
                p_whnf,
                q,
                r_original,
            } => {
                let r_whnf = curr;
                if let Some(res) = triage_apply(g, state.redex, p_whnf, q, r_original, r_whnf, ctx) {
                    state.head = res;
                    state.changed = true;
                    stack.push(WhnfCont::Apply(state));
                    curr = res;
                    continue;
                }
                curr = state.redex;
                if stack.is_empty() {
                    return curr;
                }
                continue;
            }
            WhnfCont::Prim {
                mut state,
                prim,
                need_idxs,
                mut pos,
                mut reduced,
                strict,
            } => {
                if pos < reduced.len() {
                    reduced[pos] = Some(curr);
                    pos += 1;
                }
                if pos < need_idxs.len() {
                    let idx = need_idxs[pos];
                    let arg = state.pending[idx];
                    stack.push(WhnfCont::Prim {
                        state,
                        prim,
                        need_idxs,
                        pos,
                        reduced,
                        strict,
                    });
                    curr = arg;
                    continue;
                }
                let mut eval_args = state.pending.clone();
                for (idx_pos, arg_idx) in need_idxs.iter().enumerate() {
                    if let Some(arg) = reduced.get(idx_pos).and_then(|v| *v) {
                        if *arg_idx < eval_args.len() {
                            eval_args[*arg_idx] = arg;
                        }
                    }
                }
                if let Some(res) = apply_primitive_whnf(g, prim, &eval_args) {
                    if let Some(trace) = &mut ctx.exec_trace {
                        trace.record(RuleId::Prim, state.redex, eval_args.to_vec(), res);
                    }
                    state.head = res;
                    state.pending.clear();
                    state.changed = true;
                    stack.push(WhnfCont::Apply(state));
                    curr = res;
                    continue;
                }
                if strict {
                    if state.changed {
                        let res = if state.pending.is_empty() {
                            state.head
                        } else {
                            mk_seq(g, state.head, state.pending)
                        };
                        curr = res;
                        ctx.steps = ctx.steps.saturating_add(1);
                        continue;
                    }
                    curr = state.redex;
                    if stack.is_empty() {
                        return curr;
                    }
                    continue;
                }
                state.skip_tag = true;
                let head = state.head;
                stack.push(WhnfCont::Apply(state));
                curr = head;
                continue;
            }
        }
    }

    curr
}

fn triage_apply(
    g: &mut Graph,
    redex_id: NodeId,
    p_whnf: NodeId,
    q: NodeId,
    r_original: NodeId,
    r_whnf: NodeId,
    ctx: &mut EvalContext,
) -> Option<NodeId> {
    let p_node = g.get(p_whnf).clone();
    let (res, rule_id, captured_args) = match p_node {
        Node::Leaf => (Some(q), Some(RuleId::K), vec![q, r_original]),
        Node::Stem(x) => {
            let xz = mk_seq(g, x, smallvec::smallvec![r_original]);
            let yz = mk_seq(g, q, smallvec::smallvec![r_original]);
            (
                Some(mk_seq(g, xz, smallvec::smallvec![yz])),
                Some(RuleId::S),
                vec![x, q, r_original],
            )
        }
        Node::Fork(w, x) => match g.get(r_whnf).clone() {
            Node::Leaf => {
                if let Some(trace) = &mut ctx.trace {
                    trace.record(r_original, Branch::Leaf);
                }
                (Some(w), Some(RuleId::TriageLeaf), vec![w])
            }
            Node::Stem(u) => {
                if let Some(trace) = &mut ctx.trace {
                    trace.record(r_original, Branch::Stem);
                }
                (
                    Some(mk_seq(g, x, smallvec::smallvec![u])),
                    Some(RuleId::TriageStem),
                    vec![x, u],
                )
            }
            Node::Fork(u, v) => {
                if let Some(trace) = &mut ctx.trace {
                    trace.record(r_original, Branch::Fork);
                }
                (
                    Some(mk_seq(g, q, smallvec::smallvec![u, v])),
                    Some(RuleId::TriageFork),
                    vec![q, u, v],
                )
            }
            _ => (None, None, vec![]),
        },
        _ => (None, None, vec![]),
    };

    if let Some(result) = res {
        if let (Some(trace), Some(rule)) = (&mut ctx.exec_trace, rule_id) {
            trace.record(rule, redex_id, captured_args, result);
        }
        Some(result)
    } else {
        None
    }
}
