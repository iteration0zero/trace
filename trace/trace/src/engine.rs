
use crate::arena::{Graph, Node, NodeId, Primitive};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;
use smallvec::SmallVec;
use crate::trace::{ExecutionTrace, RuleId, Branch, TraceEvent};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

static REDUCE_PROGRESS_MS_OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);
static REDUCE_DEBUG_LEVEL_OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);

const TREE_HASH_CACHE_MAX: usize = 200_000;

const DEBUG_UNPARSE_MAX_DEPTH: usize = 6;
const DEBUG_UNPARSE_MAX_NODES: usize = 200;
const DEBUG_UNPARSE_MAX_ARGS: usize = 6;

// ... encoding helpers (omitted for brevity) ...

pub fn zigzag(n: &BigInt) -> BigUint {
    match n.sign() {
        Sign::NoSign => BigUint::zero(),
        Sign::Plus => n.magnitude() << 1,
        Sign::Minus => (n.magnitude() << 1) - 1u32, 
    }
}

pub fn encode_raw_nat(g: &mut Graph, n: &BigUint) -> NodeId {
    if n.is_zero() {
        return g.add(Node::Leaf);
    }
    let mut acc = g.add(Node::Leaf);
    let bits = n.bits();
    let mut idx = bits;
    while idx > 0 {
        idx -= 1;
        if n.bit(idx) {
            let leaf = g.add(Node::Leaf);
            acc = g.add(Node::Fork(acc, leaf));
        } else {
            acc = g.add(Node::Stem(acc));
        }
    }
    acc
}

pub fn encode_int(g: &mut Graph, n: &BigInt) -> NodeId {
    let z = zigzag(n);
    encode_raw_nat(g, &z)
}

pub fn encode_str(g: &mut Graph, s: &str) -> NodeId {
    let mut rest = g.add(Node::Leaf);
    for c in s.chars().rev() {
        let n_val = c as u32;
        let nat = BigUint::from(n_val);
        let nat_node = encode_raw_nat(g, &nat);
        rest = g.add(Node::Fork(nat_node, rest));
    }
    rest
}

fn prim_tag_name(p: Primitive) -> Option<&'static str> {
    match p {
        Primitive::Add => Some("add"),
        Primitive::Sub => Some("sub"),
        Primitive::Mul => Some("mul"),
        Primitive::Div => Some("div"),
        Primitive::Eq => Some("eq"),
        Primitive::Gt => Some("gt"),
        Primitive::Lt => Some("lt"),
        Primitive::If => Some("if"),
        Primitive::S => Some("s"),
        Primitive::K => Some("k"),
        Primitive::I => Some("i"),
        Primitive::First => Some("first"),
        Primitive::Rest => Some("rest"),
        Primitive::Trace => Some("trace"),
        _ => None,
    }
}

fn prim_from_tag_name(name: &str) -> Option<Primitive> {
    match name {
        "add" => Some(Primitive::Add),
        "sub" => Some(Primitive::Sub),
        "mul" => Some(Primitive::Mul),
        "div" => Some(Primitive::Div),
        "eq" => Some(Primitive::Eq),
        "gt" => Some(Primitive::Gt),
        "lt" => Some(Primitive::Lt),
        "if" => Some(Primitive::If),
        "s" => Some(Primitive::S),
        "k" => Some(Primitive::K),
        "i" => Some(Primitive::I),
        "first" => Some(Primitive::First),
        "rest" => Some(Primitive::Rest),
        "trace" => Some(Primitive::Trace),
        _ => None,
    }
}

fn prim_tag_tree(g: &mut Graph, p: Primitive) -> Option<NodeId> {
    let name = prim_tag_name(p)?;
    let tag = encode_str(g, name);
    let leaf = g.add(Node::Leaf);
    let kk = g.add(Node::Fork(leaf, leaf));
    Some(g.add(Node::Fork(tag, kk)))
}

fn is_kk(g: &Graph, id: NodeId) -> bool {
    let id = g.resolve(id);
    match g.get(id) {
        Node::Fork(l, r) => matches!(g.get(*l), Node::Leaf) && matches!(g.get(*r), Node::Leaf),
        _ => false,
    }
}

fn decode_prim_tag_tree(g: &Graph, id: NodeId) -> Option<Primitive> {
    let id = g.resolve(id);
    if let Node::Fork(tag, kk) = g.get(id) {
        if !is_kk(g, *kk) {
            return None;
        }
        if let Some(name) = decode_str_pure(g, *tag) {
            return prim_from_tag_name(&name);
        }
    }
    None
}


pub struct EvalContext<'a> {
    pub steps: usize,
    pub step_limit: usize,
    pub node_limit: usize,
    pub node_limit_hit: bool,
    pub step_limit_hit: bool,
    pub base_nodes: usize,
    pub trace: Option<&'a mut crate::trace::Trace>, // Sensitivity trace
    pub exec_trace: Option<&'a mut ExecutionTrace>, // Execution trace for blame
    pub redex_cache: Option<&'a dyn RedexMemo>,
    pub depth: usize,
    pub depth_limit: usize,
}


impl Default for EvalContext<'_> {
    fn default() -> Self {
        Self {
            steps: 0,
            step_limit: 10_000_000,
            node_limit: 0,
            node_limit_hit: false,
            step_limit_hit: false,
            base_nodes: 0,
            depth: 0,
            depth_limit: 500,
            trace: None,
            exec_trace: None,
            redex_cache: None,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct RedexKey {
    pub hash: u64,
    pub max_steps: usize,
    pub max_nodes: usize,
}

#[derive(Clone)]
pub struct CachedRedex {
    pub graph: Graph,
    pub result: NodeId,
    pub events: Vec<TraceEvent>,
    pub snapshot_paths: Vec<(NodeId, Vec<u8>)>,
}

pub trait RedexMemo {
    fn get_redex(&self, key: &RedexKey) -> Option<Arc<CachedRedex>>;
    fn insert_redex(&self, key: RedexKey, value: Arc<CachedRedex>, size: usize, nodes: usize);
    fn max_redex_nodes(&self) -> usize;
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
    hash_memo: &mut HashMap<NodeId, u64>,
) -> Option<NodeId> {
    if args.is_empty() {
        return None;
    }
    let mut curr_head = g.resolve(head);
    let mut idx = 0usize;
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

fn collect_top_spine(g: &mut Graph, root: NodeId) -> (NodeId, SmallVec<[NodeId; 2]>) {
    let mut curr = g.resolve(root);
    let mut chunks: Vec<SmallVec<[NodeId; 2]>> = Vec::new();
    loop {
        match g.get(curr) {
            Node::App { func, args } => {
                if !args.is_empty() {
                    chunks.push(args.clone());
                }
                curr = g.resolve(*func);
            }
            Node::Ind(inner) => {
                curr = g.resolve(*inner);
            }
            _ => break,
        }
    }

    let mut args: SmallVec<[NodeId; 2]> = SmallVec::new();
    for chunk in chunks.into_iter().rev() {
        args.extend(chunk);
    }
    (curr, args)
}

fn mk_seq(g: &mut Graph, head: NodeId, args: SmallVec<[NodeId; 2]>) -> NodeId {
    if args.is_empty() {
        return head;
    }
    let (h, extra) = collect_top_spine(g, head);
    if !extra.is_empty() {
        let mut merged = extra;
        merged.extend(args);
        return g.add(Node::App { func: h, args: merged });
    }
    g.add(Node::App { func: h, args })
}

fn ensure_tree(g: &mut Graph, id: NodeId, ctx: &mut EvalContext, context: &str) -> NodeId {
    let _ = ctx;
    let _ = context;
    g.resolve(id)
}

fn attempt_reduction(
    g: &mut Graph,
    root: NodeId,
    head: NodeId,
    args: &SmallVec<[NodeId; 2]>,
    ctx: &mut EvalContext,
    hash_memo: &mut HashMap<NodeId, u64>,
) -> Option<NodeId> {
    let resolved_head = g.resolve(head);
    let head_node = g.get(resolved_head).clone();

    if args.is_empty() {
        return None;
    }
    if reduce_debug(2) {
        eprintln!(
            "REDUCE_TRY root={} head={} kind={} args_len={}",
            root.0,
            resolved_head.0,
            node_kind(g, resolved_head),
            args.len()
        );
    }

    let reducible = match head_node {
        Node::Leaf | Node::Stem(_) => true,
        Node::Fork(_, _) => true,
        Node::Prim(p) => match primitive_min_args(p) {
            Some(min) => args.len() >= min,
            None => false,
        },
        Node::App { .. } => true,
        _ => false,
    };
    if !reducible {
        if reduce_debug(2) {
            eprintln!(
                "REDUCE_NOT_REDUCIBLE root={} head_kind={} args_len={}",
                root.0,
                node_kind(g, resolved_head),
                args.len()
            );
        }
        return None;
    }

    let redex_id = root;

    if let Some(cache) = ctx.redex_cache {
        let key = RedexKey {
            hash: tree_hash_cached(g, redex_id, hash_memo),
            max_steps: ctx.step_limit,
            max_nodes: ctx.node_limit,
        };
        if let Some(entry) = cache.get_redex(&key) {
            if replay_cached_redex(g, root, redex_id, ctx, &entry) {
                return Some(g.resolve(root));
            }
        }
        let max_nodes = cache.max_redex_nodes();
        let too_big = max_nodes > 0 && count_nodes(g, redex_id, max_nodes) > max_nodes;
        if !too_big {
            if let Some(entry) = build_cached_redex(g, redex_id, ctx) {
                let entry = Arc::new(entry);
                let size = redex_entry_size(&entry);
                cache.insert_redex(key, entry.clone(), size, entry.graph.nodes.len());
                if replay_cached_redex(g, root, redex_id, ctx, &entry) {
                    return Some(g.resolve(root));
                }
            }
        }
    }

    if let Some(p) = decode_prim_tag_tree(g, resolved_head) {
        if let Some(res) = apply_primitive(g, p, args, ctx) {
            if let Some(trace) = &mut ctx.exec_trace {
                trace.record(RuleId::Prim, redex_id, args.to_vec(), res);
            }
            if reduce_debug(2) {
                eprintln!(
                    "REDUCE_RULE PrimTag root={} prim={:?} result={}",
                    redex_id.0,
                    p,
                    res.0
                );
            }
            return Some(res);
        }
    }

    match head_node {
        Node::Leaf => {
            if args.is_empty() {
                return None;
            }
            // Apply exactly one argument per step: Leaf a -> Stem(a).
            let mut result = g.add(Node::Stem(args[0]));
            if args.len() > 1 {
                let rest: SmallVec<[NodeId; 2]> = args[1..].iter().cloned().collect();
                result = mk_seq(g, result, rest);
            }
            if let Some(trace) = &mut ctx.exec_trace {
                trace.record(RuleId::App, redex_id, vec![args[0]], result);
            }
            if reduce_debug(2) {
                eprintln!(
                    "REDUCE_RULE Leaf root={} arg={} result={}",
                    redex_id.0,
                    args[0].0,
                    result.0
                );
            }
            Some(result)
        }
        Node::Stem(p) => {
            if args.is_empty() {
                return None;
            }
            // Apply exactly one argument per step: Stem(p) a -> Fork(p, a).
            let mut result = g.add(Node::Fork(p, args[0]));
            if args.len() > 1 {
                let rest: SmallVec<[NodeId; 2]> = args[1..].iter().cloned().collect();
                result = mk_seq(g, result, rest);
            }
            if let Some(trace) = &mut ctx.exec_trace {
                trace.record(RuleId::App, redex_id, vec![args[0]], result);
            }
            if reduce_debug(2) {
                eprintln!(
                    "REDUCE_RULE Stem root={} arg={} result={}",
                    redex_id.0,
                    args[0].0,
                    result.0
                );
            }
            Some(result)
        }
        Node::Fork(p, q) => {
            if args.is_empty() {
                return None;
            }
            if let Some(mut result) = triage_reduce(g, redex_id, p, q, args[0], ctx) {
                if args.len() > 1 {
                    let rest: SmallVec<[NodeId; 2]> = args[1..].iter().cloned().collect();
                    result = mk_seq(g, result, rest);
                }
                if reduce_debug(2) {
                    eprintln!(
                        "REDUCE_RULE Fork root={} arg={} result={}",
                        redex_id.0,
                        args[0].0,
                        result.0
                    );
                }
                Some(result)
            } else {
                None
            }
        }
        Node::Prim(p) => {
            if let Some(res) = apply_primitive(g, p, args, ctx) {
                if let Some(trace) = &mut ctx.exec_trace {
                    trace.record(RuleId::Prim, redex_id, args.to_vec(), res);
                }
                if reduce_debug(2) {
                    eprintln!(
                        "REDUCE_RULE Prim root={} prim={:?} result={}",
                        redex_id.0,
                        p,
                        res.0
                    );
                }
                return Some(res);
            }
            None
        }
        Node::App { func: inner_f, args: inner_args } => {
            let mut new_args = inner_args.clone();
            new_args.extend(args.iter().cloned());
            let new_id = mk_seq(g, inner_f, new_args);
            // This is effectively re-association, conceptually App rule.
            if let Some(trace) = &mut ctx.exec_trace {
                trace.record(RuleId::App, redex_id, args.to_vec(), new_id);
            }
            if reduce_debug(2) {
                eprintln!(
                    "REDUCE_RULE Reassoc root={} result={}",
                    redex_id.0,
                    new_id.0
                );
            }
            Some(new_id)
        }
        _ => None,
    }
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


fn hash_mix(seed: u64, v: u64) -> u64 {
    seed ^ (v.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(seed << 6).wrapping_add(seed >> 2))
}

fn encode_raw_nat_hash_u32(n: u32) -> u64 {
    if n == 0 {
        return 0x9e37_01;
    }
    let mut acc = 0x9e37_01;
    let mut started = false;
    for shift in (0..32).rev() {
        let bit = (n >> shift) & 1;
        if !started {
            if bit == 0 {
                continue;
            }
            started = true;
        }
        if bit == 1 {
            let mut next = 0x9e37_06;
            next = hash_mix(next, acc);
            next = hash_mix(next, 0x9e37_01);
            acc = next;
        } else {
            let mut next = 0x9e37_05;
            next = hash_mix(next, acc);
            acc = next;
        }
    }
    acc
}

fn encode_str_hash(s: &str) -> u64 {
    let mut rest = 0x9e37_01;
    for c in s.chars().rev() {
        let nat = encode_raw_nat_hash_u32(c as u32);
        let mut acc = 0x9e37_06;
        acc = hash_mix(acc, nat);
        acc = hash_mix(acc, rest);
        rest = acc;
    }
    rest
}

fn prim_tag_hash(p: Primitive) -> Option<u64> {
    let name = prim_tag_name(p)?;
    let tag = encode_str_hash(name);
    let mut kk = 0x9e37_06;
    kk = hash_mix(kk, 0x9e37_01);
    kk = hash_mix(kk, 0x9e37_01);
    let mut acc = 0x9e37_06;
    acc = hash_mix(acc, tag);
    acc = hash_mix(acc, kk);
    Some(acc)
}

pub(crate) fn tree_hash(g: &Graph, root: NodeId) -> u64 {
    let root = g.resolve(root);
    let key = TreeHashKey {
        graph_id: g.id,
        epoch: g.epoch,
        node: root.0,
    };
    if let Some(v) = tree_hash_cache_get(&key) {
        return v;
    }
    let mut memo: HashMap<NodeId, u64> = HashMap::new();
    let h = tree_hash_cached(g, root, &mut memo);
    tree_hash_cache_insert(key, h);
    h
}

fn tree_hash_cached(g: &Graph, root: NodeId, memo: &mut HashMap<NodeId, u64>) -> u64 {
    let mut stack: Vec<(NodeId, bool)> = Vec::new();
    let root = g.resolve(root);
    stack.push((root, false));
    while let Some((id, done)) = stack.pop() {
        let id = g.resolve(id);
        if memo.contains_key(&id) {
            continue;
        }
        if !done {
            stack.push((id, true));
            match g.get(id) {
                Node::Ind(inner) => {
                    stack.push((g.resolve(*inner), false));
                }
                Node::Stem(child) => stack.push((g.resolve(*child), false)),
                Node::Fork(l, r) => {
                    stack.push((g.resolve(*r), false));
                    stack.push((g.resolve(*l), false));
                }
                Node::App { func, args } => {
                    for arg in args.iter().rev() {
                        stack.push((g.resolve(*arg), false));
                    }
                    stack.push((g.resolve(*func), false));
                }
                _ => {}
            }
        } else {
            let h = match g.get(id) {
                Node::Leaf => 0x9e37_01,
                Node::Prim(p) => prim_tag_hash(*p).unwrap_or_else(|| hash_mix(0x9e37_02, *p as u64)),
                Node::Float(f) => hash_mix(0x9e37_03, f.to_bits()),
                Node::Handle(h) => hash_mix(0x9e37_04, *h as u64),
                Node::Ind(inner) => memo[&g.resolve(*inner)],
                Node::Stem(child) => {
                    let mut acc = 0x9e37_05;
                    acc = hash_mix(acc, memo[&g.resolve(*child)]);
                    acc
                }
                Node::Fork(l, r) => {
                    let mut acc = 0x9e37_06;
                    acc = hash_mix(acc, memo[&g.resolve(*l)]);
                    acc = hash_mix(acc, memo[&g.resolve(*r)]);
                    acc
                }
                Node::App { func, args } => {
                    let mut acc = hash_mix(0x9e37_07, memo[&g.resolve(*func)]);
                    acc = hash_mix(acc, args.len() as u64);
                    for arg in args.iter() {
                        acc = hash_mix(acc, memo[&g.resolve(*arg)]);
                    }
                    acc
                }
            };
            memo.insert(id, h);
        }
    }
    memo[&root]
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TreeHashKey {
    graph_id: u64,
    epoch: u64,
    node: u32,
}

fn tree_hash_cache() -> &'static Mutex<HashMap<TreeHashKey, u64>> {
    static CACHE: OnceLock<Mutex<HashMap<TreeHashKey, u64>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn tree_hash_cache_get(key: &TreeHashKey) -> Option<u64> {
    let cache = tree_hash_cache();
    let guard = cache.lock().unwrap();
    guard.get(key).copied()
}

fn tree_hash_cache_insert(key: TreeHashKey, value: u64) {
    let cache = tree_hash_cache();
    let mut guard = cache.lock().unwrap();
    if guard.len() >= TREE_HASH_CACHE_MAX {
        guard.clear();
    }
    guard.insert(key, value);
}

fn primitive_min_args(p: Primitive) -> Option<usize> {
    match p {
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Div
        | Primitive::Eq
        | Primitive::Gt
        | Primitive::Lt => Some(2),
        Primitive::If => Some(3),
        Primitive::S => Some(3),
        Primitive::K => Some(2),
        Primitive::I => Some(1),
        Primitive::First | Primitive::Rest => Some(1),
        _ => None,
    }
}

fn collect_paths(g: &Graph, root: NodeId) -> HashMap<NodeId, Vec<u8>> {
    let mut map: HashMap<NodeId, Vec<u8>> = HashMap::new();
    let mut stack: Vec<(NodeId, Vec<u8>)> = Vec::new();
    stack.push((root, Vec::new()));

    while let Some((id, path)) = stack.pop() {
        let resolved = g.resolve(id);
        if map.contains_key(&resolved) {
            continue;
        }
        map.insert(resolved, path.clone());
        match g.get(resolved) {
            Node::Stem(inner) => {
                let mut p = path.clone();
                p.push(0);
                stack.push((*inner, p));
            }
            Node::Fork(l, r) => {
                let mut p0 = path.clone();
                p0.push(0);
                stack.push((*l, p0));
                let mut p1 = path.clone();
                p1.push(1);
                stack.push((*r, p1));
            }
            Node::App { func, args } => {
                let mut pf = path.clone();
                pf.push(0);
                stack.push((*func, pf));
                for (idx, arg) in args.iter().enumerate() {
                    if idx >= 250 {
                        break;
                    }
                    let mut pa = path.clone();
                    pa.push((idx as u8) + 1);
                    stack.push((*arg, pa));
                }
            }
            Node::Ind(inner) => {
                stack.push((*inner, path));
            }
            _ => {}
        }
    }
    map
}

fn count_nodes(g: &Graph, root: NodeId, limit: usize) -> usize {
    let mut stack = Vec::new();
    let mut seen = HashSet::new();
    let mut count = 0usize;
    stack.push(root);
    while let Some(id) = stack.pop() {
        let resolved = g.resolve(id);
        if !seen.insert(resolved) {
            continue;
        }
        count += 1;
        if limit > 0 && count > limit {
            return count;
        }
        match g.get(resolved) {
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
    count
}

fn clone_subtree_engine(
    g_src: &Graph,
    g_dst: &mut Graph,
    id: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    let resolved = g_src.resolve(id);
    if let Some(&cached) = memo.get(&resolved) {
        return cached;
    }

    #[derive(Clone, Copy)]
    enum Frame {
        Enter(NodeId),
        Exit(NodeId),
    }

    let mut stack: Vec<Frame> = Vec::new();
    let mut visiting: HashSet<NodeId> = HashSet::new();
    stack.push(Frame::Enter(resolved));

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(id) => {
                let id = g_src.resolve(id);
                if memo.contains_key(&id) {
                    continue;
                }
                if !visiting.insert(id) {
                    continue;
                }
                match g_src.get(id) {
                    Node::Ind(inner) => {
                        visiting.remove(&id);
                        stack.push(Frame::Enter(*inner));
                    }
                    Node::Leaf | Node::Prim(_) | Node::Float(_) | Node::Handle(_) => {
                        stack.push(Frame::Exit(id));
                    }
                    Node::Stem(inner) => {
                        stack.push(Frame::Exit(id));
                        stack.push(Frame::Enter(*inner));
                    }
                    Node::Fork(l, r) => {
                        stack.push(Frame::Exit(id));
                        stack.push(Frame::Enter(*r));
                        stack.push(Frame::Enter(*l));
                    }
                    Node::App { func, args } => {
                        stack.push(Frame::Exit(id));
                        for arg in args.iter().rev() {
                            stack.push(Frame::Enter(*arg));
                        }
                        stack.push(Frame::Enter(*func));
                    }
                }
            }
            Frame::Exit(id) => {
                if memo.contains_key(&id) {
                    visiting.remove(&id);
                    continue;
                }
                let node = g_src.get(id).clone();
                let new_id = match node {
                    Node::Leaf => g_dst.add_raw(Node::Leaf),
                    Node::Stem(inner) => {
                        let c = *memo.get(&g_src.resolve(inner)).unwrap();
                        g_dst.add_raw(Node::Stem(c))
                    }
                    Node::Fork(l, r) => {
                        let nl = *memo.get(&g_src.resolve(l)).unwrap();
                        let nr = *memo.get(&g_src.resolve(r)).unwrap();
                        g_dst.add_raw(Node::Fork(nl, nr))
                    }
                    Node::Prim(p) => {
                        if let Some(tagged) = prim_tag_tree(g_dst, p) {
                            tagged
                        } else {
                            g_dst.add_raw(Node::Prim(p))
                        }
                    }
                    Node::Float(f) => g_dst.add_raw(Node::Float(f)),
                    Node::Handle(h) => g_dst.add_raw(Node::Handle(h)),
                    Node::Ind(inner) => *memo.get(&g_src.resolve(inner)).unwrap(),
                    Node::App { func, args } => {
                        let nf = *memo.get(&g_src.resolve(func)).unwrap();
                        let mut new_args: SmallVec<[NodeId; 2]> = SmallVec::new();
                        for arg in args {
                            let na = *memo.get(&g_src.resolve(arg)).unwrap();
                            new_args.push(na);
                        }
                        g_dst.add_raw(Node::App { func: nf, args: new_args })
                    }
                };
                memo.insert(id, new_id);
                visiting.remove(&id);
            }
        }
    }

    memo[&resolved]
}

fn redex_entry_size(entry: &Arc<CachedRedex>) -> usize {
    let node_bytes = entry.graph.nodes.len().saturating_mul(64);
    let event_bytes = entry.events.len().saturating_mul(64);
    let path_bytes = entry
        .snapshot_paths
        .iter()
        .map(|(_, p)| p.len())
        .sum::<usize>()
        .saturating_mul(8);
    node_bytes
        .saturating_add(event_bytes)
        .saturating_add(path_bytes)
        .max(1)
}

fn build_cached_redex(g: &Graph, redex: NodeId, ctx: &EvalContext) -> Option<CachedRedex> {
    let orig_paths = collect_paths(g, redex);
    let mut cached_graph = Graph::new_uninterned();
    let mut memo: HashMap<NodeId, NodeId> = HashMap::new();
    let cached_root = clone_subtree_engine(g, &mut cached_graph, redex, &mut memo);
    let mut snapshot_paths = Vec::with_capacity(orig_paths.len());
    for (orig_id, path) in orig_paths {
        let resolved = g.resolve(orig_id);
        if let Some(&cached_id) = memo.get(&resolved) {
            snapshot_paths.push((cached_id, path));
        }
    }

    let mut trace = ExecutionTrace::new();
    let mut local_ctx = EvalContext::default();
    local_ctx.step_limit = ctx.step_limit;
    local_ctx.node_limit = ctx.node_limit;
    local_ctx.exec_trace = Some(&mut trace);
    local_ctx.redex_cache = None;
    let result = reduce(&mut cached_graph, cached_root, &mut local_ctx);
    if local_ctx.node_limit_hit {
        return None;
    }
    if local_ctx.steps >= local_ctx.step_limit {
        return None;
    }
    if trace.events.is_empty() {
        return None;
    }

    Some(CachedRedex {
        graph: cached_graph,
        result,
        events: trace.events,
        snapshot_paths,
    })
}

fn replay_cached_redex(
    g: &mut Graph,
    root: NodeId,
    redex: NodeId,
    ctx: &mut EvalContext,
    cached: &CachedRedex,
) -> bool {
    let current_paths = collect_paths(g, redex);
    let mut current_by_path: HashMap<Vec<u8>, NodeId> = HashMap::new();
    for (id, path) in current_paths {
        current_by_path.insert(path, id);
    }

    let mut map: Vec<NodeId> = vec![NodeId::NULL; cached.graph.nodes.len()];
    let mut is_snapshot: Vec<bool> = vec![false; cached.graph.nodes.len()];
    for (cached_id, path) in cached.snapshot_paths.iter() {
        if let Some(&current_id) = current_by_path.get(path) {
            let idx = cached_id.0 as usize;
            if idx < map.len() {
                map[idx] = current_id;
                is_snapshot[idx] = true;
            }
        } else {
            return false;
        }
    }

    for idx in 0..cached.graph.nodes.len() {
        if map[idx] == NodeId::NULL {
            let new_id = g.add_raw(Node::Leaf);
            map[idx] = new_id;
        }
    }

    for (idx, node) in cached.graph.nodes.iter().enumerate() {
        if is_snapshot[idx] {
            continue;
        }
        let mapped = match node.clone() {
            Node::Leaf => Node::Leaf,
            Node::Stem(inner) => Node::Stem(map[inner.0 as usize]),
            Node::Fork(l, r) => Node::Fork(map[l.0 as usize], map[r.0 as usize]),
            Node::Prim(p) => Node::Prim(p),
            Node::Float(f) => Node::Float(f),
            Node::Handle(h) => Node::Handle(h),
            Node::Ind(inner) => Node::Ind(map[inner.0 as usize]),
            Node::App { func, args } => {
                let nf = map[func.0 as usize];
                let mut new_args: SmallVec<[NodeId; 2]> = SmallVec::new();
                for arg in args {
                    new_args.push(map[arg.0 as usize]);
                }
                Node::App { func: nf, args: new_args }
            }
        };
        g.replace(map[idx], mapped);
    }

    let result = map[cached.result.0 as usize];
    g.replace(root, Node::Ind(result));

    if let Some(trace) = &mut ctx.exec_trace {
        for event in cached.events.iter() {
            let redex_m = map[event.redex.0 as usize];
            let result_m = map[event.result.0 as usize];
            let mut args_m = Vec::with_capacity(event.args.len());
            for arg in event.args.iter() {
                args_m.push(map[arg.0 as usize]);
            }
            trace.record(event.rule, redex_m, args_m, result_m);
        }
    }

    let extra = cached.events.len().saturating_sub(1);
    ctx.steps = ctx.steps.saturating_add(extra);
    if ctx.node_limit > 0 && g.nodes.len() > ctx.node_limit {
        ctx.node_limit_hit = true;
    }
    true
}



// Removed try_reduce_rule as it is now integrated into reduce_step logic

pub fn reduce_whnf(g: &mut Graph, id: NodeId) -> NodeId {
    let mut ctx = EvalContext::default();
    ctx.step_limit = 100;
    reduce_whnf_depth_with_ctx(g, id, 0, &mut ctx)
}

pub fn reduce_whnf_with_ctx(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> NodeId {
    reduce_whnf_depth_with_ctx(g, id, 0, ctx)
}

#[derive(Debug)]
struct ApplyState {
    redex: NodeId,
    head: NodeId,
    pending: SmallVec<[NodeId; 2]>,
    changed: bool,
    skip_tag: bool,
}

#[derive(Debug)]
enum WhnfCont {
    Apply(ApplyState),
    TriageP {
        state: ApplyState,
        q: NodeId,
        r: NodeId,
    },
    TriageR {
        state: ApplyState,
        p_whnf: NodeId,
        q: NodeId,
        r_original: NodeId,
    },
    Prim {
        state: ApplyState,
        prim: Primitive,
        need_idxs: SmallVec<[usize; 2]>,
        pos: usize,
        reduced: Vec<Option<NodeId>>,
        strict: bool,
    },
}

fn prim_need_whnf_indices(p: Primitive) -> SmallVec<[usize; 2]> {
    match p {
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Div
        | Primitive::Eq
        | Primitive::Gt
        | Primitive::Lt => smallvec::smallvec![0, 1],
        Primitive::If => smallvec::smallvec![0],
        Primitive::First | Primitive::Rest => smallvec::smallvec![0],
        _ => smallvec::smallvec![],
    }
}

fn decode_number_any_whnf(g: &Graph, id: NodeId) -> Option<DecodedNumber> {
    let (payload, tag) = unwrap_data(g, id);
    match tag {
        Some(Primitive::TagInt) => decode_int_pure(g, payload).map(DecodedNumber::Int),
        Some(Primitive::TagFloat) => match g.get(payload) {
            Node::Float(f) => Some(DecodedNumber::Float(*f)),
            _ => None,
        },
        _ => {
            if let Node::Float(f) = g.get(id) {
                return Some(DecodedNumber::Float(*f));
            }
            decode_int_pure(g, id).map(DecodedNumber::Int)
        }
    }
}

fn apply_primitive_whnf(
    g: &mut Graph,
    p: Primitive,
    args: &SmallVec<[NodeId; 2]>,
) -> Option<NodeId> {
    match p {
        Primitive::Add | Primitive::Sub | Primitive::Mul | Primitive::Div => {
            if args.len() < 2 {
                return None;
            }
            let val_a = decode_number_any_whnf(g, args[0]);
            let val_b = decode_number_any_whnf(g, args[1]);
            if let (Some(nav), Some(nbv)) = (val_a, val_b) {
                match (nav, nbv) {
                    (DecodedNumber::Int(a), DecodedNumber::Int(b)) => {
                        let res = match p {
                            Primitive::Add => a + b,
                            Primitive::Sub => a - b,
                            Primitive::Mul => a * b,
                            Primitive::Div => {
                                if b.is_zero() {
                                    BigInt::zero()
                                } else {
                                    a / b
                                }
                            }
                            _ => BigInt::zero(),
                        };
                        let raw = encode_int(g, &res);
                        Some(make_tag(g, Primitive::TagInt, raw))
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Float(b)) => {
                        let res = match p {
                            Primitive::Add => a + b,
                            Primitive::Sub => a - b,
                            Primitive::Mul => a * b,
                            Primitive::Div => a / b,
                            _ => 0.0,
                        };
                        let raw = g.add(Node::Float(res));
                        Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                    (DecodedNumber::Int(a), DecodedNumber::Float(b)) => {
                        use num_traits::ToPrimitive;
                        let af = a.to_f64().unwrap_or(0.0);
                        let res = match p {
                            Primitive::Add => af + b,
                            Primitive::Sub => af - b,
                            Primitive::Mul => af * b,
                            Primitive::Div => af / b,
                            _ => 0.0,
                        };
                        let raw = g.add(Node::Float(res));
                        Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Int(b)) => {
                        use num_traits::ToPrimitive;
                        let bf = b.to_f64().unwrap_or(0.0);
                        let res = match p {
                            Primitive::Add => a + bf,
                            Primitive::Sub => a - bf,
                            Primitive::Mul => a * bf,
                            Primitive::Div => a / bf,
                            _ => 0.0,
                        };
                        let raw = g.add(Node::Float(res));
                        Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                }
            } else {
                None
            }
        }
        Primitive::Eq | Primitive::Gt | Primitive::Lt => {
            if args.len() < 2 {
                return None;
            }
            let val_a = decode_number_any_whnf(g, args[0]);
            let val_b = decode_number_any_whnf(g, args[1]);
            let check_numeric = if let (Some(nav), Some(nbv)) = (val_a, val_b) {
                Some(match (nav, nbv) {
                    (DecodedNumber::Int(a), DecodedNumber::Int(b)) => match p {
                        Primitive::Eq => a == b,
                        Primitive::Gt => a > b,
                        Primitive::Lt => a < b,
                        _ => false,
                    },
                    (DecodedNumber::Float(a), DecodedNumber::Float(b)) => match p {
                        Primitive::Eq => (a - b).abs() < f64::EPSILON,
                        Primitive::Gt => a > b,
                        Primitive::Lt => a < b,
                        _ => false,
                    },
                    (DecodedNumber::Int(a), DecodedNumber::Float(b)) => {
                        use num_traits::ToPrimitive;
                        let af = a.to_f64().unwrap_or(0.0);
                        match p {
                            Primitive::Eq => (af - b).abs() < f64::EPSILON,
                            Primitive::Gt => af > b,
                            Primitive::Lt => af < b,
                            _ => false,
                        }
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Int(b)) => {
                        use num_traits::ToPrimitive;
                        let bf = b.to_f64().unwrap_or(0.0);
                        match p {
                            Primitive::Eq => (a - bf).abs() < f64::EPSILON,
                            Primitive::Gt => a > bf,
                            Primitive::Lt => a < bf,
                            _ => false,
                        }
                    }
                })
            } else {
                None
            };
            match check_numeric {
                Some(check) => {
                    let res_node = if check {
                        let n = g.add(Node::Leaf);
                        g.add(Node::Stem(n))
                    } else {
                        g.add(Node::Leaf)
                    };
                    if args.len() > 2 {
                        let rest = args[2..].iter().cloned().collect();
                        Some(mk_seq(g, res_node, rest))
                    } else {
                        Some(res_node)
                    }
                }
                None => {
                    let (pa, ta) = unwrap_data(g, args[0]);
                    let (pb, tb) = unwrap_data(g, args[1]);
                    if ta.is_some() && ta == tb {
                        let check = pa == pb;
                        let res_node = if check {
                            let n = g.add(Node::Leaf);
                            g.add(Node::Stem(n))
                        } else {
                            g.add(Node::Leaf)
                        };
                        if args.len() > 2 {
                            let rest = args[2..].iter().cloned().collect();
                            Some(mk_seq(g, res_node, rest))
                        } else {
                            Some(res_node)
                        }
                    } else {
                        None
                    }
                }
            }
        }
        Primitive::If => {
            if args.len() < 3 {
                return None;
            }
            let cond = args[0];
            if let Node::Leaf = g.get(cond) {
                Some(args[2])
            } else if let Node::Float(f) = g.get(cond) {
                if *f == 0.0 {
                    Some(args[2])
                } else {
                    Some(args[1])
                }
            } else {
                Some(args[1])
            }
        }
        Primitive::I => {
            if args.is_empty() {
                return None;
            }
            let res = args[0];
            if args.len() > 1 {
                let rest = args[1..].iter().cloned().collect();
                Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::K => {
            if args.len() < 2 {
                return None;
            }
            let res = args[0];
            if args.len() > 2 {
                let rest = args[2..].iter().cloned().collect();
                Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::S => {
            if args.len() < 3 {
                return None;
            }
            let x = args[0];
            let y = args[1];
            let z = args[2];
            let xz = mk_seq(g, x, smallvec::smallvec![z]);
            let yz = mk_seq(g, y, smallvec::smallvec![z]);
            let res = mk_seq(g, xz, smallvec::smallvec![yz]);
            if args.len() > 3 {
                let rest = args[3..].iter().cloned().collect();
                Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::First => {
            if args.is_empty() {
                return None;
            }
            match g.get(args[0]).clone() {
                Node::Fork(head, _) => {
                    if let Node::Prim(p) = g.get(head) {
                        if matches!(
                            p,
                            Primitive::TagInt
                                | Primitive::TagFloat
                                | Primitive::TagStr
                                | Primitive::TagChar
                        ) {
                            return Some(g.add(Node::Leaf));
                        }
                    }
                    Some(head)
                }
                _ => Some(g.add(Node::Leaf)),
            }
        }
        Primitive::Rest => {
            if args.is_empty() {
                return None;
            }
            match g.get(args[0]).clone() {
                Node::Fork(head, tail) => {
                    if let Node::Prim(p) = g.get(head) {
                        if matches!(
                            p,
                            Primitive::TagInt
                                | Primitive::TagFloat
                                | Primitive::TagStr
                                | Primitive::TagChar
                        ) {
                            return Some(g.add(Node::Leaf));
                        }
                    }
                    Some(tail)
                }
                _ => Some(g.add(Node::Leaf)),
            }
        }
        _ => None,
    }
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

fn reduce_whnf_depth_with_ctx(
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

fn reduce_progress_ms() -> u64 {
    use std::sync::OnceLock;
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
    // Keep 0 as "disabled".
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

fn node_kind(g: &Graph, id: NodeId) -> &'static str {
    match g.get(g.resolve(id)) {
        Node::Leaf => "Leaf",
        Node::Stem(_) => "Stem",
        Node::Fork(_, _) => "Fork",
        Node::App { .. } => "Seq",
        Node::Prim(_) => "Prim",
        Node::Float(_) => "Float",
        Node::Ind(_) => "Ind",
        Node::Handle(_) => "Handle",
    }
}

fn debug_args(g: &Graph, args: &SmallVec<[NodeId; 2]>) -> String {
    let mut out = String::new();
    for (idx, arg) in args.iter().enumerate() {
        if idx >= DEBUG_UNPARSE_MAX_ARGS {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str("...");
            break;
        }
        if idx > 0 {
            out.push(' ');
        }
        out.push_str(&debug_unparse(g, *arg));
    }
    out
}

fn debug_unparse(g: &Graph, id: NodeId) -> String {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut budget = DEBUG_UNPARSE_MAX_NODES;
    debug_unparse_rec(g, g.resolve(id), 0, &mut budget, &mut seen)
}

fn debug_unparse_rec(
    g: &Graph,
    id: NodeId,
    depth: usize,
    budget: &mut usize,
    seen: &mut HashSet<NodeId>,
) -> String {
    enum Item<'a> {
        Node(NodeId, usize),
        Text(&'a str),
        Owned(String),
    }

    let mut out = String::new();
    let mut stack: Vec<Item<'_>> = Vec::new();
    stack.push(Item::Node(id, depth));

    while let Some(item) = stack.pop() {
        match item {
            Item::Text(s) => out.push_str(s),
            Item::Owned(s) => out.push_str(&s),
            Item::Node(curr, curr_depth) => {
                if *budget == 0 || curr_depth > DEBUG_UNPARSE_MAX_DEPTH {
                    out.push_str("...");
                    continue;
                }
                let curr = g.resolve(curr);
                if !seen.insert(curr) {
                    out.push_str(&format!("<cycle {}>", curr.0));
                    continue;
                }
                *budget = budget.saturating_sub(1);
                match g.get(curr) {
                    Node::Leaf => out.push_str("n"),
                    Node::Stem(x) => {
                        stack.push(Item::Text(")"));
                        stack.push(Item::Node(*x, curr_depth + 1));
                        stack.push(Item::Text(" "));
                        stack.push(Item::Text("n"));
                        stack.push(Item::Text("("));
                    }
                    Node::Fork(x, y) => {
                        stack.push(Item::Text(")"));
                        stack.push(Item::Node(*y, curr_depth + 1));
                        stack.push(Item::Text(" "));
                        stack.push(Item::Node(*x, curr_depth + 1));
                        stack.push(Item::Text(" "));
                        stack.push(Item::Text("n"));
                        stack.push(Item::Text("("));
                    }
                    Node::App { func, args } => {
                        let limit = DEBUG_UNPARSE_MAX_ARGS.min(args.len());
                        stack.push(Item::Text(")"));
                        if args.len() > DEBUG_UNPARSE_MAX_ARGS {
                            stack.push(Item::Text(" ..."));
                        }
                        for arg in args.iter().take(limit).rev() {
                            stack.push(Item::Node(*arg, curr_depth + 1));
                            stack.push(Item::Text(" "));
                        }
                        stack.push(Item::Node(*func, curr_depth + 1));
                        stack.push(Item::Text("("));
                    }
                    Node::Prim(p) => out.push_str(&format!("{:?}", p)),
                    Node::Float(f) => out.push_str(&format!("{}", f)),
                    Node::Ind(rec) => stack.push(Item::Node(*rec, curr_depth + 1)),
                    Node::Handle(h) => out.push_str(&format!("<Handle {}>", h)),
                }
            }
        }
    }

    out
}


// Tagging combinators
// tag{t, f} = d{t}(d{f}(KK)) where d{x} = Stem(Stem(x)) and KK = Fork(Leaf, Leaf)
// This canonicalizes to Fork(t, Fork(f, KK))
pub fn make_tag(g: &mut Graph, tag_prim: Primitive, val: NodeId) -> NodeId {
    let t = g.add(Node::Prim(tag_prim));
    // Fork(Tag, Fork(Val, Fork(Leaf, Leaf)))
    let leaf = g.add(Node::Leaf);
    let kk = g.add(Node::Fork(leaf, leaf)); // KK
    let val_kk = g.add(Node::Fork(val, kk));
    g.add(Node::Fork(t, val_kk))
}

pub fn unwrap_data(g: &Graph, id: NodeId) -> (NodeId, Option<Primitive>) {
    // Expect Fork(Tag, Fork(Val, KK))
    if let Node::Fork(tag_node, inner) = g.get(id) {
        if let Node::Prim(p) = g.get(*tag_node) {
             if let Node::Fork(val, _kk) = g.get(*inner) {
                 // Verify KK? Not strictly necessary if we trust structure
                 return (*val, Some(*p));
             }
        }
    }
    (id, None)
}


// Helper enum for numeric dispatch
#[derive(Debug)]
enum DecodedNumber {
    Int(BigInt),
    Float(f64),
}

fn decode_number_any(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> Option<DecodedNumber> {
    let reduced = reduce_whnf_with_ctx(g, id, ctx);
    let (payload, tag) = unwrap_data(g, reduced);
    
    // Check Tag
    match tag {
        Some(Primitive::TagInt) => {
            if let Some(bi) = decode_int_with_ctx(g, payload, ctx) {
                return Some(DecodedNumber::Int(bi));
            }
        },
        Some(Primitive::TagFloat) => {
             if let Node::Float(f) = g.get(payload) {
                 return Some(DecodedNumber::Float(*f));
             }
        },
        _ => {}
    }
    
    // Fallback logic
    if let Node::Float(f) = g.get(reduced) { return Some(DecodedNumber::Float(*f)); }
    if let Some(bi) = decode_int_with_ctx(g, reduced, ctx) {
         return Some(DecodedNumber::Int(bi));
    }
    None
}

// ... decode_int is unchanged ...
// unzigzag
fn unzigzag(n: BigInt) -> BigInt {
    if &n & BigInt::from(1u8) == BigInt::zero() {
        // Even: n / 2
        n >> 1
    } else {
        // Odd: -((n + 1) / 2)
        let numerator = n + BigInt::from(1u8);
        let halved: BigInt = numerator >> 1;
        -halved
    }
}

// Decodes raw natural number (the zig-zag encoded value)
fn decode_raw_nat_with_ctx(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> Option<BigInt> {
    let mut bits: Vec<bool> = Vec::new();
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut curr = reduce_whnf_with_ctx(g, id, ctx);

    loop {
        let resolved = g.resolve(curr);
        if !seen.insert(resolved) {
            return None;
        }
        match g.get(resolved) {
            Node::Leaf => break,
            Node::Stem(inner) => {
                bits.push(false);
                curr = reduce_whnf_with_ctx(g, *inner, ctx);
            }
            Node::Fork(rec, leaf) => {
                if !matches!(g.get(*leaf), Node::Leaf) {
                    return None;
                }
                bits.push(true);
                curr = reduce_whnf_with_ctx(g, *rec, ctx);
            }
            Node::Ind(inner) => {
                curr = *inner;
            }
            _ => return None,
        }
    }

    let mut val = BigInt::zero();
    for bit in bits.iter().rev() {
        val = val << 1;
        if *bit {
            val += 1;
        }
    }
    Some(val)
}

pub fn decode_raw_nat(g: &mut Graph, id: NodeId) -> Option<BigInt> {
    let mut ctx = EvalContext::default();
    ctx.step_limit = 100;
    decode_raw_nat_with_ctx(g, id, &mut ctx)
}

fn decode_int_with_ctx(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> Option<BigInt> {
    let raw = decode_raw_nat_with_ctx(g, id, ctx)?;
    Some(unzigzag(raw))
}

pub fn decode_int(g: &mut Graph, id: NodeId) -> Option<BigInt> {
    let mut ctx = EvalContext::default();
    ctx.step_limit = 100;
    let raw = decode_raw_nat_with_ctx(g, id, &mut ctx)?;
    Some(unzigzag(raw))
}

fn apply_primitive(
    g: &mut Graph,
    p: Primitive,
    args: &SmallVec<[NodeId; 2]>,
    ctx: &mut EvalContext,
) -> Option<NodeId> {
    let _p_node = g.add(Node::Prim(p)); // Keep reference if needed, but we construct App if stuck?
    // Actually caller (reduce_step) constructs App if we return None.
    
    match p {
        Primitive::Add | Primitive::Sub | Primitive::Mul | Primitive::Div => {
            if args.len() < 2 { return None; } 
            
            let val_a = decode_number_any(g, args[0], ctx);
            let val_b = decode_number_any(g, args[1], ctx);
            
            if let (Some(nav), Some(nbv)) = (val_a, val_b) {
                match (nav, nbv) {
                    (DecodedNumber::Int(a), DecodedNumber::Int(b)) => {
                        let res = match p {
                            Primitive::Add => a + b,
                            Primitive::Sub => a - b,
                            Primitive::Mul => a * b,
                            Primitive::Div => {
                                if b.is_zero() { BigInt::zero() } else { a / b }
                            },
                            _ => BigInt::zero(),
                        };
                         let raw = encode_int(g, &res);
                         Some(make_tag(g, Primitive::TagInt, raw))
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Float(b)) => {
                        let res = match p {
                            Primitive::Add => a + b,
                            Primitive::Sub => a - b,
                            Primitive::Mul => a * b,
                            Primitive::Div => a / b,
                            _ => 0.0,
                        };
                        let raw = g.add(Node::Float(res));
                        Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                    (DecodedNumber::Int(a), DecodedNumber::Float(b)) => {
                         use num_traits::ToPrimitive;
                         let af = a.to_f64().unwrap_or(0.0);
                         let res = match p {
                            Primitive::Add => af + b,
                            Primitive::Sub => af - b,
                            Primitive::Mul => af * b,
                            Primitive::Div => af / b,
                            _ => 0.0,
                         };
                         let raw = g.add(Node::Float(res));
                         Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Int(b)) => {
                         use num_traits::ToPrimitive;
                         let bf = b.to_f64().unwrap_or(0.0);
                         let res = match p {
                            Primitive::Add => a + bf,
                            Primitive::Sub => a - bf,
                            Primitive::Mul => a * bf,
                            Primitive::Div => a / bf,
                            _ => 0.0,
                         };
                         let raw = g.add(Node::Float(res));
                         Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                }
            } else {
                 None
            }
        }
        Primitive::Eq | Primitive::Gt | Primitive::Lt => {
            if args.len() < 2 { return None; }
            let val_a = decode_number_any(g, args[0], ctx);
            let val_b = decode_number_any(g, args[1], ctx);
            
            let check_numeric = if let (Some(nav), Some(nbv)) = (val_a, val_b) {
                Some(match (nav, nbv) {
                    (DecodedNumber::Int(a), DecodedNumber::Int(b)) => match p {
                        Primitive::Eq => a == b,
                        Primitive::Gt => a > b,
                        Primitive::Lt => a < b,
                        _ => false,
                    },
                    (DecodedNumber::Float(a), DecodedNumber::Float(b)) => match p {
                        Primitive::Eq => (a - b).abs() < f64::EPSILON,
                        Primitive::Gt => a > b,
                        Primitive::Lt => a < b,
                        _ => false,
                    },
                     (DecodedNumber::Int(a), DecodedNumber::Float(b)) => {
                         use num_traits::ToPrimitive;
                         let af = a.to_f64().unwrap_or(0.0);
                         match p {
                            Primitive::Eq => (af - b).abs() < f64::EPSILON,
                            Primitive::Gt => af > b,
                            Primitive::Lt => af < b,
                            _ => false,
                         }
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Int(b)) => {
                         use num_traits::ToPrimitive;
                         let bf = b.to_f64().unwrap_or(0.0);
                         match p {
                            Primitive::Eq => (a - bf).abs() < f64::EPSILON,
                            Primitive::Gt => a > bf,
                            Primitive::Lt => a < bf,
                            _ => false,
                         }
                    }
                })
            } else {
                None
            };
            
            match check_numeric {
                Some(check) => {
                     let res_node = if check {
                        let n = g.add(Node::Leaf);
                        g.add(Node::Stem(n)) 
                     } else {
                        g.add(Node::Leaf)
                     };
                    if args.len() > 2 {
                         let rest = args[2..].iter().cloned().collect();
                         Some(mk_seq(g, res_node, rest))
                    } else {
                        Some(res_node)
                    }
                }
                None => {
                     // Try Tagged Equality
                     let a_node = reduce_whnf_with_ctx(g, args[0], ctx);
                     let b_node = reduce_whnf_with_ctx(g, args[1], ctx);
                     
                     let (pa, ta) = unwrap_data(g, a_node);
                     let (pb, tb) = unwrap_data(g, b_node);
                     
                     if ta.is_some() && ta == tb {
                         // Structural check on payload? Or primitives only?
                         // Should check if payload same.
                         // For strings, payload is list structure.
                         // Does unwrap_data give payload node? Yes.
                         // But if we want DEEP equality of payload (e.g. string content), we need structural equality check?
                         // reduce_whnf returns NODEID.
                         // If we compare pa == pb (NodeId), it works if interned correctly OR same pointer.
                         // But equal strings might have different NodeIds if constructed differently?
                         // Interner handles it if structure is identical.
                         // So checking NodeId equality is usually sufficient for structural equality in hash-consed graph.
                         let check = pa == pb;
                         
                         let res_node = if check {
                             let n = g.add(Node::Leaf);
                             g.add(Node::Stem(n))
                         } else {
                             g.add(Node::Leaf)
                         };
                          if args.len() > 2 {
                             let rest = args[2..].iter().cloned().collect();
                             Some(mk_seq(g, res_node, rest))
                        } else {
                            Some(res_node)
                        }
                     } else {
                         None
                     }
                }
            }
        }

        Primitive::If => {
            // (If cond t f)
            if args.len() < 3 { return None; }
            let cond = reduce_whnf_with_ctx(g, args[0], ctx);
            
            if let Node::Leaf = g.get(cond) {
                 Some(args[2]) // False
            } else {
                 if let Node::Float(f) = g.get(cond) {
                     if *f == 0.0 { Some(args[2]) } else { Some(args[1]) }
                 } else {
                     Some(args[1]) // Assume True
                 }
            }
        }
        
        Primitive::I => {
            if args.is_empty() { return None; }
            let res = args[0];
            if args.len() > 1 {
                 let rest = args[1..].iter().cloned().collect();
                 Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::K => {
            if args.len() < 2 { return None; }
            let res = args[0];
            if args.len() > 2 {
                 let rest = args[2..].iter().cloned().collect();
                 Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::S => {
            if args.len() < 3 { return None; }
            let x = args[0];
            let y = args[1];
            let z = args[2];
            
            // x z
            let xz = mk_seq(g, x, smallvec::smallvec![z]);
            let yz = mk_seq(g, y, smallvec::smallvec![z]);
            let res = mk_seq(g, xz, smallvec::smallvec![yz]);
            
            if args.len() > 3 {
                 let rest = args[3..].iter().cloned().collect();
                 Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }

        Primitive::First => {
             if args.is_empty() { return None; }
             let arg = reduce_whnf_with_ctx(g, args[0], ctx); // Reduce argument to see struct
             
             match g.get(arg).clone() {
                 Node::Fork(head, _) => {
                     // check if header is a tag primitive (Atom)
                     if let Node::Prim(p) = g.get(head) {
                         if matches!(p, Primitive::TagInt | Primitive::TagFloat | Primitive::TagStr | Primitive::TagChar) {
                             // It's an Atom (e.g. String). First fails.
                             return Some(g.add(Node::Leaf)); // Fail type
                         }
                     }
                     // It's a List. Return head.
                     Some(head)
                 },
                 _ => Some(g.add(Node::Leaf)), // Fail on non-fork
             }
        }
        Primitive::Rest => {
             if args.is_empty() { return None; }
             let arg = reduce_whnf_with_ctx(g, args[0], ctx);
             
             match g.get(arg).clone() {
                 Node::Fork(head, tail) => {
                     if let Node::Prim(p) = g.get(head) {
                         if matches!(p, Primitive::TagInt | Primitive::TagFloat | Primitive::TagStr | Primitive::TagChar) {
                             return Some(g.add(Node::Leaf)); // Fail type
                         }
                     }
                     // It's a List. Return tail.
                     Some(tail)
                 },
                 _ => Some(g.add(Node::Leaf)), 
             }
        }



        _ => None,
    }
}

// Pure decoding for display (assumes node is already reduced/WHNF)
pub fn decode_raw_nat_pure(g: &Graph, id: NodeId) -> Option<BigInt> {
    let mut id = id;
    let mut val = BigInt::zero();
    let mut shift: usize = 0;
    loop {
        match g.get(id) {
            Node::Leaf => return Some(val),
            Node::Stem(rec) => {
                id = *rec;
                shift = shift.saturating_add(1);
            }
            Node::Fork(rec, leaf) => match g.get(*leaf) {
                Node::Leaf => {
                    val += BigInt::from(1u32) << shift;
                    id = *rec;
                    shift = shift.saturating_add(1);
                }
                _ => return None,
            },
            _ => return None,
        }
    }
}

pub fn decode_int_pure(g: &Graph, id: NodeId) -> Option<BigInt> {
    let raw = decode_raw_nat_pure(g, id)?;
    Some(unzigzag(raw))
}

pub fn decode_str_pure(g: &Graph, mut id: NodeId) -> Option<String> {
    let mut s = String::new();
    let mut limit = 10000;
    while limit > 0 {
        limit -= 1;
        match g.get(id) {
             Node::Leaf => return Some(s),
             Node::Fork(head, tail) => {
                 let code_bi = decode_raw_nat_pure(g, *head)?;
                 use num_traits::ToPrimitive;
                 let code_u32 = code_bi.to_u32()?;
                 let c = std::char::from_u32(code_u32)?;
                 s.push(c);
                 id = *tail;
             }
             _ => return None,
        }
    }
    None
}

pub fn unparse(g: &Graph, id: NodeId) -> String {
    // Check tags
    let (payload, tag) = unwrap_data(g, id);
    if let Some(t) = tag {
        match t {
            Primitive::TagInt => {
                if let Some(bi) = decode_int_pure(g, payload) {
                    return format!("{}", bi);
                } else {
                    return format!("Int(?)"); 
                }
            },
            Primitive::TagFloat => {
                if let Node::Float(f) = g.get(payload) {
                    return format!("{}", f);
                }
            },
            Primitive::TagStr => {
                 if let Some(s) = decode_str_pure(g, payload) {
                     return format!("{:?}", s);
                 } else {
                     return format!("Str(?)");
                 }
            },
             _ => {}
        }
    }

    enum Item<'a> {
        Node(NodeId),
        Text(&'a str),
        Owned(String),
    }

    let mut out = String::new();
    let mut stack: Vec<Item<'_>> = Vec::new();
    stack.push(Item::Node(id));

    while let Some(item) = stack.pop() {
        match item {
            Item::Text(s) => out.push_str(s),
            Item::Owned(s) => out.push_str(&s),
            Item::Node(curr) => match g.get(curr) {
                Node::Leaf => out.push_str("n"),
                Node::Stem(x) => {
                    stack.push(Item::Text(")"));
                    stack.push(Item::Node(*x));
                    stack.push(Item::Text(" "));
                    stack.push(Item::Text("n"));
                    stack.push(Item::Text("("));
                }
                Node::Fork(x, y) => {
                    stack.push(Item::Text(")"));
                    stack.push(Item::Node(*y));
                    stack.push(Item::Text(" "));
                    stack.push(Item::Node(*x));
                    stack.push(Item::Text(" "));
                    stack.push(Item::Text("n"));
                    stack.push(Item::Text("("));
                }
                Node::App { func, args } => {
                    stack.push(Item::Text(")"));
                    for arg in args.iter().rev() {
                        stack.push(Item::Node(*arg));
                        stack.push(Item::Text(" "));
                    }
                    stack.push(Item::Node(*func));
                    stack.push(Item::Text("("));
                }
                Node::Prim(p) => out.push_str(&format!("{:?}", p)),
                Node::Float(f) => out.push_str(&format!("{}", f)),
                Node::Ind(rec) => stack.push(Item::Node(*rec)),
                Node::Handle(h) => out.push_str(&format!("<Handle {}>", h)),
            },
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triage_rule_k() {
        // Rule 1: △△ y z -> y
        let mut g = Graph::new();
        let n = g.add(Node::Leaf);
        let y = g.add(Node::Float(2.0));
        let z = g.add(Node::Float(3.0));

        let term = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![n, y, z],
        });

        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        assert_eq!(res, y, "K rule failed: △△ y z -> y");
    }

    #[test]
    fn test_triage_rule_s() {
        // Rule 2: △(△x) y z -> x z (y z)
        let mut g = Graph::new();
        let n = g.add(Node::Leaf);
        let k = g.add(Node::Stem(n)); // K = △△

        let x = k; // K
        let y = n; // Leaf
        let z = g.add(Node::Float(42.0));

        let stem_x = g.add(Node::Stem(x));
        let term = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![stem_x, y, z],
        });

        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        assert_eq!(res, z, "S rule failed: △(△x) y z -> x z (y z)");
    }

    #[test]
    fn test_triage_fork_cases() {
        // Rules 3-5: triage on z when p is a fork
        let mut g = Graph::new();
        let n = g.add(Node::Leaf);

        // Leaf case: △(△w x) y △ -> w
        let w = g.add(Node::Float(1.0));
        let x = g.add(Node::Leaf);
        let y = g.add(Node::Leaf);
        let fork_wx = g.add(Node::Fork(w, x));
        let term_leaf = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![fork_wx, y, n],
        });
        let mut ctx = EvalContext::default();
        let res_leaf = reduce(&mut g, term_leaf, &mut ctx);
        assert_eq!(res_leaf, w, "Fork leaf case failed");

        // Stem case: △(△w x) y (△u) -> x u
        let u = g.add(Node::Float(7.0));
        let z_stem = g.add(Node::Stem(u));
        let fork_wx2 = g.add(Node::Fork(w, n)); // x = Leaf
        let term_stem = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![fork_wx2, y, z_stem],
        });
        let mut ctx = EvalContext::default();
        let res_stem = reduce(&mut g, term_stem, &mut ctx);
        let expected_stem = g.add(Node::Stem(u)); // Leaf u -> Stem(u)
        assert_eq!(res_stem, expected_stem, "Fork stem case failed");

        // Fork case: △(△w x) y (△u v) -> y u v
        let u2 = g.add(Node::Float(3.0));
        let v2 = g.add(Node::Float(4.0));
        let z_fork = g.add(Node::Fork(u2, v2));
        let fork_wx3 = g.add(Node::Fork(w, x));
        let term_fork = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![fork_wx3, y, z_fork],
        });
        let mut ctx = EvalContext::default();
        let res_fork = reduce(&mut g, term_fork, &mut ctx);
        let expected_fork = g.add(Node::Fork(u2, v2));
        assert_eq!(res_fork, expected_fork, "Fork fork case failed");
    }
    
    #[test]
    fn test_arithmetic() {
        let mut g = Graph::new();
        let one = g.add(Node::Float(1.0));
        let two = g.add(Node::Float(2.0));
        let add = g.add(Node::Prim(Primitive::Add));
        
        let term = g.add(Node::App {
            func: add,
            args: smallvec::smallvec![one, two]
        });
        
        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        let (payload, _) = unwrap_data(&g, res);
        match g.get(payload) {
            Node::Float(f) => assert_eq!(*f, 3.0),
            _ => panic!("Expected 3.0"),
        }
    }

    #[test]
    #[ignore]
    fn test_list_primitives() {
        let mut g = Graph::new();
        // Implement cons, first, rest as Terms using Triage logic
        
        // n
        let n = g.add(Node::Leaf);
        
        // K = \x y. x
        // cons = \a b. n a (K b)
        // first = \p. n p n n (Dispatches on Leaf -> returns w=a)
        // rest = \p. n p n (n n) (Dispatches on Stem(n) -> returns x n = K b n -> b)
        
        use crate::parser::Parser;
        // use crate::compiler::compile; // Implicitly used by parser
        
        let mut parse = |code| {
             let mut p = Parser::new(code);
             let res = p.parse_toplevel(&mut g, None).unwrap();
             if let crate::parser::ParseResult::Term(id) = res { id } else { panic!("Not a term") }
        };
        
        // cons wraps b in K: (n a ((fn x (fn y x)) b))
        let cons = parse("(fn a (fn b (n a ((fn x (fn y x)) b))))");
        let first = parse("(fn p (n p n n))");
        let rest = parse("(fn p (n p n (n n)))");
        
        // (first (cons 1 2)) -> 1
        let one = g.add(Node::Float(1.0));
        let two = g.add(Node::Float(2.0));
        
        // (cons 1 2)
        let cons_1_2 = g.add(Node::App {
            func: cons,
            args: smallvec::smallvec![one, two]
        });
        
        // (first (cons 1 2))
        let f = g.add(Node::App {
            func: first,
            args: smallvec::smallvec![cons_1_2]
        });
        
        let mut ctx = EvalContext::default();
        let res_f = reduce(&mut g, f, &mut ctx);
        match g.get(res_f) {
            Node::Float(v) => assert_eq!(*v, 1.0),
            _ => panic!("Expected 1.0, got {:?}", unparse(&g, res_f)),
        }
        
        // (rest (cons 1 2))
        let r = g.add(Node::App {
            func: rest,
            args: smallvec::smallvec![cons_1_2]
        });
        
        let mut ctx = EvalContext::default();
        let res_r = reduce(&mut g, r, &mut ctx);
        match g.get(res_r) {
            Node::Float(v) => assert_eq!(*v, 2.0),
            _ => panic!("Expected 2.0, got {:?}", unparse(&g, res_r)),
        }
    }

    #[test]
    fn test_arithmetic_dispatch() {
        let mut g = Graph::new();
        // Construct Tree Integer 3 using encode_int (ZigZag encoded)
        let three_bi = BigInt::from(3);
        let raw_three = encode_int(&mut g, &three_bi);
        let three_tagged = make_tag(&mut g, Primitive::TagInt, raw_three);

        // Float 2.0
        let two = g.add(Node::Float(2.0));
        let tagged_two = make_tag(&mut g, Primitive::TagFloat, two);

        let add = g.add(Node::Prim(Primitive::Add));
        
        // (+ 3 2.0) -> 5.0
        let term = g.add(Node::App {
            func: add,
            args: smallvec::smallvec![three_tagged, tagged_two]
        });
        
        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        let (payload, tag) = unwrap_data(&g, res);
        
        // 3 (TreeInt) + 2.0 (Float) should result in Float(5.0)
        // Because mixed arithmetic usually promotes to float?
        // My implementation: (Int, Float) -> Float.
        // So this test expectation remains CORRECT for 5.0.
        // But let's verify checking the TAG too.
        match tag {
             Some(Primitive::TagFloat) => {},
             _ => panic!("Expected TagFloat"),
        }
        match g.get(payload) {
            Node::Float(f) => assert_eq!(*f, 5.0),
            _ => panic!("Expected 5.0, got {:?}", unparse(&g, res)),
        }
    }
    
    #[test]
    fn test_large_integer_arithmetic() {
        let mut g = Graph::new();
        use std::str::FromStr;
        let big_n_str = "1000000000000000000000000"; // 10^24
        let big_n = BigInt::from_str(big_n_str).unwrap();
        
        let raw = encode_int(&mut g, &big_n);
        let tagged = make_tag(&mut g, Primitive::TagInt, raw);
        
        let add = g.add(Node::Prim(Primitive::Add));
        let term = g.add(Node::App {
            func: add,
            args: smallvec::smallvec![tagged, tagged]
        });
        
        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        let (payload, tag) = unwrap_data(&g, res);
        
        match tag {
             Some(Primitive::TagInt) => {},
             _ => panic!("Expected TagInt for integer arithmetic"),
        }
        
        let res_bi = decode_int(&mut g, payload).expect("Failed to decode result int");
        let expected = big_n.clone() + big_n;
        assert_eq!(res_bi, expected);
    }
    
    #[test]
    fn test_unparse_large_int() {
        let mut g = Graph::new();
        use std::str::FromStr;
        let big_n_str = "6591346719847561024756918028745614725610934275610384756103847561038475610384756103847561038476510384756103847561038476510384756013847561038475610384756103847561038475610384756103847561038475610384756";
        let big_n = BigInt::from_str(big_n_str).unwrap();
        
        let raw = encode_int(&mut g, &big_n);
        let tagged = make_tag(&mut g, Primitive::TagInt, raw);
        
        // Should decode purely
        let decoded = super::decode_int_pure(&g, raw);
        assert!(decoded.is_some(), "Pure decode failed");
        assert_eq!(decoded.unwrap(), big_n);
        
        let s = unparse(&g, tagged);
        if s == "Int(?)" {
             panic!("Unparse returned Int(?)");
        }
        assert_eq!(s, big_n_str);
    }
    
    #[test]
    fn test_tagged_structure() {
        let mut g = Graph::new();
        // Construct "Hi"
        let s_node = encode_str(&mut g, "Hi");
        let tagged = make_tag(&mut g, Primitive::TagStr, s_node);
        
        // Tagged value = Fork(TagStr, Fork(Payload, KK))
        // Verify we can access TagStr using 'first' logic: \p. n p n n
        
        // Verify that we can structurally access the tag using `unwrap_data`
        // Functional access via `\p. n p n n` fails because Fork reduces as S-combinator,
        // not as a Church Pair.
        
        let (payload, tag) = unwrap_data(&g, tagged);
        assert_eq!(tag, Some(Primitive::TagStr), "Expected TagStr");
        assert_eq!(payload, s_node, "Expected payload to match original string node");
    }
}
