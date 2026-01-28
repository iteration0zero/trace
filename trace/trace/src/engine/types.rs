use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::trace::{ExecutionTrace, TraceEvent, Trace};
use smallvec::SmallVec;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool};

pub static REDUCE_PROGRESS_MS_OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);
pub static REDUCE_DEBUG_LEVEL_OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);
pub static DEBUG_OOM: AtomicBool = AtomicBool::new(false);

// EvalContext cannot verify Clone because it contains mutable references
// #[derive(Clone)]

pub struct EvalContext<'a> {
    pub step_limit: usize,
    pub node_limit: usize,
    pub depth: usize,
    pub depth_limit: usize,
    pub base_nodes: usize,
    pub steps: usize,
    pub node_limit_hit: bool,
    pub step_limit_hit: bool,
    pub exec_trace: Option<&'a mut ExecutionTrace>,
    pub redex_cache: Option<&'a dyn RedexMemo>,
    pub trace: Option<&'a mut Trace>, // Deprecated? Some code uses .trace, others .exec_trace.
}

impl<'a> Default for EvalContext<'a> {
    fn default() -> Self {
        Self {
            step_limit: usize::MAX,
            node_limit: 0,
            depth: 0,
            depth_limit: 1000,
            base_nodes: 0,
            steps: 0,
            node_limit_hit: false,
            step_limit_hit: false,
            exec_trace: None,
            redex_cache: None,
            trace: None,
        }
    }
}

pub trait RedexMemo {
    fn get_redex(&self, key: &RedexKey) -> Option<Arc<CachedRedex>>;
    fn insert_redex(&self, key: RedexKey, value: Arc<CachedRedex>, size: usize, nodes: usize);
    fn max_redex_nodes(&self) -> usize;
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
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

#[derive(Clone)]
pub enum WhnfCont {
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

#[derive(Clone)]
pub struct ApplyState {
    pub redex: NodeId,
    pub head: NodeId,
    pub pending: SmallVec<[NodeId; 2]>,
    pub changed: bool,
    pub skip_tag: bool,
}

pub fn mk_seq(g: &mut Graph, head: NodeId, args: SmallVec<[NodeId; 2]>) -> NodeId {
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

pub fn collect_top_spine(g: &mut Graph, root: NodeId) -> (NodeId, SmallVec<[NodeId; 2]>) {
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
