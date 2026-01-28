//! Arena - Graph Storage for Tree Calculus
use smallvec::SmallVec;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};

static GRAPH_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Lightweight NodeId
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const NULL: NodeId = NodeId(u32::MAX);
}

/// Core Node Enum
#[derive(Clone, Debug)]
pub enum Node {
    Leaf,
    Stem(NodeId),
    Fork(NodeId, NodeId),
    Prim(Primitive),
    Float(f64),
    Ind(NodeId),
    Handle(usize),
    /// TreeSequence: sequence of two or more trees (f x y ...).
    /// The head tree is applied to the next tree to yield a new head.
    App {
        func: NodeId,
        args: SmallVec<[NodeId; 2]>,
    },
}

// Manual Hash/Eq for Float handling
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Leaf, Node::Leaf) => true,
            (Node::Stem(a), Node::Stem(b)) => a == b,
            (Node::Fork(a, b), Node::Fork(c, d)) => a == c && b == d,
            (Node::Prim(a), Node::Prim(b)) => a == b,
            (Node::Float(a), Node::Float(b)) => a.to_bits() == b.to_bits(),
            (Node::Ind(a), Node::Ind(b)) => a == b,
            (Node::Handle(a), Node::Handle(b)) => a == b,
            (Node::App { func: f1, args: a1 }, Node::App { func: f2, args: a2 }) => f1 == f2 && a1 == a2,
            _ => false,
        }
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Node::Leaf => {},
            Node::Stem(id) => id.hash(state),
            Node::Fork(a, b) => { a.hash(state); b.hash(state); },
            Node::Prim(p) => p.hash(state),
            Node::Float(f) => f.to_bits().hash(state),
            Node::Ind(id) => id.hash(state),
            Node::Handle(h) => h.hash(state),
            Node::App { func, args } => { func.hash(state); args.hash(state); },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primitive {
    Add, Sub, Mul, Div,
    Eq, Gt, Lt, If,
    S, K, I,
    First, Rest,
    Trace,
    TagInt, TagFloat, TagStr, TagChar,
    TypeOf, Any, Match, Mod,
}

#[derive(Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    interner: FxHashMap<Node, NodeId>,
    interning: bool,
    pub id: u64,
    pub epoch: u64,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            interner: FxHashMap::default(),
            interning: true,
            id: GRAPH_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
        }
    }

    /// Construct a graph that does not intern nodes (useful for destructive evaluation).
    pub fn new_uninterned() -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            interner: FxHashMap::default(),
            interning: false,
            id: GRAPH_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
        }
    }

    /// Disable interning for this graph (prevents stale interner during rewriting).
    pub fn disable_interning(&mut self) {
        self.interning = false;
        self.interner.clear();
    }

    pub fn add(&mut self, node: Node) -> NodeId {
        if self.interning {
            if let Some(&id) = self.interner.get(&node) {
                return id;
            }
        }
        let id = NodeId(self.nodes.len() as u32);
        // println!("DEBUG: Graph::add {:?} -> ID {}", node, id.0);
        self.nodes.push(node.clone());
        if self.interning {
            self.interner.insert(node, id);
        }
        id
    }

    /// Insert a node without canonicalization or interning (used for trace snapshots).
    pub fn add_raw(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn resolve(&self, mut id: NodeId) -> NodeId {
        loop {
            if let Node::Ind(next) = &self.nodes[id.0 as usize] {
                id = *next;
            } else {
                return id;
            }
        }
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interning() {
        let mut g = Graph::new();
        let n1 = g.add(Node::Leaf);
        let n2 = g.add(Node::Leaf);
        assert_eq!(n1, n2);

        let s1 = g.add(Node::Stem(n1));
        let s2 = g.add(Node::Stem(n2));
        assert_eq!(s1, s2);
    }


}
