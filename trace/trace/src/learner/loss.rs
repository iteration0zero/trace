//! Tree Edit Distance (TED) for Triage Calculus Trees
//! 
//! Implements Zhang-Shasha algorithm for computing the minimum edit distance
//! between two binary trees (values in Triage Calculus).
//! 
//! Used as a structural loss function for synthesis.

use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::engine::tree_hash;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

/// Cost constants for tree edits
const COST_DELETE: usize = 1;
const COST_INSERT: usize = 1;
const COST_RELABEL: usize = 1;

const TED_CACHE_MAX: usize = 200_000;

// (flatten_tree and Zhang-Shasha implementation removed; TED now uses recursive DP with memoization.)

fn label_for_node(node: &Node) -> u64 {
    match node {
        Node::Leaf => 0,
        Node::Stem(_) => 1,
        Node::Fork(_, _) => 2,
        Node::Float(f) => 0x10_0000_0000_0000u64 | f.to_bits(),
        Node::Prim(p) => 0x20_0000_0000_0000u64 | prim_label(*p),
        Node::Handle(h) => 0x30_0000_0000_0000u64 | (*h as u64),
        Node::App { .. } => 0x40_0000_0000_0000u64,
        Node::Ind(_) => 0x50_0000_0000_0000u64,
    }
}

fn label_for_node_app() -> u64 {
    0x40_0000_0000_0000u64
}

fn prim_label(p: Primitive) -> u64 {
    match p {
        Primitive::Add => 1,
        Primitive::Sub => 2,
        Primitive::Mul => 3,
        Primitive::Div => 4,
        Primitive::Eq => 5,
        Primitive::Gt => 6,
        Primitive::Lt => 7,
        Primitive::If => 8,
        Primitive::S => 9,
        Primitive::K => 10,
        Primitive::I => 11,
        Primitive::First => 12,
        Primitive::Rest => 13,
        Primitive::Trace => 14,
        Primitive::TagInt => 20,
        Primitive::TagFloat => 21,
        Primitive::TagStr => 22,
        Primitive::TagChar => 23,
        Primitive::TypeOf => 30,
        Primitive::Any => 31,
        Primitive::Match => 32,
        Primitive::Mod => 33,
    }
}

/// Computes Tree Edit Distance between two trees (recursive with memoization).
///
/// Uses ordered-tree edit distance with subtree memoization.
/// Costs are per-node insert/delete/relabeled and match Zhang-Shasha for ordered trees.
pub fn tree_edit_distance(g: &Graph, t1: NodeId, t2: NodeId) -> usize {
    match tree_edit_distance_capped(g, t1, t2, 0) {
        Ok(v) => v,
        Err((n, m)) => {
            // No cap requested; fall back to worst-case linear penalties.
            n.saturating_add(m)
        }
    }
}

/// Computes Tree Edit Distance with a cell cap (n*m) to avoid OOM.
pub fn tree_edit_distance_capped(
    g: &Graph,
    t1: NodeId,
    t2: NodeId,
    max_cells: usize,
) -> Result<usize, (usize, usize)> {
    let mut sizes: HashMap<NodeId, usize> = HashMap::new();
    let t1 = g.resolve(t1);
    let t2 = g.resolve(t2);
    let n = subtree_size(g, t1, &mut sizes);
    let m = subtree_size(g, t2, &mut sizes);

    if max_cells > 0 && (n as u128) * (m as u128) > max_cells as u128 {
        return Err((n, m));
    }

    let mut a = tree_hash(g, t1);
    let mut b = tree_hash(g, t2);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    let key = TedKey { a, b };
    if let Some(v) = ted_cache_get(&key) {
        return Ok(v);
    }

    let mut memo: HashMap<(NodeId, NodeId), usize> = HashMap::new();
    let dist = ted_distance(g, t1, t2, &mut sizes, &mut memo);
    ted_cache_insert(key, dist);
    Ok(dist)
}

fn ted_distance(
    g: &Graph,
    t1: NodeId,
    t2: NodeId,
    sizes: &mut HashMap<NodeId, usize>,
    memo: &mut HashMap<(NodeId, NodeId), usize>,
) -> usize {
    let t1 = g.resolve(t1);
    let t2 = g.resolve(t2);
    if let Some(v) = memo.get(&(t1, t2)) {
        return *v;
    }

    let mut stack: Vec<(NodeId, NodeId, bool)> = Vec::new();
    stack.push((t1, t2, false));

    while let Some((a, b, expanded)) = stack.pop() {
        let a = g.resolve(a);
        let b = g.resolve(b);
        if memo.contains_key(&(a, b)) {
            continue;
        }
        if expanded {
            let label1 = label_for_node(g.get(a));
            let label2 = label_for_node(g.get(b));
            let relabel = if label1 == label2 { 0 } else { COST_RELABEL };
            let children1 = node_children(g, a);
            let children2 = node_children(g, b);
            let forest = forest_distance(g, &children1, &children2, sizes, memo);
            memo.insert((a, b), relabel + forest);
            continue;
        }

        stack.push((a, b, true));
        let children1 = node_children(g, a);
        let children2 = node_children(g, b);
        for ca in &children1 {
            for cb in &children2 {
                let ca = g.resolve(*ca);
                let cb = g.resolve(*cb);
                if !memo.contains_key(&(ca, cb)) {
                    stack.push((ca, cb, false));
                }
            }
        }
    }

    *memo.get(&(t1, t2)).unwrap_or(&0)
}

fn forest_distance(
    g: &Graph,
    a: &[NodeId],
    b: &[NodeId],
    sizes: &mut HashMap<NodeId, usize>,
    memo: &mut HashMap<(NodeId, NodeId), usize>,
) -> usize {
    let n = a.len();
    let m = b.len();
    if n == 0 && m == 0 {
        return 0;
    }
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        dp[i][0] = dp[i - 1][0] + subtree_size(g, a[i - 1], sizes) * COST_DELETE;
    }
    for j in 1..=m {
        dp[0][j] = dp[0][j - 1] + subtree_size(g, b[j - 1], sizes) * COST_INSERT;
    }
    for i in 1..=n {
        for j in 1..=m {
            let del = dp[i - 1][j] + subtree_size(g, a[i - 1], sizes) * COST_DELETE;
            let ins = dp[i][j - 1] + subtree_size(g, b[j - 1], sizes) * COST_INSERT;
            let key = (g.resolve(a[i - 1]), g.resolve(b[j - 1]));
            let sub_cost = match memo.get(&key) {
                Some(v) => *v,
                None => {
                    // Should be memoized by ted_distance's worklist; fall back to size sum.
                    subtree_size(g, a[i - 1], sizes) + subtree_size(g, b[j - 1], sizes)
                }
            };
            let sub = dp[i - 1][j - 1] + sub_cost;
            dp[i][j] = min(del, min(ins, sub));
        }
    }
    dp[n][m]
}

fn subtree_size(g: &Graph, root: NodeId, memo: &mut HashMap<NodeId, usize>) -> usize {
    let root = g.resolve(root);
    if let Some(v) = memo.get(&root) {
        return *v;
    }

    let mut stack: Vec<(NodeId, bool)> = Vec::new();
    let mut visiting: HashSet<NodeId> = HashSet::new();
    stack.push((root, false));

    while let Some((id, done)) = stack.pop() {
        let id = g.resolve(id);
        if memo.contains_key(&id) {
            continue;
        }
        if done {
            let size = match g.get(id) {
                Node::Leaf | Node::Prim(_) | Node::Float(_) | Node::Handle(_) => 1,
                Node::Stem(child) => 1 + memo.get(&g.resolve(*child)).copied().unwrap_or(1),
                Node::Fork(left, right) => {
                    1 + memo.get(&g.resolve(*left)).copied().unwrap_or(1)
                        + memo.get(&g.resolve(*right)).copied().unwrap_or(1)
                }
                Node::App { func, args } => {
                    let mut size = 1 + memo.get(&g.resolve(*func)).copied().unwrap_or(1);
                    for arg in args {
                        size = size.saturating_add(memo.get(&g.resolve(*arg)).copied().unwrap_or(1));
                    }
                    size
                }
                Node::Ind(inner) => memo.get(&g.resolve(*inner)).copied().unwrap_or(1),
            };
            memo.insert(id, size);
            visiting.remove(&id);
            continue;
        }

        if !visiting.insert(id) {
            memo.insert(id, 1);
            continue;
        }

        stack.push((id, true));
        match g.get(id) {
            Node::Stem(child) => stack.push((*child, false)),
            Node::Fork(left, right) => {
                stack.push((*right, false));
                stack.push((*left, false));
            }
            Node::App { func, args } => {
                for arg in args.iter().rev() {
                    stack.push((*arg, false));
                }
                stack.push((*func, false));
            }
            Node::Ind(inner) => stack.push((*inner, false)),
            _ => {}
        }
    }

    memo.get(&root).copied().unwrap_or(1)
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TedKey {
    a: u64,
    b: u64,
}

fn ted_cache() -> &'static Mutex<HashMap<TedKey, usize>> {
    static CACHE: OnceLock<Mutex<HashMap<TedKey, usize>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ted_cache_get(key: &TedKey) -> Option<usize> {
    let cache = ted_cache();
    let guard = cache.lock().unwrap();
    guard.get(key).copied()
}

fn ted_cache_insert(key: TedKey, value: usize) {
    let cache = ted_cache();
    let mut guard = cache.lock().unwrap();
    if guard.len() >= TED_CACHE_MAX {
        guard.clear();
    }
    guard.insert(key, value);
}

fn node_children(g: &Graph, root: NodeId) -> Vec<NodeId> {
    let mut id = g.resolve(root);
    let mut seen: HashSet<NodeId> = HashSet::new();
    loop {
        if !seen.insert(id) {
            return Vec::new();
        }
        match g.get(id) {
            Node::Ind(inner) => id = g.resolve(*inner),
            Node::Stem(child) => return vec![*child],
            Node::Fork(left, right) => return vec![*left, *right],
            Node::App { func, args } => {
                let mut out = Vec::with_capacity(1 + args.len());
                out.push(*func);
                out.extend(args.iter().copied());
                return out;
            }
            _ => return Vec::new(),
        }
    }
}

/// Compute a normalized similarity score in [0, 1]
/// 1.0 = identical trees, 0.0 = maximally different
pub fn tree_similarity(g: &Graph, t1: NodeId, t2: NodeId) -> f64 {
    let dist = tree_edit_distance(g, t1, t2);
    let mut sizes = HashMap::new();
    let max_size = subtree_size(g, t1, &mut sizes).max(subtree_size(g, t2, &mut sizes));
    
    if max_size == 0 {
        return 1.0;
    }
    
    1.0 - (dist as f64 / max_size as f64).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_identical_trees() {
        let mut g = Graph::new();
        let leaf = g.add(Node::Leaf);
        assert_eq!(tree_edit_distance(&g, leaf, leaf), 0);
    }
    
    #[test]
    fn test_leaf_vs_stem() {
        let mut g = Graph::new();
        let leaf = g.add(Node::Leaf);
        let stem = g.add(Node::Stem(leaf));
        // stem has 2 nodes (leaf + stem), leaf has 1
        // Distance: insert 1 node + relabel or delete 1
        let dist = tree_edit_distance(&g, leaf, stem);
        assert!(dist > 0);
    }
    
    #[test]
    fn test_fork_symmetry() {
        let mut g = Graph::new();
        let l1 = g.add(Node::Leaf);
        let l2 = g.add(Node::Leaf);
        let fork1 = g.add(Node::Fork(l1, l2));
        
        let l3 = g.add(Node::Leaf);
        let l4 = g.add(Node::Leaf);
        let fork2 = g.add(Node::Fork(l3, l4));
        
        assert_eq!(tree_edit_distance(&g, fork1, fork2), 0);
    }
}
