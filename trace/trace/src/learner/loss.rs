//! Tree Edit Distance (TED) for Triage Calculus Trees
//! 
//! Implements Zhang-Shasha algorithm for computing the minimum edit distance
//! between two binary trees (values in Triage Calculus).
//! 
//! Used as the loss function for IGTC synthesis.

use crate::arena::{Graph, Node, NodeId, Primitive};
use std::cmp::min;

/// Cost constants for tree edits
const COST_DELETE: usize = 1;
const COST_INSERT: usize = 1;
const COST_RELABEL: usize = 1;

/// Represents a flattened tree node for the TED algorithm
#[derive(Debug, Clone)]
struct TedNode {
    /// Index of leftmost leaf descendant (1-indexed in postorder)
    leftmost: usize,
    /// Label: distinguishes leaf/stem/fork and primitive/value leaves
    label: u64,
}

/// Flattens a Graph tree into postorder representation for TED
fn flatten_tree(g: &Graph, root: NodeId) -> Vec<TedNode> {
    let mut nodes = Vec::new();
    flatten_recursive(g, root, &mut nodes);
    nodes
}

fn flatten_recursive(g: &Graph, node: NodeId, out: &mut Vec<TedNode>) -> usize {
    let resolved = g.resolve(node);
    match g.get(resolved).clone() {
        Node::Leaf => {
            let idx = out.len();
            out.push(TedNode {
                leftmost: idx + 1, // 1-indexed
                label: label_for_node(&Node::Leaf),
            });
            idx
        }
        Node::Stem(child) => {
            let child_idx = flatten_recursive(g, child, out);
            let idx = out.len();
            let leftmost = out[child_idx].leftmost;
            out.push(TedNode {
                leftmost,
                label: label_for_node(&Node::Stem(child)),
            });
            idx
        }
        Node::Fork(left, right) => {
            let left_idx = flatten_recursive(g, left, out);
            let _right_idx = flatten_recursive(g, right, out);
            let idx = out.len();
            let leftmost = out[left_idx].leftmost;
            out.push(TedNode {
                leftmost,
                label: label_for_node(&Node::Fork(left, right)),
            });
            idx
        }
        Node::Prim(p) => {
            let idx = out.len();
            out.push(TedNode {
                leftmost: idx + 1,
                label: label_for_node(&Node::Prim(p)),
            });
            idx
        }
        Node::Float(f) => {
            let idx = out.len();
            out.push(TedNode {
                leftmost: idx + 1,
                label: label_for_node(&Node::Float(f)),
            });
            idx
        }
        Node::Handle(h) => {
            let idx = out.len();
            out.push(TedNode {
                leftmost: idx + 1,
                label: label_for_node(&Node::Handle(h)),
            });
            idx
        }
        Node::App { .. } => {
            let idx = out.len();
            out.push(TedNode {
                leftmost: idx + 1,
                label: label_for_node_app(),
            });
            idx
        }
        Node::Ind(inner) => flatten_recursive(g, inner, out),
    }
}

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

/// Computes Tree Edit Distance between two trees
/// 
/// Uses Zhang-Shasha algorithm (O(n * m * min(depth_n, leaves_n) * min(depth_m, leaves_m)))
/// 
/// # Arguments
/// * `g` - The graph containing both trees
/// * `t1` - Root of tree 1
/// * `t2` - Root of tree 2
/// 
/// # Returns
/// The minimum edit distance (insertions + deletions + relabels)
pub fn tree_edit_distance(g: &Graph, t1: NodeId, t2: NodeId) -> usize {
    let nodes1 = flatten_tree(g, t1);
    let nodes2 = flatten_tree(g, t2);
    
    if nodes1.is_empty() && nodes2.is_empty() {
        return 0;
    }
    if nodes1.is_empty() {
        return nodes2.len() * COST_INSERT;
    }
    if nodes2.is_empty() {
        return nodes1.len() * COST_DELETE;
    }
    
    zhang_shasha(&nodes1, &nodes2)
}

/// Zhang-Shasha TED algorithm
fn zhang_shasha(t1: &[TedNode], t2: &[TedNode]) -> usize {
    let n = t1.len();
    let m = t2.len();
    
    // Compute keyroots (nodes whose leftmost is unique or they are roots)
    let kr1 = keyroots(t1);
    let kr2 = keyroots(t2);
    
    // Tree distance matrix (1-indexed)
    let mut td = vec![vec![0usize; m + 1]; n + 1];
    
    // Forest distance matrix (reused for each keyroot pair)
    let mut fd = vec![vec![0usize; m + 1]; n + 1];
    
    for &i in &kr1 {
        for &j in &kr2 {
            compute_forest_distance(t1, t2, i, j, &mut td, &mut fd);
        }
    }
    
    td[n][m]
}

/// Compute keyroots: nodes where leftmost(parent) != leftmost(node) or node is root
fn keyroots(nodes: &[TedNode]) -> Vec<usize> {
    let n = nodes.len();
    if n == 0 {
        return vec![];
    }
    
    // Track which leftmost values have been seen
    let mut seen_leftmost = vec![false; n + 2];
    let mut kr = Vec::new();
    
    // Traverse in reverse postorder (root first)
    for i in (0..n).rev() {
        let lm = nodes[i].leftmost;
        if !seen_leftmost[lm] {
            kr.push(i + 1); // 1-indexed
            seen_leftmost[lm] = true;
        }
    }
    
    kr.sort();
    kr
}

/// Compute forest distance for subproblems rooted at keyroots i, j
fn compute_forest_distance(
    t1: &[TedNode],
    t2: &[TedNode],
    i: usize,  // 1-indexed
    j: usize,  // 1-indexed
    td: &mut Vec<Vec<usize>>,
    fd: &mut Vec<Vec<usize>>,
) {
    let l1 = t1[i - 1].leftmost;
    let l2 = t2[j - 1].leftmost;
    
    // Initialize forest distance boundaries
    fd[l1 - 1][l2 - 1] = 0;
    
    for x in l1..=i {
        fd[x][l2 - 1] = fd[x - 1][l2 - 1] + COST_DELETE;
    }
    for y in l2..=j {
        fd[l1 - 1][y] = fd[l1 - 1][y - 1] + COST_INSERT;
    }
    
    for x in l1..=i {
        for y in l2..=j {
            let lx = t1[x - 1].leftmost;
            let ly = t2[y - 1].leftmost;
            
            let cost_del = fd[x - 1][y] + COST_DELETE;
            let cost_ins = fd[x][y - 1] + COST_INSERT;
            
            if lx == l1 && ly == l2 {
                // Both are roots of their respective subtrees
                let relabel_cost = if t1[x - 1].label == t2[y - 1].label { 0 } else { COST_RELABEL };
                let cost_match = fd[x - 1][y - 1] + relabel_cost;
                fd[x][y] = min(min(cost_del, cost_ins), cost_match);
                td[x][y] = fd[x][y];
            } else {
                // Use previously computed tree distance
                let cost_tree = fd[lx - 1][ly - 1] + td[x][y];
                fd[x][y] = min(min(cost_del, cost_ins), cost_tree);
            }
        }
    }
}

/// Compute a normalized similarity score in [0, 1]
/// 1.0 = identical trees, 0.0 = maximally different
pub fn tree_similarity(g: &Graph, t1: NodeId, t2: NodeId) -> f64 {
    let dist = tree_edit_distance(g, t1, t2);
    let nodes1 = flatten_tree(g, t1);
    let nodes2 = flatten_tree(g, t2);
    let max_size = nodes1.len().max(nodes2.len());
    
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
