use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::quantum::LinearCombination;
use smallvec::smallvec;

/// Helper to get Identity I = Node::Leaf
pub fn get_identity(g: &mut Graph) -> NodeId {
    g.add(Node::Leaf)
}

pub fn diff(g: &mut Graph, program: NodeId, variable: NodeId) -> LinearCombination {
    use std::collections::HashMap;

    enum Frame {
        Enter(NodeId),
        Exit(NodeId),
    }

    let mut memo: HashMap<NodeId, LinearCombination> = HashMap::new();
    let mut stack = vec![Frame::Enter(program)];

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(curr) => {
                if memo.contains_key(&curr) {
                    continue;
                }
                if curr == variable {
                    let i = get_identity(g);
                    memo.insert(curr, LinearCombination::from_node(i));
                    continue;
                }
                match g.get(curr).clone() {
                    Node::Fork(u, v) => {
                        stack.push(Frame::Exit(curr));
                        stack.push(Frame::Enter(v));
                        stack.push(Frame::Enter(u));
                    }
                    Node::Stem(u) => {
                        stack.push(Frame::Exit(curr));
                        stack.push(Frame::Enter(u));
                    }
                    Node::App { func, args } => {
                        if let Node::Prim(p) = g.get(func) {
                            match p {
                                Primitive::Eq | Primitive::Gt | Primitive::Lt | Primitive::If => {
                                    memo.insert(curr, LinearCombination::zero());
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        stack.push(Frame::Exit(curr));
                        for &arg in args.iter().rev() {
                            stack.push(Frame::Enter(arg));
                        }
                        stack.push(Frame::Enter(func));
                    }
                    _ => {
                        memo.insert(curr, LinearCombination::zero());
                    }
                }
            }
            Frame::Exit(curr) => {
                if memo.contains_key(&curr) {
                    continue;
                }
                let lc = match g.get(curr).clone() {
                    Node::Fork(u, v) => {
                        let du = memo.get(&u).cloned().unwrap_or_else(LinearCombination::zero);
                        let dv = memo.get(&v).cloned().unwrap_or_else(LinearCombination::zero);
                        du.add(dv)
                    }
                    Node::Stem(u) => {
                        let du = memo.get(&u).cloned().unwrap_or_else(LinearCombination::zero);
                        let leaf = g.add(Node::Leaf);
                        apply_func_to_lc(g, leaf, &du)
                    }
                    Node::App { func, args } => {
                        let mut current_lc =
                            memo.get(&func).cloned().unwrap_or_else(LinearCombination::zero);
                        let mut p_curr = func;
                        for &arg in &args {
                            let da =
                                memo.get(&arg).cloned().unwrap_or_else(LinearCombination::zero);
                            let term1 = apply_lc_to_arg(g, &current_lc, arg);
                            let term2 = apply_func_to_lc(g, p_curr, &da);
                            current_lc = term1.add(term2);
                            p_curr = g.add(Node::App { func: p_curr, args: smallvec![arg] });
                        }
                        current_lc
                    }
                    _ => LinearCombination::zero(),
                };
                memo.insert(curr, lc);
            }
        }
    }

    memo.remove(&program).unwrap_or_else(LinearCombination::zero)
}

fn apply_lc_to_arg(g: &mut Graph, lc: &LinearCombination, arg: NodeId) -> LinearCombination {
    let mut result = LinearCombination::zero();
    for (term_id, coeff) in &lc.terms {
        let new_node = g.add(Node::App { 
            func: *term_id, 
            args: smallvec![arg] 
        });
        let mut part = LinearCombination::zero();
        part.terms.insert(new_node, *coeff);
        result = result.add(part);
    }
    result
}

fn apply_func_to_lc(g: &mut Graph, func: NodeId, lc: &LinearCombination) -> LinearCombination {
    let mut result = LinearCombination::zero();
    for (term_id, coeff) in &lc.terms {
        let new_node = g.add(Node::App { 
            func: func, 
            args: smallvec![*term_id] 
        });
        let mut part = LinearCombination::zero();
        part.terms.insert(new_node, *coeff);
        result = result.add(part);
    }
    result
}

pub fn lc_to_term(g: &mut Graph, lc: &LinearCombination) -> NodeId {
    // Sort terms for determinism
    let mut entries: Vec<_> = lc.terms.iter().collect();
    // Sort by ID
    entries.sort_by_key(|(id, _)| id.0);
    
    let mut list = g.add(Node::Leaf); // Nil
    
    for (term_id, coeff) in entries.into_iter().rev() {
        let c_node = g.add(Node::Float(*coeff));
        let pair = g.add(Node::Fork(c_node, *term_id)); // (coeff . term)
        list = g.add(Node::Fork(pair, list)); // (pair . rest)
    }
    
    list
}
