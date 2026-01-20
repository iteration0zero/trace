use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::quantum::LinearCombination;
use smallvec::smallvec;

/// Helper to get Identity I = Node::Leaf
pub fn get_identity(g: &mut Graph) -> NodeId {
    g.add(Node::Leaf)
}

pub fn diff(g: &mut Graph, program: NodeId, variable: NodeId) -> LinearCombination {
    if program == variable {
        let i = get_identity(g);
        return LinearCombination::from_node(i);
    }

    let node = g.get(program).clone();
    
    match node {
        Node::Fork(u, v) => {
            let du = diff(g, u, variable);
            let dv = diff(g, v, variable);
            du.add(dv)
        }
        
        Node::Stem(u) => {
            // D(4 u) = 4 (Du)
            let du = diff(g, u, variable);
            let leaf = g.add(Node::Leaf);
            apply_func_to_lc(g, leaf, &du)
        }
        
        Node::App { func, args } => {
            // Check if func is a non-differentiable primitive (Comparator)
            if let Node::Prim(p) = g.get(func) {
                match p {
                    Primitive::Eq | Primitive::Gt | Primitive::Lt | Primitive::If => return LinearCombination::zero(),
                    _ => {}
                }
            }

            let mut current_lc = diff(g, func, variable);
            let mut p_curr = func;
            
            for &arg in &args {
                let da = diff(g, arg, variable);
                // term1 = (D P_prev) arg
                let term1 = apply_lc_to_arg(g, &current_lc, arg);
                // term2 = P_prev (D arg)
                let term2 = apply_func_to_lc(g, p_curr, &da);
                
                current_lc = term1.add(term2);
                
                // Update P_curr => App(P_curr, arg)
                p_curr = g.add(Node::App { func: p_curr, args: smallvec![arg] });
            }
            current_lc
        }
        
        _ => LinearCombination::zero(),
    }
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
