use crate::arena::{Graph, Node, NodeId};
use std::collections::HashMap;

pub fn apply_gradients_struct(g: &mut Graph, prog: NodeId, sensitivity: &crate::sensitivity::SensitivityMap, lr: f64) -> Result<NodeId, String> {
    // 1. Find the "hottest" node in this program (highest gradient magnitude)
    let target = find_hottest_node(g, prog, &sensitivity.gradients);
    
    // 2. Apply mutation to that node
    let mut memo = HashMap::new();
    
    if let Some((target_id, _grad)) = target {
        mutate_rec(g, prog, target_id, &sensitivity.gradients, lr, &mut memo)
    } else {
        Ok(prog)
    }
}

fn find_hottest_node(g: &Graph, root: NodeId, gradients: &HashMap<NodeId, crate::sensitivity::SubGradients>) -> Option<(NodeId, f64)> {
    let mut max_node = None;
    let mut max_grad = -1.0;
    let mut stack = vec![root];
    let mut visited = std::collections::HashSet::new();
    
    while let Some(id) = stack.pop() {
        let resolved = g.resolve(id); 
        if !visited.insert(resolved) { continue; }
        
        let grads = gradients.get(&resolved).cloned().unwrap_or_default();
        let magnitude = grads.numeric.abs().max(grads.expansion).max(grads.pruning);
        
        if magnitude > max_grad {
            max_grad = magnitude;
            max_node = Some(resolved);
        }
        
        match g.get(resolved) {
            Node::Stem(inner) => stack.push(*inner),
            Node::Fork(l, r) => { stack.push(*l); stack.push(*r); }
            Node::App { func, args } => {
                stack.push(*func);
                for arg in args { stack.push(*arg); }
            }
            Node::Ind(inner) => stack.push(*inner),
            _ => {}
        }
    }
    
    if max_grad > 1e-9 {
        max_node.map(|n| (n, max_grad))
    } else {
        None
    }
}

fn mutate_rec(
    g: &mut Graph, 
    id: NodeId, 
    target_id: NodeId,
    gradients: &HashMap<NodeId, crate::sensitivity::SubGradients>, 
    lr: f64,
    memo: &mut HashMap<NodeId, NodeId>
) -> Result<NodeId, String> {
    if let Some(&new_id) = memo.get(&id) {
        return Ok(new_id);
    }
    
    let resolved_id = g.resolve(id);
    if resolved_id == target_id {
        let node = g.get(resolved_id).clone();
        let grads = gradients.get(&resolved_id).cloned().unwrap_or_default();
        
        let new_node_id = match node {
             Node::Float(val) => {
                 g.add(Node::Float(val - lr * grads.numeric))
             },
             Node::Leaf => {
                 if grads.expansion > 0.0 {
                     let l = g.add(Node::Leaf);
                     g.add(Node::Stem(l))
                 } else {
                     id
                 }
             },
             Node::Stem(inner) => {
                 if grads.pruning > grads.expansion {
                     inner 
                 } else if grads.expansion > 0.0 {
                     let l = g.add(Node::Leaf);
                     g.add(Node::Fork(inner, l))
                 } else {
                     id
                 }
             },
             Node::Fork(l, _r) => {
                 if grads.pruning > 0.0 {
                     g.add(Node::Stem(l))
                 } else {
                     id
                 }
             },
             Node::App { func, .. } => {
                 if grads.pruning > 0.0 {
                     func
                 } else {
                     id
                 }
             },
             _ => id
        };
        memo.insert(id, new_node_id);
        return Ok(new_node_id);
    }

    let node = g.get(resolved_id).clone();
    let new_node_id = match node {
        Node::Float(_) | Node::Leaf | Node::Prim(_) | Node::Handle(_) => id,
        
        Node::Stem(inner) => {
            let new_inner = mutate_rec(g, inner, target_id, gradients, lr, memo)?;
            if new_inner == inner { id } else { g.add(Node::Stem(new_inner)) }
        }
        Node::Fork(l, r) => {
            let new_l = mutate_rec(g, l, target_id, gradients, lr, memo)?;
            let new_r = mutate_rec(g, r, target_id, gradients, lr, memo)?;
            if new_l == l && new_r == r { id } else { g.add(Node::Fork(new_l, new_r)) }
        }
        Node::App { func, args } => {
            let new_func = mutate_rec(g, func, target_id, gradients, lr, memo)?;
            let mut new_args = smallvec::SmallVec::new();
            let mut changed = new_func != func;
            for &arg in &args {
                let new_arg = mutate_rec(g, arg, target_id, gradients, lr, memo)?;
                if new_arg != arg { changed = true; }
                new_args.push(new_arg);
            }
            if changed {
                g.add(Node::App { func: new_func, args: new_args })
            } else {
                id
            }
        }
        Node::Ind(inner) => {
             mutate_rec(g, inner, target_id, gradients, lr, memo)?
        }
    };
    
    memo.insert(id, new_node_id);
    Ok(new_node_id)
}
