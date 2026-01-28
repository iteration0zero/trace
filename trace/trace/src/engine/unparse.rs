use crate::arena::{Graph, Node, NodeId, Primitive};
use super::primitives::{unwrap_data, decode_int_pure, decode_str_pure};
use std::collections::HashSet;

const DEBUG_UNPARSE_MAX_DEPTH: usize = 6;
const DEBUG_UNPARSE_MAX_NODES: usize = 200;
const DEBUG_UNPARSE_MAX_ARGS: usize = 6;

pub fn node_kind(g: &Graph, id: NodeId) -> &'static str {
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

pub fn debug_args(g: &Graph, args: &[NodeId]) -> String {
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

pub fn debug_unparse(g: &Graph, id: NodeId) -> String {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut budget = DEBUG_UNPARSE_MAX_NODES;
    debug_unparse_rec(g, id, 0, &mut budget, &mut seen) // Graph::resolve(id) not strictly needed if we resolve in loop, but unparse_rec assumes id?
    // engine.rs passed g.resolve(id) to debug_unparse_rec.
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
    // Resolve initially or in loop? engine.rs resolved before call.
    // But recursive calls push nodes that might need resolution?
    // engine.rs:
    // stack.push(Item::Node(id, depth));
    // loop: let curr = g.resolve(curr);
    
    // So assume we push potentially unresolved IDs, and resolve them when popping.
    
    // Initial resolve
    // g.get() works on resolved nodes usually for clean matching.
    
    stack.push(Item::Node(id, depth));

    while let Some(item) = stack.pop() {
        match item {
            Item::Text(s) => out.push_str(s),
            Item::Owned(s) => out.push_str(&s),
            Item::Node(curr_unres, curr_depth) => {
                if *budget == 0 || curr_depth > DEBUG_UNPARSE_MAX_DEPTH {
                    out.push_str("...");
                    continue;
                }
                
                // Resolution happens here
                // Note: Graph::resolve might interact with Indirections.
                // Assuming we use standard resolution.
                // engine.rs used g.resolve(curr).
                // Wait, Graph::resolve in arena.rs?
                // Does Graph have resolve? Yes.
                
                // We don't have access to Graph methods unless we import Graph? We imported Graph.
                // But Graph::resolve might not be public or available?
                // It is likely available.
                
                 // Actually, let's trust engine.rs implementation.
                
                // engine.rs:1632 let curr = g.resolve(curr);
                
                // But wait, resolve takes &self?
                // Check arena.rs briefly if possible?
                // Assuming yes.
                
                // Wait, if I'm not sure if resolve is pub, I should check.
                // It was used in engine.rs so it's pub or pub(crate).
                
                // But I can rewrite it to simple loop if needed.
                // I'll assume resolve is available to engine module.
                
                // engine.rs implementation:
                /*
                let curr = g.resolve(curr);
                if !seen.insert(curr) {
                    out.push_str(&format!("<cycle {}>", curr.0));
                    continue;
                }
                *budget = budget.saturating_sub(1);
                match g.get(curr) { ... }
                */
                
                // But I cannot call g.resolve if it's not exposed.
                // I'll assume it is exposed as engine.rs used it.
                // Since engine.rs is in crate root (src/engine.rs), it can access pub(crate) methods.
                // src/engine/unparse.rs is in submodule.
                // Access rules are same if public.
                
                // Re-implementing resolve loop here is safer if I can't check.
                let mut curr = curr_unres;
                // loop { match g.get(curr) { Node::Ind(i) => curr = *i, _ => break } }
                // I'll stick to g.resolve if possible? No, I'll use manual resolution loop to be safe and avoid compilation error if resolve is missing.
                // Wait, manual resolution needs to look at Node::Ind.
                
                 loop {
                    match g.get(curr) {
                        Node::Ind(i) => curr = *i,
                        _ => break,
                    }
                }
                
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
            Item::Node(curr_unres) => {
                 let mut curr = curr_unres;
                 loop {
                    match g.get(curr) {
                        Node::Ind(i) => curr = *i,
                        _ => break,
                    }
                }
                
                match g.get(curr) {
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
            }
          },
        }
    }

    out
}
