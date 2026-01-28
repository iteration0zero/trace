use crate::arena::{Graph, Node, NodeId};
use std::collections::{HashMap, HashSet};
use smallvec::SmallVec;

pub const UNPARSE_MAX_NODES: usize = 1000;
pub const UNPARSE_MAX_DEPTH: usize = 20;
pub const UNPARSE_MAX_ARGS: usize = 10;

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn variance(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let mu = mean(xs);
    xs.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / xs.len() as f64
}

pub fn resolve_safe(g: &Graph, mut id: NodeId) -> NodeId {
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(id) {
            return id;
        }
        match g.get(id) {
            Node::Ind(inner) => id = *inner,
            _ => return id,
        }
    }
}

pub fn collect_paths_map(g: &Graph, root: NodeId) -> HashMap<NodeId, Vec<Vec<u8>>> {
    let mut map: HashMap<NodeId, Vec<Vec<u8>>> = HashMap::new();
    let mut stack: Vec<(NodeId, Vec<u8>)> = Vec::new();
    stack.push((root, Vec::new()));

    while let Some((id, path)) = stack.pop() {
        let resolved = resolve_safe(g, id);
        map.entry(resolved).or_default().push(path.clone());
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

pub fn node_at_path(g: &Graph, root: NodeId, path: &[u8]) -> Option<NodeId> {
    let mut curr = g.resolve(root);
    for &dir in path {
        match g.get(curr) {
            Node::Stem(inner) => {
                if dir != 0 {
                    return None;
                }
                curr = g.resolve(*inner);
            }
            Node::Fork(l, r) => {
                curr = if dir == 0 { g.resolve(*l) } else { g.resolve(*r) };
            }
            Node::App { func, args } => {
                if dir == 0 {
                    curr = g.resolve(*func);
                } else {
                    let idx = (dir - 1) as usize;
                    if idx >= args.len() {
                        return None;
                    }
                    curr = g.resolve(args[idx]);
                }
            }
            Node::Ind(inner) => {
                curr = g.resolve(*inner);
            }
            _ => return None,
        }
    }
    Some(curr)
}

pub fn replace_at_path(g: &mut Graph, root: NodeId, path: &[u8], replacement: NodeId) -> NodeId {
    if path.is_empty() {
        return replacement;
    }

    #[derive(Clone)]
    enum Trail {
        Stem,
        ForkLeft { right: NodeId },
        ForkRight { left: NodeId },
        AppFunc { args: SmallVec<[NodeId; 2]> },
        AppArg { func: NodeId, args: SmallVec<[NodeId; 2]>, idx: usize },
    }

    let mut curr = g.resolve(root);
    let mut idx = 0usize;
    let mut trail: Vec<Trail> = Vec::new();

    loop {
        if idx >= path.len() {
            break;
        }
        match g.get(curr).clone() {
            Node::Stem(inner) => {
                if path[idx] != 0 {
                    return curr;
                }
                trail.push(Trail::Stem);
                curr = g.resolve(inner);
                idx += 1;
            }
            Node::Fork(l, r) => {
                if path[idx] == 0 {
                    trail.push(Trail::ForkLeft { right: r });
                    curr = g.resolve(l);
                    idx += 1;
                } else {
                    trail.push(Trail::ForkRight { left: l });
                    curr = g.resolve(r);
                    idx += 1;
                }
            }
            Node::App { func, args } => {
                if path[idx] == 0 {
                    trail.push(Trail::AppFunc { args });
                    curr = g.resolve(func);
                    idx += 1;
                } else {
                    let arg_idx = (path[idx] - 1) as usize;
                    if arg_idx >= args.len() {
                        return curr;
                    }
                    let arg_node = args[arg_idx];
                    trail.push(Trail::AppArg { func, args, idx: arg_idx });
                    curr = g.resolve(arg_node);
                    idx += 1;
                }
            }
            Node::Ind(inner) => {
                curr = g.resolve(inner);
            }
            _ => return curr,
        }
    }

    let mut built = replacement;
    while let Some(frame) = trail.pop() {
        built = match frame {
            Trail::Stem => g.add(Node::Stem(built)),
            Trail::ForkLeft { right } => g.add(Node::Fork(built, right)),
            Trail::ForkRight { left } => g.add(Node::Fork(left, built)),
            Trail::AppFunc { args } => g.add(Node::App { func: built, args }),
            Trail::AppArg { func, mut args, idx } => {
                if idx < args.len() {
                    args[idx] = built;
                }
                g.add(Node::App { func, args })
            }
        };
    }

    built
}

pub fn clone_subtree(
    g_src: &Graph,
    g_dst: &mut Graph,
    id: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
    eval_to_orig: Option<&mut HashMap<NodeId, NodeId>>,
) -> NodeId {
    let resolved = g_src.resolve(id);
    if let Some(&cached) = memo.get(&resolved) {
        return cached;
    }

    let mut eval_to_orig = eval_to_orig;
    let mut stack: Vec<(NodeId, bool)> = Vec::new();
    stack.push((resolved, false));

    while let Some((node, expanded)) = stack.pop() {
        let node = g_src.resolve(node);
        if memo.contains_key(&node) {
            continue;
        }
        if expanded {
            let new_id = match g_src.get(node) {
                Node::Leaf => g_dst.add(Node::Leaf),
                Node::Prim(p) => g_dst.add(Node::Prim(*p)),
                Node::Float(f) => g_dst.add(Node::Float(*f)),
                Node::Handle(h) => g_dst.add(Node::Handle(*h)),
                Node::Ind(inner) => {
                    let inner = g_src.resolve(*inner);
                    memo.get(&inner).copied().unwrap_or_else(|| g_dst.add(Node::Leaf))
                }
                Node::Stem(inner) => {
                    let inner = g_src.resolve(*inner);
                    let c = memo.get(&inner).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    g_dst.add(Node::Stem(c))
                }
                Node::Fork(l, r) => {
                    let l = g_src.resolve(*l);
                    let r = g_src.resolve(*r);
                    let nl = memo.get(&l).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    let nr = memo.get(&r).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    g_dst.add(Node::Fork(nl, nr))
                }
                Node::App { func, args } => {
                    let func = g_src.resolve(*func);
                    let nf = memo.get(&func).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                    let mut new_args: SmallVec<[NodeId; 2]> = SmallVec::new();
                    for arg in args {
                        let arg = g_src.resolve(*arg);
                        let na = memo.get(&arg).copied().unwrap_or_else(|| g_dst.add(Node::Leaf));
                        new_args.push(na);
                    }
                    g_dst.add(Node::App { func: nf, args: new_args })
                }
            };
            memo.insert(node, new_id);
            if let Some(map) = eval_to_orig.as_deref_mut() {
                map.insert(new_id, node);
            }
        } else {
            stack.push((node, true));
            match g_src.get(node) {
                Node::Stem(inner) => stack.push((g_src.resolve(*inner), false)),
                Node::Fork(l, r) => {
                    stack.push((g_src.resolve(*r), false));
                    stack.push((g_src.resolve(*l), false));
                }
                Node::App { func, args } => {
                    for arg in args.iter().rev() {
                        stack.push((g_src.resolve(*arg), false));
                    }
                    stack.push((g_src.resolve(*func), false));
                }
                Node::Ind(inner) => stack.push((g_src.resolve(*inner), false)),
                _ => {}
            }
        }
    }

    memo.get(&resolved).copied().unwrap_or_else(|| g_dst.add(Node::Leaf))
}

pub fn unparse_limited(g: &Graph, id: NodeId) -> String {
    let mut nodes_left = UNPARSE_MAX_NODES;
    let mut visiting: HashSet<NodeId> = HashSet::new();
    unparse_limited_rec(g, id, UNPARSE_MAX_DEPTH, &mut nodes_left, &mut visiting)
}

fn unparse_limited_rec(
    g: &Graph,
    id: NodeId,
    depth: usize,
    nodes_left: &mut usize,
    visiting: &mut HashSet<NodeId>,
) -> String {
    enum Frame<'a> {
        Enter(NodeId, usize),
        Exit(NodeId),
        Text(&'a str),
        Owned(String),
    }

    let mut out = String::new();
    let mut stack: Vec<Frame<'_>> = Vec::new();
    stack.push(Frame::Enter(id, depth));

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Text(s) => out.push_str(s),
            Frame::Owned(s) => out.push_str(&s),
            Frame::Exit(id) => {
                visiting.remove(&id);
            }
            Frame::Enter(curr, curr_depth) => {
                if *nodes_left == 0 || curr_depth == 0 {
                    out.push_str("...");
                    continue;
                }
                let resolved = g.resolve(curr);
                if !visiting.insert(resolved) {
                    out.push_str("<cycle>");
                    continue;
                }
                *nodes_left = nodes_left.saturating_sub(1);
                stack.push(Frame::Exit(resolved));
                match g.get(resolved) {
                    Node::Leaf => out.push_str("n"),
                    Node::Stem(inner) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(*inner, curr_depth - 1));
                        stack.push(Frame::Text(" "));
                        stack.push(Frame::Text("n"));
                        stack.push(Frame::Text("("));
                    }
                    Node::Fork(l, r) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(*r, curr_depth - 1));
                        stack.push(Frame::Text(" "));
                        stack.push(Frame::Enter(*l, curr_depth - 1));
                        stack.push(Frame::Text(" "));
                        stack.push(Frame::Text("n"));
                        stack.push(Frame::Text("("));
                    }
                    Node::App { func, args } => {
                        let limit = UNPARSE_MAX_ARGS.min(args.len());
                        stack.push(Frame::Text(")"));
                        if args.len() > UNPARSE_MAX_ARGS {
                            stack.push(Frame::Text(" ..."));
                        }
                        for arg in args.iter().take(limit).rev() {
                            stack.push(Frame::Enter(*arg, curr_depth - 1));
                            stack.push(Frame::Text(" "));
                        }
                        stack.push(Frame::Enter(*func, curr_depth - 1));
                        stack.push(Frame::Text("("));
                    }
                    Node::Prim(p) => out.push_str(&format!("{:?}", p)),
                    Node::Float(f) => out.push_str(&format!("{}", f)),
                    Node::Handle(h) => out.push_str(&format!("<Handle {}>", h)),
                    Node::Ind(inner) => stack.push(Frame::Enter(*inner, curr_depth - 1)),
                }
            }
        }
    }

    out
}
