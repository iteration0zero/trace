use crate::arena::{NodeId, Primitive, Graph, Node};
use smallvec::smallvec;
use std::collections::HashMap;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ProgramTemplate {
    Identity, // (I) or Leaf
    Constant(u64), // Float values mapped to u64 for template matching
    // Structure
    Binary(Primitive, Box<ProgramTemplate>, Box<ProgramTemplate>), 
    Trinary(Primitive, Box<ProgramTemplate>, Box<ProgramTemplate>, Box<ProgramTemplate>),
    
    Apply(Primitive, Box<ProgramTemplate>), // (Prim x)
    
    // Abstract Holes / Subtrees
    App(Box<ProgramTemplate>, Box<ProgramTemplate>),
    
    Leaf, // Leaf literal
    
    // Explicit Tree Constructors (No Primitives needed)
    Fork(Box<ProgramTemplate>, Box<ProgramTemplate>),
    Stem(Box<ProgramTemplate>),
}

impl ProgramTemplate {
    pub fn count_nodes(&self) -> usize {
        match self {
            ProgramTemplate::Identity => 1,
            ProgramTemplate::Constant(_) => 1,
            ProgramTemplate::Leaf => 1,
            ProgramTemplate::Apply(_, inner) => 1 + inner.count_nodes(),
            ProgramTemplate::Binary(_, l, r) => 1 + l.count_nodes() + r.count_nodes(),
            ProgramTemplate::Trinary(_, a, b, c) => 1 + a.count_nodes() + b.count_nodes() + c.count_nodes(),
            ProgramTemplate::App(f, a) => 1 + f.count_nodes() + a.count_nodes(),
            ProgramTemplate::Fork(l, r) => 1 + l.count_nodes() + r.count_nodes(),
            ProgramTemplate::Stem(inner) => 1 + inner.count_nodes(),
        }
    }
    
    pub fn get_sub_templates(&self) -> Vec<&ProgramTemplate> {
        let mut subs = Vec::new();
        match self {
            ProgramTemplate::Apply(_, inner) => {
                subs.push(self);
                subs.extend(inner.get_sub_templates());
            }
            ProgramTemplate::Binary(_, l, r) => {
                subs.push(self);
                subs.extend(l.get_sub_templates());
                subs.extend(r.get_sub_templates());
            }
            ProgramTemplate::Trinary(_, a, b, c) => {
                subs.push(self);
                subs.extend(a.get_sub_templates());
                subs.extend(b.get_sub_templates());
                subs.extend(c.get_sub_templates());
            }
            ProgramTemplate::App(f, a) => {
                subs.push(self);
                subs.extend(f.get_sub_templates());
                subs.extend(a.get_sub_templates());
            }
            ProgramTemplate::Fork(l, r) => {
                subs.push(self);
                subs.extend(l.get_sub_templates());
                subs.extend(r.get_sub_templates());
            }
            ProgramTemplate::Stem(inner) => {
                subs.push(self);
                subs.extend(inner.get_sub_templates());
            }
            _ => subs.push(self),
        }
        subs
    }
    
    pub fn replace_sub_template(&self, target: &ProgramTemplate, replacement: &ProgramTemplate) -> ProgramTemplate {
        if self == target {
            return replacement.clone();
        }
        match self {
            ProgramTemplate::Apply(p, inner) => ProgramTemplate::Apply(*p, Box::new(inner.replace_sub_template(target, replacement))),
            ProgramTemplate::Binary(p, l, r) => ProgramTemplate::Binary(*p, 
                Box::new(l.replace_sub_template(target, replacement)), 
                Box::new(r.replace_sub_template(target, replacement))),
            ProgramTemplate::Trinary(p, a, b, c) => ProgramTemplate::Trinary(*p, 
                Box::new(a.replace_sub_template(target, replacement)),
                Box::new(b.replace_sub_template(target, replacement)),
                Box::new(c.replace_sub_template(target, replacement))),
            ProgramTemplate::App(f, a) => ProgramTemplate::App(
                Box::new(f.replace_sub_template(target, replacement)),
                Box::new(a.replace_sub_template(target, replacement))),
            ProgramTemplate::Fork(l, r) => ProgramTemplate::Fork(
                Box::new(l.replace_sub_template(target, replacement)),
                Box::new(r.replace_sub_template(target, replacement))),
            ProgramTemplate::Stem(inner) => ProgramTemplate::Stem(
                Box::new(inner.replace_sub_template(target, replacement))),
            _ => self.clone(),
        }
    }
    pub fn get_root_key(&self) -> String {
        match self {
            ProgramTemplate::Identity => "Identity".to_string(),
            ProgramTemplate::Constant(_) => "Constant".to_string(),
            ProgramTemplate::Leaf => "Leaf".to_string(),
            ProgramTemplate::Apply(p, _) => format!("{:?}", p),
            ProgramTemplate::Binary(p, _, _) => format!("{:?}", p),
            ProgramTemplate::Trinary(p, _, _, _) => format!("{:?}", p),
            ProgramTemplate::App(_, _) => "App".to_string(),
            ProgramTemplate::Fork(_, _) => "Fork".to_string(),
            ProgramTemplate::Stem(_) => "Stem".to_string(),
        }
    }
}

pub fn build_graph(g: &mut Graph, tmpl: &ProgramTemplate, arg: NodeId) -> NodeId {
    match tmpl {
        ProgramTemplate::Identity => arg,
        ProgramTemplate::Constant(c) => g.add(Node::Float(*c as f64)),
        ProgramTemplate::Apply(prim, inner) => {
            let inner_node = build_graph(g, inner, arg);
            let prim_node = g.add(Node::Prim(*prim));
            g.add(Node::App { func: prim_node, args: smallvec![inner_node] })
        },
        ProgramTemplate::Binary(prim, l, r) => {
            let ln = build_graph(g, l, arg);
            let rn = build_graph(g, r, arg);
            let prim_node = g.add(Node::Prim(*prim));
            let pl = g.add(Node::App { func: prim_node, args: smallvec![ln] });
            g.add(Node::App { func: pl, args: smallvec![rn] })
        },
        ProgramTemplate::Trinary(prim, a, b, c) => {
            let an = build_graph(g, a, arg);
            let bn = build_graph(g, b, arg);
            let cn = build_graph(g, c, arg);
            let prim_node = g.add(Node::Prim(*prim));
            let pa = g.add(Node::App { func: prim_node, args: smallvec![an] });
            let pab = g.add(Node::App { func: pa, args: smallvec![bn] });
            g.add(Node::App { func: pab, args: smallvec![cn] })
        },
        ProgramTemplate::App(func_tmpl, arg_tmpl) => {
             let func_node = build_graph(g, func_tmpl, arg);
             let arg_node = build_graph(g, arg_tmpl, arg);
             g.add(Node::App { func: func_node, args: smallvec![arg_node] })
        },
        ProgramTemplate::Leaf => g.add(Node::Leaf),
        ProgramTemplate::Fork(l, r) => {
            let ln = build_graph(g, l, arg);
            let rn = build_graph(g, r, arg);
            g.add(Node::Fork(ln, rn))
        },
        ProgramTemplate::Stem(inner) => {
            let in_n = build_graph(g, inner, arg);
            g.add(Node::Stem(in_n))
        }
    }
}

pub fn build_graph_with_map(g: &mut Graph, tmpl: &ProgramTemplate, arg: NodeId, map: &mut HashMap<NodeId, ProgramTemplate>) -> NodeId {
    let node_id = match tmpl {
        ProgramTemplate::Identity => arg,
        ProgramTemplate::Constant(c) => g.add(Node::Float(*c as f64)),
        ProgramTemplate::Apply(prim, inner) => {
            let inner_node = build_graph_with_map(g, inner, arg, map);
            let prim_node = g.add(Node::Prim(*prim));
            g.add(Node::App { func: prim_node, args: smallvec![inner_node] })
        },
        ProgramTemplate::Binary(prim, l, r) => {
            let ln = build_graph_with_map(g, l, arg, map);
            let rn = build_graph_with_map(g, r, arg, map);
            let prim_node = g.add(Node::Prim(*prim));
            let pl = g.add(Node::App { func: prim_node, args: smallvec![ln] });
            g.add(Node::App { func: pl, args: smallvec![rn] })
        },
        ProgramTemplate::Trinary(prim, a, b, c) => {
            let an = build_graph_with_map(g, a, arg, map);
            let bn = build_graph_with_map(g, b, arg, map);
            let cn = build_graph_with_map(g, c, arg, map);
            let prim_node = g.add(Node::Prim(*prim));
            let pa = g.add(Node::App { func: prim_node, args: smallvec![an] });
            let pab = g.add(Node::App { func: pa, args: smallvec![bn] });
            g.add(Node::App { func: pab, args: smallvec![cn] })
        },
        ProgramTemplate::App(func_tmpl, arg_tmpl) => {
             let func_node = build_graph_with_map(g, func_tmpl, arg, map);
             let arg_node = build_graph_with_map(g, arg_tmpl, arg, map);
             g.add(Node::App { func: func_node, args: smallvec![arg_node] })
        },
        ProgramTemplate::Leaf => g.add(Node::Leaf),
        ProgramTemplate::Fork(l, r) => {
            let ln = build_graph_with_map(g, l, arg, map);
            let rn = build_graph_with_map(g, r, arg, map);
            g.add(Node::Fork(ln, rn))
        },
        ProgramTemplate::Stem(inner) => {
            let in_n = build_graph_with_map(g, inner, arg, map);
            g.add(Node::Stem(in_n))
        }
    };
    
    map.insert(node_id, tmpl.clone());
    node_id
}

pub fn node_to_template(g: &Graph, node: NodeId) -> Option<ProgramTemplate> {
    match g.get(node) {
        Node::Leaf => Some(ProgramTemplate::Leaf), // Or Identity? Leaf is Leaf. Identity is usage.
        Node::Float(f) => Some(ProgramTemplate::Constant(*f as u64)), 
        Node::Prim(_) => None, 
        Node::Fork(l, r) => {
            let lt = node_to_template(g, *l)?;
            let rt = node_to_template(g, *r)?;
            Some(ProgramTemplate::Fork(Box::new(lt), Box::new(rt)))
        },
        Node::Stem(inner) => {
            let it = node_to_template(g, *inner)?;
            Some(ProgramTemplate::Stem(Box::new(it)))
        },
        Node::App { func, args } => {
            if args.len() != 1 { return None; }
            let arg = args[0];
            
            if let Node::Prim(p) = g.get(*func) {
                 let inner = node_to_template(g, arg)?;
                 Some(ProgramTemplate::Apply(*p, Box::new(inner)))
            } else {
                let f_tmpl = node_to_template(g, *func)?;
                let a_tmpl = node_to_template(g, arg)?;
                Some(ProgramTemplate::App(Box::new(f_tmpl), Box::new(a_tmpl)))
            }
        }
        _ => None,
    }
}
