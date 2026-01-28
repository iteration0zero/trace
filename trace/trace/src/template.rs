use crate::arena::{Graph, Node, NodeId, Primitive};
use smallvec::smallvec;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, PartialEq, Eq, Hash)]
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

impl fmt::Debug for ProgramTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Frame<'a> {
            Enter(&'a ProgramTemplate),
            Text(&'a str),
            Owned(String),
        }

        let mut out = String::new();
        let mut stack = vec![Frame::Enter(self)];

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Text(s) => out.push_str(s),
                Frame::Owned(s) => out.push_str(&s),
                Frame::Enter(curr) => match curr {
                    ProgramTemplate::Identity => out.push_str("Identity"),
                    ProgramTemplate::Constant(v) => out.push_str(&format!("Constant({})", v)),
                    ProgramTemplate::Leaf => out.push_str("Leaf"),
                    ProgramTemplate::Apply(p, inner) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(inner));
                        stack.push(Frame::Owned(format!("Apply({:?}, ", p)));
                    }
                    ProgramTemplate::Binary(p, l, r) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(r));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(l));
                        stack.push(Frame::Owned(format!("Binary({:?}, ", p)));
                    }
                    ProgramTemplate::Trinary(p, a, b, c) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(c));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(a));
                        stack.push(Frame::Owned(format!("Trinary({:?}, ", p)));
                    }
                    ProgramTemplate::App(func, arg) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(arg));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(func));
                        stack.push(Frame::Text("App("));
                    }
                    ProgramTemplate::Fork(l, r) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(r));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(l));
                        stack.push(Frame::Text("Fork("));
                    }
                    ProgramTemplate::Stem(inner) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(inner));
                        stack.push(Frame::Text("Stem("));
                    }
                },
            }
        }

        f.write_str(&out)
    }
}

impl ProgramTemplate {
    pub fn count_nodes(&self) -> usize {
        let mut count = 0usize;
        let mut stack = vec![self];
        while let Some(curr) = stack.pop() {
            count += 1;
            match curr {
                ProgramTemplate::Identity
                | ProgramTemplate::Constant(_)
                | ProgramTemplate::Leaf => {}
                ProgramTemplate::Apply(_, inner) => stack.push(inner),
                ProgramTemplate::Binary(_, l, r) => {
                    stack.push(r);
                    stack.push(l);
                }
                ProgramTemplate::Trinary(_, a, b, c) => {
                    stack.push(c);
                    stack.push(b);
                    stack.push(a);
                }
                ProgramTemplate::App(f, a) => {
                    stack.push(a);
                    stack.push(f);
                }
                ProgramTemplate::Fork(l, r) => {
                    stack.push(r);
                    stack.push(l);
                }
                ProgramTemplate::Stem(inner) => stack.push(inner),
            }
        }
        count
    }
    
    pub fn get_sub_templates(&self) -> Vec<&ProgramTemplate> {
        let mut subs = Vec::new();
        let mut stack = vec![self];
        while let Some(curr) = stack.pop() {
            subs.push(curr);
            match curr {
                ProgramTemplate::Apply(_, inner) => stack.push(inner),
                ProgramTemplate::Binary(_, l, r) => {
                    stack.push(r);
                    stack.push(l);
                }
                ProgramTemplate::Trinary(_, a, b, c) => {
                    stack.push(c);
                    stack.push(b);
                    stack.push(a);
                }
                ProgramTemplate::App(f, a) => {
                    stack.push(a);
                    stack.push(f);
                }
                ProgramTemplate::Fork(l, r) => {
                    stack.push(r);
                    stack.push(l);
                }
                ProgramTemplate::Stem(inner) => stack.push(inner),
                ProgramTemplate::Identity
                | ProgramTemplate::Constant(_)
                | ProgramTemplate::Leaf => {}
            }
        }
        subs
    }
    
    pub fn replace_sub_template(&self, target: &ProgramTemplate, replacement: &ProgramTemplate) -> ProgramTemplate {
        enum Frame<'a> {
            Enter(&'a ProgramTemplate),
            ExitApply(Primitive),
            ExitBinary(Primitive),
            ExitTrinary(Primitive),
            ExitApp,
            ExitFork,
            ExitStem,
        }

        let mut stack = vec![Frame::Enter(self)];
        let mut results: Vec<ProgramTemplate> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(curr) => {
                    if curr == target {
                        results.push(replacement.clone());
                        continue;
                    }
                    match curr {
                        ProgramTemplate::Identity
                        | ProgramTemplate::Constant(_)
                        | ProgramTemplate::Leaf => results.push(curr.clone()),
                        ProgramTemplate::Apply(p, inner) => {
                            stack.push(Frame::ExitApply(*p));
                            stack.push(Frame::Enter(inner));
                        }
                        ProgramTemplate::Binary(p, l, r) => {
                            stack.push(Frame::ExitBinary(*p));
                            stack.push(Frame::Enter(r));
                            stack.push(Frame::Enter(l));
                        }
                        ProgramTemplate::Trinary(p, a, b, c) => {
                            stack.push(Frame::ExitTrinary(*p));
                            stack.push(Frame::Enter(c));
                            stack.push(Frame::Enter(b));
                            stack.push(Frame::Enter(a));
                        }
                        ProgramTemplate::App(f, a) => {
                            stack.push(Frame::ExitApp);
                            stack.push(Frame::Enter(a));
                            stack.push(Frame::Enter(f));
                        }
                        ProgramTemplate::Fork(l, r) => {
                            stack.push(Frame::ExitFork);
                            stack.push(Frame::Enter(r));
                            stack.push(Frame::Enter(l));
                        }
                        ProgramTemplate::Stem(inner) => {
                            stack.push(Frame::ExitStem);
                            stack.push(Frame::Enter(inner));
                        }
                    }
                }
                Frame::ExitApply(p) => {
                    let inner = results.pop().expect("missing inner");
                    results.push(ProgramTemplate::Apply(p, Box::new(inner)));
                }
                Frame::ExitBinary(p) => {
                    let right = results.pop().expect("missing right");
                    let left = results.pop().expect("missing left");
                    results.push(ProgramTemplate::Binary(p, Box::new(left), Box::new(right)));
                }
                Frame::ExitTrinary(p) => {
                    let c = results.pop().expect("missing c");
                    let b = results.pop().expect("missing b");
                    let a = results.pop().expect("missing a");
                    results.push(ProgramTemplate::Trinary(
                        p,
                        Box::new(a),
                        Box::new(b),
                        Box::new(c),
                    ));
                }
                Frame::ExitApp => {
                    let arg = results.pop().expect("missing arg");
                    let func = results.pop().expect("missing func");
                    results.push(ProgramTemplate::App(Box::new(func), Box::new(arg)));
                }
                Frame::ExitFork => {
                    let right = results.pop().expect("missing right");
                    let left = results.pop().expect("missing left");
                    results.push(ProgramTemplate::Fork(Box::new(left), Box::new(right)));
                }
                Frame::ExitStem => {
                    let inner = results.pop().expect("missing inner");
                    results.push(ProgramTemplate::Stem(Box::new(inner)));
                }
            }
        }

        results.pop().unwrap_or_else(|| self.clone())
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
    enum Frame<'a> {
        Enter(&'a ProgramTemplate),
        ExitApply(Primitive),
        ExitBinary(Primitive),
        ExitTrinary(Primitive),
        ExitApp,
        ExitFork,
        ExitStem,
    }

    let mut stack = vec![Frame::Enter(tmpl)];
    let mut results: Vec<NodeId> = Vec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(t) => match t {
                ProgramTemplate::Identity => results.push(arg),
                ProgramTemplate::Constant(c) => results.push(g.add(Node::Float(*c as f64))),
                ProgramTemplate::Leaf => results.push(g.add(Node::Leaf)),
                ProgramTemplate::Apply(prim, inner) => {
                    stack.push(Frame::ExitApply(*prim));
                    stack.push(Frame::Enter(inner));
                }
                ProgramTemplate::Binary(prim, l, r) => {
                    stack.push(Frame::ExitBinary(*prim));
                    stack.push(Frame::Enter(r));
                    stack.push(Frame::Enter(l));
                }
                ProgramTemplate::Trinary(prim, a, b, c) => {
                    stack.push(Frame::ExitTrinary(*prim));
                    stack.push(Frame::Enter(c));
                    stack.push(Frame::Enter(b));
                    stack.push(Frame::Enter(a));
                }
                ProgramTemplate::App(func_tmpl, arg_tmpl) => {
                    stack.push(Frame::ExitApp);
                    stack.push(Frame::Enter(arg_tmpl));
                    stack.push(Frame::Enter(func_tmpl));
                }
                ProgramTemplate::Fork(l, r) => {
                    stack.push(Frame::ExitFork);
                    stack.push(Frame::Enter(r));
                    stack.push(Frame::Enter(l));
                }
                ProgramTemplate::Stem(inner) => {
                    stack.push(Frame::ExitStem);
                    stack.push(Frame::Enter(inner));
                }
            },
            Frame::ExitApply(prim) => {
                let inner_node = results.pop().expect("missing apply inner");
                let prim_node = g.add(Node::Prim(prim));
                results.push(g.add(Node::App { func: prim_node, args: smallvec![inner_node] }));
            }
            Frame::ExitBinary(prim) => {
                let rn = results.pop().expect("missing binary right");
                let ln = results.pop().expect("missing binary left");
                let prim_node = g.add(Node::Prim(prim));
                let pl = g.add(Node::App { func: prim_node, args: smallvec![ln] });
                results.push(g.add(Node::App { func: pl, args: smallvec![rn] }));
            }
            Frame::ExitTrinary(prim) => {
                let cn = results.pop().expect("missing trinary c");
                let bn = results.pop().expect("missing trinary b");
                let an = results.pop().expect("missing trinary a");
                let prim_node = g.add(Node::Prim(prim));
                let pa = g.add(Node::App { func: prim_node, args: smallvec![an] });
                let pab = g.add(Node::App { func: pa, args: smallvec![bn] });
                results.push(g.add(Node::App { func: pab, args: smallvec![cn] }));
            }
            Frame::ExitApp => {
                let arg_node = results.pop().expect("missing app arg");
                let func_node = results.pop().expect("missing app func");
                results.push(g.add(Node::App { func: func_node, args: smallvec![arg_node] }));
            }
            Frame::ExitFork => {
                let rn = results.pop().expect("missing fork right");
                let ln = results.pop().expect("missing fork left");
                results.push(g.add(Node::Fork(ln, rn)));
            }
            Frame::ExitStem => {
                let in_n = results.pop().expect("missing stem inner");
                results.push(g.add(Node::Stem(in_n)));
            }
        }
    }

    results.pop().unwrap_or(arg)
}

pub fn build_graph_with_map(g: &mut Graph, tmpl: &ProgramTemplate, arg: NodeId, map: &mut HashMap<NodeId, ProgramTemplate>) -> NodeId {
    enum Frame<'a> {
        Enter(&'a ProgramTemplate),
        ExitApply(&'a ProgramTemplate, Primitive),
        ExitBinary(&'a ProgramTemplate, Primitive),
        ExitTrinary(&'a ProgramTemplate, Primitive),
        ExitApp(&'a ProgramTemplate),
        ExitFork(&'a ProgramTemplate),
        ExitStem(&'a ProgramTemplate),
    }

    let mut stack = vec![Frame::Enter(tmpl)];
    let mut results: Vec<NodeId> = Vec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(t) => match t {
                ProgramTemplate::Identity => {
                    results.push(arg);
                    map.insert(arg, t.clone());
                }
                ProgramTemplate::Constant(c) => {
                    let node_id = g.add(Node::Float(*c as f64));
                    results.push(node_id);
                    map.insert(node_id, t.clone());
                }
                ProgramTemplate::Leaf => {
                    let node_id = g.add(Node::Leaf);
                    results.push(node_id);
                    map.insert(node_id, t.clone());
                }
                ProgramTemplate::Apply(prim, inner) => {
                    stack.push(Frame::ExitApply(t, *prim));
                    stack.push(Frame::Enter(inner));
                }
                ProgramTemplate::Binary(prim, l, r) => {
                    stack.push(Frame::ExitBinary(t, *prim));
                    stack.push(Frame::Enter(r));
                    stack.push(Frame::Enter(l));
                }
                ProgramTemplate::Trinary(prim, a, b, c) => {
                    stack.push(Frame::ExitTrinary(t, *prim));
                    stack.push(Frame::Enter(c));
                    stack.push(Frame::Enter(b));
                    stack.push(Frame::Enter(a));
                }
                ProgramTemplate::App(func_tmpl, arg_tmpl) => {
                    stack.push(Frame::ExitApp(t));
                    stack.push(Frame::Enter(arg_tmpl));
                    stack.push(Frame::Enter(func_tmpl));
                }
                ProgramTemplate::Fork(l, r) => {
                    stack.push(Frame::ExitFork(t));
                    stack.push(Frame::Enter(r));
                    stack.push(Frame::Enter(l));
                }
                ProgramTemplate::Stem(inner) => {
                    stack.push(Frame::ExitStem(t));
                    stack.push(Frame::Enter(inner));
                }
            },
            Frame::ExitApply(t, prim) => {
                let inner_node = results.pop().expect("missing apply inner");
                let prim_node = g.add(Node::Prim(prim));
                let node_id = g.add(Node::App { func: prim_node, args: smallvec![inner_node] });
                results.push(node_id);
                map.insert(node_id, t.clone());
            }
            Frame::ExitBinary(t, prim) => {
                let rn = results.pop().expect("missing binary right");
                let ln = results.pop().expect("missing binary left");
                let prim_node = g.add(Node::Prim(prim));
                let pl = g.add(Node::App { func: prim_node, args: smallvec![ln] });
                let node_id = g.add(Node::App { func: pl, args: smallvec![rn] });
                results.push(node_id);
                map.insert(node_id, t.clone());
            }
            Frame::ExitTrinary(t, prim) => {
                let cn = results.pop().expect("missing trinary c");
                let bn = results.pop().expect("missing trinary b");
                let an = results.pop().expect("missing trinary a");
                let prim_node = g.add(Node::Prim(prim));
                let pa = g.add(Node::App { func: prim_node, args: smallvec![an] });
                let pab = g.add(Node::App { func: pa, args: smallvec![bn] });
                let node_id = g.add(Node::App { func: pab, args: smallvec![cn] });
                results.push(node_id);
                map.insert(node_id, t.clone());
            }
            Frame::ExitApp(t) => {
                let arg_node = results.pop().expect("missing app arg");
                let func_node = results.pop().expect("missing app func");
                let node_id = g.add(Node::App { func: func_node, args: smallvec![arg_node] });
                results.push(node_id);
                map.insert(node_id, t.clone());
            }
            Frame::ExitFork(t) => {
                let rn = results.pop().expect("missing fork right");
                let ln = results.pop().expect("missing fork left");
                let node_id = g.add(Node::Fork(ln, rn));
                results.push(node_id);
                map.insert(node_id, t.clone());
            }
            Frame::ExitStem(t) => {
                let in_n = results.pop().expect("missing stem inner");
                let node_id = g.add(Node::Stem(in_n));
                results.push(node_id);
                map.insert(node_id, t.clone());
            }
        }
    }

    results.pop().unwrap_or(arg)
}

pub fn node_to_template(g: &Graph, node: NodeId) -> Option<ProgramTemplate> {
    enum Frame {
        Enter(NodeId),
        ExitFork,
        ExitStem,
        ExitApply(Primitive),
        ExitApp,
    }

    let mut stack = vec![Frame::Enter(node)];
    let mut results: Vec<ProgramTemplate> = Vec::new();
    let mut ok = true;

    while let Some(frame) = stack.pop() {
        if !ok {
            break;
        }
        match frame {
            Frame::Enter(id) => match g.get(id) {
                Node::Leaf => results.push(ProgramTemplate::Leaf),
                Node::Float(f) => results.push(ProgramTemplate::Constant(*f as u64)),
                Node::Prim(_) => ok = false,
                Node::Fork(l, r) => {
                    stack.push(Frame::ExitFork);
                    stack.push(Frame::Enter(*r));
                    stack.push(Frame::Enter(*l));
                }
                Node::Stem(inner) => {
                    stack.push(Frame::ExitStem);
                    stack.push(Frame::Enter(*inner));
                }
                Node::App { func, args } => {
                    if args.len() != 1 {
                        ok = false;
                        continue;
                    }
                    let arg = args[0];
                    match g.get(*func) {
                        Node::Prim(p) => {
                            stack.push(Frame::ExitApply(*p));
                            stack.push(Frame::Enter(arg));
                        }
                        _ => {
                            stack.push(Frame::ExitApp);
                            stack.push(Frame::Enter(arg));
                            stack.push(Frame::Enter(*func));
                        }
                    }
                }
                _ => ok = false,
            },
            Frame::ExitFork => {
                let right = results.pop()?;
                let left = results.pop()?;
                results.push(ProgramTemplate::Fork(Box::new(left), Box::new(right)));
            }
            Frame::ExitStem => {
                let inner = results.pop()?;
                results.push(ProgramTemplate::Stem(Box::new(inner)));
            }
            Frame::ExitApply(p) => {
                let inner = results.pop()?;
                results.push(ProgramTemplate::Apply(p, Box::new(inner)));
            }
            Frame::ExitApp => {
                let arg_tmpl = results.pop()?;
                let func_tmpl = results.pop()?;
                results.push(ProgramTemplate::App(Box::new(func_tmpl), Box::new(arg_tmpl)));
            }
        }
    }

    if !ok {
        None
    } else {
        results.pop()
    }
}
