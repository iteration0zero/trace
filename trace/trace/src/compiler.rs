use crate::arena::{Graph, Node, NodeId};
use smallvec::SmallVec;
use std::fmt;

#[derive(Clone)]
pub enum CompileTerm {
    Var(String),
    Const(NodeId),
    App(Box<CompileTerm>, Box<CompileTerm>),
    Lam(String, Box<CompileTerm>),
}

// Bracket Expression (Internal)
#[derive(Clone)]
enum BExpr {
    Var(String),
    Const(NodeId),
    App(Box<BExpr>, Box<BExpr>),
}

impl fmt::Debug for CompileTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Frame<'a> {
            Enter(&'a CompileTerm),
            Text(&'a str),
            Owned(String),
        }

        let mut out = String::new();
        let mut stack = vec![Frame::Enter(self)];

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Text(s) => out.push_str(s),
                Frame::Owned(s) => out.push_str(&s),
                Frame::Enter(term) => match term {
                    CompileTerm::Var(name) => out.push_str(&format!("Var({})", name)),
                    CompileTerm::Const(id) => out.push_str(&format!("Const({:?})", id)),
                    CompileTerm::App(fx, arg) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(arg));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(fx));
                        stack.push(Frame::Text("App("));
                    }
                    CompileTerm::Lam(name, body) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(body));
                        stack.push(Frame::Owned(format!("Lam({}, ", name)));
                    }
                },
            }
        }

        f.write_str(&out)
    }
}

impl fmt::Debug for BExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Frame<'a> {
            Enter(&'a BExpr),
            Text(&'a str),
            Owned(String),
        }

        let mut out = String::new();
        let mut stack = vec![Frame::Enter(self)];

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Text(s) => out.push_str(s),
                Frame::Owned(s) => out.push_str(&s),
                Frame::Enter(expr) => match expr {
                    BExpr::Var(name) => out.push_str(&format!("Var({})", name)),
                    BExpr::Const(id) => out.push_str(&format!("Const({:?})", id)),
                    BExpr::App(l, r) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(r));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(l));
                        stack.push(Frame::Text("App("));
                    }
                },
            }
        }

        f.write_str(&out)
    }
}

fn leaf_node(g: &mut Graph) -> NodeId {
    g.add(Node::Leaf)
}

fn stem_node(g: &mut Graph, inner: NodeId) -> NodeId {
    g.add(Node::Stem(inner))
}

fn fork_node(g: &mut Graph, left: NodeId, right: NodeId) -> NodeId {
    g.add(Node::Fork(left, right))
}

// Canonical tree combinators (pure, factorable)
fn k_node(g: &mut Graph) -> NodeId {
    let leaf = leaf_node(g);
    stem_node(g, leaf)
}

fn s1_node(g: &mut Graph) -> NodeId {
    let leaf = leaf_node(g);
    let kk = fork_node(g, leaf, leaf);          // Δ Δ Δ
    let stem = stem_node(g, kk);                // Δ (Δ Δ Δ)
    fork_node(g, stem, leaf)                    // Δ (Δ(ΔΔΔ)) Δ
}

fn i_node(g: &mut Graph) -> NodeId {
    // I tree (normal form of S K K) as a pure tree:
    // (n (n (n n)) (n n)) = Fork(Stem(Stem(Leaf)), Stem(Leaf))
    let leaf = g.add(Node::Leaf);
    let stem = g.add(Node::Stem(leaf));
    let stemstem = g.add(Node::Stem(stem));
    g.add(Node::Fork(stemstem, stem))
}

fn s_node(g: &mut Graph) -> NodeId {
    // In this evaluator, the stem rule is the standard S (xz (yz)),
    // and the canonical tree for S is s1_node's shape.
    s1_node(g)
}

fn bexpr_k(g: &mut Graph, u: BExpr) -> BExpr {
    let k_const = BExpr::Const(k_node(g));
    BExpr::App(Box::new(k_const), Box::new(u))
}

fn bexpr_s(g: &mut Graph, u: BExpr, v: BExpr) -> BExpr {
    let s_const = BExpr::Const(s_node(g));
    let su = BExpr::App(Box::new(s_const), Box::new(u));
    BExpr::App(Box::new(su), Box::new(v))
}

fn bexpr_i(g: &mut Graph) -> BExpr {
    BExpr::Const(i_node(g))
}



fn bexpr_abstract(g: &mut Graph, name: &str, e: BExpr) -> BExpr {
    enum Frame {
        Enter(BExpr),
        ExitApp { left: BExpr, right: BExpr },
    }

    struct ResultItem {
        expr: BExpr,
        occurs: bool,
    }

    let mut stack = vec![Frame::Enter(e)];
    let mut results: Vec<ResultItem> = Vec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(curr) => match curr {
                BExpr::Var(n) => {
                    let occurs = n == name;
                    let expr = if occurs {
                        bexpr_i(g)
                    } else {
                        bexpr_k(g, BExpr::Var(n))
                    };
                    results.push(ResultItem { expr, occurs });
                }
                BExpr::Const(n) => {
                    results.push(ResultItem {
                        expr: bexpr_k(g, BExpr::Const(n)),
                        occurs: false,
                    });
                }
                BExpr::App(l, r) => {
                    let left = *l;
                    let right = *r;
                    stack.push(Frame::ExitApp { left: left.clone(), right: right.clone() });
                    stack.push(Frame::Enter(right));
                    stack.push(Frame::Enter(left));
                }
            },
            Frame::ExitApp { left, right } => {
                let right_res = results.pop().expect("missing rhs");
                let left_res = results.pop().expect("missing lhs");
                let occurs = left_res.occurs || right_res.occurs;
                if !occurs {
                    results.push(ResultItem {
                        expr: bexpr_k(g, BExpr::App(Box::new(left), Box::new(right))),
                        occurs,
                    });
                    continue;
                }
                if let BExpr::Var(rn) = &right {
                    if rn == name && !left_res.occurs {
                        results.push(ResultItem { expr: left, occurs });
                        continue;
                    }
                }
                let expr = bexpr_s(g, left_res.expr, right_res.expr);
                results.push(ResultItem { expr, occurs });
            }
        }
    }

    results.pop().map(|r| r.expr).unwrap_or_else(|| bexpr_i(g))
}

fn compile_to_bexpr(g: &mut Graph, t: CompileTerm) -> BExpr {
    enum Frame {
        Enter(CompileTerm),
        ExitApp,
        ExitLam(String),
    }

    let mut stack = vec![Frame::Enter(t)];
    let mut results: Vec<BExpr> = Vec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(curr) => match curr {
                CompileTerm::Var(s) => results.push(BExpr::Var(s)),
                CompileTerm::Const(n) => results.push(BExpr::Const(n)),
                CompileTerm::App(f, a) => {
                    stack.push(Frame::ExitApp);
                    stack.push(Frame::Enter(*a));
                    stack.push(Frame::Enter(*f));
                }
                CompileTerm::Lam(name, body) => {
                    stack.push(Frame::ExitLam(name));
                    stack.push(Frame::Enter(*body));
                }
            },
            Frame::ExitApp => {
                let right = results.pop().expect("missing rhs");
                let left = results.pop().expect("missing lhs");
                results.push(BExpr::App(Box::new(left), Box::new(right)));
            }
            Frame::ExitLam(name) => {
                let body = results.pop().expect("missing body");
                results.push(bexpr_abstract(g, &name, body));
            }
        }
    }

    results.pop().expect("missing compiled bexpr")
}

fn bexpr_to_node(g: &mut Graph, e: BExpr) -> Result<NodeId, String> {
    enum Frame {
        Enter(BExpr),
        ExitApp,
    }

    let mut stack = vec![Frame::Enter(e)];
    let mut results: Vec<NodeId> = Vec::new();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(curr) => match curr {
                BExpr::Const(n) => results.push(n),
                BExpr::Var(name) => return Err(format!("Unbound variable: {}", name)),
                BExpr::App(l, r) => {
                    stack.push(Frame::ExitApp);
                    stack.push(Frame::Enter(*r));
                    stack.push(Frame::Enter(*l));
                }
            },
            Frame::ExitApp => {
                let rn = results.pop().expect("missing rhs");
                let ln = results.pop().expect("missing lhs");
                let mut args = SmallVec::new();
                args.push(rn);
                results.push(g.add(Node::App { func: ln, args }));
            }
        }
    }

    Ok(*results.last().unwrap())
}

pub fn compile(g: &mut Graph, t: CompileTerm) -> Result<NodeId, String> {
    let bexpr = compile_to_bexpr(g, t);
    bexpr_to_node(g, bexpr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Node;
    use crate::parser::{Parser, ParseResult};
    use std::collections::HashMap;
    use crate::engine::{reduce, EvalContext};

    #[test]
    fn test_compile_identity() {
        let mut g = Graph::new();
        let env = HashMap::new();
        let mut p = Parser::new("(fn x x)");
        let root = p.parse_toplevel(&mut g, Some(&env)).unwrap();
        
        if let ParseResult::Term(node) = root {
            let compiled = compile(&mut g, CompileTerm::Const(node)).expect("Compilation failed");
            
            // compiled should behave as I (Identity)
            // Apply it to a value (e.g. Float 42)
            let val = g.add(Node::Float(42.0));
            // App(compiled, val)
            let app = g.add(Node::App { 
                func: compiled, 
                args: smallvec::smallvec![val] 
            });
            
            let mut ctx = EvalContext::default();
            let res = reduce(&mut g, app, &mut ctx);
            assert_eq!(res, val, "Identity function should return argument");
        } else {
            panic!("Parse failed");
        }
    }
}
