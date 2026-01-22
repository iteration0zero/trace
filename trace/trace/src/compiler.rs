use crate::arena::{Graph, Node, NodeId};
use crate::engine::{reduce, EvalContext};
use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub enum CompileTerm {
    Var(String),
    Const(NodeId),
    App(Box<CompileTerm>, Box<CompileTerm>),
    Lam(String, Box<CompileTerm>),
}

// Bracket Expression (Internal)
#[derive(Debug, Clone)]
enum BExpr {
    Var(String),
    Const(NodeId),
    App(Box<BExpr>, Box<BExpr>),
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
    // I = S1 K K (reduces to a pure tree under triage rules)
    let s1 = s1_node(g);
    let k = k_node(g);
    let app1 = g.add(Node::App { func: s1, args: smallvec::smallvec![k] });
    let app2 = g.add(Node::App { func: app1, args: smallvec::smallvec![k] });
    let mut ctx = EvalContext::default();
    reduce(g, app2, &mut ctx)
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

fn bexpr_occurs(name: &str, e: &BExpr) -> bool {
    match e {
        BExpr::Var(n) => n == name,
        BExpr::Const(_) => false,
        BExpr::App(l, r) => bexpr_occurs(name, l) || bexpr_occurs(name, r),
    }
}

fn bexpr_abstract(g: &mut Graph, name: &str, e: BExpr) -> BExpr {
    if !bexpr_occurs(name, &e) {
        bexpr_k(g, e)
    } else {
        match e {
            BExpr::Var(_) => bexpr_i(g),
            BExpr::App(l, r) => {
                if let BExpr::Var(rn) = &*r {
                    if rn == name && !bexpr_occurs(name, &*l) {
                        return *l;
                    }
                }
                let al = bexpr_abstract(g, name, *l);
                let ar = bexpr_abstract(g, name, *r);
                bexpr_s(g, al, ar)
            }
            BExpr::Const(_) => bexpr_k(g, e),
        }
    }
}

fn compile_to_bexpr(g: &mut Graph, t: CompileTerm) -> BExpr {
    match t {
        CompileTerm::Var(s) => BExpr::Var(s),
        CompileTerm::Const(n) => BExpr::Const(n),
        CompileTerm::App(f, a) => BExpr::App(
            Box::new(compile_to_bexpr(g, *f)),
            Box::new(compile_to_bexpr(g, *a))
        ),
        CompileTerm::Lam(name, body) => {
            let body_bexpr = compile_to_bexpr(g, *body);
            bexpr_abstract(g, &name, body_bexpr)
        }
    }
}

fn bexpr_to_node(g: &mut Graph, e: BExpr) -> Result<NodeId, String> {
    match e {
        BExpr::Const(n) => Ok(n),
        BExpr::App(l, r) => {
            let ln = bexpr_to_node(g, *l)?;
            let rn = bexpr_to_node(g, *r)?;
            let mut args = SmallVec::new();
            args.push(rn);
            Ok(g.add(Node::App { func: ln, args }))
        }
        BExpr::Var(name) => Err(format!("Unbound variable: {}", name)),
    }
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
