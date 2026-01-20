use crate::arena::{Graph, Node, NodeId};
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

fn bexpr_k(g: &mut Graph, u: BExpr) -> BExpr {
    // K = (n n)
    let n = g.add(Node::Leaf);
    // K = Stem(n).
    let k_node = g.add(Node::Stem(n)); 
    let k_const = BExpr::Const(k_node);
    BExpr::App(Box::new(k_const), Box::new(u))
}

fn bexpr_s(g: &mut Graph, u: BExpr, v: BExpr) -> BExpr {
    // Canonical S x z = Leaf (Leaf x z)
    // Corresponds to Stem(Fork(x, z))
    // We construct (n (n u v))
    // (n u v) -> Fork(u, v) (via Stem fallback)
    // (n Fork(u, v)) -> Stem(Fork(u, v)) -> S u v
    
    let n = g.add(Node::Leaf);
    let n_const = BExpr::Const(n);
    
    // (n u)
    let nu = BExpr::App(Box::new(n_const.clone()), Box::new(u));
    // (n u v)
    let nuv = BExpr::App(Box::new(nu), Box::new(v));
    
    // (n (n u v))
    BExpr::App(Box::new(n_const), Box::new(nuv))
}

fn bexpr_i(g: &mut Graph) -> BExpr {
    // I = Leaf Leaf
    // (n n) -> Stem(n) -> I
    let n = g.add(Node::Leaf);
    let n_const = BExpr::Const(n);
    BExpr::App(Box::new(n_const.clone()), Box::new(n_const))
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
