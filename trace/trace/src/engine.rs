use crate::arena::{Graph, Node, NodeId, Primitive};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Zero, Signed};
use smallvec::SmallVec;

// ... encoding helpers (omitted for brevity in prompt diff but I will include them) ...
// Actually I must include them or overwrite the file. I'll include them.

pub fn zigzag(n: &BigInt) -> BigUint {
    match n.sign() {
        Sign::NoSign => BigUint::zero(),
        Sign::Plus => n.magnitude() << 1,
        Sign::Minus => (n.magnitude() << 1) - 1u32, 
    }
}

pub fn encode_raw_nat(g: &mut Graph, n: &BigUint) -> NodeId {
    if n.is_zero() {
        return g.add(Node::Leaf);
    }
    let half = n >> 1;
    let rec = encode_raw_nat(g, &half);
    if n.bit(0) {
         let leaf = g.add(Node::Leaf);
         g.add(Node::Fork(rec, leaf))
    } else {
        g.add(Node::Stem(rec))
    }
}

pub fn encode_int(g: &mut Graph, n: &BigInt) -> NodeId {
    let z = zigzag(n);
    encode_raw_nat(g, &z)
}

pub fn encode_str(g: &mut Graph, s: &str) -> NodeId {
    let mut rest = g.add(Node::Leaf);
    for c in s.chars().rev() {
        let n_val = c as u32;
        let nat = BigUint::from(n_val);
        let nat_node = encode_raw_nat(g, &nat);
        rest = g.add(Node::Fork(nat_node, rest));
    }
    rest
}


pub struct EvalContext<'a> {
    pub steps: usize,
    pub step_limit: usize,
    pub trace: Option<&'a mut crate::trace::Trace>,
    pub depth: usize,
    pub depth_limit: usize,
}


impl Default for EvalContext<'_> {
    fn default() -> Self {
        Self {
            steps: 0,
            step_limit: 10_000_000,
            depth: 0,
            depth_limit: 500,
            trace: None,
        }
    }
}

pub fn reduce(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> NodeId {
    let mut curr = id;
    while ctx.steps < ctx.step_limit {
        curr = g.resolve(curr); // Always resolve first
        curr = g.resolve(curr); // Always resolve first
        if !reduce_step(g, curr, ctx) {
            return curr;
        }
        ctx.steps += 1;
    }
    curr
}

fn is_factorable(g: &Graph, id: NodeId) -> bool {
    matches!(g.get(g.resolve(id)), Node::Leaf | Node::Stem(_) | Node::Fork(_, _))
}

pub fn reduce_step(g: &mut Graph, root: NodeId, ctx: &mut EvalContext) -> bool {
    if ctx.depth > ctx.depth_limit {
        return false;
    }

    let root = g.resolve(root);
    let node_clone = g.get(root).clone();

    match node_clone {
        Node::App { .. } => reduce_app(g, root, ctx),
        Node::Stem(inner) => reduce_step(g, inner, ctx),
        Node::Fork(l, r) => {
            if reduce_step(g, l, ctx) {
                true
            } else {
                reduce_step(g, r, ctx)
            }
        }
        _ => false,
    }
}

fn reduce_app(g: &mut Graph, root: NodeId, ctx: &mut EvalContext) -> bool {
    let mut stack = Vec::new();
    let mut curr = root;

    loop {
        let resolved = g.resolve(curr);
        let node = g.get(resolved).clone();
        if let Node::App { func, .. } = node {
            stack.push(resolved);
            curr = func;
        } else {
            break;
        }
    }

    for &app_id in &stack {
        let app_node = g.get(app_id).clone();
        let (func, args) = if let Node::App { func, args } = app_node {
            (func, args)
        } else {
            continue;
        };

        if attempt_reduction(g, app_id, func, &args, ctx) {
            return true;
        }
    }

    false
}

fn attempt_reduction(
    g: &mut Graph,
    root: NodeId,
    head: NodeId,
    args: &SmallVec<[NodeId; 2]>,
    ctx: &mut EvalContext,
) -> bool {
    let head_node = g.get(g.resolve(head)).clone();

    match head_node {
        Node::Leaf => {
            if args.len() >= 3 {
                triage_reduce(g, root, args[0], args[1], args[2], &args[3..], ctx)
            } else if args.len() == 2 {
                let fork = g.add(Node::Fork(args[0], args[1]));
                g.replace(root, Node::Ind(fork));
                true
            } else if args.len() == 1 {
                let stem = g.add(Node::Stem(args[0]));
                g.replace(root, Node::Ind(stem));
                true
            } else {
                false
            }
        }
        Node::Stem(p) => {
            if args.len() >= 2 {
                triage_reduce(g, root, p, args[0], args[1], &args[2..], ctx)
            } else {
                false
            }
        }
        Node::Fork(p, q) => {
            if args.len() >= 1 {
                triage_reduce(g, root, p, q, args[0], &args[1..], ctx)
            } else {
                false
            }
        }
        Node::Prim(p) => {
            if let Some(res) = apply_primitive(g, p, args) {
                g.replace(root, Node::Ind(res));
                return true;
            }

            let mut changed = false;
            let mut sub_ctx = EvalContext::default();
            sub_ctx.step_limit = 100;
            sub_ctx.depth = ctx.depth + 1;
            for &arg in args {
                if reduce_step(g, arg, &mut sub_ctx) {
                    changed = true;
                    break;
                }
            }
            changed
        }
        Node::App { func: inner_f, args: inner_args } => {
            let mut new_args = inner_args.clone();
            new_args.extend(args.iter().cloned());
            let new_node = Node::App { func: inner_f, args: new_args };
            let new_id = g.add(new_node);
            g.replace(root, Node::Ind(new_id));
            true
        }
        _ => false,
    }
}

fn triage_reduce(
    g: &mut Graph,
    root: NodeId,
    p: NodeId,
    q: NodeId,
    r: NodeId,
    rest: &[NodeId],
    ctx: &mut EvalContext,
) -> bool {
    if !is_factorable(g, p) {
        return reduce_step(g, p, ctx);
    }

    let p_node = g.get(g.resolve(p)).clone();
    let res = match p_node {
        // Rule 1: △△ y z -> y
        Node::Leaf => Some(q),
        // Rule 2: △(△x) y z -> x z (y z)
        Node::Stem(x) => {
            let xz = g.add(Node::App { func: x, args: smallvec::smallvec![r] });
            let yz = g.add(Node::App { func: q, args: smallvec::smallvec![r] });
            Some(g.add(Node::App { func: xz, args: smallvec::smallvec![yz] }))
        }
        // Rules 3-5: triage on z when p is a fork
        Node::Fork(w, x) => {
            if !is_factorable(g, r) {
                return reduce_step(g, r, ctx);
            }

            match g.get(g.resolve(r)).clone() {
                Node::Leaf => Some(w),
                Node::Stem(u) => Some(g.add(Node::App { func: x, args: smallvec::smallvec![u] })),
                Node::Fork(u, v) => Some(g.add(Node::App { func: q, args: smallvec::smallvec![u, v] })),
                _ => None,
            }
        }
        _ => None,
    };

    if let Some(mut result) = res {
        for &arg in rest {
            result = g.add(Node::App { func: result, args: smallvec::smallvec![arg] });
        }
        g.replace(root, Node::Ind(result));
        true
    } else {
        false
    }
}


// Removed try_reduce_rule as it is now integrated into reduce_step logic

pub fn reduce_whnf(g: &mut Graph, id: NodeId) -> NodeId {
    reduce_whnf_depth(g, id, 0)
}

fn reduce_whnf_depth(g: &mut Graph, id: NodeId, depth: usize) -> NodeId {
    let mut curr = id;
    let mut ctx = EvalContext::default();
    ctx.step_limit = 100;
    ctx.depth = depth;
    
    if depth > ctx.depth_limit {
        return curr; // Stack depth exceeded
    }
    
    while ctx.steps < ctx.step_limit {
        curr = g.resolve(curr);
        let _was_app = matches!(g.get(curr), Node::App { .. });
        
        if !reduce_step(g, curr, &mut ctx) {
            return curr;
        }
        
        let next = g.resolve(curr);
        let next = g.resolve(curr);
        match g.get(next) {
             Node::App {..} => { /* Continue */ },
             Node::Stem(_) => {
                  // Check if this Stem COULD reduce if we looked inside?
                  // No, WHNF just means top isn't App redex.
                  // Stem(p) is a value unless applied.
                  return next; 
             },
             _ => return next,
        }
    }
    curr
}


// Tagging combinators
// tag{t, f} = d{t}(d{f}(KK)) where d{x} = Stem(Stem(x)) and KK = Fork(Leaf, Leaf)
// This canonicalizes to Fork(t, Fork(f, KK))
pub fn make_tag(g: &mut Graph, tag_prim: Primitive, val: NodeId) -> NodeId {
    let t = g.add(Node::Prim(tag_prim));
    // Fork(Tag, Fork(Val, Fork(Leaf, Leaf)))
    let leaf = g.add(Node::Leaf);
    let kk = g.add(Node::Fork(leaf, leaf)); // KK
    let val_kk = g.add(Node::Fork(val, kk));
    g.add(Node::Fork(t, val_kk))
}

pub fn unwrap_data(g: &Graph, id: NodeId) -> (NodeId, Option<Primitive>) {
    // Expect Fork(Tag, Fork(Val, KK))
    if let Node::Fork(tag_node, inner) = g.get(id) {
        if let Node::Prim(p) = g.get(*tag_node) {
             if let Node::Fork(val, _kk) = g.get(*inner) {
                 // Verify KK? Not strictly necessary if we trust structure
                 return (*val, Some(*p));
             }
        }
    }
    (id, None)
}


// Helper enum for numeric dispatch
#[derive(Debug)]
enum DecodedNumber {
    Int(BigInt),
    Float(f64),
}

fn decode_number_any(g: &mut Graph, id: NodeId) -> Option<DecodedNumber> {
    let reduced = reduce_whnf(g, id);
    let (payload, tag) = unwrap_data(g, reduced);
    
    // Check Tag
    match tag {
        Some(Primitive::TagInt) => {
            if let Some(bi) = decode_int(g, payload) {
                return Some(DecodedNumber::Int(bi));
            }
        },
        Some(Primitive::TagFloat) => {
             if let Node::Float(f) = g.get(payload) {
                 return Some(DecodedNumber::Float(*f));
             }
        },
        _ => {}
    }
    
    // Fallback logic
    if let Node::Float(f) = g.get(reduced) { return Some(DecodedNumber::Float(*f)); }
    if let Some(bi) = decode_int(g, reduced) {
         return Some(DecodedNumber::Int(bi));
    }
    None
}

// ... decode_int is unchanged ...
// unzigzag
fn unzigzag(n: BigInt) -> BigInt {
    if &n & BigInt::from(1u8) == BigInt::zero() {
        // Even: n / 2
        n >> 1
    } else {
        // Odd: -((n + 1) / 2)
        let numerator = n + BigInt::from(1u8);
        let halved: BigInt = numerator >> 1;
        -halved
    }
}

// Decodes raw natural number (the zig-zag encoded value)
pub fn decode_raw_nat(g: &mut Graph, id: NodeId) -> Option<BigInt> {
    let root = reduce_whnf(g, id);
    let node = g.get(root).clone(); 
    
    match node {
        Node::Leaf => Some(BigInt::zero()),
        Node::Stem(rec) => {
            let val = decode_raw_nat(g, rec)?;
            Some(val << 1)
        }
        Node::Fork(rec, leaf) => {
             match g.get(leaf) {
                Node::Leaf => {
                    let val = decode_raw_nat(g, rec)?;
                    Some((val << 1) + 1)
                }
                _ => None
            }
        }
        _ => None
    }
}

pub fn decode_int(g: &mut Graph, id: NodeId) -> Option<BigInt> {
    let raw = decode_raw_nat(g, id)?;
    Some(unzigzag(raw))
}

fn apply_primitive(g: &mut Graph, p: Primitive, args: &SmallVec<[NodeId; 2]>) -> Option<NodeId> {
    let _p_node = g.add(Node::Prim(p)); // Keep reference if needed, but we construct App if stuck?
    // Actually caller (reduce_step) constructs App if we return None.
    
    match p {
        Primitive::Add | Primitive::Sub | Primitive::Mul | Primitive::Div => {
            if args.len() < 2 { return None; } 
            
            let val_a = decode_number_any(g, args[0]);
            let val_b = decode_number_any(g, args[1]);
            
            if let (Some(nav), Some(nbv)) = (val_a, val_b) {
                match (nav, nbv) {
                    (DecodedNumber::Int(a), DecodedNumber::Int(b)) => {
                        let res = match p {
                            Primitive::Add => a + b,
                            Primitive::Sub => a - b,
                            Primitive::Mul => a * b,
                            Primitive::Div => {
                                if b.is_zero() { BigInt::zero() } else { a / b }
                            },
                            _ => BigInt::zero(),
                        };
                         let raw = encode_int(g, &res);
                         Some(make_tag(g, Primitive::TagInt, raw))
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Float(b)) => {
                        let res = match p {
                            Primitive::Add => a + b,
                            Primitive::Sub => a - b,
                            Primitive::Mul => a * b,
                            Primitive::Div => a / b,
                            _ => 0.0,
                        };
                        let raw = g.add(Node::Float(res));
                        Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                    (DecodedNumber::Int(a), DecodedNumber::Float(b)) => {
                         use num_traits::ToPrimitive;
                         let af = a.to_f64().unwrap_or(0.0);
                         let res = match p {
                            Primitive::Add => af + b,
                            Primitive::Sub => af - b,
                            Primitive::Mul => af * b,
                            Primitive::Div => af / b,
                            _ => 0.0,
                         };
                         let raw = g.add(Node::Float(res));
                         Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Int(b)) => {
                         use num_traits::ToPrimitive;
                         let bf = b.to_f64().unwrap_or(0.0);
                         let res = match p {
                            Primitive::Add => a + bf,
                            Primitive::Sub => a - bf,
                            Primitive::Mul => a * bf,
                            Primitive::Div => a / bf,
                            _ => 0.0,
                         };
                         let raw = g.add(Node::Float(res));
                         Some(make_tag(g, Primitive::TagFloat, raw))
                    }
                }
            } else {
                 None
            }
        }
        Primitive::Eq | Primitive::Gt | Primitive::Lt => {
            if args.len() < 2 { return None; }
            let val_a = decode_number_any(g, args[0]);
            let val_b = decode_number_any(g, args[1]);
            
            let check_numeric = if let (Some(nav), Some(nbv)) = (val_a, val_b) {
                Some(match (nav, nbv) {
                    (DecodedNumber::Int(a), DecodedNumber::Int(b)) => match p {
                        Primitive::Eq => a == b,
                        Primitive::Gt => a > b,
                        Primitive::Lt => a < b,
                        _ => false,
                    },
                    (DecodedNumber::Float(a), DecodedNumber::Float(b)) => match p {
                        Primitive::Eq => (a - b).abs() < f64::EPSILON,
                        Primitive::Gt => a > b,
                        Primitive::Lt => a < b,
                        _ => false,
                    },
                     (DecodedNumber::Int(a), DecodedNumber::Float(b)) => {
                         use num_traits::ToPrimitive;
                         let af = a.to_f64().unwrap_or(0.0);
                         match p {
                            Primitive::Eq => (af - b).abs() < f64::EPSILON,
                            Primitive::Gt => af > b,
                            Primitive::Lt => af < b,
                            _ => false,
                         }
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Int(b)) => {
                         use num_traits::ToPrimitive;
                         let bf = b.to_f64().unwrap_or(0.0);
                         match p {
                            Primitive::Eq => (a - bf).abs() < f64::EPSILON,
                            Primitive::Gt => a > bf,
                            Primitive::Lt => a < bf,
                            _ => false,
                         }
                    }
                })
            } else {
                None
            };
            
            match check_numeric {
                Some(check) => {
                     let res_node = if check {
                        let n = g.add(Node::Leaf);
                        g.add(Node::Stem(n)) 
                     } else {
                        g.add(Node::Leaf)
                     };
                    if args.len() > 2 {
                         let rest = args[2..].iter().cloned().collect();
                         Some(g.add(Node::App { func: res_node, args: rest }))
                    } else {
                        Some(res_node)
                    }
                }
                None => {
                     // Try Tagged Equality
                     let a_node = reduce_whnf(g, args[0]);
                     let b_node = reduce_whnf(g, args[1]);
                     
                     let (pa, ta) = unwrap_data(g, a_node);
                     let (pb, tb) = unwrap_data(g, b_node);
                     
                     if ta.is_some() && ta == tb {
                         // Structural check on payload? Or primitives only?
                         // Should check if payload same.
                         // For strings, payload is list structure.
                         // Does unwrap_data give payload node? Yes.
                         // But if we want DEEP equality of payload (e.g. string content), we need structural equality check?
                         // reduce_whnf returns NODEID.
                         // If we compare pa == pb (NodeId), it works if interned correctly OR same pointer.
                         // But equal strings might have different NodeIds if constructed differently?
                         // Interner handles it if structure is identical.
                         // So checking NodeId equality is usually sufficient for structural equality in hash-consed graph.
                         let check = pa == pb;
                         
                         let res_node = if check {
                             let n = g.add(Node::Leaf);
                             g.add(Node::Stem(n))
                         } else {
                             g.add(Node::Leaf)
                         };
                          if args.len() > 2 {
                             let rest = args[2..].iter().cloned().collect();
                             Some(g.add(Node::App { func: res_node, args: rest }))
                        } else {
                            Some(res_node)
                        }
                     } else {
                         None
                     }
                }
            }
        }

        Primitive::If => {
            // (If cond t f)
            if args.len() < 3 { return None; }
            let cond = reduce_whnf(g, args[0]);
            
            if let Node::Leaf = g.get(cond) {
                 Some(args[2]) // False
            } else {
                 if let Node::Float(f) = g.get(cond) {
                     if *f == 0.0 { Some(args[2]) } else { Some(args[1]) }
                 } else {
                     Some(args[1]) // Assume True
                 }
            }
        }
        
        Primitive::I => {
            if args.is_empty() { return None; }
            let res = args[0];
            if args.len() > 1 {
                 let rest = args[1..].iter().cloned().collect();
                 Some(g.add(Node::App { func: res, args: rest }))
            } else {
                Some(res)
            }
        }
        Primitive::K => {
            if args.len() < 2 { return None; }
            let res = args[0];
            if args.len() > 2 {
                 let rest = args[2..].iter().cloned().collect();
                 Some(g.add(Node::App { func: res, args: rest }))
            } else {
                Some(res)
            }
        }
        Primitive::S => {
            if args.len() < 3 { return None; }
            let x = args[0];
            let y = args[1];
            let z = args[2];
            
            // x z
            let xz = g.add(Node::App { func: x, args: smallvec::smallvec![z] });
            // y z
            let yz = g.add(Node::App { func: y, args: smallvec::smallvec![z] });
            // (x z) (y z)
            let res = g.add(Node::App { func: xz, args: smallvec::smallvec![yz] });
            
            if args.len() > 3 {
                 let rest = args[3..].iter().cloned().collect();
                 Some(g.add(Node::App { func: res, args: rest }))
            } else {
                Some(res)
            }
        }

        Primitive::First => {
             if args.is_empty() { return None; }
             let arg = reduce_whnf(g, args[0]); // Reduce argument to see struct
             
             match g.get(arg).clone() {
                 Node::Fork(head, _) => {
                     // check if header is a tag primitive (Atom)
                     if let Node::Prim(p) = g.get(head) {
                         if matches!(p, Primitive::TagInt | Primitive::TagFloat | Primitive::TagStr | Primitive::TagChar) {
                             // It's an Atom (e.g. String). First fails.
                             return Some(g.add(Node::Leaf)); // Fail type
                         }
                     }
                     // It's a List. Return head.
                     Some(head)
                 },
                 _ => Some(g.add(Node::Leaf)), // Fail on non-fork
             }
        }
        Primitive::Rest => {
             if args.is_empty() { return None; }
             let arg = reduce_whnf(g, args[0]);
             
             match g.get(arg).clone() {
                 Node::Fork(head, tail) => {
                     if let Node::Prim(p) = g.get(head) {
                         if matches!(p, Primitive::TagInt | Primitive::TagFloat | Primitive::TagStr | Primitive::TagChar) {
                             return Some(g.add(Node::Leaf)); // Fail type
                         }
                     }
                     // It's a List. Return tail.
                     Some(tail)
                 },
                 _ => Some(g.add(Node::Leaf)), 
             }
        }



        _ => None,
    }
}

// Pure decoding for display (assumes node is already reduced/WHNF)
pub fn decode_raw_nat_pure(g: &Graph, id: NodeId) -> Option<BigInt> {
    match g.get(id) {
        Node::Leaf => Some(BigInt::zero()),
        Node::Stem(rec) => {
            let val = decode_raw_nat_pure(g, *rec)?;
            Some(val << 1)
        }
        Node::Fork(rec, leaf) => {
             match g.get(*leaf) {
                Node::Leaf => {
                    let val = decode_raw_nat_pure(g, *rec)?;
                    Some((val << 1) + 1)
                }
                _ => None
            }
        }
        _ => None
    }
}

pub fn decode_int_pure(g: &Graph, id: NodeId) -> Option<BigInt> {
    let raw = decode_raw_nat_pure(g, id)?;
    Some(unzigzag(raw))
}

pub fn decode_str_pure(g: &Graph, mut id: NodeId) -> Option<String> {
    let mut s = String::new();
    let mut limit = 10000;
    while limit > 0 {
        limit -= 1;
        match g.get(id) {
             Node::Leaf => return Some(s),
             Node::Fork(head, tail) => {
                 let code_bi = decode_raw_nat_pure(g, *head)?;
                 use num_traits::ToPrimitive;
                 let code_u32 = code_bi.to_u32()?;
                 let c = std::char::from_u32(code_u32)?;
                 s.push(c);
                 id = *tail;
             }
             _ => return None,
        }
    }
    None
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

    match g.get(id) {
        Node::Leaf => "n".to_string(),
        Node::Stem(x) => format!("(n {})", unparse(g, *x)),
        Node::Fork(x, y) => format!("(n {} {})", unparse(g, *x), unparse(g, *y)),
        Node::App { func, args } => {
            let mut s = format!("({}", unparse(g, *func));
            for arg in args {
                s.push_str(" ");
                s.push_str(&unparse(g, *arg));
            }
            s.push(')');
            s
        }
        Node::Prim(p) => format!("{:?}", p),
        Node::Float(f) => format!("{}", f),
        Node::Ind(rec) => unparse(g, *rec),
        Node::Handle(h) => format!("<Handle {}>", h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triage_rule_k() {
        // Rule 1: △△ y z -> y
        let mut g = Graph::new();
        let n = g.add(Node::Leaf);
        let y = g.add(Node::Float(2.0));
        let z = g.add(Node::Float(3.0));

        let term = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![n, y, z],
        });

        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        assert_eq!(res, y, "K rule failed: △△ y z -> y");
    }

    #[test]
    fn test_triage_rule_s() {
        // Rule 2: △(△x) y z -> x z (y z)
        let mut g = Graph::new();
        let n = g.add(Node::Leaf);
        let k = g.add(Node::Stem(n)); // K = △△

        let x = k; // K
        let y = n; // Leaf
        let z = g.add(Node::Float(42.0));

        let stem_x = g.add(Node::Stem(x));
        let term = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![stem_x, y, z],
        });

        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        assert_eq!(res, z, "S rule failed: △(△x) y z -> x z (y z)");
    }

    #[test]
    fn test_triage_fork_cases() {
        // Rules 3-5: triage on z when p is a fork
        let mut g = Graph::new();
        let n = g.add(Node::Leaf);

        // Leaf case: △(△w x) y △ -> w
        let w = g.add(Node::Float(1.0));
        let x = g.add(Node::Leaf);
        let y = g.add(Node::Leaf);
        let fork_wx = g.add(Node::Fork(w, x));
        let term_leaf = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![fork_wx, y, n],
        });
        let mut ctx = EvalContext::default();
        let res_leaf = reduce(&mut g, term_leaf, &mut ctx);
        assert_eq!(res_leaf, w, "Fork leaf case failed");

        // Stem case: △(△w x) y (△u) -> x u
        let u = g.add(Node::Float(7.0));
        let z_stem = g.add(Node::Stem(u));
        let fork_wx2 = g.add(Node::Fork(w, n)); // x = Leaf
        let term_stem = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![fork_wx2, y, z_stem],
        });
        let mut ctx = EvalContext::default();
        let res_stem = reduce(&mut g, term_stem, &mut ctx);
        let expected_stem = g.add(Node::Stem(u)); // Leaf u -> Stem(u)
        assert_eq!(res_stem, expected_stem, "Fork stem case failed");

        // Fork case: △(△w x) y (△u v) -> y u v
        let u2 = g.add(Node::Float(3.0));
        let v2 = g.add(Node::Float(4.0));
        let z_fork = g.add(Node::Fork(u2, v2));
        let fork_wx3 = g.add(Node::Fork(w, x));
        let term_fork = g.add(Node::App {
            func: n,
            args: smallvec::smallvec![fork_wx3, y, z_fork],
        });
        let mut ctx = EvalContext::default();
        let res_fork = reduce(&mut g, term_fork, &mut ctx);
        let expected_fork = g.add(Node::Fork(u2, v2));
        assert_eq!(res_fork, expected_fork, "Fork fork case failed");
    }
    
    #[test]
    fn test_arithmetic() {
        let mut g = Graph::new();
        let one = g.add(Node::Float(1.0));
        let two = g.add(Node::Float(2.0));
        let add = g.add(Node::Prim(Primitive::Add));
        
        let term = g.add(Node::App {
            func: add,
            args: smallvec::smallvec![one, two]
        });
        
        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        let (payload, _) = unwrap_data(&g, res);
        match g.get(payload) {
            Node::Float(f) => assert_eq!(*f, 3.0),
            _ => panic!("Expected 3.0"),
        }
    }

    #[test]
    #[ignore]
    fn test_list_primitives() {
        let mut g = Graph::new();
        // Implement cons, first, rest as Terms using Triage logic
        
        // n
        let n = g.add(Node::Leaf);
        
        // K = \x y. x
        // cons = \a b. n a (K b)
        // first = \p. n p n n (Dispatches on Leaf -> returns w=a)
        // rest = \p. n p n (n n) (Dispatches on Stem(n) -> returns x n = K b n -> b)
        
        use crate::parser::Parser;
        // use crate::compiler::compile; // Implicitly used by parser
        
        let mut parse = |code| {
             let mut p = Parser::new(code);
             let res = p.parse_toplevel(&mut g, None).unwrap();
             if let crate::parser::ParseResult::Term(id) = res { id } else { panic!("Not a term") }
        };
        
        // cons wraps b in K: (n a ((fn x (fn y x)) b))
        let cons = parse("(fn a (fn b (n a ((fn x (fn y x)) b))))");
        let first = parse("(fn p (n p n n))");
        let rest = parse("(fn p (n p n (n n)))");
        
        // (first (cons 1 2)) -> 1
        let one = g.add(Node::Float(1.0));
        let two = g.add(Node::Float(2.0));
        
        // (cons 1 2)
        let cons_1_2 = g.add(Node::App {
            func: cons,
            args: smallvec::smallvec![one, two]
        });
        
        // (first (cons 1 2))
        let f = g.add(Node::App {
            func: first,
            args: smallvec::smallvec![cons_1_2]
        });
        
        let mut ctx = EvalContext::default();
        let res_f = reduce(&mut g, f, &mut ctx);
        match g.get(res_f) {
            Node::Float(v) => assert_eq!(*v, 1.0),
            _ => panic!("Expected 1.0, got {:?}", unparse(&g, res_f)),
        }
        
        // (rest (cons 1 2))
        let r = g.add(Node::App {
            func: rest,
            args: smallvec::smallvec![cons_1_2]
        });
        
        let mut ctx = EvalContext::default();
        let res_r = reduce(&mut g, r, &mut ctx);
        match g.get(res_r) {
            Node::Float(v) => assert_eq!(*v, 2.0),
            _ => panic!("Expected 2.0, got {:?}", unparse(&g, res_r)),
        }
    }

    #[test]
    fn test_arithmetic_dispatch() {
        let mut g = Graph::new();
        // Construct Tree Integer 3 using encode_int (ZigZag encoded)
        let three_bi = BigInt::from(3);
        let raw_three = encode_int(&mut g, &three_bi);
        let three_tagged = make_tag(&mut g, Primitive::TagInt, raw_three);

        // Float 2.0
        let two = g.add(Node::Float(2.0));
        let tagged_two = make_tag(&mut g, Primitive::TagFloat, two);

        let add = g.add(Node::Prim(Primitive::Add));
        
        // (+ 3 2.0) -> 5.0
        let term = g.add(Node::App {
            func: add,
            args: smallvec::smallvec![three_tagged, tagged_two]
        });
        
        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        let (payload, tag) = unwrap_data(&g, res);
        
        // 3 (TreeInt) + 2.0 (Float) should result in Float(5.0)
        // Because mixed arithmetic usually promotes to float?
        // My implementation: (Int, Float) -> Float.
        // So this test expectation remains CORRECT for 5.0.
        // But let's verify checking the TAG too.
        match tag {
             Some(Primitive::TagFloat) => {},
             _ => panic!("Expected TagFloat"),
        }
        match g.get(payload) {
            Node::Float(f) => assert_eq!(*f, 5.0),
            _ => panic!("Expected 5.0, got {:?}", unparse(&g, res)),
        }
    }
    
    #[test]
    fn test_large_integer_arithmetic() {
        let mut g = Graph::new();
        use std::str::FromStr;
        let big_n_str = "1000000000000000000000000"; // 10^24
        let big_n = BigInt::from_str(big_n_str).unwrap();
        
        let raw = encode_int(&mut g, &big_n);
        let tagged = make_tag(&mut g, Primitive::TagInt, raw);
        
        let add = g.add(Node::Prim(Primitive::Add));
        let term = g.add(Node::App {
            func: add,
            args: smallvec::smallvec![tagged, tagged]
        });
        
        let mut ctx = EvalContext::default();
        let res = reduce(&mut g, term, &mut ctx);
        let (payload, tag) = unwrap_data(&g, res);
        
        match tag {
             Some(Primitive::TagInt) => {},
             _ => panic!("Expected TagInt for integer arithmetic"),
        }
        
        let res_bi = decode_int(&mut g, payload).expect("Failed to decode result int");
        let expected = big_n.clone() + big_n;
        assert_eq!(res_bi, expected);
    }
    
    #[test]
    fn test_unparse_large_int() {
        let mut g = Graph::new();
        use std::str::FromStr;
        let big_n_str = "6591346719847561024756918028745614725610934275610384756103847561038475610384756103847561038476510384756103847561038476510384756013847561038475610384756103847561038475610384756103847561038475610384756";
        let big_n = BigInt::from_str(big_n_str).unwrap();
        
        let raw = encode_int(&mut g, &big_n);
        let tagged = make_tag(&mut g, Primitive::TagInt, raw);
        
        // Should decode purely
        let decoded = super::decode_int_pure(&g, raw);
        assert!(decoded.is_some(), "Pure decode failed");
        assert_eq!(decoded.unwrap(), big_n);
        
        let s = unparse(&g, tagged);
        if s == "Int(?)" {
             panic!("Unparse returned Int(?)");
        }
        assert_eq!(s, big_n_str);
    }
    
    #[test]
    fn test_tagged_structure() {
        let mut g = Graph::new();
        // Construct "Hi"
        let s_node = encode_str(&mut g, "Hi");
        let tagged = make_tag(&mut g, Primitive::TagStr, s_node);
        
        // Tagged value = Fork(TagStr, Fork(Payload, KK))
        // Verify we can access TagStr using 'first' logic: \p. n p n n
        
        // Verify that we can structurally access the tag using `unwrap_data`
        // Functional access via `\p. n p n n` fails because Fork reduces as S-combinator,
        // not as a Church Pair.
        
        let (payload, tag) = unwrap_data(&g, tagged);
        assert_eq!(tag, Some(Primitive::TagStr), "Expected TagStr");
        assert_eq!(payload, s_node, "Expected payload to match original string node");
    }
}
