use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::trace::RuleId;
use crate::engine::types::{mk_seq, EvalContext};
// Cyclic dependency handled via mod.rs in parent module
// We assume reduce module exists and exports reduce_whnf_with_ctx
// If simpler, we can pass reduce function as argument, but that's cumbersome.
// Rust allows imports from sibling modules.
use super::reduce::reduce_whnf_with_ctx;

use smallvec::SmallVec;
use num_bigint::{BigInt, BigUint};
use num_traits::{Zero, ToPrimitive};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

pub fn prim_need_whnf_indices(p: Primitive) -> SmallVec<[usize; 2]> {
    match p {
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Div
        | Primitive::Eq
        | Primitive::Gt
        | Primitive::Lt => smallvec::smallvec![0, 1],
        Primitive::If => smallvec::smallvec![0],
        Primitive::First | Primitive::Rest => smallvec::smallvec![0],
        _ => smallvec::smallvec![],
    }
}

pub fn primitive_min_args(p: Primitive) -> Option<usize> {
    match p {
        Primitive::Add | Primitive::Sub | Primitive::Mul | Primitive::Div => Some(2),
        Primitive::Eq | Primitive::Gt | Primitive::Lt => Some(2),
        Primitive::If => Some(3),
        Primitive::I => Some(1),
        Primitive::K => Some(2),
        Primitive::S => Some(3),
        Primitive::First => Some(1),
        Primitive::Rest => Some(1),
        _ => None,
    }
}

pub fn prim_tag_name(p: Primitive) -> Option<&'static str> {
    match p {
        Primitive::TagInt => Some("Int"),
        Primitive::TagFloat => Some("Float"),
        Primitive::TagStr => Some("Str"),
        Primitive::TagChar => Some("Char"),
        _ => None,
    }
}

pub fn decode_prim_tag_tree(g: &Graph, id: NodeId) -> Option<Primitive> {
    let (_, tag) = unwrap_data(g, id);
    tag
}

#[derive(Debug)]
pub enum DecodedNumber {
    Int(BigInt),
    Float(f64),
}

fn decode_number_any_whnf(g: &mut Graph, id: NodeId) -> Option<DecodedNumber> {
    let (payload, tag) = unwrap_data(g, id);
    match tag {
        Some(Primitive::TagInt) => decode_int_pure(g, payload).map(DecodedNumber::Int),
        Some(Primitive::TagFloat) => match g.get(payload) {
            Node::Float(f) => Some(DecodedNumber::Float(*f)),
            _ => None,
        },
        _ => {
            if let Node::Float(f) = g.get(id) {
                return Some(DecodedNumber::Float(*f));
            }
            decode_int_pure(g, id).map(DecodedNumber::Int)
        }
    }
}

fn decode_number_any(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> Option<DecodedNumber> {
    let reduced = reduce_whnf_with_ctx(g, id, ctx);
    let (payload, tag) = unwrap_data(g, reduced);
    
    // Check Tag
    match tag {
        Some(Primitive::TagInt) => {
            if let Some(bi) = decode_int_with_ctx(g, payload, ctx) {
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
    if let Some(bi) = decode_int_with_ctx(g, reduced, ctx) {
         return Some(DecodedNumber::Int(bi));
    }
    None
}

pub fn apply_primitive_whnf(g: &mut Graph, p: Primitive, args: &SmallVec<[NodeId; 2]>) -> Option<NodeId> {
    match p {
        Primitive::Add | Primitive::Sub | Primitive::Mul | Primitive::Div => {
            if args.len() < 2 {
                return None;
            }
            let val_a = decode_number_any_whnf(g, args[0]);
            let val_b = decode_number_any_whnf(g, args[1]);
            if let (Some(nav), Some(nbv)) = (val_a, val_b) {
                match (nav, nbv) {
                    (DecodedNumber::Int(a), DecodedNumber::Int(b)) => {
                        let res = match p {
                            Primitive::Add => a + b,
                            Primitive::Sub => a - b,
                            Primitive::Mul => a * b,
                            Primitive::Div => {
                                if b.is_zero() {
                                    BigInt::zero()
                                } else {
                                    a / b
                                }
                            }
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
        _ => None,
    }
}

pub fn apply_primitive(
    g: &mut Graph,
    p: Primitive,
    args: &SmallVec<[NodeId; 2]>,
    ctx: &mut EvalContext,
) -> Option<NodeId> {
    match p {
        Primitive::Add | Primitive::Sub | Primitive::Mul | Primitive::Div => {
            if args.len() < 2 { return None; } 
            
            let val_a = decode_number_any(g, args[0], ctx);
            let val_b = decode_number_any(g, args[1], ctx);
            
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
            let val_a = decode_number_any(g, args[0], ctx);
            let val_b = decode_number_any(g, args[1], ctx);
            
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
                         let af = a.to_f64().unwrap_or(0.0);
                         match p {
                            Primitive::Eq => (af - b).abs() < f64::EPSILON,
                            Primitive::Gt => af > b,
                            Primitive::Lt => af < b,
                            _ => false,
                         }
                    }
                    (DecodedNumber::Float(a), DecodedNumber::Int(b)) => {
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
                         Some(mk_seq(g, res_node, rest))
                    } else {
                        Some(res_node)
                    }
                }
                None => {
                     let a_node = reduce_whnf_with_ctx(g, args[0], ctx);
                     let b_node = reduce_whnf_with_ctx(g, args[1], ctx);
                     let (pa, ta) = unwrap_data(g, a_node);
                     let (pb, tb) = unwrap_data(g, b_node);
                     if ta.is_some() && ta == tb {
                         let check = pa == pb;
                         let res_node = if check {
                             let n = g.add(Node::Leaf);
                             g.add(Node::Stem(n))
                         } else {
                             g.add(Node::Leaf)
                         };
                          if args.len() > 2 {
                             let rest = args[2..].iter().cloned().collect();
                             Some(mk_seq(g, res_node, rest))
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
            if args.len() < 3 { return None; }
            let cond = reduce_whnf_with_ctx(g, args[0], ctx);
            
            if let Node::Leaf = g.get(cond) {
                 Some(args[2])
            } else {
                 if let Node::Float(f) = g.get(cond) {
                     if *f == 0.0 { Some(args[2]) } else { Some(args[1]) }
                 } else {
                     Some(args[1])
                 }
            }
        }
        Primitive::I => {
            if args.is_empty() { return None; }
            let res = args[0];
            if args.len() > 1 {
                 let rest = args[1..].iter().cloned().collect();
                 Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::K => {
            if args.len() < 2 { return None; }
            let res = args[0];
            if args.len() > 2 {
                 let rest = args[2..].iter().cloned().collect();
                 Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::S => {
            if args.len() < 3 { return None; }
            let x = args[0];
            let y = args[1];
            let z = args[2];
            let xz = mk_seq(g, x, smallvec::smallvec![z]);
            let yz = mk_seq(g, y, smallvec::smallvec![z]);
            let res = mk_seq(g, xz, smallvec::smallvec![yz]);
            if args.len() > 3 {
                 let rest = args[3..].iter().cloned().collect();
                 Some(mk_seq(g, res, rest))
            } else {
                Some(res)
            }
        }
        Primitive::First => {
             if args.is_empty() { return None; }
             let arg = reduce_whnf_with_ctx(g, args[0], ctx);
             match g.get(arg).clone() {
                 Node::Fork(head, _) => {
                     if let Node::Prim(p) = g.get(head) {
                         if matches!(p, Primitive::TagInt | Primitive::TagFloat | Primitive::TagStr | Primitive::TagChar) {
                             return Some(g.add(Node::Leaf));
                         }
                     }
                     Some(head)
                 },
                 _ => Some(g.add(Node::Leaf)),
             }
        }
        Primitive::Rest => {
             if args.is_empty() { return None; }
             let arg = reduce_whnf_with_ctx(g, args[0], ctx);
             match g.get(arg).clone() {
                 Node::Fork(head, tail) => {
                     if let Node::Prim(p) = g.get(head) {
                         if matches!(p, Primitive::TagInt | Primitive::TagFloat | Primitive::TagStr | Primitive::TagChar) {
                             return Some(g.add(Node::Leaf));
                         }
                     }
                     Some(tail)
                 },
                 _ => Some(g.add(Node::Leaf)), 
             }
        }
        _ => None,
    }
}

pub fn make_tag(g: &mut Graph, tag_prim: Primitive, val: NodeId) -> NodeId {
    let t = g.add(Node::Prim(tag_prim));
    let leaf = g.add(Node::Leaf);
    let kk = g.add(Node::Fork(leaf, leaf));
    let val_kk = g.add(Node::Fork(val, kk));
    g.add(Node::Fork(t, val_kk))
}

pub fn unwrap_data(g: &Graph, id: NodeId) -> (NodeId, Option<Primitive>) {
    // Expect Fork(Tag, Fork(Val, KK))
    if let Node::Fork(tag_node, inner) = g.get(id) {
        if let Node::Prim(p) = g.get(*tag_node) {
             if let Node::Fork(val, _kk) = g.get(*inner) {
                 return (*val, Some(*p));
             }
        }
    }
    (id, None)
}

fn unzigzag(n: BigInt) -> BigInt {
    if &n & BigInt::from(1u8) == BigInt::zero() {
        n >> 1
    } else {
        let numerator = n + BigInt::from(1u8);
        let halved: BigInt = numerator >> 1;
        -halved
    }
}

fn zigzag(n: &BigInt) -> BigUint {
    use num_bigint::Sign;
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
    let mut acc = g.add(Node::Leaf);
    let bits = n.bits();
    let mut idx = bits;
    while idx > 0 {
        idx -= 1;
        if n.bit(idx) {
            let leaf = g.add(Node::Leaf);
            acc = g.add(Node::Fork(acc, leaf));
        } else {
            acc = g.add(Node::Stem(acc));
        }
    }
    acc
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

fn decode_raw_nat_with_ctx(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> Option<BigInt> {
    let mut bits: Vec<bool> = Vec::new();
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut curr = reduce_whnf_with_ctx(g, id, ctx);

    loop {
        let resolved = g.resolve(curr);
        if !seen.insert(resolved) {
            return None;
        }
        match g.get(resolved) {
            Node::Leaf => break,
            Node::Stem(inner) => {
                bits.push(false);
                curr = reduce_whnf_with_ctx(g, *inner, ctx);
            }
            Node::Fork(rec, leaf) => {
                if !matches!(g.get(*leaf), Node::Leaf) {
                    return None;
                }
                bits.push(true);
                curr = reduce_whnf_with_ctx(g, *rec, ctx);
            }
            Node::Ind(inner) => {
                curr = *inner;
            }
            _ => return None,
        }
    }

    let mut val = BigInt::zero();
    for bit in bits.iter().rev() {
        val = val << 1;
        if *bit {
            val += 1;
        }
    }
    Some(val)
}

pub fn decode_raw_nat(g: &mut Graph, id: NodeId) -> Option<BigInt> {
    let mut ctx = EvalContext::default();
    ctx.step_limit = 100;
    decode_raw_nat_with_ctx(g, id, &mut ctx)
}

fn decode_int_with_ctx(g: &mut Graph, id: NodeId, ctx: &mut EvalContext) -> Option<BigInt> {
    let raw = decode_raw_nat_with_ctx(g, id, ctx)?;
    Some(unzigzag(raw))
}

pub fn decode_int(g: &mut Graph, id: NodeId) -> Option<BigInt> {
    let mut ctx = EvalContext::default();
    ctx.step_limit = 100;
    let raw = decode_raw_nat_with_ctx(g, id, &mut ctx)?;
    Some(unzigzag(raw))
}

pub fn decode_raw_nat_pure(g: &Graph, id: NodeId) -> Option<BigInt> {
    let mut id = id;
    let mut val = BigInt::zero();
    let mut shift: usize = 0;
    loop {
        match g.get(id) {
            Node::Leaf => return Some(val),
            Node::Stem(rec) => {
                id = *rec;
                shift = shift.saturating_add(1);
            }
            Node::Fork(rec, leaf) => match g.get(*leaf) {
                Node::Leaf => {
                    val += BigInt::from(1u32) << shift;
                    id = *rec;
                    shift = shift.saturating_add(1);
                }
                _ => return None,
            },
            _ => return None,
        }
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

const TREE_HASH_CACHE_MAX: usize = 200_000;

fn hash_mix(seed: u64, v: u64) -> u64 {
    seed ^ (v.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(seed << 6).wrapping_add(seed >> 2))
}

fn encode_raw_nat_hash_u32(n: u32) -> u64 {
    if n == 0 {
        return 0x9e37_01;
    }
    let mut acc = 0x9e37_01;
    let mut started = false;
    for shift in (0..32).rev() {
        let bit = (n >> shift) & 1;
        if !started {
            if bit == 0 {
                continue;
            }
            started = true;
        }
        if bit == 1 {
            let mut next = 0x9e37_06;
            next = hash_mix(next, acc);
            next = hash_mix(next, 0x9e37_01);
            acc = next;
        } else {
            let mut next = 0x9e37_05;
            next = hash_mix(next, acc);
            acc = next;
        }
    }
    acc
}

fn encode_str_hash(s: &str) -> u64 {
    let mut rest = 0x9e37_01;
    for c in s.chars().rev() {
        let nat = encode_raw_nat_hash_u32(c as u32);
        let mut acc = 0x9e37_06;
        acc = hash_mix(acc, nat);
        acc = hash_mix(acc, rest);
        rest = acc;
    }
    rest
}

fn prim_tag_hash(p: Primitive) -> Option<u64> {
    let name = prim_tag_name(p)?;
    let tag = encode_str_hash(name);
    let mut kk = 0x9e37_06;
    kk = hash_mix(kk, 0x9e37_01);
    kk = hash_mix(kk, 0x9e37_01);
    let mut acc = 0x9e37_06;
    acc = hash_mix(acc, tag);
    acc = hash_mix(acc, kk);
    Some(acc)
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TreeHashKey {
    graph_id: u64,
    epoch: u64,
    node: u32,
}

fn tree_hash_cache() -> &'static Mutex<HashMap<TreeHashKey, u64>> {
    static CACHE: OnceLock<Mutex<HashMap<TreeHashKey, u64>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn tree_hash_cache_get(key: &TreeHashKey) -> Option<u64> {
    let cache = tree_hash_cache();
    let guard = cache.lock().unwrap();
    guard.get(key).copied()
}

fn tree_hash_cache_insert(key: TreeHashKey, value: u64) {
    let cache = tree_hash_cache();
    let mut guard = cache.lock().unwrap();
    if guard.len() >= TREE_HASH_CACHE_MAX {
        guard.clear();
    }
    guard.insert(key, value);
}

pub fn tree_hash(g: &Graph, root: NodeId) -> u64 {
    let root = g.resolve(root);
    let key = TreeHashKey {
        graph_id: g.id,
        epoch: g.epoch,
        node: root.0,
    };
    if let Some(v) = tree_hash_cache_get(&key) {
        return v;
    }
    let mut memo: HashMap<NodeId, u64> = HashMap::new();
    let h = tree_hash_cached(g, root, &mut memo);
    tree_hash_cache_insert(key, h);
    h
}

fn tree_hash_cached(g: &Graph, root: NodeId, memo: &mut HashMap<NodeId, u64>) -> u64 {
    let mut stack: Vec<(NodeId, bool)> = Vec::new();
    let root = g.resolve(root);
    stack.push((root, false));
    while let Some((id, done)) = stack.pop() {
        let id = g.resolve(id);
        if memo.contains_key(&id) {
            continue;
        }
        if !done {
            stack.push((id, true));
            match g.get(id) {
                Node::Ind(inner) => {
                    stack.push((g.resolve(*inner), false));
                }
                Node::Stem(child) => stack.push((g.resolve(*child), false)),
                Node::Fork(l, r) => {
                    stack.push((g.resolve(*r), false));
                    stack.push((g.resolve(*l), false));
                }
                Node::App { func, args } => {
                    for arg in args.iter().rev() {
                        stack.push((g.resolve(*arg), false));
                    }
                    stack.push((g.resolve(*func), false));
                }
                _ => {}
            }
        } else {
            let h = match g.get(id) {
                Node::Leaf => 0x9e37_01,
                Node::Prim(p) => prim_tag_hash(*p).unwrap_or_else(|| hash_mix(0x9e37_02, *p as u64)),
                Node::Float(f) => hash_mix(0x9e37_03, f.to_bits()),
                Node::Handle(h) => hash_mix(0x9e37_04, *h as u64),
                Node::Ind(inner) => memo[&g.resolve(*inner)],
                Node::Stem(child) => {
                    let mut acc = 0x9e37_05;
                    acc = hash_mix(acc, memo[&g.resolve(*child)]);
                    acc
                }
                Node::Fork(l, r) => {
                    let mut acc = 0x9e37_06;
                    acc = hash_mix(acc, memo[&g.resolve(*l)]);
                    acc = hash_mix(acc, memo[&g.resolve(*r)]);
                    acc
                }
                Node::App { func, args } => {
                    let mut acc = hash_mix(0x9e37_07, memo[&g.resolve(*func)]);
                    acc = hash_mix(acc, args.len() as u64);
                    for arg in args.iter() {
                        acc = hash_mix(acc, memo[&g.resolve(*arg)]);
                    }
                    acc
                }
            };
            memo.insert(id, h);
        }
    }
    memo[&root]
}
