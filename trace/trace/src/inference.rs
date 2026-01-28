//! # Inference - Path-Polymorphic Type Inference
//!
//! This module implements type inference for the Triage Calculus with support
//! for structural (path-polymorphic) types.

use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::types::{Type, TypeEnv};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Constraint {
    /// T1 must be a subtype of T2 (T1 <= T2)
    Subtype(Type, Type),
    /// T1 must be equal to T2
    Equality(Type, Type),
    /// T must be applicable (T = A -> B)
    Applicable(Type, Type, Type), // FuncType, ArgType, ResultType
}

pub struct InferenceEngine {
    next_var_id: usize,
    pub constraints: Vec<Constraint>,
    substitutions: HashMap<usize, Type>,
}

impl InferenceEngine {
    fn subst_generic(ty: &Type, name: &str, replacement: &Type) -> Type {
        enum Frame<'a> {
            Enter(&'a Type),
            ExitForall(String),
            ExitArrow,
            ExitStem,
            ExitPair,
            ExitUnion(usize),
            ExitRec(usize),
        }

        let mut stack = vec![Frame::Enter(ty)];
        let mut results: Vec<Type> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(t) => match t {
                    Type::Generic(n) if n == name => results.push(replacement.clone()),
                    Type::Forall(n, body) => {
                        if n == name {
                            results.push(Type::Forall(n.clone(), body.clone()));
                        } else {
                            stack.push(Frame::ExitForall(n.clone()));
                            stack.push(Frame::Enter(body));
                        }
                    }
                    Type::Arrow(a, b) => {
                        stack.push(Frame::ExitArrow);
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Enter(a));
                    }
                    Type::Stem(a) => {
                        stack.push(Frame::ExitStem);
                        stack.push(Frame::Enter(a));
                    }
                    Type::Pair(a, b) => {
                        stack.push(Frame::ExitPair);
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Enter(a));
                    }
                    Type::Union(ts) => {
                        stack.push(Frame::ExitUnion(ts.len()));
                        for t in ts.iter().rev() {
                            stack.push(Frame::Enter(t));
                        }
                    }
                    Type::Rec(id, body) => {
                        stack.push(Frame::ExitRec(*id));
                        stack.push(Frame::Enter(body));
                    }
                    _ => results.push(t.clone()),
                },
                Frame::ExitForall(n) => {
                    let body = results.pop().expect("missing forall body");
                    results.push(Type::Forall(n, Box::new(body)));
                }
                Frame::ExitArrow => {
                    let b = results.pop().expect("missing arrow rhs");
                    let a = results.pop().expect("missing arrow lhs");
                    results.push(Type::Arrow(Box::new(a), Box::new(b)));
                }
                Frame::ExitStem => {
                    let a = results.pop().expect("missing stem inner");
                    results.push(Type::Stem(Box::new(a)));
                }
                Frame::ExitPair => {
                    let b = results.pop().expect("missing pair rhs");
                    let a = results.pop().expect("missing pair lhs");
                    results.push(Type::Pair(Box::new(a), Box::new(b)));
                }
                Frame::ExitUnion(count) => {
                    let mut ts = Vec::with_capacity(count);
                    for _ in 0..count {
                        ts.push(results.pop().expect("missing union member"));
                    }
                    ts.reverse();
                    results.push(Type::Union(ts));
                }
                Frame::ExitRec(id) => {
                    let body = results.pop().expect("missing rec body");
                    results.push(Type::Rec(id, Box::new(body)));
                }
            }
        }

        results.pop().unwrap_or_else(|| ty.clone())
    }

    fn subst_rec(ty: &Type, id: usize, replacement: &Type) -> Type {
        enum Frame<'a> {
            Enter(&'a Type),
            ExitForall(String),
            ExitArrow,
            ExitStem,
            ExitPair,
            ExitUnion(usize),
            ExitRec(usize),
        }

        let mut stack = vec![Frame::Enter(ty)];
        let mut results: Vec<Type> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(t) => match t {
                    Type::RecVar(i) if *i == id => results.push(replacement.clone()),
                    Type::Rec(i, body) => {
                        if *i == id {
                            results.push(Type::Rec(*i, body.clone()));
                        } else {
                            stack.push(Frame::ExitRec(*i));
                            stack.push(Frame::Enter(body));
                        }
                    }
                    Type::Arrow(a, b) => {
                        stack.push(Frame::ExitArrow);
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Enter(a));
                    }
                    Type::Stem(a) => {
                        stack.push(Frame::ExitStem);
                        stack.push(Frame::Enter(a));
                    }
                    Type::Pair(a, b) => {
                        stack.push(Frame::ExitPair);
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Enter(a));
                    }
                    Type::Union(ts) => {
                        stack.push(Frame::ExitUnion(ts.len()));
                        for t in ts.iter().rev() {
                            stack.push(Frame::Enter(t));
                        }
                    }
                    Type::Forall(n, body) => {
                        stack.push(Frame::ExitForall(n.clone()));
                        stack.push(Frame::Enter(body));
                    }
                    _ => results.push(t.clone()),
                },
                Frame::ExitForall(n) => {
                    let body = results.pop().expect("missing forall body");
                    results.push(Type::Forall(n, Box::new(body)));
                }
                Frame::ExitArrow => {
                    let b = results.pop().expect("missing arrow rhs");
                    let a = results.pop().expect("missing arrow lhs");
                    results.push(Type::Arrow(Box::new(a), Box::new(b)));
                }
                Frame::ExitStem => {
                    let a = results.pop().expect("missing stem inner");
                    results.push(Type::Stem(Box::new(a)));
                }
                Frame::ExitPair => {
                    let b = results.pop().expect("missing pair rhs");
                    let a = results.pop().expect("missing pair lhs");
                    results.push(Type::Pair(Box::new(a), Box::new(b)));
                }
                Frame::ExitUnion(count) => {
                    let mut ts = Vec::with_capacity(count);
                    for _ in 0..count {
                        ts.push(results.pop().expect("missing union member"));
                    }
                    ts.reverse();
                    results.push(Type::Union(ts));
                }
                Frame::ExitRec(rec_id) => {
                    let body = results.pop().expect("missing rec body");
                    results.push(Type::Rec(rec_id, Box::new(body)));
                }
            }
        }

        results.pop().unwrap_or_else(|| ty.clone())
    }

    pub fn fresh_skolem(&mut self, name: &str) -> Type {
        let id = self.next_var_id;
        self.next_var_id += 1;
        Type::Generic(format!("{}#{}", name, id))
    }

    pub fn new() -> Self {
        Self {
            next_var_id: 0,
            constraints: Vec::new(),
            substitutions: HashMap::new(),
        }
    }
    
    pub fn fresh_var(&mut self) -> Type {
        let id = self.next_var_id;
        self.next_var_id += 1;
        Type::Var(id)
    }
    
    pub fn infer(&mut self, g: &Graph, root: NodeId, env: &TypeEnv) -> Result<Type, String> {
        // Map visited nodes to their types to handle DAG sharing/recursion
        let mut node_types: HashMap<NodeId, Type> = HashMap::new();
        
        self.generate_constraints(g, root, env, &mut node_types)?;
        
        // Solve collected constraints
        self.solve()?;
        
        // Return resolved type of root
        let raw_ty = node_types.get(&root).unwrap().clone();
        Ok(self.resolve_type(raw_ty))
    }
    
    pub fn generate_constraints(
        &mut self, 
        g: &Graph, 
        node: NodeId, 
        env: &TypeEnv, 
        node_types: &mut HashMap<NodeId, Type>
    ) -> Result<(), String> {
        enum Frame {
            Enter(NodeId),
            ExitStem(NodeId, NodeId),
            ExitFork(NodeId, NodeId, NodeId),
            ExitApp(NodeId, NodeId, Vec<NodeId>),
        }

        let mut stack = vec![Frame::Enter(node)];

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(n) => {
                    if node_types.contains_key(&n) {
                        continue;
                    }

                    let ty = self.fresh_var();
                    node_types.insert(n, ty.clone());

                    if let Some(special_ty) = env.specials.get(&n.0).cloned() {
                        self.constraints.push(Constraint::Equality(ty, special_ty));
                        continue;
                    }

                    match g.get(n) {
                        Node::Leaf => {
                            self.constraints.push(Constraint::Equality(ty, Type::Leaf));
                        }
                        Node::Float(_) => {
                            self.constraints.push(Constraint::Equality(ty, Type::Float));
                        }
                        Node::Stem(inner) => {
                            stack.push(Frame::ExitStem(n, *inner));
                            stack.push(Frame::Enter(*inner));
                        }
                        Node::Fork(head, tail) => {
                            stack.push(Frame::ExitFork(n, *head, *tail));
                            stack.push(Frame::Enter(*tail));
                            stack.push(Frame::Enter(*head));
                        }
                        Node::App { func, args } => {
                            let mut arg_list: Vec<NodeId> = args.iter().copied().collect();
                            stack.push(Frame::ExitApp(n, *func, arg_list.clone()));
                            stack.push(Frame::Enter(*func));
                            arg_list.reverse();
                            for arg in arg_list {
                                stack.push(Frame::Enter(arg));
                            }
                        }
                        Node::Prim(p) => {
                            let prim_ty = self.get_primitive_type(*p);
                            self.constraints.push(Constraint::Equality(ty, prim_ty));
                        }
                        _ => {}
                    }
                }
                Frame::ExitStem(n, inner) => {
                    let ty = node_types.get(&n).unwrap().clone();
                    let inner_ty = node_types.get(&inner).unwrap().clone();
                    self.constraints.push(Constraint::Equality(ty, Type::Stem(Box::new(inner_ty))));
                }
                Frame::ExitFork(n, head, tail) => {
                    let ty = node_types.get(&n).unwrap().clone();
                    let h_ty = node_types.get(&head).unwrap().clone();
                    let t_ty = node_types.get(&tail).unwrap().clone();
                    let pair_ty = Type::Pair(Box::new(h_ty), Box::new(t_ty));
                    self.constraints.push(Constraint::Equality(ty, pair_ty));
                }
                Frame::ExitApp(n, func, args) => {
                    let ty = node_types.get(&n).unwrap().clone();
                    let func_ty = node_types.get(&func).unwrap().clone();
                    let mut curr_ty = func_ty;
                    for arg in args {
                        let arg_ty = node_types.get(&arg).unwrap().clone();
                        let res_ty = self.fresh_var();
                        self.constraints.push(Constraint::Applicable(curr_ty.clone(), arg_ty, res_ty.clone()));
                        curr_ty = res_ty;
                    }
                    self.constraints.push(Constraint::Equality(ty, curr_ty));
                }
            }
        }

        Ok(())
    }
    
    pub fn solve(&mut self) -> Result<(), String> {
        self.constraints.reverse();
        let mut visited_subtypes: std::collections::HashSet<(Type, Type)> = std::collections::HashSet::new();

        while let Some(constraint) = self.constraints.pop() {
            match constraint {
                Constraint::Equality(t1, t2) => {
                    let t1_res = self.resolve_type(t1);
                    let t2_res = self.resolve_type(t2);
                    self.unify(t1_res, t2_res)?;
                }
                Constraint::Applicable(func_ty, arg_ty, res_ty) => {
                    let f_res = self.resolve_type(func_ty);
                    match f_res {
                        Type::Arrow(param, ret) => {
                             // Application Rule: Arg <= Param, Result = Ret
                             self.constraints.push(Constraint::Subtype(arg_ty, *param));
                             self.constraints.push(Constraint::Equality(res_ty, *ret));
                        }
                        f => {
                             let param_var = self.fresh_var();
                             let arrow = Type::Arrow(Box::new(param_var.clone()), Box::new(res_ty));
                             
                             // Constrain f to be usable as a function via subtyping axioms.
                             self.constraints.push(Constraint::Subtype(f, arrow));
                             
                             // Constrain arg to be subtype of param_var
                             self.constraints.push(Constraint::Subtype(arg_ty, param_var));
                        }
                    }
                }
                Constraint::Subtype(t1, t2) => {
                    let t1_res = self.resolve_type(t1);
                    let t2_res = self.resolve_type(t2);
                    match (t1_res.clone(), t2_res.clone()) {
                        (Type::Var(i), t) if !matches!(t, Type::Var(_)) => {
                             self.unify(Type::Var(i), t)?;
                        }
                        (t, Type::Var(i)) if !matches!(t, Type::Var(_)) => {
                             self.unify(Type::Var(i), t)?;
                        }
                        (Type::Pair(a1, b1), Type::Pair(a2, b2)) => {
                             self.constraints.push(Constraint::Subtype(*a1, *a2));
                             self.constraints.push(Constraint::Subtype(*b1, *b2));
                        }
                        (Type::Stem(a), Type::Stem(b)) => {
                             self.constraints.push(Constraint::Subtype(*a, *b));
                        }
                        (Type::Arrow(a1, r1), Type::Arrow(a2, r2)) => {
                             // Contravariant Args: a2 <= a1
                             // Covariant Result: r1 <= r2
                             self.constraints.push(Constraint::Subtype(*a2, *a1));
                             self.constraints.push(Constraint::Subtype(*r1, *r2));
                        }
                        _ => {
                            if !self.is_subtype(t1_res.clone(), t2_res.clone(), &mut visited_subtypes) {
                                 if let (Type::Var(_), Type::Var(_)) = (&t1_res, &t2_res) {
                                     self.unify(t1_res, t2_res)?;
                                 } else {
                                     return Err(format!("Subtype Check Failed: {:?} <= {:?}", t1_res, t2_res));
                                 }
                            }
                        }
                    }
                }
            }
            }

        Ok(())
    }
    
    // Coinductive Subtyping (Algorithm 3.4.2) with support for Forall and Rec
    pub fn is_subtype(&mut self, t1: Type, t2: Type, visited: &mut std::collections::HashSet<(Type, Type)>) -> bool {
        enum Frame {
            Check(Type, Type),
            AllPairs { pairs: Vec<(Type, Type)>, idx: usize },
            AnyPairs { pairs: Vec<(Type, Type)>, idx: usize },
        }

        let mut stack = vec![Frame::Check(t1, t2)];
        let mut results: Vec<bool> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Check(a, b) => {
                    if a == b {
                        results.push(true);
                        continue;
                    }
                    if visited.contains(&(a.clone(), b.clone())) {
                        results.push(true);
                        continue;
                    }
                    visited.insert((a.clone(), b.clone()));

                    match (a, b) {
                        (Type::Leaf, Type::Leaf) => results.push(true),
                        (Type::Rec(id, body), t2) => {
                            let rec_ty = Type::Rec(id, body.clone());
                            let unrolled = Self::subst_rec(&body, id, &rec_ty);
                            stack.push(Frame::Check(unrolled, t2));
                        }
                        (t1, Type::Rec(id, body)) => {
                            let rec_ty = Type::Rec(id, body.clone());
                            let unrolled = Self::subst_rec(&body, id, &rec_ty);
                            stack.push(Frame::Check(t1, unrolled));
                        }
                        (Type::Forall(name, body), t2) => {
                            let alpha = self.fresh_var();
                            let instantiated = Self::subst_generic(&body, &name, &alpha);
                            stack.push(Frame::Check(instantiated, t2));
                        }
                        (t1, Type::Forall(name, body)) => {
                            let skolem = self.fresh_skolem(&name);
                            let instantiated = Self::subst_generic(&body, &name, &skolem);
                            stack.push(Frame::Check(t1, instantiated));
                        }
                        (t1, Type::Union(ts)) => {
                            if ts.is_empty() {
                                results.push(false);
                            } else {
                                let pairs: Vec<(Type, Type)> = ts.into_iter().map(|t| (t1.clone(), t)).collect();
                                let first = pairs[0].clone();
                                stack.push(Frame::AnyPairs { pairs, idx: 0 });
                                stack.push(Frame::Check(first.0, first.1));
                            }
                        }
                        (Type::Union(ts), t2) => {
                            if ts.is_empty() {
                                results.push(true);
                            } else {
                                let pairs: Vec<(Type, Type)> = ts.into_iter().map(|t| (t, t2.clone())).collect();
                                let first = pairs[0].clone();
                                stack.push(Frame::AllPairs { pairs, idx: 0 });
                                stack.push(Frame::Check(first.0, first.1));
                            }
                        }
                        (Type::Var(i), t) => {
                            if self.occurs_in(i, &t) {
                                results.push(false);
                            } else {
                                self.substitutions.insert(i, t.clone());
                                self.substitute(i, &t);
                                results.push(true);
                            }
                        }
                        (t, Type::Var(i)) => {
                            if self.occurs_in(i, &t) {
                                results.push(false);
                            } else {
                                self.substitutions.insert(i, t.clone());
                                self.substitute(i, &t);
                                results.push(true);
                            }
                        }
                        (Type::Generic(a), Type::Generic(b)) => results.push(a == b),
                        (Type::Arrow(a1, r1), Type::Arrow(a2, r2)) => {
                            let pairs = vec![(*a2, *a1), (*r1, *r2)];
                            let first = pairs[0].clone();
                            stack.push(Frame::AllPairs { pairs, idx: 0 });
                            stack.push(Frame::Check(first.0, first.1));
                        }
                        (Type::Stem(a), Type::Stem(b)) => {
                            stack.push(Frame::Check(*a, *b));
                        }
                        (Type::Pair(a1, b1), Type::Pair(a2, b2)) => {
                            let pairs = vec![(*a1, *a2), (*b1, *b2)];
                            let first = pairs[0].clone();
                            stack.push(Frame::AllPairs { pairs, idx: 0 });
                            stack.push(Frame::Check(first.0, first.1));
                        }
                        (Type::Leaf, Type::Arrow(a, b)) => {
                            let stem_a = Type::Stem(a);
                            stack.push(Frame::Check(stem_a, *b));
                        }
                        (Type::Stem(inner), Type::Arrow(arg, res)) => {
                            let pair = Type::Pair(inner, arg);
                            stack.push(Frame::Check(pair, *res));
                        }
                        (Type::Pair(left, right), Type::Arrow(arg, res)) => {
                            if matches!(left.as_ref(), Type::Leaf) {
                                stack.push(Frame::Check(*right.clone(), *res));
                                continue;
                            }
                            if matches!(arg.as_ref(), Type::Leaf) {
                                if let Type::Pair(t, _v) = left.as_ref() {
                                    stack.push(Frame::Check(*t.clone(), *res));
                                    continue;
                                }
                            }
                            if let Type::Stem(v_arg) = arg.as_ref() {
                                if let Type::Pair(_u, v_arrow) = left.as_ref() {
                                    if let Type::Arrow(v, t) = v_arrow.as_ref() {
                                        let pairs = vec![(*v_arg.clone(), *v.clone()), (*t.clone(), *res.clone())];
                                        let first = pairs[0].clone();
                                        stack.push(Frame::AllPairs { pairs, idx: 0 });
                                        stack.push(Frame::Check(first.0, first.1));
                                        continue;
                                    }
                                }
                            }
                            if let Type::Pair(w1, w2) = arg.as_ref() {
                                if let Type::Pair(_u, _v) = left.as_ref() {
                                    if let Type::Arrow(w1_t, w2_arrow) = right.as_ref() {
                                        if let Type::Arrow(w2_t, t) = w2_arrow.as_ref() {
                                            let pairs = vec![
                                                (*w1.clone(), *w1_t.clone()),
                                                (*w2.clone(), *w2_t.clone()),
                                                (*t.clone(), *res.clone()),
                                            ];
                                            let first = pairs[0].clone();
                                            stack.push(Frame::AllPairs { pairs, idx: 0 });
                                            stack.push(Frame::Check(first.0, first.1));
                                            continue;
                                        }
                                    }
                                }
                            }
                            if let Type::Stem(s_inner) = left.as_ref() {
                                if let Type::Arrow(u, uv) = s_inner.as_ref() {
                                    if let Type::Arrow(v, t) = uv.as_ref() {
                                        let pairs = vec![
                                            (*right.clone(), Type::Arrow(u.clone(), v.clone())),
                                            (*arg.clone(), *u.clone()),
                                            (*t.clone(), *res.clone()),
                                        ];
                                        let first = pairs[0].clone();
                                        stack.push(Frame::AllPairs { pairs, idx: 0 });
                                        stack.push(Frame::Check(first.0, first.1));
                                        continue;
                                    }
                                }
                            }
                            results.push(false);
                        }
                        (Type::Leaf, Type::Bool) => results.push(true),
                        (Type::Stem(inner), Type::Bool) => {
                            if matches!(inner.as_ref(), Type::Leaf) {
                                results.push(true);
                            } else {
                                stack.push(Frame::Check(*inner.clone(), Type::Leaf));
                            }
                        }
                        (Type::Bool, Type::Bool) => results.push(true),
                        _ => results.push(false),
                    }
                }
                Frame::AllPairs { pairs, idx } => {
                    let left = results.pop().unwrap_or(false);
                    if !left {
                        results.push(false);
                    } else {
                        let next = idx + 1;
                        if next >= pairs.len() {
                            results.push(true);
                        } else {
                            let pair = pairs[next].clone();
                            stack.push(Frame::AllPairs { pairs, idx: next });
                            stack.push(Frame::Check(pair.0, pair.1));
                        }
                    }
                }
                Frame::AnyPairs { pairs, idx } => {
                    let left = results.pop().unwrap_or(false);
                    if left {
                        results.push(true);
                    } else {
                        let next = idx + 1;
                        if next >= pairs.len() {
                            results.push(false);
                        } else {
                            let pair = pairs[next].clone();
                            stack.push(Frame::AnyPairs { pairs, idx: next });
                            stack.push(Frame::Check(pair.0, pair.1));
                        }
                    }
                }
            }
        }

        results.pop().unwrap_or(false)
    }
    
    pub fn resolve_type(&self, ty: Type) -> Type {
        enum Frame {
            Enter(Type),
            ExitArrow,
            ExitStem,
            ExitPair,
            ExitUnion(usize),
            ExitRec(usize),
            ExitForall(String),
        }

        let mut stack = vec![Frame::Enter(ty)];
        let mut results: Vec<Type> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(t) => match t {
                    Type::Var(id) => {
                        if let Some(sub) = self.substitutions.get(&id) {
                            stack.push(Frame::Enter(sub.clone()));
                        } else {
                            results.push(Type::Var(id));
                        }
                    }
                    Type::Arrow(a, b) => {
                        stack.push(Frame::ExitArrow);
                        stack.push(Frame::Enter(*b));
                        stack.push(Frame::Enter(*a));
                    }
                    Type::Stem(a) => {
                        stack.push(Frame::ExitStem);
                        stack.push(Frame::Enter(*a));
                    }
                    Type::Pair(a, b) => {
                        stack.push(Frame::ExitPair);
                        stack.push(Frame::Enter(*b));
                        stack.push(Frame::Enter(*a));
                    }
                    Type::Union(ts) => {
                        stack.push(Frame::ExitUnion(ts.len()));
                        for t in ts.into_iter().rev() {
                            stack.push(Frame::Enter(t));
                        }
                    }
                    Type::Rec(id, body) => {
                        stack.push(Frame::ExitRec(id));
                        stack.push(Frame::Enter(*body));
                    }
                    Type::Forall(name, body) => {
                        stack.push(Frame::ExitForall(name));
                        stack.push(Frame::Enter(*body));
                    }
                    _ => results.push(t),
                },
                Frame::ExitArrow => {
                    let b = results.pop().expect("missing arrow rhs");
                    let a = results.pop().expect("missing arrow lhs");
                    results.push(Type::Arrow(Box::new(a), Box::new(b)));
                }
                Frame::ExitStem => {
                    let a = results.pop().expect("missing stem inner");
                    results.push(Type::Stem(Box::new(a)));
                }
                Frame::ExitPair => {
                    let b = results.pop().expect("missing pair rhs");
                    let a = results.pop().expect("missing pair lhs");
                    results.push(Type::Pair(Box::new(a), Box::new(b)));
                }
                Frame::ExitUnion(count) => {
                    let mut ts = Vec::with_capacity(count);
                    for _ in 0..count {
                        ts.push(results.pop().expect("missing union member"));
                    }
                    ts.reverse();
                    results.push(Type::Union(ts));
                }
                Frame::ExitRec(id) => {
                    let body = results.pop().expect("missing rec body");
                    results.push(Type::Rec(id, Box::new(body)));
                }
                Frame::ExitForall(name) => {
                    let body = results.pop().expect("missing forall body");
                    results.push(Type::Forall(name, Box::new(body)));
                }
            }
        }

        results.pop().unwrap_or(Type::Var(0))
    }
    
    fn unify(&mut self, t1: Type, t2: Type) -> Result<(), String> {
        enum Frame {
            Unify(Type, Type),
            AllPairs { pairs: Vec<(Type, Type)>, idx: usize },
            AnyUnify { t: Type, members: Vec<Type>, idx: usize },
        }

        let mut stack = vec![Frame::Unify(t1, t2)];
        let mut results: Vec<bool> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Unify(a, b) => match (a, b) {
                    (Type::Var(i), Type::Var(j)) if i == j => results.push(true),
                    (Type::Var(i), t) | (t, Type::Var(i)) => {
                        if self.occurs_in(i, &t) {
                            results.push(false);
                        } else {
                            self.substitutions.insert(i, t.clone());
                            self.substitute(i, &t);
                            results.push(true);
                        }
                    }
                    (Type::Leaf, Type::Leaf) => results.push(true),
                    (Type::Float, Type::Float) => results.push(true),
                    (Type::Bool, Type::Bool) => results.push(true),
                    (Type::Arrow(a1, r1), Type::Arrow(a2, r2)) => {
                        let pairs = vec![(*a1, *a2), (*r1, *r2)];
                        let first = pairs[0].clone();
                        stack.push(Frame::AllPairs { pairs, idx: 0 });
                        stack.push(Frame::Unify(first.0, first.1));
                    }
                    (Type::Stem(i1), Type::Stem(i2)) => {
                        stack.push(Frame::Unify(*i1, *i2));
                    }
                    (Type::Pair(a1, b1), Type::Pair(a2, b2)) => {
                        let pairs = vec![(*a1, *a2), (*b1, *b2)];
                        let first = pairs[0].clone();
                        stack.push(Frame::AllPairs { pairs, idx: 0 });
                        stack.push(Frame::Unify(first.0, first.1));
                    }
                    (t, Type::Union(ts)) | (Type::Union(ts), t) => {
                        let mut visited = std::collections::HashSet::new();
                        let mut ok = false;
                        for member in &ts {
                            if self.is_subtype(t.clone(), member.clone(), &mut visited) {
                                ok = true;
                                break;
                            }
                        }
                        if ok {
                            results.push(true);
                        } else if ts.is_empty() {
                            results.push(false);
                        } else {
                            let members = ts;
                            let first = members[0].clone();
                            stack.push(Frame::AnyUnify { t: t.clone(), members, idx: 0 });
                            stack.push(Frame::Unify(t, first));
                        }
                    }
                    _ => results.push(false),
                },
                Frame::AllPairs { pairs, idx } => {
                    let left = results.pop().unwrap_or(false);
                    if !left {
                        results.push(false);
                    } else {
                        let next = idx + 1;
                        if next >= pairs.len() {
                            results.push(true);
                        } else {
                            let pair = pairs[next].clone();
                            stack.push(Frame::AllPairs { pairs, idx: next });
                            stack.push(Frame::Unify(pair.0, pair.1));
                        }
                    }
                }
                Frame::AnyUnify { t, members, idx } => {
                    let left = results.pop().unwrap_or(false);
                    if left {
                        results.push(true);
                    } else {
                        let next = idx + 1;
                        if next >= members.len() {
                            results.push(false);
                        } else {
                            let member = members[next].clone();
                            stack.push(Frame::AnyUnify { t: t.clone(), members, idx: next });
                            stack.push(Frame::Unify(t, member));
                        }
                    }
                }
            }
        }

        if results.pop().unwrap_or(false) {
            Ok(())
        } else {
            Err("Type Mismatch".to_string())
        }
    }
    
    fn occurs_in(&self, var: usize, ty: &Type) -> bool {
        let mut stack = vec![ty];
        while let Some(t) = stack.pop() {
            match t {
                Type::Var(i) => {
                    if *i == var {
                        return true;
                    }
                }
                Type::Arrow(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                Type::Stem(a) => {
                    stack.push(a);
                }
                Type::Pair(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                Type::Union(ts) => {
                    for t in ts {
                        stack.push(t);
                    }
                }
                Type::Rec(_, body) => {
                    stack.push(body);
                }
                Type::Forall(_, body) => {
                    stack.push(body);
                }
                _ => {}
            }
        }
        false
    }
    
    fn substitute(&mut self, var: usize, replacement: &Type) {
        for c in &mut self.constraints {
            match c {
                Constraint::Equality(a, b) => {
                    *a = Self::subst_ty(a, var, replacement);
                    *b = Self::subst_ty(b, var, replacement);
                }
                Constraint::Applicable(f, a, r) => {
                    *f = Self::subst_ty(f, var, replacement);
                    *a = Self::subst_ty(a, var, replacement);
                    *r = Self::subst_ty(r, var, replacement);
                }
                Constraint::Subtype(a, b) => {
                    *a = Self::subst_ty(a, var, replacement);
                    *b = Self::subst_ty(b, var, replacement);
                }
            }
        }
    }
    
    fn subst_ty(ty: &mut Type, var: usize, replacement: &Type) -> Type {
        enum Frame<'a> {
            Enter(&'a Type),
            ExitArrow,
            ExitStem,
            ExitPair,
            ExitForall(String),
        }

        let mut stack = vec![Frame::Enter(ty)];
        let mut results: Vec<Type> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(t) => match t {
                    Type::Var(i) if *i == var => results.push(replacement.clone()),
                    Type::Arrow(a, b) => {
                        stack.push(Frame::ExitArrow);
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Enter(a));
                    }
                    Type::Stem(a) => {
                        stack.push(Frame::ExitStem);
                        stack.push(Frame::Enter(a));
                    }
                    Type::Pair(a, b) => {
                        stack.push(Frame::ExitPair);
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Enter(a));
                    }
                    Type::Forall(n, body) => {
                        stack.push(Frame::ExitForall(n.clone()));
                        stack.push(Frame::Enter(body));
                    }
                    _ => results.push(t.clone()),
                },
                Frame::ExitArrow => {
                    let b = results.pop().expect("missing arrow rhs");
                    let a = results.pop().expect("missing arrow lhs");
                    results.push(Type::Arrow(Box::new(a), Box::new(b)));
                }
                Frame::ExitStem => {
                    let a = results.pop().expect("missing stem inner");
                    results.push(Type::Stem(Box::new(a)));
                }
                Frame::ExitPair => {
                    let b = results.pop().expect("missing pair rhs");
                    let a = results.pop().expect("missing pair lhs");
                    results.push(Type::Pair(Box::new(a), Box::new(b)));
                }
                Frame::ExitForall(n) => {
                    let body = results.pop().expect("missing forall body");
                    results.push(Type::Forall(n, Box::new(body)));
                }
            }
        }

        results.pop().unwrap_or_else(|| ty.clone())
    }

    fn get_number_type(&mut self) -> Type {
        // All numbers are Tagged Values: Pair(Tag, Pair(Value, KK))
        // Tag = Leaf (for numeric types)
        // KK = Pair(Leaf, Leaf) (continuation structure)
        // Value = Any (Variable) - covers Float payload or IntTree
        
        let val_var = self.fresh_var();
        
        let kk = Type::Pair(Box::new(Type::Leaf), Box::new(Type::Leaf));
        let tagged_payload = Type::Pair(
            Box::new(val_var),
            Box::new(kk)
        );
        
        // TaggedNumber = Pair(Leaf, Pair(Value, Pair(Leaf, Leaf)))
        Type::Pair(
            Box::new(Type::Leaf), // Tag
            Box::new(tagged_payload)
        )
    }

    fn get_primitive_type(&mut self, p: Primitive) -> Type {
        match p {
            Primitive::Add | Primitive::Sub | Primitive::Mul | Primitive::Div => {
                let n1 = self.get_number_type();
                let n2 = self.get_number_type();
                let n3 = self.get_number_type();
                
                Type::Arrow(
                    Box::new(n1),
                    Box::new(Type::Arrow(Box::new(n2), Box::new(n3)))
                )
            }
            Primitive::I => {
                let a = "A".to_string();
                let var_a = Type::Generic(a.clone());
                // I: ∀A. A -> A
                Type::Forall(a, Box::new(Type::Arrow(Box::new(var_a.clone()), Box::new(var_a))))
            }
            Primitive::K => {
                let a = "A".to_string();
                let b = "B".to_string();
                let var_a = Type::Generic(a.clone());
                let var_b = Type::Generic(b.clone());
                // K: ∀A.∀B. A -> B -> A
                Type::Forall(a, Box::new(Type::Forall(b, Box::new(
                    Type::Arrow(Box::new(var_a.clone()), Box::new(Type::Arrow(Box::new(var_b), Box::new(var_a))))
                ))))
            }
            Primitive::S => {
                let a = "A".to_string();
                let b = "B".to_string();
                let c = "C".to_string();
                let var_a = Type::Generic(a.clone());
                let var_b = Type::Generic(b.clone());
                let var_c = Type::Generic(c.clone());
                
                // S: ∀A.∀B.∀C. (A->B->C) -> (A->B) -> A->C
                // A->B->C
                let abc = Type::Arrow(Box::new(var_a.clone()), Box::new(Type::Arrow(Box::new(var_b.clone()), Box::new(var_c.clone()))));
                // A->B
                let ab = Type::Arrow(Box::new(var_a.clone()), Box::new(var_b));
                // A->C
                let ac = Type::Arrow(Box::new(var_a), Box::new(var_c));
                
                Type::Forall(a, Box::new(Type::Forall(b, Box::new(Type::Forall(c, Box::new(
                    Type::Arrow(Box::new(abc), Box::new(Type::Arrow(Box::new(ab), Box::new(ac))))
                ))))))
            }
            Primitive::First => {
                let a = "A".to_string();
                let b = "B".to_string();
                let var_a = Type::Generic(a.clone());
                let var_b = Type::Generic(b.clone());
                // First: ∀A.∀B. Pair(A, B) -> A
                Type::Forall(a, Box::new(Type::Forall(b, Box::new(
                   Type::Arrow(Box::new(Type::Pair(Box::new(var_a.clone()), Box::new(var_b))), Box::new(var_a))
                ))))
            }
            Primitive::Rest => {
               let a = "A".to_string();
               let b = "B".to_string();
               let var_a = Type::Generic(a.clone());
               let var_b = Type::Generic(b.clone());
               // Rest: ∀A.∀B. Pair(A, B) -> B
               Type::Forall(a, Box::new(Type::Forall(b, Box::new(
                  Type::Arrow(Box::new(Type::Pair(Box::new(var_a), Box::new(var_b.clone()))), Box::new(var_b))
               ))))
            }
            _ => Type::Leaf, 
        }
    }
    pub fn propagate_type_constraints(ty: &Type) -> (Type, Type, Type) {
        // Returns (StemChild, ForkLeft, ForkRight) constraints
        // Type::Var(0) represents "Any" / Unconstrained
        let any = || Type::Var(0);
        
        enum Frame<'a> {
            Enter(&'a Type),
            ExitUnion(usize),
        }

        let mut stack = vec![Frame::Enter(ty)];
        let mut results: Vec<(Type, Type, Type)> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(t) => match t {
                    Type::Leaf => results.push((any(), any(), any())),
                    Type::Stem(inner) => results.push((inner.as_ref().clone(), any(), any())),
                    Type::Pair(l, r) => results.push((any(), l.as_ref().clone(), r.as_ref().clone())),
                    Type::Arrow(_, _) => results.push((any(), any(), any())),
                    Type::Bool => results.push((Type::Leaf, any(), any())),
                    Type::Union(ts) => {
                        stack.push(Frame::ExitUnion(ts.len()));
                        for t in ts.iter().rev() {
                            stack.push(Frame::Enter(t));
                        }
                    }
                    Type::Forall(_, body) => {
                        stack.push(Frame::Enter(body));
                    }
                    _ => results.push((any(), any(), any())),
                },
                Frame::ExitUnion(count) => {
                    let mut stems = Vec::with_capacity(count);
                    let mut lefts = Vec::with_capacity(count);
                    let mut rights = Vec::with_capacity(count);
                    for _ in 0..count {
                        let (s, l, r) = results.pop().expect("missing union constraint");
                        stems.push(s);
                        lefts.push(l);
                        rights.push(r);
                    }
                    stems.reverse();
                    lefts.reverse();
                    rights.reverse();
                    results.push((Type::Union(stems), Type::Union(lefts), Type::Union(rights)));
                }
            }
        }

        results.pop().unwrap_or_else(|| (any(), any(), any()))
    }

    pub fn get_structural_mask(ty: &Type) -> [f64; 3] {
        let mut stack: Vec<&Type> = vec![ty];
        let mut mask = [-f64::INFINITY; 3];

        while let Some(t) = stack.pop() {
            match t {
                Type::Leaf => {
                    mask[0] = mask[0].max(0.0);
                }
                Type::Stem(_) => {
                    mask[1] = mask[1].max(0.0);
                }
                Type::Pair(_, _) => {
                    mask[2] = mask[2].max(0.0);
                }
                Type::Arrow(_, _) | Type::Rec(_, _) | Type::Var(_) | Type::RecVar(_) | Type::Forall(_, _) | Type::Generic(_) => {
                    for i in 0..3 {
                        mask[i] = mask[i].max(0.0);
                    }
                }
                Type::Bool => {
                    mask[0] = mask[0].max(0.0);
                    mask[1] = mask[1].max(0.0);
                }
                Type::Union(ts) => {
                    for inner in ts {
                        stack.push(inner);
                    }
                }
                _ => {
                    for i in 0..3 {
                        mask[i] = mask[i].max(0.0);
                    }
                }
            }
        }

        mask
    }
}
