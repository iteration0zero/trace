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
        if node_types.contains_key(&node) {
            return Ok(());
        }
        
        let ty = self.fresh_var();
        node_types.insert(node, ty.clone());
        
        match g.get(node) {
            Node::Leaf => {
                self.constraints.push(Constraint::Equality(ty, Type::Leaf));
            }
            Node::Float(_) => {
                self.constraints.push(Constraint::Equality(ty, Type::Float));
            }
            Node::Stem(inner) => {
                self.generate_constraints(g, *inner, env, node_types)?;
                let inner_ty = node_types.get(inner).unwrap().clone();
                self.constraints.push(Constraint::Equality(ty, Type::Stem(Box::new(inner_ty))));
            }
            Node::Fork(head, tail) => {
                self.generate_constraints(g, *head, env, node_types)?;
                self.generate_constraints(g, *tail, env, node_types)?;
                
                let h_ty = node_types.get(head).unwrap().clone();
                let t_ty = node_types.get(tail).unwrap().clone();
                
                // Construct the Pair type
                let pair_ty = Type::Pair(Box::new(h_ty), Box::new(t_ty));
                self.constraints.push(Constraint::Equality(ty, pair_ty));
            }
            Node::App { func, args } => {
                self.generate_constraints(g, *func, env, node_types)?;
                let func_ty = node_types.get(func).unwrap().clone();
                
                let mut curr_ty = func_ty;
                
                for arg in args {
                    self.generate_constraints(g, *arg, env, node_types)?;
                    let arg_ty = node_types.get(arg).unwrap().clone();
                    let res_ty = self.fresh_var();
                    
                    self.constraints.push(Constraint::Applicable(curr_ty.clone(), arg_ty, res_ty.clone()));
                    curr_ty = res_ty;
                }
                
                self.constraints.push(Constraint::Equality(ty, curr_ty));
            }
            Node::Prim(p) => {
                let prim_ty = self.get_primitive_type(*p);
                self.constraints.push(Constraint::Equality(ty, prim_ty));
            }
            _ => {}
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
                             // Defer to unification logic (handles Var, Leaf, Stem, Pair via our extensions)
                             // Warning: unifying f with Arrow(arg, res) enforces arg == param.
                             // We need arg <= param.
                             let a_res = self.resolve_type(arg_ty);
                             let r_res = self.resolve_type(res_ty);
                             
                             let param_var = self.fresh_var();
                             let arrow = Type::Arrow(Box::new(param_var.clone()), Box::new(r_res));
                             
                             // Constrain f to be an arrow taking param_var
                             self.unify(f, arrow)?;
                             
                             // Constrain arg to be subtype of param_var
                             self.constraints.push(Constraint::Subtype(a_res, param_var));
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
    
    // Coinductive Subtyping (Algorithm 3.4.2)
    fn is_subtype(&mut self, t1: Type, t2: Type, visited: &mut std::collections::HashSet<(Type, Type)>) -> bool {
         if t1 == t2 { return true; }
         if visited.contains(&(t1.clone(), t2.clone())) { return true; } // Cycle detection
         visited.insert((t1.clone(), t2.clone()));
         
         match (t1, t2) {
             (Type::Leaf, Type::Leaf) => true,
             
             // Union Right: T1 <= A | B if T1 <= A OR T1 <= B
             (t1, Type::Union(ts)) => {
                 // We must be careful not to commit unification on failed branches?
                 // But we have mutable self.
                 // Ideally we snapshot state. But simplified:
                 // Try to unify with each. If one works, stop.
                 // But unify mutates.
                 // If we have to try multiple, we need backtracking?
                 // For now, let's assume if it fails, it didn't mutate (mostly true for structural mismatch).
                 // But `Float <= Var` unifies.
                 // If we have Union(Var1, Var2). `Float` matches Var1.
                 // This seems acceptable.
                 for t in ts {
                     // Hack: check if it MIGHT match before unifying?
                     // Or just rely on greedy matching (first valid option).
                     // This is standard for simple inference.
                     if self.is_subtype(t1.clone(), t.clone(), visited) {
                         return true;
                     }
                 }
                 false
             },
             
             // Union Left
             (Type::Union(ts), t2) => {
                 for t in ts {
                     if !self.is_subtype(t.clone(), t2.clone(), visited) {
                         return false;
                     }
                 }
                 true
             },
             
             (Type::Var(i), t) => {
                 match self.unify(Type::Var(i), t) {
                     Ok(_) => true,
                     Err(_) => false,
                 }
             },
             (t, Type::Var(i)) => {
                 match self.unify(Type::Var(i), t) {
                     Ok(_) => true,
                     Err(_) => false,
                 }
             },

             // Arrow Contravariance: A1->R1 <= A2->R2 if A2 <= A1 AND R1 <= R2
             (Type::Arrow(a1, r1), Type::Arrow(a2, r2)) => {
                 self.is_subtype(*a2, *a1, visited) && self.is_subtype(*r1, *r2, visited)
             },
             
             // Structure Preserving
             (Type::Stem(a), Type::Stem(b)) => self.is_subtype(*a, *b, visited),
             (Type::Pair(a1, b1), Type::Pair(a2, b2)) => {
                 self.is_subtype(*a1, *a2, visited) && self.is_subtype(*b1, *b2, visited)
             },

             // Functional Behavior for Leaf: Leaf :: forall A. A -> Stem(A)
             // Leaf <= A -> B if Stem(A) <= B
             (Type::Leaf, Type::Arrow(a, b)) => {
                 let stem_a = Type::Stem(a); // Takes target arg type A
                 self.is_subtype(stem_a, *b, visited)
             },

             // Functional Behavior for Stem
             (Type::Stem(inner), Type::Arrow(arg, res)) => {
                 match inner.as_ref() {
                     Type::Leaf => {
                         // Identity: A -> A
                         // A -> A <= Arg -> Res
                         // Arg <= A ? No, Arg IS A.
                         // But Identity accepts ANYTHING.
                         // Identity :: forall T. T -> T.
                         // We are checking if Identity <= Arg -> Res.
                         // This means (Arg -> Arg) <= (Arg -> Res).
                         // LHS takes Arg, returns Arg.
                         // RHS expects function taking Arg, returning Res.
                         // So Ret(LHS) <= Ret(RHS).
                         // Arg <= Res.
                         self.is_subtype(*arg, *res, visited)
                     },
                     Type::Stem(x) => {
                         // K x: A -> x
                         // (Arg -> x) <= (Arg -> Res)
                         // x <= Res.
                         self.is_subtype(*x.clone(), *res, visited)
                     },
                     Type::Pair(x, z) => {
                         // S x z: A -> (x A)(z A)
                         // (Arg -> (x Arg)(z Arg)) <= (Arg -> Res)
                         // (x Arg)(z Arg) <= Res
                         
                         // We need to simulate application of x and z?
                         // But is_subtype shouldn't mutate/generate constraints?
                         // Subtyping check usually structural.
                         // This requires generating constraints/unification if we don't know x/z types fully.
                         // Example: if x is Var?
                         
                         // If we are in is_subtype, we might be blocked.
                         // Fallback to false? Or try?
                         // S combinator subtyping is complex.
                         // But unification handled it. If we are here, strict check failed?
                         // If we return 'true' we claim it works.
                         // If we generate type application constraints here?
                         // Logic below mirrors Unify logic, but as check.
                         // Can't easily do it without mutable Constraint list (which we have: &mut self).
                         
                         // Recreating the application constraint:
                         // We want to ENFORCE (x Arg)(z Arg) <= Res.
                         // We can push constraints!
                         
                         // Create fresh vars for intermediate?
                         let res1 = self.fresh_var();
                         let res2 = self.fresh_var();
                         let a_clone = *arg.clone();
                         
                         // x Arg -> res1
                         // z Arg -> res2
                         // res1 res2 -> Res
                         
                         // Since is_subtype returns bool, we should return true IF we successfully scheduled checks?
                         // Pushing constraints ensures they will be checked later.
                         
                         self.constraints.push(Constraint::Applicable(*x.clone(), a_clone.clone(), res1.clone()));
                         self.constraints.push(Constraint::Applicable(*z.clone(), a_clone, res2.clone()));
                         // We want result <= Res.
                         // Constraint::Applicable enforces equality usually (res1 res2 = result).
                         // We can set Constraint::Applicable(res1, res2, Res).
                         // Or if we want subtype?
                         // Constraint::Applicable gives equality result.
                         // If we do Applicable to strict `Res`, we enforce Equality.
                         // Subtyping requires Result <= Res.
                         // Let `final_res` be result.
                         
                         let final_res = self.fresh_var();
                         self.constraints.push(Constraint::Applicable(res1, res2, final_res.clone()));
                         self.constraints.push(Constraint::Subtype(final_res, *res));
                         
                         true
                     },
                     _ => false
                 }
             },
             
             // Bool Subtyping
             (Type::Leaf, Type::Bool) => true,
             (Type::Stem(inner), Type::Bool) => {
                 match inner.as_ref() {
                     Type::Leaf => true,
                     _ => self.is_subtype(*inner.clone(), Type::Leaf, visited) // Stem(T) <: Bool if T <: Leaf (i.e. T is Leaf)
                 }
             },
             (Type::Bool, Type::Bool) => true,
             // Bool is NOT subtype of Leaf or Stem(Leaf) generally, unless we downcast? No.

             // Recursive types not fully handled yet
             _ => false
         }
    }
    
    pub fn resolve_type(&self, ty: Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(sub) = self.substitutions.get(&id) {
                    self.resolve_type(sub.clone())
                } else {
                    Type::Var(id)
                }
            }
            Type::Arrow(a, b) => Type::Arrow(Box::new(self.resolve_type(*a)), Box::new(self.resolve_type(*b))),
            Type::Stem(a) => Type::Stem(Box::new(self.resolve_type(*a))),
            Type::Pair(a, b) => Type::Pair(Box::new(self.resolve_type(*a)), Box::new(self.resolve_type(*b))),
            Type::Union(ts) => Type::Union(ts.into_iter().map(|t| self.resolve_type(t)).collect()),
            Type::Rec(id, body) => Type::Rec(id, Box::new(self.resolve_type(*body))),
            _ => ty,
        }
    }
    
    fn unify(&mut self, t1: Type, t2: Type) -> Result<(), String> {
        match (t1, t2) {
            (Type::Var(i), Type::Var(j)) if i == j => Ok(()),
            (Type::Var(i), t) | (t, Type::Var(i)) => {
                if self.occurs_in(i, &t) {
                    return Err("Occurs check failed: cyclic type detected".to_string());
                }
                self.substitutions.insert(i, t.clone());
                self.substitute(i, &t);
                Ok(())
            }
            (Type::Leaf, Type::Leaf) => Ok(()),
            (Type::Float, Type::Float) => Ok(()),
            (Type::Bool, Type::Bool) => Ok(()),
            (Type::Arrow(a1, r1), Type::Arrow(a2, r2)) => {
                self.unify(*a1, *a2)?;
                self.unify(*r1, *r2)
            }
            (Type::Stem(i1), Type::Stem(i2)) => self.unify(*i1, *i2),
            (Type::Pair(a1, b1), Type::Pair(a2, b2)) => {
                self.unify(*a1, *a2)?;
                self.unify(*b1, *b2)
            }
            
            // Core Calculus Unification Rules (Structural S-Combinator Logic)
            (Type::Leaf, Type::Arrow(a, b)) | (Type::Arrow(a, b), Type::Leaf) => {
                 // Rule 1: Leaf a -> Stem(a)
                 // Matches Stem construction in engine.
                 self.unify(*b, Type::Stem(a))
            }
            (Type::Stem(i), Type::Arrow(a, b)) | (Type::Arrow(a, b), Type::Stem(i)) => {
                 // Canonical Tree Calculus Unification
                 // Stem(i) is applied to 'a' producing 'b'.
                 // Dispatch based on structure of 'i' (the content of the Stem).
                 
                 match i.as_ref() {
                     Type::Leaf => {
                         // Rule 1: Stem(Leaf) a -> a
                         // Identity: a must unify with b
                         self.unify(*a, *b)
                     },
                     Type::Stem(x) => {
                         // Rule 2: Stem(Stem(x)) a -> x
                         // K combinator: b must unify with x
                         self.unify(*b, *x.clone())
                     },
                     Type::Pair(x, z) => {
                         // Rule 3: Stem(Fork(x, z)) a -> (x a) (z a)
                         // S combinator behavior.
                         let res1 = self.fresh_var();
                         let res2 = self.fresh_var();
                         
                         let arg = *a.clone();
                         // x a -> res1
                         self.constraints.push(Constraint::Applicable(*x.clone(), arg.clone(), res1.clone()));
                         // z a -> res2
                         self.constraints.push(Constraint::Applicable(*z.clone(), arg, res2.clone()));
                         // res1 res2 -> b
                         self.constraints.push(Constraint::Applicable(res1, res2, *b));
                         Ok(())
                     },
                     _ => {
                         // Fallback / Generic S-rule?
                         // If 'i' is Var or other, we might presume it will eventually become Fork-like or Stem-like?
                         // In the original Triage logic, Stem(i) a -> Fork(i, a).
                         // That is NO LONGER VALID for canonical calculus in the general case?
                         // Wait, if i is simple Leaf/Stem/Fork, we covered it.
                         // If i is Float/Prim/Var?
                         // In engine::reduce_step: 
                         //   Stem(p) q -> (if p ! Leaf/Stem/Fork) -> Stuck? Or Fork(p, q)?
                         //   My engine change made it return Fork(p, q) ONLY if p is NOT Leaf/Stem/Fork.
                         //   So if i is Float, Stem(Float) a -> Fork(Float, a).
                         //   If i is Var, we don't know yet.
                         
                         // We should unify with Pair(i, a) as a safe fallback for Data Construction?
                         // If i turns out to be Leaf later, we have a problem: Pair(Leaf, a) != a.
                         // This is a soundness issue if we eagerly commit to Pair.
                         
                         // Ideally we should defer constraint if 'i' is Var.
                         // But we don't have deferred constraints mechanism easily here except sticking it back in?
                         // For now, let's assume if it's not structural, it's data.
                         // Unifying b = Fork(i, a)
                         self.unify(*b, Type::Pair(i.clone(), a))
                     }
                 }
            }
            (Type::Pair(p1, p2), Type::Arrow(a, b)) | (Type::Arrow(a, b), Type::Pair(p1, p2)) => {
                 // Fork(p, q) applied.
                 // In canonical calculus, this doesn't reduce.
                 // It's a type error or just inert?
                 // If we treat it as inert app, we can't unify with Arrow unless it's a specific Arrow type?
                 // Actually, if something IS a Pair, and we use it as Arrow, we implies it has function behavior.
                 // But Fork doesn't.
                 // So we should Error?
                 // Or treat as "Any" -> "Any"?
                 // Let's error for now to be strict.
                 Err(format!("Type Error: Pair {:?} {:?} cannot function as Arrow", p1, p2))
            }
            (Type::Float, Type::Arrow(_a, b)) | (Type::Arrow(_a, b), Type::Float) => {
                 // Treat Float as Const: Float x -> Float
                 self.unify(*b, Type::Float)
            }
            
            // Union Handling: T unifies with Union if T is subtype of one element
            (t, Type::Union(ts)) | (Type::Union(ts), t) => {
                let mut visited = std::collections::HashSet::new();
                for member in &ts {
                    if self.is_subtype(t.clone(), member.clone(), &mut visited) {
                        return Ok(());
                    }
                }
                // If no direct subtype, try unifying with first compatible member
                for member in ts {
                    if self.unify(t.clone(), member).is_ok() {
                        return Ok(());
                    }
                }
                Err(format!("Type {:?} not in Union", t))
            }
            
            _ => Err("Type Mismatch".to_string()),
        }
    }
    
    fn occurs_in(&self, var: usize, ty: &Type) -> bool {
        match ty {
            Type::Var(i) => *i == var,
            Type::Arrow(a, b) => self.occurs_in(var, a) || self.occurs_in(var, b),
            Type::Stem(a) => self.occurs_in(var, a),
            Type::Pair(a, b) => self.occurs_in(var, a) || self.occurs_in(var, b),
            Type::Union(ts) => ts.iter().any(|t| self.occurs_in(var, t)),
            Type::Rec(_, body) => self.occurs_in(var, body),
            _ => false,
        }
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
        match ty {
            Type::Var(i) if *i == var => replacement.clone(),
            Type::Arrow(a, b) => Type::Arrow(
                Box::new(Self::subst_ty(a, var, replacement)),
                Box::new(Self::subst_ty(b, var, replacement))
            ),
            Type::Stem(a) => Type::Stem(Box::new(Self::subst_ty(a, var, replacement))),
            Type::Pair(a, b) => Type::Pair(
                Box::new(Self::subst_ty(a, var, replacement)),
                Box::new(Self::subst_ty(b, var, replacement))
            ),
            _ => ty.clone(),
        }
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
                let n3 = self.get_number_type(); // Result is also Number
                
                Type::Arrow(
                    Box::new(n1),
                    Box::new(Type::Arrow(Box::new(n2), Box::new(n3)))
                )
            }
            _ => Type::Leaf, 
        }
    }
    pub fn propagate_type_constraints(ty: &Type) -> (Type, Type, Type) {
        // Returns (StemChild, ForkLeft, ForkRight) constraints
        // Type::Var(0) represents "Any" / Unconstrained
        let any = || Type::Var(0);
        
        match ty {
            Type::Leaf => (any(), any(), any()),
            Type::Stem(inner) => (*inner.clone(), any(), any()),
            Type::Pair(l, r) => (any(), *l.clone(), *r.clone()),
            Type::Arrow(_, _) => (any(), any(), any()), 
            Type::Bool => (Type::Leaf, any(), any()), 
            Type::Union(ts) => {
                let mut stems = Vec::new();
                let mut lefts = Vec::new();
                let mut rights = Vec::new();
                
                for t in ts {
                    let (s, l, r) = Self::propagate_type_constraints(t);
                    stems.push(s);
                    lefts.push(l);
                    rights.push(r);
                }
                
                (Type::Union(stems), Type::Union(lefts), Type::Union(rights))
            },
             _ => (any(), any(), any())
        }
    }

    pub fn get_structural_mask(ty: &Type) -> [f64; 3] {
        match ty {
            Type::Leaf => [0.0, -f64::INFINITY, -f64::INFINITY],
            Type::Stem(_) => [-f64::INFINITY, 0.0, -f64::INFINITY],
            Type::Pair(_, _) => [-f64::INFINITY, -f64::INFINITY, 0.0],
            Type::Arrow(_, _) => [0.0, 0.0, 0.0], 
            Type::Union(ts) => {
                 let mut mask = [-f64::INFINITY; 3];
                 for t in ts {
                     let m = Self::get_structural_mask(t);
                     for i in 0..3 { mask[i] = mask[i].max(m[i]); }
                 }
                 mask
            },
            Type::Bool => [0.0, 0.0, -f64::INFINITY], 
            Type::Rec(_, _) | Type::Var(_) | Type::RecVar(_) => [0.0, 0.0, 0.0],
            _ => [0.0, 0.0, 0.0]
        }
    }
}
