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
        match ty {
            Type::Generic(n) if n == name => replacement.clone(),
            Type::Forall(n, body) => {
                if n == name {
                    Type::Forall(n.clone(), body.clone())
                } else {
                    Type::Forall(n.clone(), Box::new(Self::subst_generic(body, name, replacement)))
                }
            }
            Type::Arrow(a, b) => Type::Arrow(
                Box::new(Self::subst_generic(a, name, replacement)),
                Box::new(Self::subst_generic(b, name, replacement))
            ),
            Type::Stem(a) => Type::Stem(Box::new(Self::subst_generic(a, name, replacement))),
            Type::Pair(a, b) => Type::Pair(
                Box::new(Self::subst_generic(a, name, replacement)),
                Box::new(Self::subst_generic(b, name, replacement))
            ),
            Type::Union(ts) => Type::Union(ts.iter().map(|t| Self::subst_generic(t, name, replacement)).collect()),
            Type::Rec(_id, body) => Type::Rec(*_id, Box::new(Self::subst_generic(body, name, replacement))),
            _ => ty.clone(),
        }
    }

    fn subst_rec(ty: &Type, id: usize, replacement: &Type) -> Type {
        match ty {
            Type::RecVar(i) if *i == id => replacement.clone(),
            Type::Rec(i, body) => {
                // If nested Rec shadows ID, stop.
                if *i == id {
                    Type::Rec(*i, body.clone())
                } else {
                    Type::Rec(*i, Box::new(Self::subst_rec(body, id, replacement)))
                }
            }
            Type::Arrow(a, b) => Type::Arrow(
                Box::new(Self::subst_rec(a, id, replacement)),
                Box::new(Self::subst_rec(b, id, replacement))
            ),
            Type::Stem(a) => Type::Stem(Box::new(Self::subst_rec(a, id, replacement))),
            Type::Pair(a, b) => Type::Pair(
                Box::new(Self::subst_rec(a, id, replacement)),
                Box::new(Self::subst_rec(b, id, replacement))
            ),
            Type::Union(ts) => Type::Union(ts.iter().map(|t| Self::subst_rec(t, id, replacement)).collect()),
            Type::Forall(n, body) => Type::Forall(n.clone(), Box::new(Self::subst_rec(body, id, replacement))),
            _ => ty.clone(),
        }
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
        if node_types.contains_key(&node) {
            return Ok(());
        }
        
        let ty = self.fresh_var();
        node_types.insert(node, ty.clone());

        // Special-case typings for canonical tree combinators
        if let Some(special_ty) = env.specials.get(&node.0).cloned() {
            self.constraints.push(Constraint::Equality(ty, special_ty));
            return Ok(());
        }
        
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
         if t1 == t2 { return true; }
         if visited.contains(&(t1.clone(), t2.clone())) { return true; } // Cycle detection
         visited.insert((t1.clone(), t2.clone()));
         
         match (t1, t2) {
             (Type::Leaf, Type::Leaf) => true,
             
             // Rec Unfolding
             (Type::Rec(id, body), t2) => {
                 let rec_ty = Type::Rec(id, body.clone());
                 let unrolled = Self::subst_rec(&body, id, &rec_ty);
                 self.is_subtype(unrolled, t2, visited)
             },
             (t1, Type::Rec(id, body)) => {
                 // Unrolling right:
                 let rec_ty = Type::Rec(id, body.clone());
                 let unrolled = Self::subst_rec(&body, id, &rec_ty);
                 self.is_subtype(t1, unrolled, visited)
             },
             
             // Forall Instantiation (Left)
             // ∀X. T <= U  if  T[alpha/X] <= U  (where alpha is fresh existential/unification var)
             (Type::Forall(name, body), t2) => {
                 let alpha = self.fresh_var();
                 let instantiated = Self::subst_generic(&body, &name, &alpha);
                 self.is_subtype(instantiated, t2, visited)
             },
             
             // Forall Generalization (Right)
             // U <= ∀X. T  if  U <= T[skolem/X] (where skolem is fresh constant)
             (t1, Type::Forall(name, body)) => {
                 let skolem = self.fresh_skolem(&name);
                 let instantiated = Self::subst_generic(&body, &name, &skolem);
                 self.is_subtype(t1, instantiated, visited)
             },
             
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
             (Type::Generic(a), Type::Generic(b)) => a == b,

             // Arrow Contravariance: A1->R1 <= A2->R2 if A2 <= A1 AND R1 <= R2
             (Type::Arrow(a1, r1), Type::Arrow(a2, r2)) => {
                 self.is_subtype(*a2, *a1, visited) && self.is_subtype(*r1, *r2, visited)
             },
             
             // Structure Preserving
             (Type::Stem(a), Type::Stem(b)) => self.is_subtype(*a, *b, visited),
             (Type::Pair(a1, b1), Type::Pair(a2, b2)) => {
                 self.is_subtype(*a1, *a2, visited) && self.is_subtype(*b1, *b2, visited)
             },

             // Leaf < U -> Stem(U)
             (Type::Leaf, Type::Arrow(a, b)) => {
                 let stem_a = Type::Stem(a);
                 self.is_subtype(stem_a, *b, visited)
             },

             // Stem(U) < V -> Pair(U, V)
             (Type::Stem(inner), Type::Arrow(arg, res)) => {
                 let pair = Type::Pair(inner, arg);
                 self.is_subtype(pair, *res, visited)
             },

             // Fork subtyping axioms for triage calculus
             (Type::Pair(left, right), Type::Arrow(arg, res)) => {
                 // K axiom: F L U < V -> U
                 if matches!(left.as_ref(), Type::Leaf) {
                     return self.is_subtype(*right.clone(), *res, visited);
                 }

                 // Leaf-case triage: F (F T V) W < L -> T
                 if matches!(arg.as_ref(), Type::Leaf) {
                     if let Type::Pair(t, _v) = left.as_ref() {
                         return self.is_subtype(*t.clone(), *res, visited);
                     }
                 }

                 // Stem-case triage: F (F U (V -> T)) W < S V -> T
                 if let Type::Stem(v_arg) = arg.as_ref() {
                     if let Type::Pair(_u, v_arrow) = left.as_ref() {
                         if let Type::Arrow(v, t) = v_arrow.as_ref() {
                             return self.is_subtype(*v_arg.clone(), *v.clone(), visited)
                                 && self.is_subtype(*t.clone(), *res, visited);
                         }
                     }
                 }

                 // Fork-case triage: F (F U V) (W1 -> W2 -> T) < F W1 W2 -> T
                 if let Type::Pair(w1, w2) = arg.as_ref() {
                     if let Type::Pair(_u, _v) = left.as_ref() {
                         if let Type::Arrow(w1_t, w2_arrow) = right.as_ref() {
                             if let Type::Arrow(w2_t, t) = w2_arrow.as_ref() {
                                 return self.is_subtype(*w1.clone(), *w1_t.clone(), visited)
                                     && self.is_subtype(*w2.clone(), *w2_t.clone(), visited)
                                     && self.is_subtype(*t.clone(), *res, visited);
                             }
                         }
                     }
                 }

                 // S axiom: F (S (U -> V -> T)) (U -> V) < U -> T
                 if let Type::Stem(s_inner) = left.as_ref() {
                     if let Type::Arrow(u, uv) = s_inner.as_ref() {
                         if let Type::Arrow(v, t) = uv.as_ref() {
                             let s_ok = self.is_subtype(*right.clone(), Type::Arrow(u.clone(), v.clone()), visited)
                                 && self.is_subtype(*arg.clone(), *u.clone(), visited)
                                 && self.is_subtype(*t.clone(), *res, visited);
                             if s_ok {
                                 return true;
                             }
                         }
                     }
                 }

                 false
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
            Type::Forall(name, body) => Type::Forall(name, Box::new(self.resolve_type(*body))),
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
            
            // Structural/function relationships are handled via subtyping axioms.
            
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
            Type::Forall(_, body) => self.occurs_in(var, body),
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
            Type::Forall(n, body) => Type::Forall(n.clone(), Box::new(Self::subst_ty(body, var, replacement))),
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
            Type::Forall(_, body) => Self::propagate_type_constraints(body),
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
            Type::Rec(_, _) | Type::Var(_) | Type::RecVar(_) | Type::Forall(_, _) | Type::Generic(_) => [0.0, 0.0, 0.0],
            _ => [0.0, 0.0, 0.0]
        }
    }
}
