use std::fmt;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Var(usize),
    Leaf,              // nil / false (Δ)
    Stem(Box<Type>),   // (n x)
    Pair(Box<Type>, Box<Type>), // (n x y) / Fork
    Arrow(Box<Type>, Box<Type>),
    Union(Vec<Type>),
    Intersection(Vec<Type>),
    Rec(usize, Box<Type>), // μX. T
    RecVar(usize),         // X
    
    // Primitive Types
    Float,
    Int,               // BigInt
    Bool,              // Constrained Union: Leaf | Stem(Leaf)
    Str,               // String (TagStr)
    Char,              // Char (TagChar)
    
    // Explicit Polymorphism
    Forall(String, Box<Type>), // ∀X. T
    Generic(String),           // X (bound by Forall)
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Frame<'a> {
            Enter(&'a Type),
            Text(&'a str),
            Owned(String),
        }

        let mut out = String::new();
        let mut stack = vec![Frame::Enter(self)];

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Text(s) => out.push_str(s),
                Frame::Owned(s) => out.push_str(&s),
                Frame::Enter(t) => match t {
                    Type::Var(id) => out.push_str(&format!("α{}", id)),
                    Type::Leaf => out.push_str("Δ"),
                    Type::Stem(inner) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(inner));
                        stack.push(Frame::Text("Stem("));
                    }
                    Type::Pair(a, b) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Text(", "));
                        stack.push(Frame::Enter(a));
                        stack.push(Frame::Text("Pair("));
                    }
                    Type::Arrow(a, b) => {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Enter(b));
                        stack.push(Frame::Text(" -> "));
                        stack.push(Frame::Enter(a));
                        stack.push(Frame::Text("("));
                    }
                    Type::Union(ts) => {
                        stack.push(Frame::Text(")"));
                        for idx in (0..ts.len()).rev() {
                            stack.push(Frame::Enter(&ts[idx]));
                            if idx > 0 {
                                stack.push(Frame::Text(" | "));
                            }
                        }
                        stack.push(Frame::Text("("));
                    }
                    Type::Rec(id, body) => {
                        stack.push(Frame::Enter(body));
                        stack.push(Frame::Owned(format!("μX{}.", id)));
                    }
                    Type::RecVar(id) => out.push_str(&format!("X{}", id)),
                    Type::Float => out.push_str("Float"),
                    Type::Int => out.push_str("Int"),
                    Type::Bool => out.push_str("Bool"),
                    Type::Str => out.push_str("String"),
                    Type::Char => out.push_str("Char"),
                    Type::Forall(var, body) => {
                        stack.push(Frame::Enter(body));
                        stack.push(Frame::Owned(format!("∀{}.", var)));
                    }
                    Type::Generic(name) => out.push_str(name),
                    _ => out.push_str("?"), // Intersection not fully supported in display yet
                },
            }
        }

        f.write_str(&out)
    }
}

pub struct TypeEnv {
    // Map variable names to Types
    pub vars: std::collections::HashMap<String, Type>,
    // Special-case typings for canonical tree combinators (NodeId -> Type)
    pub specials: std::collections::HashMap<u32, Type>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self { 
            vars: std::collections::HashMap::new(),
            specials: std::collections::HashMap::new(),
        }
    }
}
