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
        match self {
            Type::Var(id) => write!(f, "α{}", id),
            Type::Leaf => write!(f, "Δ"),
            Type::Stem(inner) => write!(f, "Stem({:?})", inner),
            Type::Pair(a, b) => write!(f, "Pair({:?}, {:?})", a, b),
            Type::Arrow(a, b) => write!(f, "({:?} -> {:?})", a, b),
            Type::Union(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 { write!(f, " | ")?; }
                    write!(f, "{:?}", t)?;
                }
                write!(f, ")")
            }
            Type::Rec(id, body) => write!(f, "μX{}.{:?}", id, body),
            Type::RecVar(id) => write!(f, "X{}", id),
            Type::Float => write!(f, "Float"),
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
            Type::Str => write!(f, "String"),
            Type::Char => write!(f, "Char"),
            Type::Forall(var, body) => write!(f, "∀{}.{:?}", var, body),
            Type::Generic(name) => write!(f, "{}", name),
            _ => write!(f, "?"), // Intersection not fully supported in display yet
        }
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
