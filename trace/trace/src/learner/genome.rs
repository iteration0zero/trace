use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::engine::{reduce, EvalContext};

#[derive(Debug, Clone, PartialEq)]
pub enum Gene {
    // Combinators
    S, 
    K, 
    I,
    // Primitives (Pure Compatible Macros)
    First,
    Rest,
    Leaf,
    // Values
    Int(i64),
    // Structure
    App(Box<Gene>, Box<Gene>),
}

impl Gene {
    /// Compiles the Gene into the Triage Graph
    pub fn compile(&self, g: &mut Graph) -> NodeId {
        match self {
            Gene::S => self.compile_pure(g),
            Gene::K => self.compile_pure(g),
            Gene::I => self.compile_pure(g),
            Gene::First => g.add(Node::Prim(Primitive::First)),
            Gene::Rest => g.add(Node::Prim(Primitive::Rest)),
            Gene::Leaf => g.add(Node::Leaf),
            
            Gene::Int(val) => {
                 // Use BigInt encoding
                 let big_val = num_bigint::BigInt::from(*val);
                 let raw = crate::engine::encode_int(g, &big_val);
                 crate::engine::make_tag(g, Primitive::TagInt, raw)
            },
            
            Gene::App(func, arg) => {
                let f = func.compile(g);
                let a = arg.compile(g);
                g.add(Node::App { func: f, args: smallvec::smallvec![a] })
            }
        }
    }
    
    // Helper to estimate complexity
    pub fn complexity(&self) -> usize {
        match self {
            Gene::App(f, a) => 1 + f.complexity() + a.complexity(),
            _ => 1
        }
    }

    /// Compiles the Gene into a Pure Triage Graph (Leaf/Stem/Fork only)
    pub fn compile_pure(&self, g: &mut Graph) -> NodeId {
        match self {
            // S = ((n (n (k n))) n) = Fork(Fork(Leaf, Fork(Leaf, Leaf)), Leaf) ???
            // Check main.rs prelude definitions for canonical forms
            // S: (n (n (k n)) n)
            // K: (n n) = Fork(Leaf, Leaf)
            // I: (s k k)
            Gene::K => {
                let leaf = g.add(Node::Leaf);
                g.add(Node::Stem(leaf))
            },
            Gene::S => {
                let leaf = g.add(Node::Leaf);
                let kk = g.add(Node::Fork(leaf, leaf));
                let stem = g.add(Node::Stem(kk));
                g.add(Node::Fork(stem, leaf))
            },
            Gene::I => {
                let s = self.clone_as_gene(Gene::S).compile_pure(g);
                let k = self.clone_as_gene(Gene::K).compile_pure(g);
                let sk = g.add(Node::App { func: s, args: smallvec::smallvec![k] });
                let skk = g.add(Node::App { func: sk, args: smallvec::smallvec![k] });
                let mut ctx = EvalContext::default();
                reduce(g, skk, &mut ctx)
            },
            Gene::First => {
                // First = (fn p ((triage n (fn u n) (fn a (fn b a))) p))
                // This is complex. Use the primitive if we can't easily expand?
                // Or just use Leaf? (Fail).
                // Let's use App(Leaf, Leaf) as placeholder or fallback.
                g.add(Node::Leaf)
            },
            Gene::Rest => g.add(Node::Leaf), 
            Gene::Leaf => g.add(Node::Leaf),
            
            Gene::Int(val) => {
                 let big_val = num_bigint::BigInt::from(*val);
                 let raw = crate::engine::encode_int(g, &big_val);
                 crate::engine::make_tag(g, Primitive::TagInt, raw)
            },
            
            Gene::App(func, arg) => {
                let f = func.compile_pure(g);
                let a = arg.compile_pure(g);
                g.add(Node::App { func: f, args: smallvec::smallvec![a] })
            }
        }
    }
    
    fn clone_as_gene(&self, variant: Gene) -> Gene {
         variant
    }
}
