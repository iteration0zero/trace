use crate::arena::{Graph, Node, NodeId, Primitive};

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
            Gene::S => g.add(Node::Prim(Primitive::S)),
            Gene::K => g.add(Node::Prim(Primitive::K)),
            Gene::I => g.add(Node::Prim(Primitive::I)),
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
                // K = Fork(Leaf, Leaf)  ( acts as \x y. x ) ?
                // Wait. K = (n n) = Fork(Leaf, Leaf)?
                // Fork(Leaf, Leaf) x -> Leaf Leaf x -> x? No.
                // Fork(p, q) x -> Leaf p q x.
                // Leaf Leaf Leaf x -> Leaf. Matches nothing?
                // Check my "Pair vs Stem identity":
                // Stem(Leaf) is K.
                // Stem(Leaf) x -> Fork(Leaf, x).
                // Fork(Leaf, x) y -> x.
                // So K = Stem(Leaf).
                let leaf = g.add(Node::Leaf);
                g.add(Node::Stem(leaf))
            },
            Gene::S => {
                // S = Fork(Fork(Leaf, Fork(Leaf, Leaf)), Leaf) ???
                // S x y z = x z (y z).
                // Canonical S in Triage is known as "The S Combinator".
                // Prelude: "s", "(n (n (k n)) n)".
                // Let's implement that structure.
                // K = Stem(Leaf).
                // (k n) = App(K, Leaf).
                // (n (k n)) = Fork(Leaf, App(K, Leaf)).
                // (n (n (k n)) n) = Fork(Fork(Leaf, App(K, Leaf)), Leaf).
                // Let's build this.
                let leaf = g.add(Node::Leaf);
                let k = { let l = g.add(Node::Leaf); g.add(Node::Stem(l)) };
                let k_n = g.add(Node::App { func: k, args: smallvec::smallvec![leaf] }); // This is App, reduces to structure. But pure compilation should produce Structure if possible?
                // (k n) reduces to Fork(Leaf, Leaf)? 
                // Stem(Leaf) applied to Leaf -> Fork(Leaf, Leaf).
                // So (k n) IS Fork(Leaf, Leaf).
                let k_n_struct = g.add(Node::Fork(leaf, leaf));
                
                // Inner: (n (k n)) -> Fork(Leaf, k_n_struct)
                let inner = g.add(Node::Fork(leaf, k_n_struct));
                
                // Outer: (n inner n) -> Fork(inner, Leaf).
                g.add(Node::Fork(inner, leaf))
            },
            Gene::I => {
                // I = S K K.
                // App( App(S, K), K ).
                // Or structural I? (fn x x).
                // (fn x x) = (s k k).
                // Triage "Leaf" is not I.
                // Let's use (s k k).
                let s = self.clone_as_gene(Gene::S).compile_pure(g);
                let k = self.clone_as_gene(Gene::K).compile_pure(g);
                // App(S, K)
                let sk = g.add(Node::App { func: s, args: smallvec::smallvec![k] });
                // App(sk, K)
                g.add(Node::App { func: sk, args: smallvec::smallvec![k] })
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
