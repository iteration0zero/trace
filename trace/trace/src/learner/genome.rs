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
        enum Frame<'a> {
            Enter(&'a Gene),
            ExitApp,
        }

        let mut stack = vec![Frame::Enter(self)];
        let mut results: Vec<NodeId> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(gene) => match gene {
                    Gene::S | Gene::K | Gene::I => results.push(gene.compile_pure(g)),
                    Gene::First => results.push(g.add(Node::Prim(Primitive::First))),
                    Gene::Rest => results.push(g.add(Node::Prim(Primitive::Rest))),
                    Gene::Leaf => results.push(g.add(Node::Leaf)),
                    Gene::Int(val) => {
                        let big_val = num_bigint::BigInt::from(*val);
                        let raw = crate::engine::encode_int(g, &big_val);
                        results.push(crate::engine::make_tag(g, Primitive::TagInt, raw));
                    }
                    Gene::App(func, arg) => {
                        stack.push(Frame::ExitApp);
                        stack.push(Frame::Enter(arg));
                        stack.push(Frame::Enter(func));
                    }
                },
                Frame::ExitApp => {
                    let a = results.pop().expect("missing app arg");
                    let f = results.pop().expect("missing app func");
                    results.push(g.add(Node::App { func: f, args: smallvec::smallvec![a] }));
                }
            }
        }

        results.pop().unwrap_or_else(|| g.add(Node::Leaf))
    }
    
    // Helper to estimate complexity
    pub fn complexity(&self) -> usize {
        let mut count = 0usize;
        let mut stack = vec![self];
        while let Some(gene) = stack.pop() {
            count += 1;
            if let Gene::App(f, a) = gene {
                stack.push(f);
                stack.push(a);
            }
        }
        count
    }

    /// Compiles the Gene into a Pure Triage Graph (Leaf/Stem/Fork only)
    pub fn compile_pure(&self, g: &mut Graph) -> NodeId {
        enum Frame<'a> {
            Enter(&'a Gene),
            ExitApp,
        }

        let mut stack = vec![Frame::Enter(self)];
        let mut results: Vec<NodeId> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(gene) => match gene {
                    Gene::K => results.push(build_pure_k(g)),
                    Gene::S => results.push(build_pure_s(g)),
                    Gene::I => results.push(build_pure_i(g)),
                    Gene::First => results.push(g.add(Node::Leaf)),
                    Gene::Rest => results.push(g.add(Node::Leaf)),
                    Gene::Leaf => results.push(g.add(Node::Leaf)),
                    Gene::Int(val) => {
                        let big_val = num_bigint::BigInt::from(*val);
                        let raw = crate::engine::encode_int(g, &big_val);
                        results.push(crate::engine::make_tag(g, Primitive::TagInt, raw));
                    }
                    Gene::App(func, arg) => {
                        stack.push(Frame::ExitApp);
                        stack.push(Frame::Enter(arg));
                        stack.push(Frame::Enter(func));
                    }
                },
                Frame::ExitApp => {
                    let a = results.pop().expect("missing app arg");
                    let f = results.pop().expect("missing app func");
                    results.push(g.add(Node::App { func: f, args: smallvec::smallvec![a] }));
                }
            }
        }

        results.pop().unwrap_or_else(|| g.add(Node::Leaf))
    }
    
    fn clone_as_gene(&self, variant: Gene) -> Gene {
         variant
    }
}

fn build_pure_k(g: &mut Graph) -> NodeId {
    let leaf = g.add(Node::Leaf);
    g.add(Node::Stem(leaf))
}

fn build_pure_s(g: &mut Graph) -> NodeId {
    let leaf = g.add(Node::Leaf);
    let kk = g.add(Node::Fork(leaf, leaf));
    let stem = g.add(Node::Stem(kk));
    g.add(Node::Fork(stem, leaf))
}

fn build_pure_i(g: &mut Graph) -> NodeId {
    let s = build_pure_s(g);
    let k = build_pure_k(g);
    let sk = g.add(Node::App { func: s, args: smallvec::smallvec![k] });
    let skk = g.add(Node::App { func: sk, args: smallvec::smallvec![k] });
    let mut ctx = EvalContext::default();
    reduce(g, skk, &mut ctx)
}
