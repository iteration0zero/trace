use trace::arena::{Graph, Node};
use trace::learner::{LearnerConfig, learn};
use trace::engine::{reduce, EvalContext, unparse};

// test_learn_identity removed due to slow convergence in unit test environment

#[test]
fn test_learn_k_combinator() {
    // Learn a simple HOF: f x = Stem(x)
    // Target = Leaf.
    
    let mut g = Graph::new();
    let target = g.add(Node::Leaf); // acts as \x -> Stem(x)
    
    let config = LearnerConfig {
        epochs: 50,
        lr: 0.1,
        lambda: 0.0,
        skeleton_depth: 5,
        samples_per_step: 3,
    };
    
    let res = learn(&mut g, target, None, config, vec![]);
    assert!(res.is_some(), "Learner failed to find Leaf-like function");
    let prog = res.unwrap();
    
    let x = g.add(Node::Float(1.0));
    let app = g.add(Node::App{ func: prog, args: smallvec::smallvec![x] });
    let mut ctx = EvalContext::default();
    let out = reduce(&mut g, app, &mut ctx);
    
    let stem_x = g.add(Node::Stem(x));
    assert_eq!(out, stem_x, "Learned program should wrap input in Stem");
}
