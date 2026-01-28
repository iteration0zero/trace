use trace::arena::{Graph, Node};
use trace::learner::{CounterfactualConfig, counterfactual_synthesize};
use trace::engine::{reduce, EvalContext};

#[test]
fn test_learn_stem_function() {
    // Learn: f x = Stem(x)
    // In Triage, Leaf applied to x yields Stem(x).
    // So the target program is just Leaf.
    
    let mut g = Graph::new();
    
    // Create examples: x -> Stem(x)
    let mut examples = Vec::new();
    
    // Ex 1: Leaf -> Stem(Leaf)
    let leaf = g.add(Node::Leaf);
    let stem_leaf = g.add(Node::Stem(leaf));
    examples.push((leaf, stem_leaf));
    
    // Ex 2: Stem(Leaf) -> Stem(Stem(Leaf))
    let stem_stem_leaf = g.add(Node::Stem(stem_leaf));
    examples.push((stem_leaf, stem_stem_leaf));
    
    let config = CounterfactualConfig {
        max_iterations: 50,
        verbose: false,
        ..Default::default()
    };
    
    let res = counterfactual_synthesize(&mut g, examples, config);
    
    assert!(res.is_some(), "Learner failed to find solution");
    let prog = res.unwrap();
    
    // Verify on new input
    let x = g.add(Node::Float(1.0));
    let app = g.add(Node::App{ func: prog, args: smallvec::smallvec![x] });
    let mut ctx = EvalContext::default();
    let out = reduce(&mut g, app, &mut ctx);
    
    let stem_x = g.add(Node::Stem(x));
    assert_eq!(out, stem_x, "Learned program should wrap input in Stem");
}
