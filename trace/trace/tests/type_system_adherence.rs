use trace::arena::{Graph, Node};
use trace::types::{Type, TypeEnv};
use trace::inference::{InferenceEngine, Constraint};

fn assert_subtype(t1: Type, t2: Type) {
    let mut engine = InferenceEngine::new();
    engine.constraints.push(Constraint::Subtype(t1, t2));
    assert!(engine.solve().is_ok(), "Expected subtype constraint to hold");
}

// Helper: Run inference on a graph rooted at `root` and return the type.
fn infer_helper(g: &Graph, root: trace::arena::NodeId) -> Result<Type, String> {
    let mut engine = InferenceEngine::new();
    let env = TypeEnv::new();
    let ty = engine.infer(g, root, &env)?;
    engine.solve()?;
    Ok(engine.resolve_type(ty))
}

#[test]
fn test_atomic_inference() {
    let mut g = Graph::new();
    let leaf = g.add(Node::Leaf);
    let float = g.add(Node::Float(42.0));
    
    let ty_leaf = infer_helper(&g, leaf).expect("Leaf inference failed");
    assert_eq!(ty_leaf, Type::Leaf);
    
    let ty_float = infer_helper(&g, float).expect("Float inference failed");
    assert_eq!(ty_float, Type::Float);
}

#[test]
fn test_structural_inference() {
    let mut g = Graph::new();
    let leaf = g.add(Node::Leaf);
    let float = g.add(Node::Float(3.14));
    
    // Test Stem: Stem(Leaf) -> Stem(Leaf)
    let stem = g.add(Node::Stem(leaf));
    let ty_stem = infer_helper(&g, stem).expect("Stem inference failed");
    match ty_stem {
        Type::Stem(inner) => assert_eq!(*inner, Type::Leaf),
        _ => panic!("Expected Stem(Leaf), got {:?}", ty_stem),
    }
    
    // Test Fork: Fork(Leaf, Float) -> Pair(Leaf, Float)
    let fork = g.add(Node::Fork(leaf, float));
    let ty_fork = infer_helper(&g, fork).expect("Fork inference failed");
    match ty_fork {
        Type::Pair(a, b) => {
            assert_eq!(*a, Type::Leaf);
            assert_eq!(*b, Type::Float);
        },
        _ => panic!("Expected Pair(Leaf, Float), got {:?}", ty_fork),
    }
}

#[test]
fn test_leaf_subtyping_axiom() {
    // L < U -> S U
    let u = Type::Float;
    let target = Type::Arrow(Box::new(u.clone()), Box::new(Type::Stem(Box::new(u))));
    assert_subtype(Type::Leaf, target);
}

#[test]
fn test_stem_subtyping_axiom() {
    // S U < V -> F U V
    let u = Type::Leaf;
    let v = Type::Float;
    let target = Type::Arrow(
        Box::new(v.clone()),
        Box::new(Type::Pair(Box::new(u.clone()), Box::new(v))),
    );
    assert_subtype(Type::Stem(Box::new(u)), target);
}

#[test]
fn test_k_subtyping_axiom() {
    // F L U < V -> U
    let u = Type::Float;
    let target = Type::Arrow(Box::new(Type::Leaf), Box::new(u.clone()));
    let fork = Type::Pair(Box::new(Type::Leaf), Box::new(u));
    assert_subtype(fork, target);
}

#[test]
fn test_triage_subtyping_axioms() {
    // Leaf-case: F (F T V) W < L -> T
    let t = Type::Float;
    let v = Type::Leaf;
    let w = Type::Leaf;
    let left = Type::Pair(Box::new(t.clone()), Box::new(v));
    let fork = Type::Pair(Box::new(left), Box::new(w));
    let target = Type::Arrow(Box::new(Type::Leaf), Box::new(t.clone()));
    assert_subtype(fork, target);

    // Stem-case: F (F U (V -> T)) W < S V -> T
    let u = Type::Leaf;
    let v2 = Type::Float;
    let t2 = Type::Float;
    let left2 = Type::Pair(
        Box::new(u),
        Box::new(Type::Arrow(Box::new(v2.clone()), Box::new(t2.clone()))),
    );
    let fork2 = Type::Pair(Box::new(left2), Box::new(Type::Leaf));
    let target2 = Type::Arrow(Box::new(Type::Stem(Box::new(v2))), Box::new(t2));
    assert_subtype(fork2, target2);

    // Fork-case: F (F U V) (W1 -> W2 -> T) < F W1 W2 -> T
    let w1 = Type::Float;
    let w2 = Type::Leaf;
    let t3 = Type::Float;
    let left3 = Type::Pair(Box::new(Type::Leaf), Box::new(Type::Leaf));
    let right3 = Type::Arrow(
        Box::new(w1.clone()),
        Box::new(Type::Arrow(Box::new(w2.clone()), Box::new(t3.clone()))),
    );
    let fork3 = Type::Pair(Box::new(left3), Box::new(right3));
    let target3 = Type::Arrow(
        Box::new(Type::Pair(Box::new(w1), Box::new(w2))),
        Box::new(t3),
    );
    assert_subtype(fork3, target3);

    // S axiom: F (S (U -> V -> T)) (U -> V) < U -> T
    let u4 = Type::Float;
    let v4 = Type::Leaf;
    let t4 = Type::Float;
    let left4 = Type::Stem(Box::new(Type::Arrow(
        Box::new(u4.clone()),
        Box::new(Type::Arrow(Box::new(v4.clone()), Box::new(t4.clone()))),
    )));
    let right4 = Type::Arrow(Box::new(u4.clone()), Box::new(v4));
    let fork4 = Type::Pair(Box::new(left4), Box::new(right4));
    let target4 = Type::Arrow(Box::new(u4), Box::new(t4));
    assert_subtype(fork4, target4);
}

#[test]
fn test_nested_structure_types() {
    let mut g = Graph::new();
    let l = g.add(Node::Leaf);
    let f = g.add(Node::Float(1.2));
    
    // Fork(Stem(Leaf), Fork(Float, Leaf))
    // Type should be Pair(Stem(Leaf), Pair(Float, Leaf))
    
    let stem_leaf = g.add(Node::Stem(l));
    let inner_fork = g.add(Node::Fork(f, l));
    let outer = g.add(Node::Fork(stem_leaf, inner_fork));
    
    let ty = infer_helper(&g, outer).expect("Nested structure inference failed");
    
    if let Type::Pair(head_ty, tail_ty) = ty {
        match *head_ty {
            Type::Stem(s) => assert_eq!(*s, Type::Leaf),
            _ => panic!("Expected Head Stem(Leaf)"),
        }
        match *tail_ty {
            Type::Pair(a, b) => {
                assert_eq!(*a, Type::Float);
                assert_eq!(*b, Type::Leaf);
            },
            _ => panic!("Expected Tail Pair(Float, Leaf)"),
        }
    } else {
        panic!("Expected outer Pair");
    }
}
