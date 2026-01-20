use trace::arena::{Graph, Node, Primitive};
use trace::types::{Type, TypeEnv};
use trace::inference::{InferenceEngine, Constraint};

// Helper: Check if type of `node` unifies with `expected`.
fn checks_unifies(g: &Graph, node: trace::arena::NodeId, expected: Type) -> bool {
    let mut engine = InferenceEngine::new();
    let env = TypeEnv::new(); // Empty environment
    match engine.infer(g, node, &env) {
        Ok(ty) => {
             engine.constraints.push(Constraint::Equality(ty, expected));
             engine.solve().is_ok()
        },
        Err(_) => false,
    }
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
fn test_identity_polymorphism() {
    let mut g = Graph::new();
    let leaf = g.add(Node::Leaf); 
    let identity = g.add(Node::Stem(leaf)); // Rule: Leaf Leaf y -> y. So Stem(Leaf) is Identity.
    
    // Identity should unify with Float -> Float
    let target = Type::Arrow(Box::new(Type::Float), Box::new(Type::Float));
    assert!(checks_unifies(&g, identity, target), "Stem(Leaf) should act as Identity (Float -> Float)");
    
    // Identity should unify with Leaf -> Leaf
    let target2 = Type::Arrow(Box::new(Type::Leaf), Box::new(Type::Leaf));
    assert!(checks_unifies(&g, identity, target2), "Stem(Leaf) should act as Identity (Leaf -> Leaf)");
}

#[test]
fn test_k_combinator_polymorphism() {
    let mut g = Graph::new();
    let val_x = g.add(Node::Float(10.0));
    let stem_x = g.add(Node::Stem(val_x)); 
    let k_x = g.add(Node::Stem(stem_x)); // Rule: Leaf (Leaf x) y -> x. Stem(Stem(x)) is K x.
    
    // K x should unify with Any -> x
    // Test: K(Float) unifies with Leaf -> Float
    let target = Type::Arrow(Box::new(Type::Leaf), Box::new(Type::Float));
    assert!(checks_unifies(&g, k_x, target), "Stem(Stem(x)) should act as K x (y -> x)");
}

#[test]
fn test_s_combinator_inference() {
    // S behavior: Leaf (Leaf x y) z -> x z (y z)
    // Matches: Stem(Fork(x, y)) z -> x z (y z)
    // S I I = Stem(Fork(I, I))
    
    let mut g = Graph::new();
    let l = g.add(Node::Leaf);
    let i = g.add(Node::Stem(l)); // I = Stem(Leaf)
    
    let fork_ii = g.add(Node::Fork(i, i));
    let s_ii = g.add(Node::Stem(fork_ii)); // S I I
    
    // S I I should unify with I -> I (which is Leaf -> Leaf)
    // Logic: S I I x -> I x (I x) -> x x.
    // Use x = Leaf? x x = Stem(Leaf) = I. 
    // No. Type system.
    // S I I :: (Leaf -> Leaf) -> (Leaf -> Leaf) ?
    // I :: A -> A.
    // S I I returns I (approx).
    
    let target = Type::Arrow(Box::new(Type::Leaf), Box::new(Type::Leaf));
    
    let app = g.add(Node::App { func: s_ii, args: smallvec::smallvec![l] });
    // Note: Applying to 'Leaf' (l).
    // if x=Leaf is Identity? No, I=Stem(Leaf).
    // Application is S I I Leaf. Leaf is passed as arg.
    // Result I Leaf (I Leaf) -> Leaf Leaf -> I (Stem(Leaf)).
    // Wait. I Leaf -> Leaf? No. I x -> x.
    // I Leaf -> Leaf.
    // So result is Leaf.
    // Type of Leaf is Type::Leaf.
    
    let ty = infer_helper(&g, app).expect("S combinator application failed");
    // S I I Leaf -> Stem(Leaf) (Identity)
    assert_eq!(ty, Type::Stem(Box::new(Type::Leaf)));
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
