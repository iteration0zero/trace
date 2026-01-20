use trace::arena::{Graph, Node};
use trace::types::{Type, TypeEnv};
use trace::inference::{InferenceEngine, Constraint};

fn infer(g: &Graph, node: trace::arena::NodeId) -> Result<Type, String> {
    let mut engine = InferenceEngine::new();
    let env = TypeEnv::new();
    let ty = engine.infer(g, node, &env)?;
    engine.solve()?;
    Ok(engine.resolve_type(ty))
}

#[test]
fn test_bool_structure() {
    let mut g = Graph::new();
    // False = Leaf
    let false_node = g.add(Node::Leaf);
    // True = Stem(Leaf)
    let true_child = g.add(Node::Leaf);
    let true_node = g.add(Node::Stem(true_child));
    
    // Invalid = Fork(Leaf, Leaf)
    let inv_child = g.add(Node::Leaf);
    let invalid_node = g.add(Node::Fork(inv_child, inv_child));
    
    // Check unification with Expectation
    // We expect:
    // Leaf <: Bool
    // Stem(Leaf) <: Bool
    // Fork(_, _) NOT <: Bool
    
    // Testing via Unification constraints with explicit Type::Bool
    
    // 1. Leaf should unify with Bool
    {
        let mut engine = InferenceEngine::new();
        let env = TypeEnv::new();
        let ty = engine.infer(&g, false_node, &env).unwrap();
        // Leaf <: Bool
        engine.constraints.push(Constraint::Subtype(ty, Type::Bool));
        assert!(engine.solve().is_ok(), "Leaf should be subtype of Bool");
    }
    
    // 2. Stem(Leaf) should unify with Bool
    {
        let mut engine = InferenceEngine::new();
        let env = TypeEnv::new();
        let ty = engine.infer(&g, true_node, &env).unwrap();
        // Stem(Leaf) <: Bool
        engine.constraints.push(Constraint::Subtype(ty, Type::Bool));
        assert!(engine.solve().is_ok(), "Stem(Leaf) should be subtype of Bool");
    }
    
    // 3. Fork should fail
    {
        let mut engine = InferenceEngine::new();
        let env = TypeEnv::new();
        let ty = engine.infer(&g, invalid_node, &env).unwrap();
        engine.constraints.push(Constraint::Subtype(ty, Type::Bool));
        assert!(engine.solve().is_err(), "Fork should NOT be subtype of Bool");
    }
}

#[test]
fn test_bool_propagation() {
    // Check Propagate Constraints for Bool
    // Should be (Leaf, Any, Any)
    let (s, l, r) = InferenceEngine::propagate_type_constraints(&Type::Bool);
    
    match s {
        Type::Leaf => {}, // Correct
        _ => panic!("Bool stem constraint should be Leaf, got {:?}", s)
    }
    
    match l {
        Type::Var(_) => {}, // Correct (Any)
        _ => panic!("Bool left constraint should be Var (Any), got {:?}", l)
    }
}

#[test]
fn test_bool_mask() {
    // Check Mask
    // Should be [0.0, 0.0, -inf]
    let mask = InferenceEngine::get_structural_mask(&Type::Bool);
    assert_eq!(mask[0], 0.0, "Leaf allowed");
    assert_eq!(mask[1], 0.0, "Stem allowed");
    assert_eq!(mask[2], f64::NEG_INFINITY, "Fork disallowed");
}
