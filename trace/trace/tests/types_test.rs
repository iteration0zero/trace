use trace::types::{Type, TypeEnv};
use trace::inference::{InferenceEngine};
use trace::arena::{Graph, Node};

fn infer_program(code: &str) -> Type {
    // Ideally we'd use the full parser, but to keep tests focused on type engine independent of parser:
    // We will verify subtyping rules directly.
    unimplemented!("Use inference engine unit tests pattern");
}

#[test]
fn test_datatype_subtyping() {
    let mut engine = InferenceEngine::new();
    let visited = &mut std::collections::HashSet::new();

    // Leaf <= Leaf
    assert!(engine.is_subtype(Type::Leaf, Type::Leaf, visited));

    // Stem(Leaf) <= Stem(Leaf)
    assert!(engine.is_subtype(
        Type::Stem(Box::new(Type::Leaf)),
        Type::Stem(Box::new(Type::Leaf)),
        visited
    ));

    // Leaf is NOT subtype of Stem(Leaf)
    assert!(!engine.is_subtype(
        Type::Leaf,
        Type::Stem(Box::new(Type::Leaf)),
        visited
    ));
}

#[test]
fn test_forall_subtyping_instantiation() {
    let mut engine = InferenceEngine::new();
    let visited = &mut std::collections::HashSet::new();

    // ∀X. X -> X
    let id_type = Type::Forall(
        "X".to_string(),
        Box::new(Type::Arrow(
            Box::new(Type::Generic("X".to_string())),
            Box::new(Type::Generic("X".to_string()))
        ))
    );

    // Int -> Int
    let int_arrow_int = Type::Arrow(
        Box::new(Type::Int),
        Box::new(Type::Int)
    );

    // ∀X. X->X <= Int->Int (Specialization)
    visited.clear();
    assert!(engine.is_subtype(id_type.clone(), int_arrow_int.clone(), visited), "Specialization failed");

    // Int->Int is NOT subtype of ∀X. X->X
    // unless Int->Int is generic enough? No.
    visited.clear();
    assert!(!engine.is_subtype(int_arrow_int.clone(), id_type.clone(), visited), "Generalization should fail for concrete type");
}

#[test]
fn test_forall_alpha_equivalence() {
    let mut engine = InferenceEngine::new();
    let visited = &mut std::collections::HashSet::new();

    // ∀X. X -> X
    let forall_x = Type::Forall(
        "X".to_string(),
        Box::new(Type::Arrow(
            Box::new(Type::Generic("X".to_string())),
            Box::new(Type::Generic("X".to_string()))
        ))
    );

    // ∀Y. Y -> Y
    let forall_y = Type::Forall(
        "Y".to_string(),
        Box::new(Type::Arrow(
            Box::new(Type::Generic("Y".to_string())),
            Box::new(Type::Generic("Y".to_string()))
        ))
    );

    visited.clear();
    assert!(engine.is_subtype(forall_x.clone(), forall_y.clone(), visited));
    
    visited.clear();
    assert!(engine.is_subtype(forall_y.clone(), forall_x.clone(), visited));
}

#[test]
fn test_recursive_type_structural() {
    let mut engine = InferenceEngine::new();
    let visited = &mut std::collections::HashSet::new();

    // μX. Stem(X)
    let rec_stem = Type::Rec(
        0,
        Box::new(Type::Stem(Box::new(Type::RecVar(0))))
    );

    // Unrolled once: Stem(μX. Stem(X))
    let unrolled_once = Type::Stem(Box::new(rec_stem.clone()));

    // Should be equivalent
    visited.clear();
    assert!(engine.is_subtype(rec_stem.clone(), unrolled_once.clone(), visited));
    
    visited.clear();
    assert!(engine.is_subtype(unrolled_once.clone(), rec_stem.clone(), visited));
}

#[test]
fn test_triage_axioms() {
    let mut engine = InferenceEngine::new();
    let visited = &mut std::collections::HashSet::new();
    
    // Axiom K: Pair(Leaf, U) < V -> U
    // Let U = Int, V = Float
    let pair_l_u = Type::Pair(
        Box::new(Type::Leaf),
        Box::new(Type::Int)
    );
    let float_arrow_int = Type::Arrow(
        Box::new(Type::Float),
        Box::new(Type::Int)
    );
    
    visited.clear();
    assert!(engine.is_subtype(pair_l_u, float_arrow_int, visited), "K Axiom failed");
    
    // Axiom S: F (S (U -> V -> T)) (U -> V) < U -> T
    // Let U=Leaf, V=Leaf, T=Leaf
    // Left: Pair(Stem(Leaf->Leaf->Leaf), Leaf->Leaf)
    let u = Type::Leaf;
    let v = Type::Leaf;
    let t = Type::Leaf;
    
    let u_v_t = Type::Arrow(
        Box::new(u.clone()),
        Box::new(Type::Arrow(Box::new(v.clone()), Box::new(t.clone())))
    );
    let u_v = Type::Arrow(Box::new(u.clone()), Box::new(v.clone()));
    let u_t = Type::Arrow(Box::new(u.clone()), Box::new(t.clone()));
    
    let s_inner = Type::Stem(Box::new(u_v_t));
    let left = Type::Pair(
        Box::new(s_inner),
        Box::new(u_v)
    );
    
    visited.clear();
    assert!(engine.is_subtype(left, u_t, visited), "S Axiom failed");
}
