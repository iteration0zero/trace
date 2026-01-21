# Trace: Tree Calculus Implementation

This project implements the **Triage Calculus**, a variant of Tree Calculus described in the paper *"Typed Program Analysis without Encodings"* (PEPM '25).

## Triage Calculus vs Original Tree Calculus

While based on the fundamental concepts of Tree Calculus (Natural Trees, minimal operators), this implementation follows the Triage Calculus specification for better type system integration.

### Combinator Behavior

- **K ("k")**: `△ △ y z -> y`
- **S ("s")**: `△ (△ x) y z -> x z (y z)`
    - *Note:* This differs from the original Tree Calculus where `S` reduces to `(x z) (y z)` or similar variants depending on the source. We strictly follow the Triage Calculus rule.
- **Triage**: `△ (△ w x) y z`
    - Dispatches based on the structure of `z`:
        - If `z` is a **Leaf** (`△`), reduces to `w`.
        - If `z` is a **Stem** (`△ u`), reduces to `x u`.
        - If `z` is a **Fork** (`△ u v`), reduces to `y u v`.

## Type System

The project implements a path-polymorphic type system that corresponds to the Triage Calculus specification.

### Types
- **Base Types**: `Leaf` (nil), `Float`, `Int`, `Bool`, `Char`, `String`.
- **Structural Types**: `Stem(T)`, `Pair(T1, T2)` (Fork).
- **Function Types**: `T1 -> T2`.
- **Polymorphism**: `∀X. T` (Universal Quantification) and `Generic(X)`.
- **Recursive Types**: `μX. T` and `RecVar(X)`.
- **Set-Theoretic Types**: `Union` and `Intersection`.

### Subtyping
The type inference engine implements coinductive subtyping rules (Algorithm 3.4.2) with support for:
- **Triage Axioms**: Implements the specific K, S, and Fork reduction axioms of Triage Calculus.
- **Parametric Polymorphism**: Supports instantiation and generalization of `Forall` types.
- **Recursive Unrolling**: Handles recursive type subtyping via lazy unrolling and cycle detection.

### Inference
Type inference is constraint-based, generating constraints from the AST and solving them via unification and subtyping checks.
