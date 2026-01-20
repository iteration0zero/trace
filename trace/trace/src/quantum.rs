use crate::arena::NodeId;
use std::collections::HashMap;
use std::fmt;

/// Represents a linear combination of NodeIds: P = Σ α_i * t_i
/// The NodeIds must be valid in the associated Graph.
#[derive(Clone, Default, PartialEq)]
pub struct LinearCombination {
    pub terms: HashMap<NodeId, f64>,
}

impl fmt::Debug for LinearCombination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<_> = self.terms.iter().collect();
        // Sort by ID for deterministic output
        entries.sort_by_key(|(id, _)| id.0);
        
        write!(f, "{{")?;
        for (i, (node_id, coeff)) in entries.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{:.2} * {:?}", coeff, node_id)?;
        }
        write!(f, "}}")
    }
}

impl LinearCombination {
    pub fn zero() -> Self {
        LinearCombination {
            terms: HashMap::new(),
        }
    }

    pub fn from_node(id: NodeId) -> Self {
        let mut terms = HashMap::new();
        terms.insert(id, 1.0);
        LinearCombination { terms }
    }

    pub fn scaled(mut self, scalar: f64) -> Self {
        for val in self.terms.values_mut() {
            *val *= scalar;
        }
        self.terms.retain(|_, v| v.abs() > 1e-10);
        self
    }

    pub fn add(mut self, other: LinearCombination) -> Self {
        for (id, coeff) in other.terms {
            *self.terms.entry(id).or_insert(0.0) += coeff;
        }
        self.terms.retain(|_, v| v.abs() > 1e-10);
        self
    }
}

// Allow operations like P + Q
impl std::ops::Add for LinearCombination {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}
