use crate::arena::{Graph, Node, NodeId};
use crate::learner::genome::Gene;
use crate::sensitivity::SensitivityMap;
use rand::prelude::*;
use std::collections::HashMap;

const CHOICES: usize = 6;
// 0: S, 1: K, 2: I, 3: Leaf, 4: First, 5: Rest

#[derive(Clone, Debug)]
pub struct SoftGene {
    pub logits: Vec<f64>,
}

impl Default for SoftGene {
    fn default() -> Self {
        Self { logits: vec![0.0; CHOICES] }
    }
}

pub struct SoftGenome {
    pub leaves: Vec<SoftGene>,
    pub depth: usize,
}

impl SoftGenome {
    pub fn new(depth: usize) -> Self {
        // Number of leaves in a full binary tree of depth D is 2^D ?
        // Or if tree is App structure.
        // A full tree of Apps of depth D has 2^D leaves.
        let num_leaves = 1 << depth;
        Self {
            leaves: vec![SoftGene::default(); num_leaves],
            depth,
        }
    }
    
    /// Instantiate the genome into a Graph as a Pure Tree (expanded)
    /// Returns (Root NodeId, Map<NodeId, LeafIdx>, Choices)
    pub fn instantiate(&self, g: &mut Graph, sample: bool) -> (NodeId, HashMap<NodeId, usize>, Vec<usize>) {
        let mut blamer = HashMap::new();
        let mut choices = Vec::new();
        let mut leaf_iter = self.leaves.iter().enumerate();
        
        // Recursive builder
        let root = build_tree(g, self.depth, &mut leaf_iter, &mut blamer, &mut choices, sample);
        (root, blamer, choices)
    }
    
    // Fix: update signature requires knowing choices.
    pub fn update_with_choices(&mut self, sensitivity: &SensitivityMap, blamer: &HashMap<NodeId, usize>, choices: &[usize], lr: f64) {
         let mut leaf_badness = vec![0.0; self.leaves.len()];
         
         for (node_id, &leaf_idx) in blamer {
            let grads = sensitivity.get(*node_id);
            // Magnitude of gradient indicates contribution to error flow
            let mag = grads.numeric.abs() + grads.expansion.abs() + grads.pruning.abs();
            leaf_badness[leaf_idx] += mag;
         }
         
         for (i, leaf) in self.leaves.iter_mut().enumerate() {
             if i >= choices.len() { break; }
             let choice = choices[i];
             let badness = leaf_badness[i]; 
             // Simple: logit[choice] -= lr * badness.
             // Usually sensitivity is "gradient of loss". So subtracting moves roughly in right direction.
             // Note: This is an approximation of REINFORCE policy gradient
             leaf.logits[choice] -= lr * badness;
         }
    }
}

fn build_tree<'a, I>(
    g: &mut Graph, 
    depth: usize, 
    leaves: &mut I, 
    blamer: &mut HashMap<NodeId, usize>,
    choices_out: &mut Vec<usize>,
    sample: bool
) -> NodeId 
where I: Iterator<Item = (usize, &'a SoftGene)> {
    if depth == 0 {
        if let Some((idx, gene)) = leaves.next() {
            let choice = if sample {
                sample_logits(&gene.logits)
            } else {
                argmax(&gene.logits)
            };
            
            choices_out.push(choice);
            
            // Compile Gene Variant
            let g_gene = match choice {
                0 => Gene::S,
                1 => Gene::K,
                2 => Gene::I,
                3 => Gene::Leaf,
                4 => Gene::First,
                5 => Gene::Rest,
                _ => Gene::Leaf,
            };
            
            // Compile Pure and track nodes
            let root = g_gene.compile_pure(g);
            
            // Mark root as blamed on this leaf
            blamer.insert(root, idx);
            
            return root;
        }
        g.add(Node::Leaf)
    } else {
        let l = build_tree(g, depth - 1, leaves, blamer, choices_out, sample);
        let r = build_tree(g, depth - 1, leaves, blamer, choices_out, sample);
        g.add(Node::App { func: l, args: smallvec::smallvec![r] })
    }
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

pub fn sample_logits(logits: &[f64]) -> usize {
    let probs = softmax(logits);
    let mut rng = thread_rng();
    let r: f64 = rng.gen();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc { return i; }
    }
    probs.len() - 1
}

pub fn argmax(logits: &[f64]) -> usize {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}
