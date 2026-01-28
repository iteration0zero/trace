use super::genome::Gene;
use super::soft::{SoftGenome, argmax};
use crate::arena::{Graph, NodeId, Node};
use crate::engine::{reduce, EvalContext};
use crate::sensitivity::compute_sensitivity;

pub struct SearchConfig {
    pub max_depth: usize,
    pub max_epochs: usize,
    pub lr: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 3, 
            max_epochs: 100,
            lr: 0.1,
        }
    }
}

pub fn evolve(
    _examples: &[(NodeId, NodeId)], 
    config: SearchConfig
) -> Option<Gene> {
    let _genome = SoftGenome::new(config.max_depth);
    None
}

// Fixed signature with Graph
pub fn evolve_with_graph(
    g: &mut Graph,
    examples: &[(NodeId, NodeId)], 
    config: SearchConfig
) -> Option<Gene> {
    let mut genome = SoftGenome::new(config.max_depth);
    println!("Gradient Learner Started: Depth {}, Epochs {}, Examples {}", config.max_depth, config.max_epochs, examples.len());
    
    for epoch in 0..config.max_epochs {
        let mut total_loss = 0.0;
        let mut perfect = true;
        
        for (inp, expected) in examples {
            // 1. Instantiate Candidate
            let (root, blamer, choices) = genome.instantiate(g, true); // Sampled
            
            // 2. Prepare Application
            let base_nodes = g.nodes.len();
            let app = g.add(Node::App { func: root, args: smallvec::smallvec![*inp] });
            
            // 3. Run with Trace
            let mut trace = crate::trace::Trace::new();
            let mut ctx = EvalContext {
                steps: 0,
                step_limit: 500,
                node_limit: 0,
                node_limit_hit: false,
                step_limit_hit: false,
                depth: 0,
                depth_limit: 1000,
                trace: Some(&mut trace),
                exec_trace: None,
                redex_cache: None,
                base_nodes,
            };
            
            let res = reduce(g, app, &mut ctx);
            
            // 4. Compute Loss
            // Simple loss: distinct from expected?
            // Or numeric difference if Int?
            // "Type-Safe" Loss:
            let loss = calculate_loss(g, res, *expected);
            
            if loss > 1e-6 {
                perfect = false;
                total_loss += loss;
                
                // 5. Backprop
                if let Some(trace) = &ctx.trace {
                    let sensitivity = compute_sensitivity(trace, loss);
                    genome.update_with_choices(&sensitivity, &blamer, &choices, config.lr);
                }
            }
        }
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, total_loss);
        }
        
        if perfect {
            println!("Solution found at Epoch {}", epoch);
            return Some(reconstruct_gene(&genome));
        }
    }
    
    Some(reconstruct_gene(&genome))
}

fn calculate_loss(g: &Graph, actual: NodeId, expected: NodeId) -> f64 {
    // If exact match
    if actual == expected { return 0.0; }
    
    // Check values
    let _n1 = g.get(g.resolve(actual));
    let _n2 = g.get(g.resolve(expected));
    
    // Tagged Ints comparison
    // We need to unwrap data?
    // Assuming simple structural equality or numeric difference.
    // For now: 1.0 if different.
    1.0
}

fn reconstruct_gene(sg: &SoftGenome) -> Gene {
    // Return ArgMax gene
    // We need to build the tree of Genes.
    // Hard to reconstruct pure App tree into Gene tree?
    // Gene::App is structural.
    // SoftGenome `instantiate` builds a Tree of Apps.
    // We can replicate that structure with Genes.
    
    // Recursive builder
    let mut leaves = sg.leaves.iter();
    build_gene_tree(sg.depth, &mut leaves)
}

fn build_gene_tree<'a, I>(depth: usize, leaves: &mut I) -> Gene
where I: Iterator<Item = &'a super::soft::SoftGene> {
    enum Frame {
        Enter(usize),
        ExitApp,
    }

    let mut stack: Vec<Frame> = Vec::new();
    let mut results: Vec<Gene> = Vec::new();
    stack.push(Frame::Enter(depth));

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(d) => {
                if d == 0 {
                    let gene = if let Some(soft_gene) = leaves.next() {
                        let choice = argmax(&soft_gene.logits);
                        match choice {
                            0 => Gene::S,
                            1 => Gene::K,
                            2 => Gene::I,
                            3 => Gene::Leaf,
                            4 => Gene::First,
                            5 => Gene::Rest,
                            _ => Gene::Leaf,
                        }
                    } else {
                        Gene::Leaf
                    };
                    results.push(gene);
                } else {
                    stack.push(Frame::ExitApp);
                    stack.push(Frame::Enter(d - 1));
                    stack.push(Frame::Enter(d - 1));
                }
            }
            Frame::ExitApp => {
                let right = results.pop().unwrap_or(Gene::Leaf);
                let left = results.pop().unwrap_or(Gene::Leaf);
                results.push(Gene::App(Box::new(left), Box::new(right)));
            }
        }
    }

    results.pop().unwrap_or(Gene::Leaf)
}
