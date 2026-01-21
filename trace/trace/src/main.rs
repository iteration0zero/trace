use trace::arena::{Graph, NodeId};
use trace::parser::{Parser, ParseResult};
use trace::engine::{reduce, unparse, EvalContext};
use std::collections::HashMap;
use std::io::{self, Write};

use trace::inference::InferenceEngine;
use trace::types::{TypeEnv, Type};

use trace::learner::{learn, learn_from_examples, learn_from_examples_with_library, LearnerConfig};

fn main() {
    println!("Trace REPL");
    println!("(n (n n)) based syntax. Type (exit) or Ctrl+C to quit.");

    let mut g = Graph::new();
    let mut env: HashMap<String, NodeId> = HashMap::new();
    let mut type_env = TypeEnv::new();

    // Standard Library (Prelude)
    let prelude = [
        ("k", "(n n)"),
        ("s", "(n (n (k n)) n)"),
        ("i", "(s k k)"),
        ("true", "k"),
        ("false", "n"),
        ("triage", "(fn w (fn x (fn y (n (n w x) y))))"),
        ("is-leaf", "(fn z ((triage true (fn u false) (fn u (fn v false))) z))"),
        ("is-stem", "(fn z ((triage false (fn u true) (fn u (fn v false))) z))"),
        ("is-fork", "(fn z ((triage false (fn u false) (fn u (fn v true))) z))"),
        ("first", "(fn p ((triage n (fn u n) (fn a (fn b a))) p))"),
        ("rest", "(fn p ((triage n (fn u n) (fn a (fn b b))) p))"),
        ("stem", "(fn x (n x))"),
        ("fork", "(fn a (fn b (n a b)))"),
        ("cons", "fork"),
        ("leaf", "n"),
    ];

    // Library of learned/defined programs for curriculum learning
    let mut library: Vec<NodeId> = Vec::new();

    println!("Loading standard library...");
    for (name, code) in prelude {
        let mut p = Parser::new(code);
         if let Ok(ParseResult::Term(node)) = p.parse_toplevel(&mut g, Some(&env)) {
             // Verify type
             let mut engine = InferenceEngine::new();
             if let Ok(ty) = engine.infer(&g, node, &type_env) {
                 type_env.vars.insert(name.to_string(), ty);
             }
             
             let mut ctx = EvalContext::default();
             let reduced = reduce(&mut g, node, &mut ctx);
             env.insert(name.to_string(), reduced);
             
             // Add valid functions to library
             library.push(reduced);
         } else {
             println!("Failed to load {}", name);
         }
    }

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        // Multi-line input: accumulate until parentheses are balanced
        let mut input = String::new();
        let mut paren_depth = 0i32;
        let mut in_comment = false;
        
        loop {
            let mut line = String::new();
            if io::stdin().read_line(&mut line).unwrap() == 0 {
                if input.is_empty() { 
                    // EOF on first line - exit
                    return; 
                }
                break; // EOF while collecting - try to parse what we have
            }
            
            // Check for exit on first line
            if input.is_empty() {
                let trimmed = line.trim();
                if trimmed == "(exit)" || trimmed == "exit" {
                    return;
                }
                // Skip empty lines and comment-only lines
                if trimmed.is_empty() || trimmed.starts_with(';') {
                    break; // Skip this line
                }
            }
            
            // Count parentheses (ignoring comments)
            for c in line.chars() {
                match c {
                    ';' => in_comment = true,
                    '\n' => in_comment = false,
                    '(' if !in_comment => paren_depth += 1,
                    ')' if !in_comment => paren_depth -= 1,
                    _ => {}
                }
            }
            in_comment = false; // Reset at end of line
            
            input.push_str(&line);
            
            // If balanced (or negative - error), stop collecting
            if paren_depth <= 0 {
                break;
            }
            
            // Continuation prompt
            print!("  ");
            io::stdout().flush().unwrap();
        }
        
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        
        // Handle (learn ...) and (learn-from ...) commands
        if trimmed.starts_with("(learn ") || trimmed.starts_with("(learn-from ") {
            if let Some(learned) = handle_learn_command(trimmed, &mut g, &env, &type_env, &library) {
                 // If a new program was learned, add it to the library
                 println!("Added learned program to curriculum library (Size: {})", library.len() + 1);
                 library.push(learned);
            }
            continue;
        }

        let mut parser = Parser::new(&input);
        match parser.parse_toplevel(&mut g, Some(&env)) {
            Ok(res) => {
                match res {
                    ParseResult::Term(node) => {
                        // Infer Type
                        let mut engine = InferenceEngine::new();
                        match engine.infer(&g, node, &type_env) {
                            Ok(ty) => println!(" : {:?}", ty),
                            Err(e) => println!("Type Error: {}", e),
                        }
                        
                        let mut ctx = EvalContext::default();
                        let result = reduce(&mut g, node, &mut ctx);
                        println!("= {}", unparse(&g, result));
                    }
                    ParseResult::Def(name, node) => {
                         // Infer Type
                         let mut engine = InferenceEngine::new();
                         match engine.infer(&g, node, &type_env) {
                             Ok(ty) => {
                                 type_env.vars.insert(name.clone(), ty.clone());
                                 println!(" : {:?}", ty);
                             },
                             Err(e) => println!("Type Error: {}", e),
                         }
                         
                         let mut ctx = EvalContext::default();
                         let result = reduce(&mut g, node, &mut ctx); 
                         env.insert(name.clone(), result);
                         println!("Defined {}", name);
                    }
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
}

/// Handle learn command: (learn fn epochs) or (learn-from ((in1 out1) ...) [type] epochs)
/// Returns learned program NodeId if successful
fn handle_learn_command(
    input: &str, 
    g: &mut Graph, 
    env: &HashMap<String, NodeId>,
    type_env: &TypeEnv,
    library: &Vec<NodeId>
) -> Option<NodeId> {
    // Check for learn-from syntax
    if input.starts_with("(learn-from ") {
        return handle_learn_from_command(input, g, env, type_env, library);
    }
    
    // Parse: (learn name epochs [depth] [samples])
    // Example: (learn id 500) or (learn id 500 3 10)
    let inner = input.trim_start_matches("(learn ").trim_end_matches(')').trim();
    let parts: Vec<&str> = inner.split_whitespace().collect();
    
    if parts.is_empty() {
        println!("Usage: (learn <name> [epochs] [depth] [samples])");
        println!("       (learn id 500)");
        println!("       (learn id 500 3 10)  ; depth 3, 10 samples/step");
        println!("       (learn-from ((in out) ...) [type])");
        return None;
    }
    
    let name = parts[0];
    let epochs: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let depth: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let samples: usize = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    
    if let Some(&target_func) = env.get(name) {
        // Get type constraint from type_env if available
        let target_type = type_env.vars.get(name).cloned();
        
        let config = LearnerConfig {
            epochs,
            skeleton_depth: depth,
            samples_per_step: samples,
            ..Default::default()
        };

        // Filter library to exclude the target function itself (to prevent trivial solution)
        let filtered_library: Vec<NodeId> = library.iter()
            .cloned()
            .filter(|&id| id != target_func)
            .collect();

        learn(g, target_func, target_type, config, filtered_library)
    } else {
        println!("Error: '{}' not defined. Define a function first with (def {} ...)", name, name);
        None
    }
}

/// Handle learn-from: (learn-from ((in1 out1) (in2 out2) ...) [epochs] [depth])
/// Returns learned program NodeId if successful
fn handle_learn_from_command(
    input: &str,
    g: &mut Graph,
    env: &HashMap<String, NodeId>,
    _type_env: &TypeEnv,
    library: &Vec<NodeId>
) -> Option<NodeId> {
    // Parse examples from input
    // Format: (learn-from ((in1 out1) (in2 out2) ...) [epochs] [depth])
    let inner = input.trim_start_matches("(learn-from ").trim_end_matches(')').trim();
    
    // Find the examples list (starts with '((' and ends with '))')
    if !inner.starts_with('(') {
        println!("Usage: (learn-from ((in1 out1) (in2 out2) ...) [epochs] [depth])");
        return None;
    }
    
    // Find matching closing paren for examples list
    let mut depth = 0;
    let mut examples_end = 0;
    for (i, c) in inner.chars().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    examples_end = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    
    if examples_end == 0 {
        println!("Error: Malformed examples list");
        return None;
    }
    
    let examples_str = &inner[..examples_end];
    let rest = inner[examples_end..].trim();
    let rest_parts: Vec<&str> = rest.split_whitespace().collect();
    
    let epochs: usize = rest_parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(500);
    let skeleton_depth: usize = rest_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let samples: usize = rest_parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    
    // Parse examples
    let mut train_data: Vec<(NodeId, NodeId)> = Vec::new();
    
    // examples_str is like "((in1 out1) (in2 out2) ...)" 
    // Strip one layer of outer parens to get "(in1 out1) (in2 out2) ..."
    let examples_inner = examples_str.trim();
    if examples_inner.starts_with('(') && examples_inner.ends_with(')') {
        let inner = &examples_inner[1..examples_inner.len()-1];
        train_data = parse_pairs(inner.trim(), g, env);
    }
    
    if train_data.is_empty() {
        // Debug: print what we're trying to parse
        eprintln!("DEBUG: examples_str = '{}'", examples_str);
    }
    
    if train_data.is_empty() {
        println!("Error: No valid examples found");
        return None;
    }
    
    println!("Parsed {} examples", train_data.len());
    
    let config = LearnerConfig {
        epochs,
        skeleton_depth,
        samples_per_step: samples,
        ..Default::default()
    };
    
    learn_from_examples_with_library(g, train_data, None, config, library.clone())
}

/// Parse pairs like (in1 out1) (in2 out2) from a string
fn parse_pairs(input: &str, g: &mut Graph, env: &HashMap<String, NodeId>) -> Vec<(NodeId, NodeId)> {
    let mut pairs = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    
    for (i, c) in input.chars().enumerate() {
        match c {
            '(' => {
                if depth == 0 { start = i; }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let pair_str = &input[start..=i];
                    if let Some((inp, out)) = parse_single_pair(pair_str, g, env) {
                        pairs.push((inp, out));
                    }
                }
            }
            _ => {}
        }
    }
    
    pairs
}

/// Parse a single (in out) pair - handles nested parentheses correctly
fn parse_single_pair(input: &str, g: &mut Graph, env: &HashMap<String, NodeId>) -> Option<(NodeId, NodeId)> {
    let trimmed = input.trim();
    // Only strip one layer of parens: "(expr1 expr2)" -> "expr1 expr2"
    if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
        return None;
    }
    let inner = &trimmed[1..trimmed.len()-1];
    
    // Find the boundary between input and output expressions
    // They are separated by whitespace, but we need to respect parentheses
    let mut depth = 0;
    let mut split_pos = None;
    let mut in_whitespace = false;
    
    for (i, c) in inner.chars().enumerate() {
        match c {
            '(' => {
                depth += 1;
                in_whitespace = false;
            },
            ')' => {
                depth -= 1;
                in_whitespace = false;
            },
            c if c.is_whitespace() => {
                if depth == 0 && !in_whitespace {
                    split_pos = Some(i);
                    in_whitespace = true;
                }
            },
            _ => {
                in_whitespace = false;
                // For atoms at depth 0, the next whitespace is the split
                if depth == 0 && split_pos.is_some() {
                    break;
                }
            }
        }
        // If we found a split and we're starting a new expression, we're done
        if split_pos.is_some() && depth == 0 && !c.is_whitespace() {
            break;
        }
    }
    
    let split_pos = split_pos?;
    let inp_str = inner[..split_pos].trim();
    let out_str = inner[split_pos..].trim();
    
    if inp_str.is_empty() || out_str.is_empty() {
        return None;
    }
    
    let mut p1 = Parser::new(inp_str);
    let mut p2 = Parser::new(out_str);
    
    let inp = match p1.parse_toplevel(g, Some(env)) {
        Ok(ParseResult::Term(n)) => n,
        _ => return None,
    };
    
    let out = match p2.parse_toplevel(g, Some(env)) {
        Ok(ParseResult::Term(n)) => n,
        _ => return None,
    };
    
    // Reduce both to get canonical form
    use trace::engine::{reduce, EvalContext};
    let mut ctx = EvalContext::default();
    let inp_reduced = reduce(g, inp, &mut ctx);
    let out_reduced = reduce(g, out, &mut ctx);
    
    Some((inp_reduced, out_reduced))
}
