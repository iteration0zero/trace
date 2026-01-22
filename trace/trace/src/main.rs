use trace::arena::{Graph, NodeId, Node, Primitive};
use trace::parser::{Parser, ParseResult};
use trace::engine::{reduce, unparse, EvalContext};
use std::collections::HashMap;
use std::io::{self, Write};
use num_traits::ToPrimitive;

use trace::inference::InferenceEngine;
use trace::types::{TypeEnv, Type};

use trace::learner::{evolve, SearchConfig};

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
        // tree_book: self_apply = λ* w. w w = d{I} I, and d{x} = n (n x)
        ("self_apply", "(n (n i) i)"),
        ("true", "k"),
        ("false", "n"),
        ("triage", "(fn w (fn x (fn y (n (n w x) y))))"),
        ("is-leaf", "(fn z ((triage true (fn u false) (fn u (fn v false))) z))"),
        ("is-stem", "(fn z ((triage false (fn u true) (fn u (fn v false))) z))"),
        ("is-fork", "(fn z ((triage false (fn u false) (fn u (fn v true))) z))"),
        // first/rest are now primitives
        ("d", "(fn x (n (n x)))"),
        // Fixpoint helpers from tree_book
        ("self_apply", "(n (n i) i)"), // d{I}I
        ("omega", "(d (k i) (d self_apply (k d)))"), // ω = d{K I}(d{self_apply}(K d))
        // wait{x,y} = d{I}(d{K y}(K x))
        ("wait", "(fn x (fn y (d i (d (k y) (k x)))))"),
        // wait1{x} = d{d{K(K x)}(d{d{K}(K△)}(K△))}(K(d{△KK}))
        ("wait1", "(fn x (d (d (k (k x)) (d (d (k) (k n)) (k n))) (k (d (n (k k))))))"),
        // wait2{x,y} = d{d{K(d{K y}(K x))} d{d{K}(K△)}(K△)}(K(d{I}))
        ("wait2", "(fn x (fn y (d (d (k (d (k y) (k x))) (d (d (k) (k n)) (k n))) (k (d i)))))"),
        // swap{f} = d{K f}d{d{K}(K△)}(K△)
        ("swap", "(fn f (d (k f) (d (d (k) (k n)) (k n))))"),
        // Z{f} = wait{self_apply, d{wait1{self_apply}(K f)}}
        ("Z", "(fn f (wait self_apply (d (wait1 self_apply (k f)))))"),
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
        
        // Handle (learn-from ...)
        if trimmed.starts_with("(learn-from") {
             handle_learn_from_command(trimmed, &mut g, &mut env, &mut library);
             continue;
        }
        // Handle (learn-self ...)
        if trimmed.starts_with("(learn-self") {
             handle_learn_self_command(trimmed, &mut g, &mut env, &mut library);
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
                        if ctx.steps >= ctx.step_limit {
                            println!("Warning: step limit reached; result may be a partial reduction.");
                        }
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



/// Handle learn-from command: (learn-from [epochs] ((in out) ...))
fn handle_learn_from_command(
    input: &str, 
    g: &mut Graph, 
    env: &mut HashMap<String, NodeId>,
    library: &mut Vec<NodeId>,
) {
    // Format: (learn-from ( (in1 out1) (in2 out2) ... ) epochs)
    // We need to extract the examples list and epochs
    
    let inner = input.trim_start_matches("(learn-from").trim_end_matches(')').trim();
    
    // Find the examples list (first parenthesized expression) and epochs (last number)
    let mut depth = 0;
    let mut examples_start = None;
    let mut examples_end = None;
    
    for (i, c) in inner.chars().enumerate() {
        match c {
            '(' => {
                if depth == 0 {
                    examples_start = Some(i);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 && examples_start.is_some() && examples_end.is_none() {
                    examples_end = Some(i + 1);
                }
            }
            _ => {}
        }
    }
    
    let (examples_str, rest) = if let (Some(start), Some(end)) = (examples_start, examples_end) {
        (&inner[start..end], inner[end..].trim())
    } else {
        println!("Error: Could not find examples list in learn-from");
        return;
    };
    
    // Parse epochs from rest (if any)
    let epochs: usize = rest.parse().unwrap_or(100);
    
    // Now parse the examples list content (without outer parens)
    let examples_content = examples_str.trim_start_matches('(').trim_end_matches(')').trim();
    
    // Parse each example pair: (input output)
    let mut examples = Vec::new();
    let mut p = Parser::new(examples_content);
    let mut ctx = EvalContext::default();
    
    while p.has_more() {
        // Each example is a parenthesized pair (input output)
        if let Ok(ParseResult::Term(pair_node)) = p.parse_toplevel(g, Some(env)) {
            // pair_node is compiled as App(input, output) in Triage syntax
            // We need to:
            // 1. Look at the App structure (unreduced)
            // 2. Extract the two parts
            
            // Actually, let's use a simpler approach: parse TWO consecutive terms
            // and treat them as (input, output) - this matches how (a b) is parsed
            
            // The pair_node from parse is already App(input, output)
            // But after compilation, it might have been reduced
            // Let's peek at the node structure
            
            match g.get(pair_node).clone() {
                Node::App { func, args } if args.len() == 1 => {
                    // Compiled as App(input, output)
                    let in_val = reduce(g, func, &mut ctx);
                    let out_val = reduce(g, args[0], &mut ctx);
                    examples.push((in_val, out_val));
                }
                Node::Fork(left, right) => {
                    // Might have been reduced to Fork
                    let in_val = reduce(g, left, &mut ctx);
                    let out_val = reduce(g, right, &mut ctx);
                    examples.push((in_val, out_val));
                }
                _ => {
                    println!("Warning: Skipping malformed example pair");
                }
            }
        } else {
            break;
        }
    }
    
    if examples.is_empty() {
        println!("Error: Empty examples list.");
        return;
    }

    println!("Parsed {} examples. Running IGTC synthesis for {} iterations...", examples.len(), epochs);
    
    // Use IGTC synthesizer
    // Use IGTC synthesizer
    let mut config = trace::learner::IgtcConfig::default();
    config.max_iterations = epochs;
    // Keep other defaults which are now tuned (LR=0.01, support=500, edits=15)
    
    if let Some(learned_node) = trace::learner::igtc_synthesize_with_seeds(g, examples, config, &library) {
        println!("Success! Learned Node: {}", unparse(g, learned_node));
        
        env.insert("learned".to_string(), learned_node);
        println!("Saved as 'learned'.");
        
        // Curriculum: add to library for future synthesis
        library.push(learned_node);
    } else {
        println!("IGTC synthesis failed to converge.");
    }
}

/// Handle learn-self command: (learn-self (t1 t2 ...) epochs)
/// Builds examples where output = (t t) reduced, i.e. self-application.
fn handle_learn_self_command(
    input: &str,
    g: &mut Graph,
    env: &mut HashMap<String, NodeId>,
    library: &mut Vec<NodeId>,
) {
    let inner = input.trim_start_matches("(learn-self").trim_end_matches(')').trim();

    // Find the term list (first parenthesized expression) and epochs (last number)
    let mut depth = 0;
    let mut list_start = None;
    let mut list_end = None;

    for (i, c) in inner.chars().enumerate() {
        match c {
            '(' => {
                if depth == 0 {
                    list_start = Some(i);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 && list_start.is_some() && list_end.is_none() {
                    list_end = Some(i + 1);
                }
            }
            _ => {}
        }
    }

    let (list_str, rest) = if let (Some(start), Some(end)) = (list_start, list_end) {
        (&inner[start..end], inner[end..].trim())
    } else {
        println!("Error: Could not find term list in learn-self");
        return;
    };

    let epochs: usize = rest.parse().unwrap_or(100);
    let list_content = list_str.trim_start_matches('(').trim_end_matches(')').trim();

    let mut examples = Vec::new();
    let mut p = Parser::new(list_content);
    let mut ctx = EvalContext::default();
    ctx.step_limit = 10_000;

    while p.has_more() {
        if let Ok(ParseResult::Term(term)) = p.parse_toplevel(g, Some(env)) {
            let input = reduce(g, term, &mut ctx);
            let app = g.add(Node::App { func: input, args: smallvec::smallvec![input] });
            let mut out_ctx = EvalContext::default();
            out_ctx.step_limit = 10_000;
            let output = reduce(g, app, &mut out_ctx);
            if out_ctx.steps >= out_ctx.step_limit {
                println!("Warning: self-apply example hit step limit; skipping.");
                continue;
            }
            examples.push((input, output));
        } else {
            break;
        }
    }

    if examples.is_empty() {
        println!("Error: Empty examples list.");
        return;
    }

    println!("Parsed {} examples. Running IGTC synthesis for {} iterations...", examples.len(), epochs);

    let mut config = trace::learner::IgtcConfig::default();
    config.max_iterations = epochs;

    if let Some(learned_node) = trace::learner::igtc_synthesize_with_seeds(g, examples, config, &library) {
        println!("Success! Learned Node: {}", unparse(g, learned_node));
        env.insert("learned".to_string(), learned_node);
        println!("Saved as 'learned'.");
        library.push(learned_node);
    } else {
        println!("IGTC synthesis failed to converge.");
    }
}
