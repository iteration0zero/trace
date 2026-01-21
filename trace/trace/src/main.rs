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
        ("true", "k"),
        ("false", "n"),
        ("triage", "(fn w (fn x (fn y (n (n w x) y))))"),
        ("is-leaf", "(fn z ((triage true (fn u false) (fn u (fn v false))) z))"),
        ("is-stem", "(fn z ((triage false (fn u true) (fn u (fn v false))) z))"),
        ("is-fork", "(fn z ((triage false (fn u false) (fn u (fn v true))) z))"),
        // first/rest are now primitives
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
             handle_learn_from_command(trimmed, &mut g, &mut env);
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



/// Handle learn-from command: (learn-from [epochs] ((in out) ...))
fn handle_learn_from_command(
    input: &str, 
    g: &mut Graph, 
    env: &mut HashMap<String, NodeId>
) {
    let inner = input.trim_start_matches("(learn-from").trim_end_matches(')').trim();
    
    // Parse: Count and Examples (Auto-detected via type)

    // Robust Parsing using Parser loop
    let mut p = Parser::new(inner);
    let mut terms = Vec::new();
    
    // Parse all terms in inner
    while p.has_more() {
        if let Ok(ParseResult::Term(node)) = p.parse_toplevel(g, Some(env)) {
             terms.push(node);
        } else {
             break;
        }
    }
    
    let mut epochs = 100;
    let mut examples_raw = None;
    
    for term in terms {
        let mut ctx = EvalContext::default();
        let val = reduce(g, term, &mut ctx);
        match g.get(g.resolve(val)).clone() {
            Node::Prim(Primitive::TagInt) => {
                if let Some(bi) = trace::engine::decode_int(g, val) {
                     if let Some(n) = bi.to_usize() { epochs = n; }
                }
            },
            _ => {
                // Assume structure is the examples list
                // It might not be reduced to List logic, but Application logic
                examples_raw = Some(term); // Use unreduced term to preserve App structure?
                // reduce() evaluates (cons a b) to Fork.
                // But ( (a b) (c d) ) reduces to... application?
                // If I use unreduced 'term', I can flatten Apps.
            }
        }
    }
    
    let raw_list = if let Some(n) = examples_raw { n } else {
        println!("Error: No examples found.");
        return;
    };
    
    // Flatten App chain: (a b c) -> App(App(a,b),c)
    // We want [a, b, c]
    fn flatten_apps(g: &Graph, node: NodeId, acc: &mut Vec<NodeId>) {
        // Resolve? App is structural.
        match g.get(node).clone() {
            Node::App { func, args } => {
                // Triage App has vec of args. App(f, [a1, a2]) -> f a1 a2
                // Flatten f, then add args.
                flatten_apps(g, func, acc);
                for arg in args {
                    acc.push(arg);
                }
            },
            _ => acc.push(node)
        }
    }
    
    let mut items = Vec::new();
    flatten_apps(g, raw_list, &mut items);
    
    let mut examples = Vec::new();
    let mut ctx = EvalContext::default();
    
    for item in items {
        // Each item is an Example Pair (in out).
        // Parsed as App(in, out)
        // Check structural App
        match g.get(item).clone() {
            Node::App { func, args } => {
                if args.len() == 1 {
                     // App(in, out)
                     let input = func;
                     let output = args[0];
                     
                     // We should REDUCE them now to get Values
                     let in_val = reduce(g, input, &mut ctx);
                     let out_val = reduce(g, output, &mut ctx);
                     examples.push((in_val, out_val));
                } else {
                    println!("Error: Malformed example pair (multi-args): {:?}", g.get(item));
                }
            },
            // Maybe it reduced to Fork if valid cons?
            Node::Fork(cur_in, cur_tail) => {
                 // Handle list-style (cons in (cons out n)) if user used cons
                 // But (learn-from ( (a b) )) uses parens, likely App.
                 // This branch is fallback.
                 let out_node = match g.get(g.resolve(cur_tail)).clone() {
                     Node::Fork(out, _) => out,
                     _ => cur_tail
                 };
                 examples.push((cur_in, out_node));
            }
            _ => {
                println!("Error: Example item is not a pair (App or Fork). Got {:?}", g.get(item));
            }
        }
    }
    
    if examples.is_empty() {
        println!("Error: Empty examples list.");
        return;
    }

    println!("Parsed {} examples. Evolving for {} epochs...", examples.len(), epochs);
    
    let config = SearchConfig {
        max_epochs: epochs,
        max_depth: 3, 
        lr: 0.1,
    };
    
    if let Some(gene) = evolve(g, &examples, config) {
        println!("Success! Learned Gene: {:?}", gene);
        let learned_node = gene.compile(g); 
        println!("Compiled Node: {}", unparse(g, learned_node));
        
        env.insert("learned".to_string(), learned_node);
        println!("Saved as 'learned'.");
    } else {
        println!("Evolution failed to converge.");
    }
}
