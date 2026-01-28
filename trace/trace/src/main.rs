use trace::arena::{Graph, NodeId, Node};
use trace::parser::{Parser, ParseResult};
use trace::engine::{reduce, unparse, EvalContext};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

use trace::inference::InferenceEngine;
use trace::types::{TypeEnv, Type};



fn main() {
    println!("Trace REPL");
    println!("(n (n n)) based syntax. Type (exit) or Ctrl+C to quit.");

    let mut g = Graph::new();
    let mut env: HashMap<String, NodeId> = HashMap::new();
    let mut type_env = TypeEnv::new();
    let mut typecheck_enabled = true;

    // Standard Library (Prelude)
    let prelude = [
        // Pure tree combinators (factorable)
        ("k_tree", "(n n)"),
        ("s_tree", "(n (n (k_tree n)) n)"),
        ("i_tree", "(s_tree k_tree k_tree)"),
        // Standard names (all tree-based)
        ("k", "k_tree"),
        ("s", "s_tree"),
        ("i", "i_tree"),
        ("K", "k_tree"),
        ("S", "s_tree"),
        ("I", "i_tree"),
        // tree_book: self_apply = λ* w. w w = d{I} I, and d{x} = n (n x)
        ("self_apply", "(n (n i) i)"),
        ("true", "k"),
        ("false", "n"),
        ("triage", "(fn w (fn x (fn y (n (n w x) y))))"),
        ("is-leaf", "(fn z ((triage true (fn u false) (fn u (fn v false))) z))"),
        ("is-stem", "(fn z ((triage false (fn u true) (fn u (fn v false))) z))"),
        ("is-fork", "(fn z ((triage false (fn u false) (fn u (fn v true))) z))"),
        ("d", "(fn x (n (n x)))"),
        // Fixpoint helpers from tree_book
        // ω = λ* z. λ* f. f (z z f)
        ("omega", "(fn z (fn f (f (z z f))))"),
        // wait{x,y} = d{I}(d{K y}(K x))
        ("wait", "(fn x (fn y (d i (d (k y) (k x)))))"),
        // wait1{x} = d{d{K(K x)}(d{d{K}(K△)}(K△))}(K(d{△KK}))
        ("wait1", "(fn x (d (d (k (k x)) (d (d (k) (k n)) (k n))) (k (d (n (k k))))))"),
        // wait2{x,y} = d{d{K(d{K y}(K x))} d{d{K}(K△)}(K△)}(K(d{I}))
        ("wait2", "(fn x (fn y (d (d (k (d (k y) (k x))) (d (d (k) (k n)) (k n))) (k (d i)))))"),
        // swap{f} = d{K f}d{d{K}(K△)}(K△)
        ("swap", "(fn f (d (k f) (d (d (k) (k n)) (k n))))"),
        // head/tail from tree_book (triage on fork)
        ("head", "(triage n n k)"),
        ("tail", "(triage n n (k i))"),
        ("first", "head"),
        ("rest", "tail"),
        // Z{f} = wait{self_apply, d{wait1{self_apply}(K f)}}
        ("Z", "(fn f (wait self_apply (d (wait1 self_apply (k f)))))"),
        ("stem", "(fn x (n x))"),
        ("fork", "(fn a (fn b (n a b)))"),
        ("cons", "fork"),
        ("leaf", "n"),
        // tree wrappers for primitives
        ("add", "(fn x (fn y (+ x y)))"),
        ("sub", "(fn x (fn y (- x y)))"),
        ("mul", "(fn x (fn y (* x y)))"),
        ("div", "(fn x (fn y (/ x y)))"),
        ("if_tree", "(fn c (fn t (fn f (if c t f))))"),
        ("trace_tree", "(fn x (trace x))"),
    ];

    // Library of learned/defined programs for curriculum learning
    let mut library: Vec<NodeId> = Vec::new();

    // Polymorphic types for core tree combinators
    let type_k = Type::Forall(
        "A".into(),
        Box::new(Type::Forall(
            "B".into(),
            Box::new(Type::Arrow(
                Box::new(Type::Generic("A".into())),
                Box::new(Type::Arrow(
                    Box::new(Type::Generic("B".into())),
                    Box::new(Type::Generic("A".into())),
                )),
            )),
        )),
    );
    let type_i = Type::Forall(
        "A".into(),
        Box::new(Type::Arrow(
            Box::new(Type::Generic("A".into())),
            Box::new(Type::Generic("A".into())),
        )),
    );
    let type_s = Type::Forall(
        "A".into(),
        Box::new(Type::Forall(
            "B".into(),
            Box::new(Type::Forall(
                "C".into(),
                Box::new(Type::Arrow(
                    Box::new(Type::Arrow(
                        Box::new(Type::Generic("A".into())),
                        Box::new(Type::Arrow(
                            Box::new(Type::Generic("B".into())),
                            Box::new(Type::Generic("C".into())),
                        )),
                    )),
                    Box::new(Type::Arrow(
                        Box::new(Type::Arrow(
                            Box::new(Type::Generic("A".into())),
                            Box::new(Type::Generic("B".into())),
                        )),
                        Box::new(Type::Arrow(
                            Box::new(Type::Generic("A".into())),
                            Box::new(Type::Generic("C".into())),
                        )),
                    )),
                )),
            )),
        )),
    );

    println!("Loading standard library...");
    let std_names: HashSet<String> = prelude.iter().map(|(name, _)| (*name).to_string()).collect();
    for (name, code) in prelude {
        let mut p = Parser::new(code);
         if let Ok(ParseResult::Term(node)) = p.parse_toplevel(&mut g, Some(&env)) {
             // Verify type
             let mut engine = InferenceEngine::new();
             
             let mut ctx = EvalContext::default();
             let reduced = reduce(&mut g, node, &mut ctx);
             env.insert(name.to_string(), reduced);

             // Install special polymorphic types for core combinators
             if name == "k_tree" {
                 type_env.specials.insert(reduced.0, type_k.clone());
                 type_env.vars.insert(name.to_string(), type_k.clone());
             } else if name == "i_tree" {
                 type_env.specials.insert(reduced.0, type_i.clone());
                 type_env.vars.insert(name.to_string(), type_i.clone());
             } else if name == "s_tree" {
                 type_env.specials.insert(reduced.0, type_s.clone());
                 type_env.vars.insert(name.to_string(), type_s.clone());
             } else {
                 if let Ok(ty) = engine.infer(&g, reduced, &type_env) {
                     type_env.vars.insert(name.to_string(), ty);
                 }
             }
             
             // Add valid functions to library, BUT EXCLUDE 'swap' so it must be learned
             // if name != "swap" {
             //     library.push(reduced);
             // }
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
        
        // Handle typecheck toggle
        if trimmed == "(typecheck off)" {
            typecheck_enabled = false;
            println!("Typecheck disabled.");
            continue;
        }
        if trimmed == "(typecheck on)" {
            typecheck_enabled = true;
            println!("Typecheck enabled.");
            continue;
        }

        // Handle (learn-from ...)
        if trimmed.starts_with("(learn-from") {
             handle_learn_from_command(trimmed, &mut g, &mut env, &std_names, &mut library);
             continue;
        }
        // Handle (learn-self ...)
        if trimmed.starts_with("(learn-self") {
             handle_learn_self_command(trimmed, &mut g, &mut env, &std_names, &mut library);
             continue;
        }


        let mut parser = Parser::new(&input);
        match parser.parse_toplevel(&mut g, Some(&env)) {
            Ok(res) => {
                match res {
                    ParseResult::Term(node) => {
                        if typecheck_enabled {
                            let mut engine = InferenceEngine::new();
                            match engine.infer(&g, node, &type_env) {
                                Ok(ty) => println!(" : {:?}", ty),
                                Err(e) => println!("Type Error: {}", e),
                            }
                        }
                        
                        let mut ctx = EvalContext::default();
                        let result = reduce(&mut g, node, &mut ctx);
                        if ctx.steps >= ctx.step_limit {
                            println!("Warning: step limit reached; result may be a partial reduction.");
                        }
                        println!("= {}", unparse(&g, result));
                    }
                    ParseResult::Def(name, node) => {
                         if typecheck_enabled {
                             let mut engine = InferenceEngine::new();
                             match engine.infer(&g, node, &type_env) {
                                 Ok(ty) => {
                                     type_env.vars.insert(name.clone(), ty.clone());
                                     println!(" : {:?}", ty);
                                 },
                                 Err(e) => println!("Type Error: {}", e),
                             }
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



fn filtered_env(env: &HashMap<String, NodeId>, allowed: &HashSet<String>) -> HashMap<String, NodeId> {
    let mut out = HashMap::new();
    for name in allowed {
        if let Some(id) = env.get(name) {
            out.insert(name.clone(), *id);
        }
    }
    out
}

/// Handle learn-from command: (learn-from ((in out) ...) epochs [support_cap] [max_eval_steps])
fn handle_learn_from_command(
    input: &str, 
    g: &mut Graph, 
    env: &mut HashMap<String, NodeId>,
    std_names: &HashSet<String>,
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
    
    // Parse epochs, optional support cap, and optional max_eval_steps from rest.
    // Also accept key=value tokens like steps=500.
    let mut nums: Vec<usize> = Vec::new();
    let mut max_eval_steps_override: Option<usize> = None;
    for tok in rest.split_whitespace() {
        if let Some((key, value)) = tok.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            if matches!(key, "steps" | "max_steps" | "max_eval_steps") {
                if let Ok(n) = value.parse() {
                    max_eval_steps_override = Some(n);
                }
            }
            continue;
        }
        if let Ok(n) = tok.parse() {
            nums.push(n);
        }
    }
    let epochs: usize = nums.get(0).cloned().unwrap_or(100);
    let support_cap: Option<usize> = nums.get(1).cloned();
    let max_eval_steps: Option<usize> = max_eval_steps_override.or_else(|| nums.get(2).cloned());
    
    // Now parse the examples list content (without outer parens)
    let examples_content = examples_str.trim_start_matches('(').trim_end_matches(')').trim();

    // Split into top-level parenthesized pairs to avoid App/Fork canonicalization
    let mut pairs: Vec<String> = Vec::new();
    let mut depth = 0i32;
    let mut start: Option<usize> = None;
    for (i, c) in examples_content.char_indices() {
        match c {
            '(' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        pairs.push(examples_content[s..=i].to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let learn_env = filtered_env(env, std_names);
    // Parse each example pair: (input output) by parsing two terms inside
    let mut examples = Vec::new();
    for pair in pairs {
        let mut inner = pair.trim();
        if let Some(stripped) = inner.strip_prefix('(') {
            inner = stripped;
        }
        if let Some(stripped) = inner.strip_suffix(')') {
            inner = stripped;
        }
        let inner = inner.trim();
        let mut p = Parser::new(inner);
        let in_term = match p.parse_toplevel(g, Some(&learn_env)) {
            Ok(ParseResult::Term(t)) => t,
            _ => {
                println!("Warning: Skipping malformed example pair: {}", inner);
                continue;
            }
        };
        let out_term = match p.parse_toplevel(g, Some(&learn_env)) {
            Ok(ParseResult::Term(t)) => t,
            _ => {
                println!("Warning: Skipping malformed example pair: {}", inner);
                continue;
            }
        };
        let mut in_ctx = EvalContext::default();
        let in_val = reduce(g, in_term, &mut in_ctx);
        let mut out_ctx = EvalContext::default();
        let out_val = reduce(g, out_term, &mut out_ctx);
        examples.push((in_val, out_val));
    }
    
    if examples.is_empty() {
        println!("Error: Empty examples list.");
        return;
    }

    println!(
        "Parsed {} examples. Running counterfactual synthesis for {} iterations...",
        examples.len(),
        epochs
    );
    
    // Use counterfactual single-candidate learner
    let mut config = trace::learner::CounterfactualConfig::default();
    config.max_iterations = epochs;
    config.verbose = true;
    if let Some(cap) = support_cap {
        config.max_edits_per_iter = cap;
    }
    if let Some(steps) = max_eval_steps {
        config.max_eval_steps = steps.max(1);
        println!("  max_eval_steps={}", config.max_eval_steps);
    }
    // Keep other defaults (thresholds, variance weight)
    
    if let Some(learned_node) = trace::learner::counterfactual_synthesize(g, examples, config) {
        println!("Success! Learned Node: {}", unparse(g, learned_node));
        
        env.insert("learned".to_string(), learned_node);
        println!("Saved as 'learned'.");
        
        // Curriculum: add to library for future synthesis
        library.push(learned_node);
    } else {
        println!("Counterfactual synthesis failed to converge.");
    }
}

/// Handle learn-self command: (learn-self (t1 t2 ...) epochs [max_eval_steps])
/// Builds examples where output = (t t) reduced, i.e. self-application.
fn handle_learn_self_command(
    input: &str,
    g: &mut Graph,
    env: &mut HashMap<String, NodeId>,
    std_names: &HashSet<String>,
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

    let mut nums: Vec<usize> = Vec::new();
    let mut max_eval_steps_override: Option<usize> = None;
    for tok in rest.split_whitespace() {
        if let Some((key, value)) = tok.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            if matches!(key, "steps" | "max_steps" | "max_eval_steps") {
                if let Ok(n) = value.parse() {
                    max_eval_steps_override = Some(n);
                }
            }
            continue;
        }
        if let Ok(n) = tok.parse() {
            nums.push(n);
        }
    }
    let epochs: usize = nums.get(0).cloned().unwrap_or(100);
    let max_eval_steps: Option<usize> = max_eval_steps_override.or_else(|| nums.get(1).cloned());
    let list_content = list_str.trim_start_matches('(').trim_end_matches(')').trim();

    let learn_env = filtered_env(env, std_names);
    let mut examples = Vec::new();
    let mut p = Parser::new(list_content);
    while p.has_more() {
        if let Ok(ParseResult::Term(term)) = p.parse_toplevel(g, Some(&learn_env)) {
            let mut in_ctx = EvalContext::default();
            in_ctx.step_limit = 10_000;
            let input = reduce(g, term, &mut in_ctx);
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

    println!(
        "Parsed {} examples. Running counterfactual synthesis for {} iterations...",
        examples.len(),
        epochs
    );

    let mut config = trace::learner::CounterfactualConfig::default();
    config.max_iterations = epochs;
    config.verbose = true;
    let steps = max_eval_steps.unwrap_or(10_000);
    config.max_eval_steps = steps.max(1);
    println!("  max_eval_steps={}", config.max_eval_steps);

    if let Some(learned_node) = trace::learner::counterfactual_synthesize(g, examples, config) {
        println!("Success! Learned Node: {}", unparse(g, learned_node));
        env.insert("learned".to_string(), learned_node);
        println!("Saved as 'learned'.");
        library.push(learned_node);
    } else {
        println!("Counterfactual synthesis failed to converge.");
    }
}
