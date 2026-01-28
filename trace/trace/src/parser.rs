use crate::arena::{Graph, Node, NodeId, Primitive};
use crate::compiler::CompileTerm;
use crate::engine::{encode_int, encode_str};
use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;
use num_bigint::BigInt;

#[derive(Debug, PartialEq)]
pub enum Token {
    LParen,
    RParen,
    Symbol(String),
    String(String),
}

struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { chars: input.chars().peekable() }
    }

    fn next_token(&mut self) -> Option<Token> {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() || c == ',' {
                self.chars.next();
                continue;
            }
            match c {
                '(' => { self.chars.next(); return Some(Token::LParen); }
                ')' => { self.chars.next(); return Some(Token::RParen); }
                ';' => {
                    while let Some(&x) = self.chars.peek() {
                        if x == '\n' { break; }
                        self.chars.next();
                    }
                    continue;
                }
                '"' => {
                    self.chars.next();
                    let mut s = String::new();
                    while let Some(&x) = self.chars.peek() {
                        if x == '"' { self.chars.next(); return Some(Token::String(s)); }
                        if x == '\\' {
                             self.chars.next();
                             if let Some(&nc) = self.chars.peek() {
                                 s.push(nc); // Simplified escape
                                 self.chars.next();
                             }
                        } else {
                            s.push(x);
                            self.chars.next();
                        }
                    }
                    return Some(Token::String(s));
                }
                _ => {
                    let mut s = String::new();
                    while let Some(&x) = self.chars.peek() {
                        if x.is_whitespace() || x == '(' || x == ')' || x == ';' { break; }
                        s.push(x);
                        self.chars.next();
                    }
                    return Some(Token::Symbol(s));
                }
            }
        }
        None
    }
}

pub struct Parser<'a> {
    undo: Option<Token>,
    lexer: Lexer<'a>,
}

pub enum ParseResult {
    Term(NodeId),
    Def(String, NodeId),
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { undo: None, lexer: Lexer::new(input) }
    }

    fn peek(&mut self) -> Option<&Token> {
        if self.undo.is_none() {
            self.undo = self.lexer.next_token();
        }
        self.undo.as_ref()
    }

    fn consume(&mut self) -> Option<Token> {
        if let Some(t) = self.undo.take() { Some(t) } else { self.lexer.next_token() }
    }
    
    pub fn has_more(&mut self) -> bool {
        self.peek().is_some()
    }

    pub fn parse_toplevel(&mut self, g: &mut Graph, env: Option<&HashMap<String, NodeId>>) -> Result<ParseResult, String> {
         // Check for (def ...)
         if let Some(Token::LParen) = self.peek() {
             // We can't easily peek deep.
             // But we can parse as expression.
             // If manual check needed:
             self.consume();
             if let Some(Token::Symbol(s)) = self.peek() {
                 if s == "def" {
                     self.consume();
                     let name = match self.consume() {
                         Some(Token::Symbol(n)) => n,
                         _ => return Err("Expected name after def".into()),
                     };
                     let val_term = self.parse_expr(g, env, &[])?;
                     if let Some(Token::RParen) = self.consume() {
                         let val = crate::compiler::compile(g, val_term)?;
                         return Ok(ParseResult::Def(name, val));
                     } else {
                         return Err("Expected )".into());
                     }
                 }
             }
             // Not def, backtrack?
             // Actually `parse_list_inner` expects to start AFTER LParen?
             // Or `parse_expr` handles it?
             // `parse_expr` consumes LParen.
             // Since I consumed LParen already to check for 'def', I am inside the list.
             // I need to parse the rest of the list.
             // But my `parse_expr` expects to consume LParen.
             // I'll create `parse_list_inner`.
             let expr = self.parse_list_rest(g, env, &[])?;
             let val = crate::compiler::compile(g, expr)?;
             return Ok(ParseResult::Term(val));
         }
         
         let expr = self.parse_expr(g, env, &[])?;
         let val = crate::compiler::compile(g, expr)?;
         Ok(ParseResult::Term(val))
    }

    fn parse_list_rest(&mut self, g: &mut Graph, env: Option<&HashMap<String, NodeId>>, bound: &[String]) -> Result<CompileTerm, String> {
        self.parse_with_stack(g, env, bound, vec![Context::new(bound)])
    }

    fn parse_expr(&mut self, g: &mut Graph, env: Option<&HashMap<String, NodeId>>, bound: &[String]) -> Result<CompileTerm, String> {
        self.parse_with_stack(g, env, bound, Vec::new())
    }

    fn parse_with_stack(
        &mut self,
        g: &mut Graph,
        env: Option<&HashMap<String, NodeId>>,
        bound: &[String],
        mut stack: Vec<Context>,
    ) -> Result<CompileTerm, String> {
        let top_bound = bound.to_vec();

        loop {
            let token = self.consume().ok_or("Unexpected EOF".to_string())?;
            match token {
                Token::LParen => {
                    if let Some(ctx) = stack.last() {
                        if matches!(ctx.stage, ListStage::FnParam) {
                            return Err("fn expects param".into());
                        }
                        stack.push(Context::new(&ctx.bound));
                    } else {
                        stack.push(Context::new(&top_bound));
                    }
                }
                Token::RParen => {
                    let ctx = stack.pop().ok_or("Unexpected )".to_string())?;
                    let term = ctx.finish()?;
                    if let Some(done) = push_or_return(term, &mut stack)? {
                        return Ok(done);
                    }
                }
                Token::Symbol(s) => {
                    if let Some(ctx) = stack.last_mut() {
                        ctx.consume_symbol(g, env, s)?;
                    } else {
                        return symbol_to_term(g, env, &top_bound, s);
                    }
                }
                Token::String(s) => {
                    let raw = encode_str(g, &s);
                    let tagged = crate::engine::make_tag(g, Primitive::TagStr, raw);
                    let term = CompileTerm::Const(tagged);
                    if let Some(done) = push_or_return(term, &mut stack)? {
                        return Ok(done);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ListStage {
    Start,
    FnParam,
    FnBody,
    Normal,
}

struct Context {
    elems: Vec<CompileTerm>,
    bound: Vec<String>,
    stage: ListStage,
    fn_param: Option<String>,
}

impl Context {
    fn new(bound: &[String]) -> Self {
        Self {
            elems: Vec::new(),
            bound: bound.to_vec(),
            stage: ListStage::Start,
            fn_param: None,
        }
    }

    fn push_elem(&mut self, term: CompileTerm) -> Result<(), String> {
        match self.stage {
            ListStage::Start => {
                self.stage = ListStage::Normal;
                self.elems.push(term);
                Ok(())
            }
            ListStage::FnParam => Err("fn expects param".into()),
            ListStage::FnBody => {
                if !self.elems.is_empty() {
                    return Err("fn expects single body".into());
                }
                self.elems.push(term);
                Ok(())
            }
            ListStage::Normal => {
                self.elems.push(term);
                Ok(())
            }
        }
    }

    fn consume_symbol(
        &mut self,
        g: &mut Graph,
        env: Option<&HashMap<String, NodeId>>,
        s: String,
    ) -> Result<(), String> {
        match self.stage {
            ListStage::Start => {
                if s == "fn" {
                    self.stage = ListStage::FnParam;
                    return Ok(());
                }
                let term = symbol_to_term(g, env, &self.bound, s)?;
                self.stage = ListStage::Normal;
                self.elems.push(term);
                Ok(())
            }
            ListStage::FnParam => {
                self.fn_param = Some(s.clone());
                self.bound.push(s);
                self.stage = ListStage::FnBody;
                Ok(())
            }
            ListStage::FnBody | ListStage::Normal => {
                let term = symbol_to_term(g, env, &self.bound, s)?;
                self.elems.push(term);
                Ok(())
            }
        }
    }

    fn finish(mut self) -> Result<CompileTerm, String> {
        match self.stage {
            ListStage::Start => Err("Empty list".into()),
            ListStage::FnParam => Err("fn missing param/body".into()),
            ListStage::FnBody => {
                let param = self.fn_param.take().ok_or("fn missing param")?;
                if self.elems.len() != 1 {
                    return Err("fn expects single body".into());
                }
                Ok(CompileTerm::Lam(param, Box::new(self.elems.remove(0))))
            }
            ListStage::Normal => {
                if self.elems.is_empty() {
                    return Err("Empty list".into());
                }
                let mut iter = self.elems.into_iter();
                let mut term = iter.next().unwrap();
                for arg in iter {
                    term = CompileTerm::App(Box::new(term), Box::new(arg));
                }
                Ok(term)
            }
        }
    }
}

fn push_or_return(term: CompileTerm, stack: &mut Vec<Context>) -> Result<Option<CompileTerm>, String> {
    if let Some(parent) = stack.last_mut() {
        parent.push_elem(term)?;
        Ok(None)
    } else {
        Ok(Some(term))
    }
}

fn symbol_to_term(
    g: &mut Graph,
    env: Option<&HashMap<String, NodeId>>,
    bound: &[String],
    s: String,
) -> Result<CompileTerm, String> {
    if bound.contains(&s) {
        return Ok(CompileTerm::Var(s));
    }
    if s == "n" {
        return Ok(CompileTerm::Const(g.add(Node::Leaf)));
    }
    if let Ok(i) = s.parse::<BigInt>() {
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            let raw = encode_int(g, &i);
            let tagged = crate::engine::make_tag(g, Primitive::TagInt, raw);
            return Ok(CompileTerm::Const(tagged));
        }
    }

    if let Ok(f) = s.parse::<f64>() {
        let raw = g.add(Node::Float(f));
        let tagged = crate::engine::make_tag(g, Primitive::TagFloat, raw);
        return Ok(CompileTerm::Const(tagged));
    }
    if let Some(e) = env {
        if let Some(&id) = e.get(&s) {
            return Ok(CompileTerm::Const(id));
        }
    }

    let prim = match s.as_str() {
        "+" => Some(Primitive::Add),
        "-" => Some(Primitive::Sub),
        "*" => Some(Primitive::Mul),
        "/" => Some(Primitive::Div),
        "if" => Some(Primitive::If),
        "trace" => Some(Primitive::Trace),
        _ => None,
    };
    if let Some(p) = prim {
        return Ok(CompileTerm::Const(g.add(Node::Prim(p))));
    }

    Err(format!("Unknown symbol {}", s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Node;

    #[test]
    fn test_parse_basic() {
        let mut g = Graph::new();
        let mut p = Parser::new("n");
        let res = p.parse_toplevel(&mut g, None).unwrap();
        match res {
            ParseResult::Term(id) => match g.get(id) {
                Node::Leaf => {},
                _ => panic!("Expected Leaf"),
            },
            _ => panic!("Expected Term"),
        }
    }

    #[test]
    fn test_parse_list() {
        let mut g = Graph::new();
        let mut p = Parser::new("(n n)");
        let res = p.parse_toplevel(&mut g, None).unwrap();
        // (n n) should parse as an App; reduction handles Leaf->Stem later.
        if let ParseResult::Term(id) = res {
             match g.get(id) {
                 Node::App { func, args } => {
                     assert!(matches!(g.get(*func), Node::Leaf));
                     assert_eq!(args.len(), 1);
                     assert!(matches!(g.get(args[0]), Node::Leaf));
                 }
                 _ => panic!("Expected App(n, [n]), got {:?}", g.get(id)),
             }
        }
    }

    #[test]
    fn test_parse_lambda() {
        let mut g = Graph::new();
        let mut p = Parser::new("(fn x x)");
        let res = p.parse_toplevel(&mut g, None).unwrap();
        // Should compile to Identity.
        // I = S (n n) n ? Or reduced?
        // Just check it parses without error.
        assert!(matches!(res, ParseResult::Term(_)));
    }
}
