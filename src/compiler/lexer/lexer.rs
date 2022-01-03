use super::{
    source::Source,
    token::{
        BinOp, Comment, Keyword, NumFlag, Operator, Token,
    },
};

use crate::prelude::{
    span::{Location, Positioned},
    traits::Peek,
};

pub const COMMENT_OUTER: char = '*';
pub const COMMENT_INNER: char = '~';

// TODO! DOC COMMENTS
// pub const COMMENT_DOC_A: char = '>';
pub const ESCAPE: char = '\\';
pub const UNDER: char = '_';
pub const PRIME: char = '\'';
pub const STR_PREFIXES: &'static str = "brfm";
pub const OP_CHARS: &'static str = "!$%^&*-+/|<>=?:\\@#`";

#[derive(Clone, Debug)]
pub struct Lexer<'t> {
    source: Source<'t>,
    current: Option<Token>,
    queue: Vec<Token>,
    pub spans: Vec<Location>,
}

impl<'t> Positioned for Lexer<'t> {
    type Loc = Location;

    fn loc(&self) -> Self::Loc {
        self.source.loc()
    }
}

impl<'t> Lexer<'t> {
    pub fn new(src: &'t str) -> Self {
        Self {
            source: Source::new(src),
            current: None,
            queue: vec![],
            spans: vec![],
        }
    }

    fn peek_char(&mut self) -> Option<&char> {
        self.source.peek()
    }

    fn next_char(&mut self) -> Option<char> {
        self.source.next()
    }

    fn next_while<F>(&mut self, pred: F) -> String
    where
        F: Fn(char) -> bool,
    {
        let mut buf = String::new();
        while !self.is_done() {
            if let Some(c) = self.peek_char() {
                if pred(*c) {
                    buf.push(*c);
                    self.next_char();
                    continue;
                }
            }
            break;
        }
        buf
    }

    /// Advances the stream of characters based on a given predicate `pred`. Note that `pred` is [`FnMut`] which, unlike the predicate in `next_while`, allows for mutating captured variables.
    fn take_while<F>(&mut self, mut pred: F) -> String
    where
        F: FnMut(char) -> bool,
    {
        let mut buf = String::new();
        while !self.is_done() {
            if let Some(c) = self.peek_char() {
                if pred(*c) {
                    buf.push(*c);
                    self.next_char();
                    continue;
                }
                break;
            }
        }
        buf
    }

    fn eat_whitespace(&mut self) {
        self.next_while(|c| c.is_whitespace());
    }

    fn token(&mut self) -> Token {
        if let Some(t) = self.queue.pop() {
            return t;
        };
        if self.is_done() {
            return Token::Eof;
        };
        self.eat_whitespace();
        // since char is Copy, and we have to mutably borrow the stream afterwards
        if let Some(c) = self.peek_char().copied() {
            self.lex(c)
        } else {
            Token::Eof
        }
    }

    fn lex(self: &mut Self, c: char) -> Token {
        match c {
            COMMENT_INNER => self.comment(),
            ';' => {
                self.next_char();
                Token::Semi
            }
            '\'' => self.character(),
            '"' => self.string(),
            ':' => self.colon(),
            c if c.is_digit(10) => self.number(),
            c if is_ident_start(c) => {
                if STR_PREFIXES.contains(c) {
                    self.prefix(c)
                } else {
                    self.ident(c)
                }
            }
            t if "()[]{},;".contains(t) => {
                self.next_char();
                use Token::*;
                match t {
                    '(' => ParenL,
                    ')' => ParenR,
                    '[' => BrackL,
                    ']' => BrackR,
                    '{' => CurlyL,
                    '}' => CurlyR,
                    ',' => Comma,
                    ';' => Semi,
                    _ => unreachable!(),
                }
            }
            '.' => self.dot(),
            '\\' => self.lambda(),
            c if OP_CHARS.contains(c) => self.operator(),
            _ => self.unknown(c),
        }
    }

    fn lambda(&mut self) -> Token {
        self.next_char();
        match self.peek_char() {
            Some(':' | '.') => {
                let rhs = self.token();
                self.queue.push(rhs);
                Token::Lambda
            }
            Some(c) if OP_CHARS.contains(*c) => {
                Token::Operator(Operator::Custom(format!("\\{}",self.next_while(|c| OP_CHARS.contains(c)))))
            }
            _ => Token::Lambda
        }
    }

    fn block_comment(&mut self) -> Token {
        // let pos = self.loc();
        let mut penult = false;
        let comment =
            self.take_while(|c| match (penult, c) {
                (true, COMMENT_INNER) => false,
                (true, COMMENT_OUTER)
                | (false, COMMENT_INNER) => true,
                (false, COMMENT_OUTER) => {
                    penult = true;
                    true
                }
                (true, _) => {
                    penult = false;
                    true
                }
                (false, _) => true,
            });
        self.next_char();
        Token::Comment(Comment::Block(comment))
    }

    fn character(&mut self) -> Token {
        let pos = self.loc();
        self.next_char();
        match self.peek_char().copied() {
            Some(c) => {
                if c == ESCAPE {
                    self.next_char();
                    match self.peek_char().copied() {
                        Some(c) if is_escapable(c) => {
                            self.next_char();
                            if let Some('\'') = self.peek_char() {
                                self.next_char();
                                Token::Char(get_escaped(c))
                            } else {
                                self.next_char();
                                Token::Invalid {
                                    data: format!("{}", c),
                                    msg: format!(
                                        "Unclosed character! Expected `'` after escape {}",
                                        c
                                    ),
                                    pos,
                                }
                            }
                        }
                        invalid @ _ => {
                            self.next_char();
                            Token::Invalid {
                                data: format!("{:?}", invalid),
                                msg: format!("Invalid character escape!"),
                                pos,
                            }
                        }
                    }
                } else {
                    self.next_char();
                    if let Some('\'') = self.next_char() {
                        Token::Char(c)
                    } else {
                        Token::Invalid {
                            data: format!("{}", c),
                            msg: format!("Unclosed character! Expected `'` after char {}", c),
                            pos,
                        }
                    }
                }
            }
            None => Token::Invalid {
                data: "'".into(),
                msg: "Unexpected end of input after first `'`!".into(),
                pos,
            },
        }
    }

    fn colon(&mut self) -> Token {
        self.next_char();
        match self.peek_char().copied() {
            Some(':') => {
                self.next_char();
                Token::Colon2
            }
            Some('=') => {
                self.next();
                Token::ColonEq
            }
            Some(c) if OP_CHARS.contains(c) => {
                Token::Ident(format!(
                    ":{}",
                    self.next_while(
                        |c| OP_CHARS.contains(c)
                    )
                ))
            }
            Some(c) if is_ident_start(c) => Token::Sym(
                self.next_while(|c| is_ident_char(c)),
            ),
            _ => Token::Colon,
        }
    }

    #[inline]
    fn dot(&mut self) -> Token {
        self.next_char();
        match self.peek_char().copied() {
            Some('.') => {
                self.next_char();
                match self.peek_char().copied() {
                    Some('.') => {
                        self.next_char();
                        Token::Dot3
                    }
                    _ => Token::Dot2,
                }
            }
            _ => Token::Dot,
        }
    }

    /// Single line comments are preceded by `~~`, while multiline comments are placed between `~*` and `*~`.
    /// Note: this method is called when encountering an COMMENT_INNER  character; i.e., we expect one of:
    /// - `COMMENT_INNER`       ==> `~`
    /// - `COMMENT_OUTER`       ==> `*`
    fn comment(&mut self) -> Token {
        let pos = self.loc();
        self.next_char();
        if let Some(c2) = self.peek_char() {
            match *c2 {
                COMMENT_INNER => {
                    self.line_comment();
                    self.token()
                }
                COMMENT_OUTER => {
                    self.block_comment();
                    self.token()
                }
                _ => {
                    let tok = self.token();
                    self.queue.push(tok);
                    Token::Ident(COMMENT_INNER.into())
                }
            }
        } else {
            Token::Invalid {
                data: COMMENT_INNER.into(),
                msg: format!(
                    "Unexpected end of input after `{}`!",
                    COMMENT_INNER
                ),
                pos,
            }
        }
    }

    fn ident(&mut self, _start: char) -> Token {
        let buf = self.next_while(|c| is_ident_char(c));
        if matches!(buf.as_str(), "_") {
            Token::Underscore
        } else if let Some(kw) = Keyword::from_str(&*buf) {
            Token::Kw(kw)
        } else {
            Token::Ident(buf)
        }
    }

    fn line_comment(&mut self) -> Token {
        // let pos = self.loc();
        let comment = self.next_while(|c| c != '\n');
        Token::Comment(Comment::Line(comment))
    }

    fn number(&mut self) -> Token {
        // numeric prefixes for base 2/8/16 unsigned integers, infixes `.` `e` `e+` or `e-` for floats
        // The default flag is `Int`, however number tokens won't *actually* be parsed into numeric values until necessary (post-parsing).
        let mut flag = NumFlag::default();
        let mut data = String::new();
        const ZERO: char = '0';
        // counter for instances of (valid) `+` or `-` encountered. Should only be 0 or 1. If this is 1, then the flag is `NumFlag::Sci`.
        let mut sciop: u8 = 0;

        if let Some(&ZERO) = self.peek_char() {
            self.next_char();
            match self.peek_char().copied() {
                Some('b' | 'B') => {
                    return self.integer(NumFlag::Bin);
                }
                Some('o' | 'O') => {
                    return self.integer(NumFlag::Oct);
                }
                Some('x' | 'X') => {
                    return self.integer(NumFlag::Hex)
                }
                Some(d @ '.') => {
                    match self.dot() {
                        dot
                        @
                        (Token::Dot2
                        | Token::Dot3) => {
                            self.queue.push(dot);
                            return Token::Num {
                                data: "0".into(),
                                flag,
                            };
                        }
                        Token::Dot => {
                            flag = NumFlag::Dec;
                            data.push(ZERO);
                            data.push(d);
                        }
                        // since we've confirmed above the preceding token was Token::Dot
                        _ => unreachable!(),
                    }
                }
                Some(exp @ ('e' | 'E')) => {
                    flag = NumFlag::Sci;
                    data.push(exp);
                    self.next_char();
                }
                Some(c) if c.is_digit(10) => {
                    let pos = self.loc();
                    return self
                        .next_char()
                        .and_then(|val| {
                            Some(Token::Invalid {
                                data,
                                msg: format!("Character {} after initial `0` not supported!", val),
                                pos,
                            })
                        })
                        .unwrap();
                }
                _ => {
                    data.push(ZERO);
                    return Token::Num { data, flag }
                },
            }
        };

        let mut first = true;
        while let Some(&c) = self.peek_char() {
            // self.next_char();
            if first {
                first = false;
                if !c.is_digit(10) {
                    return Token::Invalid{ 
                        data: c.into(), 
                        msg: format!("Invalid sequence after partial lexer result `{}`! Expected a digit, but found `{}`", data, c), 
                        pos: self.loc()};
                }
            };
            match c {
                '_' if data.ends_with(|ch: char| {
                    ch.is_digit(10)
                }) =>
                {
                    // no-op, underscores wedged between digits are separators
                }
                '0'..='9' => data.push(c),
                'e' | 'E' => {
                    if flag == NumFlag::Sci {
                        self.next_char();
                        return Token::Invalid { 
                            msg: format!("Invalid exponential! \
                            There may only be one instance of the infix `{}` in the representation of a floating point number. \n\
                            Input lexed: {}", c, &data), 
                            data, 
                            pos: self.loc() };
                    } else {
                        flag = NumFlag::Sci;
                        data.push(c);
                    }
                }
                '+' | '-' => {
                    if sciop == 0 && flag == NumFlag::Sci {
                        if data.ends_with(|c| {
                            c == 'e' || c == 'E'
                        }) {
                            sciop += 1;
                            data.push(c);
                        } else {
                            return Token::Invalid {
                                data: c.into(),
                                msg: format!("The characters `{}` may only come after an exponential infix `e` or `E` for floating point numbers. Lexed: `{}`", c, data),
                                pos: self.loc()
                            };
                        }
                    } else if data.ends_with('.') {
                        return Token::Invalid {
                            msg: format!(
                                "Invalid exponential infix! Unable to lex `{}` + `{}`",
                                &data, c
                            ),
                            data,
                            pos: self.loc(),
                        };
                    } else {
                        // the `+` and `-` as operators for next token
                        break;
                    }
                }
                '.' => {
                    self.next_char();
                    match flag {
                        NumFlag::Int => {
                            if let Some('.') =
                                self.peek_char().copied()
                            {
                                self.next_char();
                                if let Some('.') = self
                                    .peek_char()
                                    .copied()
                                {
                                    self.next_char();
                                    self.queue
                                        .push(Token::Dot3);
                                } else {
                                    self.queue
                                        .push(Token::Dot2)
                                };
                                println!("matching dot @ num cur tok is: {:?}", self.peek_char());
                                break;
                            } else {
                                flag = NumFlag::Dec;
                                data.push('.');
                                continue;
                            }
                        }
                        NumFlag::Dec | NumFlag::Sci => {
                            return Token::Invalid {
                                msg: format!(
                                    "Invalid character `.`! The number \
                                    being lexed, `{0}`, has flag `{1:?}` and does not accept further `.`. Additionally, the use of the Range operator `..` is only supported for integers (flag `Int`).", &data, flag),
                                data,
                                pos: self.loc()
                            };
                        }
                        // other numeric flags should have been re-routed to the `integer` method by now
                        _ => unreachable!(),
                    }
                }
                _ => break,
            }
            self.next_char();
        }

        Token::Num { data, flag }
    }

    // binary, octal, hexadecimal -- NEVER called for other number flags
    fn integer(&mut self, flag: NumFlag) -> Token {
        let mut data = format!("0{}", flag.as_str());
        // we start off on the 2nd character of the int prefix, so we eat it and incidentally know only 3 of the 5 NumFlag variants may show up
        self.next_char();
        let base = match flag {
            NumFlag::Bin => 2,
            NumFlag::Oct => 8,
            NumFlag::Hex => 16,
            _ => unreachable!(),
        };
        while let Some(&c) = self.peek_char() {
            self.next_char();
            match c {
                '.' => {
                    let dot = self.dot();
                    if let Token::Dot2 = dot {
                        self.queue.push(Token::Dot2);
                        break;
                    } else {
                        return Token::Invalid {
                            msg: format!(
                                "Found `{}` while lexing an \
                                integer {} with flag {:?}",
                                dot, &data, flag
                            ),
                            data,
                            pos: self.loc(),
                        };
                    }
                }
                d if d.is_digit(base) => {
                    data.push(d);
                }
                _ => break,
            }
        }

        Token::Num { data, flag }
    }

    fn operator(&mut self) -> Token {
        let buf = self.next_while(|c| OP_CHARS.contains(c));
        match buf.as_str() {
            "|" => Token::Pipe,
            "@" => Token::At,
            "=" => Token::Eq,
            "#" => Token::Pound,
            "\\" => Token::Lambda,
            "->" => Token::ArrowR,
            "<-" => Token::ArrowL,
            s => {
                if let Some(op) = BinOp::from_str(s) {
                    Token::Operator(Operator::Reserved(op))
                } else {
                    Token::Operator(buf.into())
                }
            }
        }
    }

    fn prefix(&mut self, c: char) -> Token {
        let pos = self.loc();
        match c {
            // byte string
            'b' => {
                self.next_char();
                match self.peek_char().copied() {
                    Some('"') => {
                        let tok = self.string();
                        if let Token::Str(bs) = tok {
                            Token::Bytes(bs.into_bytes())
                        } else {
                            tok
                        }
                    }
                    Some(ch) if is_ident_char(ch) => {
                        let first = self
                            .next_char()
                            .unwrap_or('\0');
                        match self.ident(ch) {
                                Token::Ident(s) => Token::Ident(format!("{}{}", c, s)),
                                _ => Token::Invalid {
                                    data: ch.into(),
                                    msg: format!(
                                        "Invalid sequence after prefix `{}`! Expected `'` or `\"`, but found {}",
                                        first, c
                                    ),
                                    pos,
                                },
                            }
                    }
                    _ => Token::Ident(c.into()),
                }
            }
            // TODO
            // // raw string
            // 'r' => {}
            // // macro?
            // 'm' => {}
            // // format string
            // 'f' => {}
            _ => self.ident(c),
        }
    }

    fn string(&mut self) -> Token {
        let mut buf = String::new();
        self.next_char();
        let mut escaped = false;
        while let Some(c) = self.next_char() {
            if escaped {
                escaped = false;
                match c {
                    esc if is_escapable(esc) => {
                        buf.push(get_escaped(esc))
                    }

                    '\0' => { /* null grapheme */ }

                    // preserve indentation/ignore whitespace on new line
                    // e.g. `"a b c\
                    //        d e"` lexes as "a b cd e"
                    '\n' => {
                        self.eat_whitespace();
                    }
                    _ => {
                        buf.push(c);
                    }
                };
            } else if c == '"' {
                break;
            } else if c == '\\' {
                escaped = true;
            } else {
                buf.push(c);
            }
        }
        Token::Str(buf)
    }

    fn unknown(&mut self, c: char) -> Token {
        let pos = self.loc();
        self.next_char();
        Token::Invalid {
            data: c.into(),
            msg: "Unknown character!".into(),
            pos,
        }
    }
}

impl<'t> Iterator for Lexer<'t> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        match self.current.take() {
            Some(t) => Some(t),
            None => match self.token() {
                Token::Eof => None,
                tok => {
                    // self.spans.push(self.loc());
                    Some(tok)
                },
            },
        }
    }
}

/// Note: The actual Lexer doesn't use either `Peek` nor `Iterator` on itself.
impl<'t> Peek for Lexer<'t> {
    type Peeked = Token;
    /// Returns a reference to the current Token, if any. Primarily used by Parser.
    fn peek(&mut self) -> Option<&Self::Peeked> {
        if let Some(ref t) = self.current {
            Some(t)
        } else {
            let token = self.token();
            self.current.replace(token);
            self.current.as_ref()
        }
    }
    fn is_done(&mut self) -> bool {
        self.source.is_done() && self.current.is_none()
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || matches!(c, UNDER)
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, UNDER | PRIME)
}

const fn get_escaped(c: char) -> char {
    match c {
        't' => '\t',
        'n' => '\n',
        'r' => '\r',
        '"' => '\"',
        '\'' => PRIME,
        '\\' => ESCAPE,
        _ => c,
    }
}

const fn is_escapable(c: char) -> bool {
    matches!(c, 't' | 'n' | 'r' | '"' | '\'' | '\\')
}

#[allow(unused)]
/// Tokenize entire input, returning a vector of Tokens. Allowing it to go unused, as idiomatically, the lexer will be accessed via iterating in sync with another iterator, namely the parser.
pub fn tokenize(input: &str) -> Vec<Token> {
    Lexer::new(input).into_iter().collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lex_ident() {
        let mut lexer = Lexer::new("b\"xuv1\" +");
        assert_eq!(
            lexer.next(),
            Some(Token::Bytes(b"bxuv1".to_vec()))
        );
        assert_eq!(
            lexer.next(),
            Some(Token::Operator(BinOp::Plus.into()))
        );
    }

    #[test]
    fn test_numbers() {
        let lexer =
            Lexer::new("1 1.2 3 4e20 0xfff 5..6 0 0..5");
        let expected = &[
            Token::Num {
                data: "1".into(),
                flag: NumFlag::Int,
            },
            Token::Num {
                data: "1.2".into(),
                flag: NumFlag::Dec,
            },
            Token::Num {
                data: "3".into(),
                flag: NumFlag::Int,
            },
            Token::Num {
                data: "4e20".into(),
                flag: NumFlag::Sci,
            },
            Token::Num {
                data: "0xfff".into(),
                flag: NumFlag::Hex,
            },
            Token::Num {
                data: "5".into(),
                flag: NumFlag::Int,
            },
            Token::Dot2,
            Token::Num {
                data: "6".into(),
                flag: NumFlag::Int,
            },
            Token::Num {
                data: "0".into(),
                flag: NumFlag::Int,
            },
            Token::Num {
                data: "0".into(),
                flag: NumFlag::Int,
            },
            Token::Dot3,
            Token::Num {
                data: "5".into(),
                flag: NumFlag::Int,
            },
        ];

        lexer
            .into_iter()
            .zip(expected.iter())
            .for_each(|(t1, t2)| assert_eq!(&t1, t2));
    }

    #[test]
    fn lex_chars() {
        let lexer = Lexer::new(
            "\\:A a 1; . | # @ ## #> .. 3...5 'a' :b -> 2.0 <-",
        );

        let expected = [
            Token::Lambda,
            Token::Sym("A".into()),
            Token::Ident("a".into()),
            Token::Num {
                data: "1".into(),
                flag: NumFlag::Int,
            },
            Token::Semi,
            Token::Dot,
            Token::Pipe,
            Token::Pound,
            Token::At,
            Token::Operator(Operator::Custom("##".into())),
            Token::Operator(Operator::Custom("#>".into())),
            Token::Dot2,
            Token::Num {
                data: "3".into(),
                flag: NumFlag::Int,
            },
            Token::Dot3,
            Token::Num {
                data: "5".into(),
                flag: NumFlag::Int,
            },
            Token::Char('a'),
            Token::Sym("b".into()),
            Token::ArrowR,
            Token::Num {
                data: "2.0".into(),
                flag: NumFlag::Dec,
            },
            Token::ArrowL,
        ];

        lexer.into_iter().zip(expected.iter()).for_each(
            |(res, tok)| {
                println!("{}", &res);
                assert_eq!(&res, tok)
            },
        );
    }

    #[test]
    fn test_lexer() {
        let mut lexer = Lexer::new("a :b c () [] {} + 1.2");
        assert_eq!(
            lexer.next(),
            Some(Token::Ident("a".into()))
        );
        assert_eq!(
            lexer.next(),
            Some(Token::Sym(":b".into()))
        );
        assert_eq!(
            lexer.next(),
            Some(Token::Ident("c".into()))
        );
        assert_eq!(lexer.next(), Some(Token::ParenL));
        assert_eq!(lexer.next(), Some(Token::ParenR));
        assert_eq!(lexer.next(), Some(Token::BrackL));
        assert_eq!(lexer.next(), Some(Token::BrackR));
        assert_eq!(lexer.next(), Some(Token::CurlyL));
        assert_eq!(lexer.next(), Some(Token::CurlyR));
        assert_eq!(
            lexer.next(),
            Some(Token::Operator(BinOp::Plus.into()))
        );
        assert_eq!(
            lexer.next(),
            Some(Token::Num {
                data: "1.2".into(),
                flag: NumFlag::Dec
            })
        );
        assert!(lexer.is_done());
        assert_eq!(lexer.next(), None)
    }

    #[test]
    fn test_spanned() {
        let src = "a     thing here is \nsuch <> *a thing*";
        let mut lexer = Lexer::new(src);
        let mut ct = 0;
        while !lexer.is_done() {
            let start = lexer.loc();
            let tok = lexer.next();
            let end = lexer.loc();
            println!("[{}]:\tstart: {}, token: {:?}, end: {}", &ct, start, tok, end);
            ct += 1; 
        }
        println!("{:#?}", lexer.spans)
    }
}
