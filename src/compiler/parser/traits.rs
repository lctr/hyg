use super::{Peek, Positioned};

pub trait Consume<'t>: Peek + Positioned {
    type Error: std::fmt::Display;

    /// Consume the provided token and return a `Result` containing a mutable reference to self if successful, or a `Self::Error` describing the error otherwise.
    fn eat(&mut self, token: &<Self as Peek>::Item) -> Result<&mut Self, Self::Error>;

    /// Returns the next item unwrapped. The underlying item to be iterated must be equivalently "null-terminated". By convention, this method should return T::Eof if the Option<&T> returned by `peek` is `None`, otherwise it returns `T`,
    fn take_next(&mut self) -> <Self as Peek>::Item;

    /// Generate an error describing the expected token, the actual token, and the position in the stream. The position is given in an `Option` type -- this is to allow flexibility in position reporting.
    ///
    /// Since it may be more convenient to capture the position in the stream prior to running into an error, any `Position<u32>` provided in a `Some` variant will be used. In the event a `None` variant is passed in, the position by default will be called using the supertrait `Positioned`.  This has the effect that the position reported will correspond to the position in the stream *after which* the error was encountered.
    ///
    /// *Tldr;* Provide custom location wrapped in a `Some` variant for accurate error reporting, as the default position may correspond to the position in the stream *after which* the error was encountered.
    // &mut self since many calling methods will have borrowed self as mutable.
    fn unexpected(
        &mut self,
        expected: &<Self as Peek>::Item,
        actual: &<Self as Peek>::Item,
        pos: Option<Self::Loc>,
    ) -> Self::Error;
}

/// Minimum constraint necessary by higher order parsing methods.
/// This is a supertrait of `Consume`, which itself is a supertrait of `Peek`, and `Positioned`. Methods in this trait do not directly produce a parent node, but are instead used to compose parser actions, the results of which are then used to form an expression.
/// Most methods in this trait are implemented by default thanks its chain of super traits, i.e., are effectively built on the Peek trait, which itself is a flavor of iterator (the `peekable` method on any iterator produces a peekable ierator, for which the implementation of `Peek` is nearly trivial).
pub trait Combinator<'t>: Consume<'t> {
    /// Given a function `parse :: &mut Self -> Result<X, E>`, return the vector of nodes of type `X` obtained by applying `parse` repeatedly, initially after consuming a token matching `start` parameter, after each delimiter matching the `sep` parameter, and terminating upon encountering a token matching the `end` parameter.
    /// For example, this method would be used with a parser `f` that parses numeric tokens to parse the input `[1, 2, 3]`
    /// On failure, all consumed tokens are discared and the error `E = Self::Error` is returned.
    fn delimited<F, X>(
        &mut self,
        start: <Self as Peek>::Item,
        sep: <Self as Peek>::Item,
        end: <Self as Peek>::Item,
        parse: &mut F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        self.eat(&start)?;
        self.sep_by_until(sep, end, parse)
    }

    fn sep_by_until<F, X>(
        &mut self,
        sep: <Self as Peek>::Item,
        end: <Self as Peek>::Item,
        parse: &mut F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![];
        let mut first = true;
        while !self.is_done() {
            if self.match_curr(&end) {
                break;
            };
            if first {
                first = false;
            } else {
                self.eat(&sep)?;
            };
            if self.match_curr(&end) {
                break;
            };
            nodes.push(parse(self)?);
        }
        self.eat(&end)?;
        Ok(nodes)
    }

    fn many_sep_by<F, X>(
        &mut self,
        sep: <Self as Peek>::Item,
        parse: &mut F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![parse(self)?];
        while !self.is_done() && self.match_curr(&sep) {
            self.eat(&sep)?;
            nodes.push(parse(self)?);
        }
        Ok(nodes)
    }

    fn many<F, X>(&mut self, parse: &mut F) -> Result<(Vec<X>, &mut Self), Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![parse(self)?];
        while let Ok(x) = parse(self) {
            nodes.push(x);
        }
        Ok((nodes, self))
    }

    fn many_while<F, X>(
        &mut self,
        pred: fn(&<Self as Peek>::Item) -> bool,
        parse: &mut F,
    ) -> Result<(Vec<X>, &mut Self), Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![];
        while let Some(true) = self.peek().and_then(|t| Some(pred(t))) {
            nodes.push(parse(self)?);
        }
        Ok((nodes, self))
    }

    fn row_sep<F, X>(
        &mut self,
        empty_rows: u32,
        f: &mut F,
    ) -> Result<(Vec<X>, &mut Self), Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![];
        let mut row = self.get_row();
        nodes.push(f(self)?);
        while self.get_row() - row < empty_rows {
            row = self.get_row();
            nodes.push(f(self)?);
        }
        Ok((nodes, self))
    }

    fn many_col_aligned<F, X>(&mut self, f: &mut F) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let col0 = self.get_column();
        let mut nodes = vec![];
        nodes.push(f(self)?);
        while !self.is_done() && self.get_column() > col0 {
            nodes.push(f(self)?)
        }
        Ok(nodes)
    }

    fn optional<F, X>(
        &mut self,
        tok: &<Self as Peek>::Item,
        parse: &mut F,
    ) -> Result<Option<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        if self.match_curr(tok) {
            self.eat(tok)?;
            parse(self).map(|x| Some(x))
        } else {
            Ok(None)
        }
    }
}

pub trait Term<'t>: Combinator<'t> {
    type Lexeme;
    type Morpheme;

    fn atom(&mut self) -> Self::Lexeme;

    fn ident(&mut self) -> Self::Lexeme;

    fn literal(&mut self) -> Self::Lexeme;

    fn pattern(&mut self) -> Self::Morpheme;
}

pub trait Expression<'t>: Term<'t> {
    fn expression(&mut self) -> Self::Lexeme;

    fn list(&mut self) -> Self::Lexeme;

    fn group(&mut self) -> Self::Lexeme;

    fn do_block(&mut self) -> Self::Lexeme;
}

pub trait ParseForm<'t>: Expression<'t> {
    fn let_expr(&mut self) -> Self::Lexeme;

    fn case_expr(&mut self) -> Self::Lexeme;

    fn if_expr(&mut self) -> Self::Lexeme;

    fn lambda_expr(&mut self) -> Self::Lexeme;
}

pub trait Binder<'t>: ParseForm<'t> {
    type Binding;
    type Branch;
    fn binding(&mut self) -> Self::Binding;
    // fn gen_pat(&mut self) -> Self::Binding;
    fn case_pat(&mut self) -> Self::Branch;
}

pub trait Annotation<'t>: Binder<'t> {
    type Decl;
    fn annotation(&mut self) -> Self::Decl;
    fn signature(&mut self) -> Self::Decl;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::lexer::*;

    #[derive(Clone, Debug)]
    pub struct SyntaxError(pub String);

    impl std::fmt::Display for SyntaxError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", &(self.0))
        }
    }
    struct Parser<'t> {
        lexer: Lexer<'t>,
    }

    impl<'t> Parser<'t> {
        fn token(&mut self) -> Option<Token> {
            self.lexer.next()
        }
    }

    impl<'t> Peek for Parser<'t> {
        type Item = Token;
        fn peek(&mut self) -> Option<&Self::Item> {
            self.lexer.peek()
        }
        fn is_done(&mut self) -> bool {
            self.lexer.is_done()
        }
    }

    impl<'t> Positioned for Parser<'t> {
        type Loc = Location;
        fn loc(&self) -> Self::Loc {
            self.lexer.loc()
        }
    }

    impl<'t> Consume<'t> for Parser<'t> {
        type Error = SyntaxError;

        fn eat(&mut self, token: &<Self as Peek>::Item) -> Result<&mut Self, Self::Error> {
            if self.match_curr(token) {
                self.lexer.next();
                Ok(self)
            } else {
                let pos = self.loc();
                let curr = self.peek().unwrap_or_else(|| &Token::Eof);
                Err(SyntaxError(format!(
                    "Expected {}, but found {} at {}",
                    token, curr, pos
                )))
            }
        }

        fn take_next(&mut self) -> <Self as Peek>::Item {
            match self.lexer.next() {
                Some(t) => t,
                _ => Token::Eof,
            }
        }

        fn unexpected(
            &mut self,
            expected: &<Self as Peek>::Item,
            actual: &<Self as Peek>::Item,
            pos: Option<Self::Loc>,
        ) -> Self::Error {
            SyntaxError(format!(
                "Expected {}, but found {} at {}",
                expected,
                actual,
                pos.unwrap_or_default()
            ))
        }
    }

    impl<'t> Combinator<'t> for Parser<'t> {}

    impl<'t> From<Lexer<'t>> for Parser<'t> {
        fn from(lexer: Lexer<'t>) -> Self {
            Self { lexer }
        }
    }

    #[test]
    fn test_many_aligned() {
        let mut parser = Parser::from(Lexer::new("a b c d"));
        let results = parser.many_col_aligned(&mut |p| {
            let pos = p.loc();
            match p.token() {
                Some(e @ Token::Error { .. }) => {
                    Err(SyntaxError(format!("Lexer error! {} at {}", e, pos)))
                }
                Some(t) => Ok((t, pos)),
                None => Err(SyntaxError(format!("Unexpected EOF at {}", pos))),
            }
        });
        println!("{:#?}", results)
    }
}
