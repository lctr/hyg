use crate::{compiler::lexer::{
    BinOp, Keyword, Lexer, Operator, Peek, Token,
}, prelude::span::{Location, Positioned}};

use crate::compiler::syntax::{expr::{Expr, Match,  Section, Var}, literal::{Literal, LitErr}, pattern::Pat};

use super::{
    fixity::{Fixity, FixityTable}, 
    scan::{Combinator, Consume},
    Comment,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxError(pub String);

impl From<(String, Location)> for SyntaxError {
    fn from((msg, loc): (String, Location)) -> Self {
        Self(format!("{} at {}", msg, loc))
    }
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &(self.0))
    }
}
pub struct Parser<'t> {
    lexer: Lexer<'t>,
    fixities: FixityTable,
    comments: Vec<Comment>,
}

impl<'t> From<&'t str> for Parser<'t> {
    fn from(s: &'t str) -> Self {
        Parser::new(s)
    }
}

impl<'t> Peek for Parser<'t> {
    type Peeked = Token;
    fn peek(&mut self) -> Option<&Self::Peeked> {
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

    fn eat(&mut self, token: &<Self as Peek>::Peeked) -> Result<&mut Self, Self::Error> {
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

    fn take_next(&mut self) -> <Self as Peek>::Peeked {
        match self.lexer.next() {
            Some(t) => t,
            _ => Token::Eof,
        }
    }

    fn unexpected(
        &mut self,
        expected: &<Self as Peek>::Peeked,
        actual: &<Self as Peek>::Peeked,
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
        Self {
            lexer,
            fixities: FixityTable::default(),
            comments: vec![],
        }
    }
}

impl<'t> Parser<'t> {
    pub fn new(source: &'t str) -> Self {
        Self {
            lexer: Lexer::new(source),
            fixities: FixityTable::default(),
            comments: vec![],
        }
    }

    pub fn expression(&mut self) -> Result<Expr, SyntaxError> {
        self.binary(0, &mut Self::terminal)
    }

    fn binary<F>(&mut self, min_prec: u8, f: &mut F) -> Result<Expr, SyntaxError>
    where
        F: FnMut(&mut Self) -> Result<Expr, SyntaxError>,
    {
        let mut expr = f(self)?;
        while let Some(&Fixity { assoc, prec }) = {
            let op = self.peek().and_then(|token| token.as_operator());
            if let Some(operator) = op {
                self.fixities.get(&operator)
            } else {
                None
            }
        } {
            if min_prec < prec || (min_prec == prec && assoc.is_right()) {
                // since we know the token will contain an `Operator`, this is safe to unwrap
                let op = self.take_next().as_operator().unwrap();
                let right = self.binary(prec, f)?;
                expr = Expr::Binary {
                    infix: op,
                    left: Box::new(expr),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn unary(&mut self) -> Result<Expr, SyntaxError> {
        if let Some(Token::Operator(Operator::Reserved(BinOp::Minus))) = self.peek() {
            let prefix = self.take_next().as_operator().unwrap();
            let right = Box::new(self.unary()?);
            Ok(Expr::Unary { prefix, right })
        } else {
            self.terminal()
        }
    }

    fn case_expr(&mut self) -> Result<Expr, SyntaxError> {
        let expr = Box::new(
            self.eat(&Token::Kw(Keyword::Case))
                .and_then(Self::expression)?,
        );
        self.eat(&Token::Kw(Keyword::Of))?;
        let arms = match self.peek() {
            Some(Token::CurlyL) => self.delimited(
                Token::CurlyL,
                Token::Semi,
                Token::CurlyR,
                &mut Self::case_arms,
            )?,
            _ => self.many_col_aligned(&mut Self::case_arms)?,
        };
        Ok(Expr::Case { expr, arms })
    }

    fn case_arms(&mut self) -> Result<(Match, Expr), SyntaxError> {
        let pattern = self.case_branch_pat()?;
        let bound = matches!(&pattern, Pat::Binder { .. });
        let alts = if let Some(Token::Pipe) = self.peek() {
            self.eat(&Token::Pipe)
                .and_then(|p| p.many_sep_by(Token::Pipe, &mut Self::case_branch_pat))?
        } else {
            vec![]
        };
        let guard = if let Some(Token::Kw(Keyword::If)) = self.peek() {
            Some(
                self.eat(&Token::Kw(Keyword::If))
                    .and_then(|p| p.expression())?,
            )
        } else {
            None
        };
        self.eat(&Token::ArrowR)?;
        let body = self.expression()?;
        Ok((
            Match {
                pattern,
                bound,
                alts,
                guard,
            },
            body,
        ))
    }

    // TODO: Optimize `Binder` variant
    fn case_pat_at(&mut self, binder: Token) -> Result<Pat, SyntaxError> {
        use Token::*;
        let loc = &self.loc();

        let no_local_vars: fn(&Location, &Token, &Token) -> SyntaxError =
            |l: &Location, t: &Token, b: &Token| {
                SyntaxError(format!(
                    "Invalid token `{1}` found at {0} while parsing case \
                expression branch pattern for binder `{2}`. \n\
                Case expressions do not evaluate patterns as expressions, \
                so the binder `{2}` cannot bind the local variable `{1}`. \n\
                If trying to match data constructors, use symbol syntax, \n\
                \ti.e., `:{1}` instead of `{1}`",
                    l, t, b
                ))
            };

        match self.peek() {
            Some(Underscore) => self
                .eat(&Underscore)
                .and_then(|_| Ok(Pat::Wild)),

            Some(ParenL) => self.eat(&ParenL)
                .and_then(|this| match this.peek() {
                    Some(Pipe) => {
                        this.many_while(
                            |tok| matches!(tok, Pipe),
                            |p| p
                                .eat(&Pipe)
                                .and_then(&mut Self::case_branch_pat))
                            .and_then(|(pattern, parser)|
                                parser.eat(&ParenR)
                                    .and_then(|_| Ok(Pat::Binder{
                                        binder, 
                                        pattern: Box::new(Pat::Union(pattern))
                                    })))
                    }

                    Some(t@Ident(_)) => Err(no_local_vars(loc, t, &binder)),

                    Some(Sym(_)) => {
                        let ctor = this.take_next();
                        let (args, _) = this.many_while(
                            |t| !matches!(t, Pipe | ParenR | Comma), 
                            &mut Self::case_branch_pat)?;
                        match this.peek() {
                            Some(Comma) => {
                                let mut items = this.delimited(
                                    Comma, 
                                    Comma, 
                                    ParenR, 
                                    &mut Self::case_branch_pat)?;
                                items.insert(0, Pat::Ctor(ctor, args));
                                Ok(Pat::Binder{ 
                                    binder, 
                                    pattern: Box::new(Pat::Tuple(items))
                                })
                            }
                            Some(Pipe) => {
                                let mut items = this.many_sep_by(Pipe, &mut Self::case_branch_pat)?;
                                items.insert(0, Pat::Ctor(ctor, args));
                                this.eat(&ParenR)?;
                                Ok(Pat::Binder { binder, pattern: Box::new(Pat::Union(items)) })
                            }
                            Some(ParenR) => {
                                this.eat(&ParenR)?;
                                Ok(Pat::Binder {
                                        binder, 
                                        pattern: Box::new(Pat::Ctor(ctor, args))
                                })
                            }
                            // since the previous rule only stopped at the above three tokens, it follows any other matches are unreachable
                            _ => unreachable!()
                        }
                    }

                    Some(Bytes(_) | Char(_) | Str(_) | Num {..}) => {
                        this.many_sep_by(Pipe, &mut Self::case_branch_pat)
                            .and_then(|alts| 
                                this.eat(&ParenR)
                                    .and_then(|_| 
                                        Ok(Pat::Binder{ binder, pattern: Box::new(Pat::Union(alts)) })))
                    }

                    Some(Token::Underscore) => this
                        .eat(&Token::Underscore)
                        .and_then(|_| Ok(Pat::Wild)),

                    Some(t) => Err(SyntaxError(format!(
                        "Invalid token `{1}` found at {0} while parsing \
                        case expression branch bound pattern for binder `{2}`. The token `{1}` does not form a valid pattern!", loc, t, binder))),

                    None => Err(SyntaxError(format!(
                        "Unexpected EOF at {} while parsing alternatives \
                        in case expression branch pattern for binder `{}`", 
                        loc, binder)))
                }),

            // we allow the first token in a union to be the union pipe `|`
            Some(Pipe) => {
                self.many_while(
                    |tok| matches!(tok, Pipe), 
                    |p| p
                            .eat(&Pipe)
                            .and_then(&mut Self::case_branch_pat))
                    .and_then(|(alts, _)| Ok(Pat::Binder{
                        binder, 
                        pattern: Box::new(Pat::Union(alts))
                    }))
            }

            Some(t@Ident(_)) => Err(no_local_vars(loc, t, &binder)),

            Some(t) => Err(SyntaxError(format!(
                "Invalid token `{1}` found at {0} while parsing \
                case expression branch pattern for binder `{2}`. \
                The token `{1}` does not form a valid pattern!", 
                loc, t, binder)
            )),

            None => Err(SyntaxError(format!(
                "Unexpected EOF at {} while parsing alternatives \
                in case expression branch pattern for binder `{}`", 
                loc, binder)))
        }
    }

    fn case_branch_pat(&mut self) -> Result<Pat, SyntaxError> {
        use Token::*;
        let loc = self.loc();
        match self.peek() {
            Some(Underscore) => self
                .eat(&Underscore)
                .and_then(|_| Ok(Pat::Wild)),

            Some(Ident(_)) => {
                let ident = self.take_next();
                if self.match_curr(&At) {
                    self.eat(&At)
                        .and_then(|this| this.case_pat_at(ident))
                } else {
                    Ok(Pat::Var(ident))
                }
            }
            Some(Char(..) | Str(..) | Bytes(..) | Num { .. }) => {
                Ok(Pat::from(match self.take_next() {
                    // Token::Sym(_) => todo!(),
                    Char(c) => Literal::Char(c),
                    Str(s) => Literal::Str(s),
                    Bytes(b) => Literal::Bytes(b),
                    Num { data, flag } => Literal::Num { data, flag },
                    _ => unreachable!(),
                }))
                
            }
            Some(Error { data, msg, pos }) => Err(SyntaxError(format!(
                "Invalid token while parsing case branches. \
                    Lexer provided the following error for `{}` at {}:\n\t{}",
                data, pos, msg
            ))),
            Some(t) => Err(SyntaxError(format!(
                "Invalid token at {}! \
                The token `{}` is not a valid pattern for the left-hand-side of a case expression branch!", loc, t))),
            None => Err(Self::unexpected_eof_while("parsing left-hand-side of a case expression branch pattern", loc))
        }
    }

    fn conditional_expr(&mut self) -> Result<Expr, SyntaxError> {
        self.eat(&Token::Kw(Keyword::If))?;
        let cond = self.expression()?;
        self.eat(&Token::Kw(Keyword::Then))?;
        let then = self.expression()?;
        self.eat(&Token::Kw(Keyword::Else))?;
        let other = self.expression()?;

        Ok(Expr::Cond {
            cond: Box::new(cond),
            then: Box::new(then),
            other: Box::new(other),
        })
    }

    /// Called when already on a literal token
    fn literal(&mut self) -> Result<Expr, SyntaxError> {
        let loc = self.loc();
        match self.peek() {
            Some(
                Token::Sym(_)
                | Token::Char(_)
                | Token::Str(_)
                | Token::Bytes(_)
                | Token::Num { .. },
            ) => Ok(Expr::Lit(Literal::from_token(self.take_next()).unwrap())),

            // in theory, these cases shouldn't come up as this method is only called when peeking a `Token::Literal`.
            Some(t) => Err(SyntaxError(format!(
                "Invalid token type! Expected a literal, \
                but found `{0}` at {1}!",
                t, loc
            ))),

            None => Err(Self::unexpected_eof_while("parsing literal", loc)),
        }
    }

    /// Parses an expression surrounded by parentheses `(` abd `)`.
    ///
    /// This method is called internally upon witnessing a `Token::ParenL`,
    /// i.e. a `Token` representing a left parenthesis `(`.
    ///
    /// An expression of the form `(` ... `)` may produce:
    /// * tuple expression
    /// * application
    /// * identifier, GIVEN the parentheses wrap a single `Token::Operator`
    /// * operator section
    ///     * left section, GIVEN (EXPR OP), equivalent to \x -> EXPR OP x
    ///     * right section, GIVEN (OP EXPR), equivalent to \x - x OP EXPR
    ///     * SELF NOTES:
    ///         * Haskell doesn't accept right sections for `-` as it
    ///           coincides with unary negation. How to handle this?
    ///         * When generating lambdas, be careful wrt naming arg(s)!
    ///         * Sections to be represented as their own Expr node and can
    ///           be desugared into lambdas later
    /// * the same expression contained, GIVEN the parentheses only wrap a
    /// single expression *without* a trailing comma. An example of when this
    /// is *necessary* is when a lambda is used as an argument in an
    /// application, e.g., `map (\n -> n + 1) [1, 2, 3]`
    ///
    ///
    /// *Note:* For `(` EXPR `)`, the future linter should flag this as redundant
    fn parentheses(&mut self) -> Result<Expr, SyntaxError> {
        self.eat(&Token::ParenL)?;
        let loc = self.loc();
        match self.peek() {
            // first thing's first -- left-recursion
            Some(Token::ParenL) => {
                self.eat(&Token::ParenL)?;
                if self.match_curr(&Token::ParenR) {
                    self.eat(&Token::ParenR)?;
                    Ok(Expr::Tuple(vec![]))
                } else {
                    let first = self.expression()?;
                    self.tuple_or_app(first)
                }
            }

            // Empty tuple -- should this be its own AST node? probably not
            Some(Token::ParenR) => self
                .eat(&Token::ParenR)
                .and_then(|_| Ok(Expr::Tuple(vec![]))),

            Some(Token::Operator(_)) => {
                let oploc = self.loc();
                let op = self.take_next().as_operator().unwrap();
                if self.match_curr(&Token::ParenR) {
                    self.take_next();
                    Ok(Expr::Ident(Var::Infix(op)))
                } else {
                    if self.fixities.get(&op).is_some() {
                        let right = Box::new(self.expression()?);
                        Ok(Expr::Section(Section::Right {
                            infix: op, 
                            right
                        }))
                    } else {
                        Err(SyntaxError(format!(
                            "Invalid operator for right section! The operator \
                            `{}` at {} has not had a fixity or precedence \
                            defined", op, oploc
                        )))
                    }

                }
            }

            // if we see something we know is NOT callable, we either have a left section, tuple, or reduncantly wrapped expression
            Some(Token::Char(_)
            | Token::Str(_)
            | Token::Num {..}
            | Token::Bytes(..)) => {
                let first = Expr::Lit(
                    // since we know it won't fail, we can unwrap this
                    Literal::from_token(self.take_next()
                ).unwrap());
                self.tuple_or_app(first)
            }

            // if we see a keyword, we know it must either be a redundantly wrapped expression, an application, or a tuple
            Some(
            Token::Kw(Keyword::Let) 
            | Token::Kw(Keyword::Case) 
            | Token::Kw(Keyword::If)
            // and any other tokens that would form either a reduntantly wrapped expression, an application, or a tuple
            | Token::Ident(_)
            | Token::BrackL
            ) => {
                let first = self.expression()?;
                self.tuple_or_app(first)
            }

            // Terminals we reject. This will probably be heavily modified as 
            // it will reject anything we haven't caught up until now.
            Some(t) => 
                Err(SyntaxError(format!(
                    "Invalid token `{}` found at {} while parsing inner \
                    contents of a grouuped expression!", t, loc
                )))
            ,
            None => 
                Err(Self::unexpected_eof_while(
                    "parsing inner contents of grouped expression", loc)),
        }
    }

    fn record_app(&mut self, proto: Token) -> Result<Expr, SyntaxError> {
        let fields = self.delimited(Token::CurlyL, Token::Comma, Token::CurlyR, |p| {
            let left = p.take_next();
            match p.peek() {
                Some(Token::Comma) => Ok((left, None)),
                _ => Ok((left, Some(p.expression()?))),
            }
        })?;
        Ok(Expr::Record { proto, fields })
    }

    fn tuple_or_app(&mut self, head: Expr) -> Result<Expr, SyntaxError> {
        let loc = self.loc();
        match self.peek() {
            Some(Token::ParenL) => {
                let args = self
                    .eat(&Token::ParenL)
                    .and_then(|p| {
                        p.many_while(|t| !matches!(t, Token::ParenR), &mut Self::expression)
                    })
                    .and_then(|(args, p)| p.eat(&Token::ParenR).and_then(|_| Ok(args)))?;
                Ok(Expr::App {
                    func: Box::new(head),
                    args,
                })
            }

            // tuple
            Some(Token::Comma) => {
                let mut rest = self.delimited(
                    Token::Comma,
                    Token::Comma,
                    Token::ParenR,
                    &mut Self::expression,
                )?;
                rest.insert(0, head);
                Ok(Expr::Tuple(rest))
            }

            // trivially wrapped
            Some(Token::ParenR) => self.eat(&Token::ParenR).and_then(|_| Ok(head)),

            Some(Token::Eof) | None => Err(Self::unexpected_eof_while(
                "parsing tail of grouped expression \
                    (`tuple_or_app`) at {}",
                loc,
            )),

            _ => self.application(head),
        }
    }

    // stops after seeing a semicolon OR after more than 1 consecutive line break
    // Comments may be used to extend the number of empty rows allowed, since comments are still lexed and lazily parsed.
    fn application(&mut self, head: Expr) -> Result<Expr, SyntaxError> {
        let mut args = vec![];
        let mut row = self.get_row();

        // let mut col = self.get_col();
        'tail: loop {
            if matches!(self.peek(), Some(Token::Comment(..))) {
                let comment = self.take_next().as_comment().unwrap();
                self.comments.push(comment);
                row = self.get_row();
                continue 'tail;
            }

            if self.is_done() || self.match_curr(&Token::Semi) {
                break;
            }

            args.push(self.expression()?);

            if matches!(self.peek(), Some(Token::Comment(..))) {
                continue 'tail;
            }

            if self.get_row() - row > 1 {
                break;
            }
        }

        Ok(Expr::App {
            func: Box::new(head),
            args,
        })
    }

    /// Named application declarations are syntactic sugar for
    /// case expressions in a (function) declaration.
    /// 
    /// Declarations must be grouped together such that the first 
    /// declaration defines the base case, and subsequent declarations
    /// form unique case branches which are matched against in order. 
    /// The last declaration marks the end of the match alternatives.
    /// 
    /// ```xg
    /// mult x 0 = 0
    /// mult 0 x = 0
    /// mult x y = x * y
    /// ```
    /// 
    /// Is equivalent to
    /// ```xg
    /// fn |mult| \x y -> case (x, y) of 
    ///     (_, 0) -> 0
    ///     (0, _) -> 0
    ///     (x, y) -> x * y
    /// ```
    fn named_pats(&mut self,) {}

    fn let_expr(&mut self) -> Result<Expr, SyntaxError> {
        todo!()
        // let named = self.peek()
        //     .and_then(|t| matches!(t, Token::Ident(..)))
        //     .unwrap_or_else(|| false);
    }

    fn brackets(&mut self) -> Result<Expr, SyntaxError> {
        todo!()
    }

    fn terminal(&mut self) -> Result<Expr, SyntaxError> {
        let loc = self.loc();
        let invalid = |t: &Token, l: &Location| {
            SyntaxError(format!(
                "Invalid token `{}` found at {} while parsing method `terminal`. This token does not begin an expression!",
                t, l
            ))
        };

        use Keyword::{Case, If, Let};
        match self.peek() {
            // might be: TUPLE, APPLY, or REDUNDANTLY GROUPED
            Some(Token::ParenL) => self.parentheses(),
            Some(Token::BrackL) => self.brackets(),
            // Some(Token::CurlyL) => todo!(),
            Some(Token::Kw(Let)) => self.let_expr(),
            Some(Token::Kw(Case)) => self.case_expr(),
            Some(Token::Kw(If)) => self.conditional_expr(),
            Some(Token::Bang | Token::Operator(Operator::Reserved(BinOp::Minus))) => self.unary(),
            Some(Token::Operator(_)) => {
                // TODO: offload to `operator` method
                // since this is predicated on having matched an operator, it is safe to unwrap the results of the `Token`'s `as_operator` method call
                Ok(Expr::Ident(Var::Infix(
                    self.take_next().as_operator().unwrap(),
                )))
            }
            Some(Token::Comment(_)) => {
                let comment = self.take_next().as_comment().unwrap();
                self.comments.push(comment);
                self.terminal()
            }

            Some(Token::Lambda) => self.lambda(),

            Some(Token::Ident(_) | Token::Underscore) => {
                Ok(Expr::Ident(Var::Ident(self.take_next().get_string())))
            }

            Some(Token::Sym(..)) => Ok(Expr::Lit(Literal::Sym(self.take_next().get_string()))),

            Some(Token::Char(_) | Token::Str(_) | Token::Bytes(_) | Token::Num { .. }) => {
                Ok(Expr::Lit(Literal::from_token(self.take_next()).unwrap()))
            }

            Some(Token::Kw(kw)) => Err(SyntaxError(format!(
                "Unexpected keyword `{0}` in terminal position at {1} while parsing expression!\n\
                Perhaps a delimiter is missing in a prior expression?",
                kw, loc
            ))),

            Some(Token::Error { data, msg, pos }) => Err(SyntaxError(format!(
                "Lexer error at Lexer[{0}] (Parser[{1}])!\n\
                \tread: {2}\n\tmessage: {3}\n\t",
                pos, loc, data, msg
            ))),

            Some(t) => Err(invalid(t, &loc)),

            None => Err(Self::unexpected_eof_while("parsing terminals", loc)),
        }
    }


    /// Parse the body of a record pattern (i.e., everything between the curly 
    /// braces) according to the given closure that returns a `Pat`, where that 
    /// closure's strictness varies based on constraints and parsing context 
    /// (namely, lambda vs let vs case).
    fn record_field_pats<F>(&mut self, mut f: F) -> Result<Vec<(Token, Option<Pat>)>, SyntaxError> 
    where F: FnMut(&mut Self) -> Result<Pat, SyntaxError> {
        use Token::*; 
        self.delimited(CurlyL, Comma, CurlyR, 
            |p| {
                let loc = p.loc();
                match p.peek() {
                    Some(Dot2) => {
                        return Ok((Dot2, Some(Pat::Rest)));
                    }
                    Some(Ident(_) | Sym(_)) => {
                        let field = p.take_next();
                        match p.peek() {
                            Some(Comma | CurlyR) => { Ok((field, None))}
                            _ => {
                                let rhs = f(p)?;
                                Ok((field, Some(rhs)))
                            }
                        }
                    }
                    Some(t) => Err(SyntaxError(format!(
                        "Invalid record accessor`{}` found at {} while parsing record constructor parameter accessors!", t, loc))),

                    None => Err(Self::unexpected_eof_while("Parsing record field patterns", loc))
                }
            })
    }

    /// Parses a lambda argument pattern beginning with `(`.
    fn grouped_lambda_arg(&mut self) -> Result<Pat, SyntaxError> {
        use Token::*; 
        self.eat(&ParenL)?;
        match self.peek() {
            Some(ParenR) => {
               self.take_next();
                return Ok(Pat::Tuple(Vec::new())); 
            }
            Some(Sym(..) | Ident(..)) => {
                let sym = self.take_next();
                match self.peek() {
                    Some(ParenR) => {
                        self.take_next();
                        Ok(Pat::Ctor(sym, Vec::new()))
                    }

                    Some(Comma) => {
                        let mut rest = self.delimited(
                            Comma, 
                            Comma, 
                            ParenR, 
                            Self::lambda_arg)?;
                        rest.insert(0, Pat::Ctor(sym, Vec::new()));
                        Ok(Pat::Tuple(rest))

                    }

                    Some(CurlyL) => {
                        Ok(Pat::Record { ctor: sym, fields: self.record_field_pats(Self::lambda_arg)? })
                    }

                    _ => {
                        let (args, _) = self
                            .many_while(|t | !matches!(t, ParenR), 
                            Self::lambda_arg)?;
                        self.eat(&ParenR)?;
                        Ok(Pat::Ctor(sym, args))
                    }
                }
            }
            _ => {
                let first = self.lambda_arg()?;
                let loc = self.loc(); 
                match self.peek() {
                    Some(Comma) => {
                        let mut rest = self
                            .delimited(Comma, Comma, ParenR, Self::lambda_arg)?;
                        rest.insert(0, first);
                        return Ok(Pat::Tuple(rest));
                    }

                    Some(ParenR) => {self.take_next(); Ok(first)}

                    t => Err(SyntaxError(format!(
                        "Invalid construuctor `{}` found at \
                        {} while parsing lambda parameters",
                        t.unwrap_or_else(|| &Eof), loc
                    )))
                }
            }
        }
    }

    /// Parses patterns used as lambda argument(s)
    fn lambda_arg(&mut self) -> Result<Pat, SyntaxError> {
        use Token::*;
        let pos = self.loc();
        match self.peek() {
            Some(Ident(_)) => Ok(Pat::Var(self.take_next())),

            Some(BrackL) => {
                let elements = self
                    .delimited(BrackL, Comma, BrackR, 
                        Self::lambda_arg)?;
                Ok(Pat::List(elements))
            }

            Some(Sym(..)) => {
                let ctor = self.take_next();
                if matches!(self.peek(), Some(&CurlyL)) {
                    Ok(Pat::Record { 
                        ctor, 
                        fields: self.record_field_pats(Self::lambda_arg)?
                    })
                } else {
                    Ok(Pat::Ctor(ctor, Vec::new()))
                }
            }

            Some(Underscore) => self.eat(&Underscore)
                .and_then(|_| Ok(Pat::Wild)),

            Some(ParenL) => {
                self.grouped_lambda_arg()
            }

            Some(Token::Error { data, msg, pos }) => Err(SyntaxError(format!(
                "Invalid token while parsing lambda arguments. \
                Lexer provided the following error for `{}` \
                at {}:\n\t{}",
                data, pos, msg
            ))),
            Some(t) => Err(SyntaxError(format!(
                "Invalid lambda arg pattern {} at {}",
                t, pos
            ))),
            None => Err(SyntaxError(format!(
                "Unexpected end of input at {} \
                while parsing lambda argument patterns!",
                pos
            ))),
        }
    }

    fn lambda(&mut self) -> Result<Expr, SyntaxError> {
        use Token::{ArrowR, Lambda};
        use Expr::Lam;

        self.eat(&Lambda)?;
        let (pats, _) = self.many_while(
            |t| !matches!(t, ArrowR), 
            Self::lambda_arg)?;

        self.eat(&ArrowR)?;
        let body = self.expression()?;
        Ok(pats
            .into_iter()
            .rev()
            .fold(body, 
                |expr, arg| Lam {
                arg, body: Box::new(expr)
            }))
    }

    fn unexpected_eof_while(action: &str, loc: Location) -> SyntaxError {
        SyntaxError(format!("Unexpected EOF at {} while {}!", loc, action))
    }
}

mod tests {
    #![allow(unused)]
    extern crate test;
    use crate::compiler::syntax::expr::Var;

    use super::*;
    use test::Bencher;

    fn echo_expr(s: &str) -> Result<Expr, SyntaxError> {
        println!("source:\n{}", s);
        Parser::new(s).expression().and_then(|expr| {
            println!("Successfully parsed:\n{:#?}", &expr);
            Ok(expr)
        }).map_err(|err| {
            println!("Parse failure:\n{}", &err);
            err
        })
    }

    fn parser_with_expression(s: &str) -> Result<(Expr, Parser), SyntaxError> {
        let mut parser = Parser::new(s);
        parser.expression()
            .and_then(|expr| Ok((expr, parser)))
    }

    #[bench]
    fn bench_binary_expression(b: &mut Bencher) {
        let lim = 2780; // 1000;
        let mut source = (0..lim).map(|i| format!("{} + ", i)).collect::<String>();
        source.push_str(&*(lim.to_string()));
        b.iter(|| {
            test::black_box(Parser::new(source.as_str()).binary(0, &mut |p| {
                Ok(Expr::Ident(Var::Ident(p.take_next().to_string())))
            }))
        });
    }

    #[test]
    fn test_binary_expression() {
        let source = "a + b / c - d";
        let mut parser = Parser::new(source);

        let expr = parser.binary(0, &mut |p| {
            Ok(Expr::Ident(Var::Ident(p.take_next().to_string())))
        });

        println!("{:#?}", &expr);
        assert_eq!(
            expr,
            Ok(Expr::Binary {
                infix: BinOp::Plus.into(),
                left: Box::new(Expr::Ident(Var::Ident("a".into()))),
                right: Box::new(Expr::Binary {
                    infix: BinOp::Minus.into(),
                    left: Box::new(Expr::Binary {
                        infix: BinOp::Div.into(),
                        left: Box::new(Expr::Ident(Var::Ident("b".into()))),
                        right: Box::new(Expr::Ident(Var::Ident("c".into())))
                    }),
                    right: Box::new(Expr::Ident(Var::Ident("d".into())))
                })
            })
        )
    }

    #[test]
    fn long_binary_expression_overflows_stack() {
        let lim = 2780; // 1000;
        let mut source = (0..lim).map(|i| format!("{} + ", i)).collect::<String>();
        source.push_str(&*(lim.to_string()));

        let mut parser = Parser::new(source.as_str());
        let _expr = parser.binary(0, &mut |p| {
            Ok(Expr::Ident(Var::Ident(p.take_next().to_string())))
        });

        // println!("{:?}", expr)
    }

    #[test]
    fn lambda_expr() {
        let _ = echo_expr("\\a b -> a + b");
        let _ = echo_expr("\\a -> \\b -> a + b");
        let _ = echo_expr("\\(:A' a) -> a");
        let _ = echo_expr("\\:A { a } -> a");
        let _ = echo_expr("(+) 1 2");
    }

    #[bench]
    fn bench_case_expr_col_aligned(b: &mut Bencher) {
        let source = "case x + y of \n\
                \t1 | 2 | 3 -> 4 \n\
                \t_ -> 5 \n\
                ";
        let _ = echo_expr(source);
        println!("{}", &source);
        b.iter(|| {
            test::black_box(Parser::new(source).expression())
        });
    }
}
