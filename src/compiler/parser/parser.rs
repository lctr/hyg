use std::convert::{TryInto, TryFrom};

use crate::{compiler::{lexer::{
    BinOp, Keyword, Lexer, Operator, Peek, Token,
}, syntax::{expr::Binding, decl::{Decl, TyPat, DataVariant}}}, prelude::span::{Location, Positioned}};

use crate::compiler::syntax::{expr::{Expr, Match,  Section, Var}, literal::{Literal, LitErr}, pattern::Pat, fixity::{Fixity, FixityTable},};

use super::{
     
    scan::{Combinator, Consume, Layout},
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
        self.lexer.is_done() || matches!(self.peek(), Some(Token::Eof))
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

impl<'t> Layout<'t> for Parser<'t> {}

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
        // self.within_offside(Token::is_terminal, Self::infix_expr)
        //     .and_then(|expr| Ok(expr.resolve(|x| x, 
        //         |(func, args)| Expr::App { func: Box::new(func), args })))

        let (start, expr) = self.subexpression()?;

        println!("[{}] {:?}", &start, &expr);
        if self.is_done() 
            || self.peek()
                   .and_then(|t| Some(!t.is_terminal()))
                   .unwrap_or_else(|| false) 
            || self.get_column() <= start.col 
            || self.get_row() > start.row { 
            Ok(expr) 
        } else {
            self.maybe_app(start, expr)
        }
    }
    fn infix_expr(&mut self) -> Result<Expr, SyntaxError> {
        self.binary(0, &mut Self::terminal)
    }

    /// First saves the current location, and then parses an expressiob, 
    /// returning both the location and the parsed results. 
    pub fn subexpression(&mut self) -> Result<(Location, Expr), SyntaxError> {
        let loc = self.loc();
        let expr = self.binary(0, &mut Self::terminal)?;
        Ok((loc, expr))
    }

    /// Given an expression, continues parsing any legal upper-right diagonally
    /// inclusive expressions as arguments to an application expression.
    /// 
    /// Legal upper-right diagonal inclusivity for expressions is demonsteated
    /// below.
    /// 
    /// ```rustignore
    /// Let r - row, c - col
    /// given an expression X with coordinates (r_x, c_x), its arguments
    /// is the maximal set of expressions Y with coordinates (r_y, c_y) such
    /// that if for all y in Y
    ///     r_x == r_y              <- notice this is a single binary expr
    ///     c_x < c_y               <- binary exprs are infix apps
    ///     r_x < r_y && c_x < x_y
    ///      
    /// Suppose we have `f x y`. 
    /// We see that the first subexpr is `f` which has coordinates
    ///     (r_x, c_x) = (1, 0)
    /// the second is `x`, with coordinates 
    ///     (r_y1, c_y1) = (1, 2)
    /// the third is `y`, with coordinates
    ///     (r,_y2, c_y2) = (1, 4)
    /// Since r_x == r_yi for all r_yi's, this expression trivially forms an 
    /// application App { func: `f`, args: [`x`, `y`]}
    /// 
    /// Now consider an expression with nested expressions.
    ///     `f h
    ///         g (j k x)`
    /// The coordinates of each expressions (1, 0), (1, 2), (2, 3), (2, 5)
    /// 
    /// ```
    fn maybe_app(&mut self, start: Location, expr: Expr) -> Result<Expr, SyntaxError> {
        let mut nodes = vec![];

        let mut row = if self.get_row() == start.row 
            || self.get_column() > start.col {
            start.row 
        } else { self.get_row() };

        while self.peek().and_then(|t| Some(t.is_terminal())).unwrap_or_else(|| false) && !self.is_done() && self.get_row() == row {
            nodes.push(self.binary(0, &mut Self::terminal)?);
            if self.get_column() > start.col {
                row = self.get_row()
            }
        }
        Ok(Expr::App { func: Box::new(expr), args: nodes })
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
                if self.is_done() {
                    return Ok(Expr::Section(Section::Left { infix: op, left: Box::new(expr) }))
                };
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
                .and_then(Self::subexpression)?.1,
        );
        self.eat(&Token::Kw(Keyword::Of))?;
        let arms = match self.peek() {
            Some(Token::CurlyL) => self.delimited(
                Token::CurlyL,
                Token::Semi,
                Token::CurlyR,
                Self::case_arms,
            )?,
            _ => self.many_col_aligned(Self::case_arms)?,
        };
        Ok(Expr::Case { expr, arms })
    }

    fn case_arms(&mut self) -> Result<(Match, Expr), SyntaxError> {
        let pattern = self.case_branch_pat()?;
        let bound = matches!(&pattern, Pat::Binder { .. });
        let alts = if let Some(Token::Pipe) = self.peek() {
            self.eat(&Token::Pipe)
                .and_then(|p| p.many_sep_by(Token::Pipe, Self::case_branch_pat))?
        } else {
            vec![]
        };
        let guard = if let Some(Token::Kw(Keyword::If)) = self.peek() {
            Some(
                self.eat(&Token::Kw(Keyword::If))
                    .and_then(|p| p.subexpression())?.1,
            )
        } else {
            None
        };
        self.eat(&Token::ArrowR)?;
        let body = self.subexpression()?.1;
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
                                .and_then(Self::case_branch_pat))
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
                            Self::case_branch_pat)?;
                        match this.peek() {
                            Some(Comma) => {
                                let mut items = this.delimited(
                                    Comma, 
                                    Comma, 
                                    ParenR, 
                                    Self::case_branch_pat)?;
                                items.insert(0, Pat::Ctor(ctor, args));
                                Ok(Pat::Binder{ 
                                    binder, 
                                    pattern: Box::new(Pat::Tuple(items))
                                })
                            }
                            Some(Pipe) => {
                                let mut items = this.many_sep_by(Pipe, Self::case_branch_pat)?;
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
                        this.many_sep_by(Pipe, Self::case_branch_pat)
                            .and_then(|alts| 
                                this.eat(&ParenR)
                                    .and_then(|_| 
                                        Ok(Pat::Binder{ 
                                            binder, 
                                            pattern: Box::new(Pat::Union(alts)) 
                                        })))
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
                            .and_then(Self::case_branch_pat))
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
            Some(ParenL) => {
                self.eat(&ParenL)?;
                match self.peek() {
                    Some(ParenR) => {
                        self.eat(&ParenR)?;
                        Ok(Pat::Tuple(vec![]))
                    }
                    Some(Ident(..) | Sym(..)) => {
                        let ctor = self.take_next();
                        match self.peek() {
                            Some(ParenR) => {
                                self.eat(&ParenR)?;
                                Ok(Pat::Lit(ctor.try_into().unwrap()))
                            }
                            Some(Comma) => {
                                let mut items = self.delimited(
                                    Comma, Comma, ParenR, 
                                    Self::case_branch_pat)?;
                                items.insert(0, Pat::Lit(ctor.try_into().unwrap()));
                                Ok(Pat::Tuple(items))
                            }
                            _ => {
                                let (args, _) = self.many_while(|t| !matches!(t, ParenR),
                                Self::case_branch_pat)?;
                                Ok(Pat::Ctor(ctor, args))
                            }
                        }
                    }
                    _ => {
                        let first = self.case_branch_pat()?;
                        match self.peek() {
                            Some(ParenR) => {
                                self.eat(&ParenR)?;
                                Ok(first)
                            }
                            Some(Comma) => {
                                let mut items = self.delimited(
                                    Comma, Comma, ParenR,
                                    Self::case_branch_pat)?;
                                items.insert(0, first);
                                Ok(Pat::Tuple(items))
                            }
                            _ => {
                                let (args, _) = self.many_while(
                                    |t| !matches!(t, ParenR), 
                                    Self::case_branch_pat)?;
                                let ctor: Result<Token, SyntaxError> = first.try_into().map_err(|x: &'static str| SyntaxError(x.into()));
                                Ok(Pat::Ctor(ctor?, args))
                            }
                        }
                    }
                }
            }
            Some(Invalid { data, msg, pos }) => Err(SyntaxError(format!(
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
        let (_, cond) = self.subexpression()?;
        self.eat(&Token::Kw(Keyword::Then))?;
        let (_, then) = self.subexpression()?;
        self.eat(&Token::Kw(Keyword::Else))?;
        let (_, other) = self.subexpression()?;

        Ok(Expr::Cond {
            cond: Box::new(cond),
            then: Box::new(then),
            other: Box::new(other),
        })
    }

    /// Called when already on a literal token
    fn literal(&mut self, token: Token) -> Result<Expr, SyntaxError> {
        let loc = self.loc();
        match &token {
            Token::Sym(_)
            | Token::Char(_)
            | Token::Str(_)
            | Token::Bytes(_)
            | Token::Num { .. }
            => Ok(Expr::Lit(Literal::from_token(token).unwrap())),

            // in theory, these cases shouldn't come up as this method is only called when peeking a `Token::Literal`.
            t => Err(SyntaxError(format!(
                "Invalid token type! Expected a literal, \
                but found `{0}` at {1}!",
                t, loc
            ))),
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
        use Token::{ParenR, ParenL, Comma, Operator}; 
        self.eat(&ParenL)?;
        match self.peek() {
            Some(ParenR) => {
                self.eat(&ParenR)?;
                Ok(Expr::Tuple(vec![]))
            }
            Some(Comma) => {
                self.eat(&Comma)?;
                self.eat(&ParenR)?;
                Ok(Expr::Ident(Var::Ident(",".into())))
            }
            Some(Operator(..)) => {
                let loc = self.loc(); 
                let infix = self.take_next().as_operator().unwrap();
                if matches!(self.peek(), Some(ParenR)) {
                    self.eat(&ParenR)?;
                    return Ok(Expr::Ident(Var::Infix(infix)))
                };
                // let fixity = self.fixities.get(&infix).copied();
                if let Some(Fixity { prec, ..}) = self
                    .fixities
                    .get(&infix)
                    .cloned() {
                    let right = Box::new(self.binary(prec, 
                        &mut Self::terminal)?);
                    self.eat(&ParenR)?;
                    let section = Expr::Section(Section::Right { infix, right });
                    self.maybe_app(loc, section)
                } else {
                    Err(SyntaxError(format!(
                        "Operator section error! The operator `{}` \
                        at {} does not have a defined fixity.", 
                        infix, loc
                    )))
                }
            }
            _ => {
                let (_, head) = self.subexpression()?;
                match self.peek() {
                    Some(ParenR) => {
                        self.eat(&ParenR)?;
                        Ok(head)
                    }
                    Some(Comma) => {
                        self.eat(&Comma)?;
                        let (mut items, _) = self.many_while(|t| matches!(t, Comma), Self::expression)?;
                        // let mut items = self.delimited(Comma, Comma, ParenR, |p| p.subexpression().and_then(|(_, x)| Ok(x)))?;
                        items.insert(0, head);
                        self.eat(&ParenR)?;
                        Ok(Expr::Tuple(items))
                    }
                    Some(Operator(..)) => {
                        let loc = self.loc();
                        let infix = self.take_next().as_operator().unwrap();
                        if matches!(self.peek(), Some(ParenR)) {
                            self.eat(&ParenR)?;
                            return Ok(Expr::Section(Section::Left { infix, left: Box::new(head) }));
                        } else {
                                Err(SyntaxError(format!(
                                    "Operator section error! The operator `{}` \
                                    at {} did not have its parentheses closed.", 
                                    infix, loc
                                )))
                            }
                    }
                    _ => {
                        let mut args = vec![];
                        while !self.is_done() && !self.match_curr(&ParenR) {
                            let (_, arg) = self.subexpression()?;
                            args.push(arg);
                        }
                        self.eat(&ParenR)?;
                        Ok(Expr::App { func: Box::new(head), args })
                    }
                }
            }
        }
    }

    fn record_app(&mut self, proto: Token) -> Result<Expr, SyntaxError> {
        let fields = self.delimited(Token::CurlyL, Token::Comma, Token::CurlyR, |p| {
            let left = p.take_next();
            match p.peek() {
                Some(Token::Comma) => Ok((left, None)),
                _ => Ok((left, Some(p.subexpression()?.1))),
            }
        })?;
        Ok(Expr::Record { proto, fields })
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
    fn named_pats(&mut self,) {
        todo!()
    }

    fn let_expr(&mut self) -> Result<Expr, SyntaxError> {
        self.eat(&Token::Kw(Keyword::Let))?;
        let mut bind = vec![];
        while !self.is_done() && !self.match_curr(&Token::Kw(Keyword::In)) {
            let pat = self.lambda_arg()?;
            self.eat(&Token::Eq)?;
            let (_, expr) = self.subexpression()?;
            bind.push(Binding { pat, expr });
            if self.match_curr(&Token::Comma) { self.take_next(); };
        }
        self.eat(&Token::Kw(Keyword::In))?;
        let body = Box::new(self.subexpression()?.1);
        Ok(Expr::Let { bind, body })
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
            // Some(Token::Operator(_)) => {
            //     // TODO: offload to `operator` method
            //     // since this is predicated on having matched an operator, it is safe to unwrap the results of the `Token`'s `as_operator` method call
            //     Ok(Expr::Ident(Var::Infix(
            //         self.take_next().as_operator().unwrap(),
            //     )))
            // }
            Some(Token::Comment(_)) => {
                let comment = self.take_next().as_comment().unwrap();
                self.comments.push(comment);
                self.terminal()
            }

            Some(Token::Lambda) => self.lambda(),

            Some(Token::Ident(_) | Token::Underscore) => {
                Ok(Expr::Ident(Var::Ident(self.take_next().get_string())))
            }

            Some(Token::Sym(..)) => Ok(Expr::Ident(Var::Cons(self.take_next().get_string()))),

            Some(Token::Char(_) | Token::Str(_) | Token::Bytes(_) | Token::Num { .. }) => {
                self.with_next(|t, p| p.literal(t))
                // self.literal(self.take_next())
                // Ok(Expr::Lit(Literal::from_token(self.take_next()).unwrap()))
            }

            Some(Token::Kw(kw)) => Err(SyntaxError(format!(
                "Unexpected keyword `{0}` in terminal position at {1} while parsing expression!\n\
                Perhaps a delimiter is missing in a prior expression?",
                kw, loc
            ))),

            Some(Token::Invalid { data, msg, pos }) => Err(SyntaxError(format!(
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
                        rest.insert(0, Pat::Var(sym));
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

            Some(Token::Invalid { data, msg, pos }) => Err(SyntaxError(format!(
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
        let (_, body) = self.subexpression()?;
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

    //-------- DECLARATIONS

    /// After parsing top-level module stuff, parse declarations
    fn declaration(&mut self) -> Result<Decl, SyntaxError> {
        let loc = self.loc(); 
        match self.peek() {
            Some(Token::Kw(Keyword::InfixL | Keyword::InfixR)) => {
                let fixity = self.fixity_spec()?;
                let infixes = self.with_fixity(fixity)?;
                Ok(Decl::Infix { fixity, infixes })
            }
            Some(Token::Kw(Keyword::Data)) => {
                self.data_decl()
            }
            Some(t) => {
                Err(SyntaxError(format!(
                    "Invalid token `{}` at {} while parsing body declarations!", t, loc
                )))
            }
            None => {
                Err(Self::unexpected_eof_while("parsing body declarations", loc))
            }
        }
    }

    /// Parses a data declaration.
    /// 
    /// A data declaration consists of
    ///     * Data type name
    ///     * Data type variables (~ generics)
    ///     * Data constructors/variants
    ///         * variants may be fieldless
    ///         * constructor + arguments
    fn data_decl(&mut self) -> Result<Decl, SyntaxError> {
        self.eat(&Token::Kw(Keyword::Data))?;
        let loc = self.loc();
        let name = match self.peek() {
            Some(Token::Sym(..)) => {
                Ok(Var::Cons(self.take_next().get_string()))
            }
            t => {
                Err(SyntaxError(format!(
                    "Invalid name used for data declaration! Expected a `Sym` token, but instead found `{:?}` at {}", t, loc
                )))
            }
        }?;
        let (poly, _) = self.many_while(
            |t| !matches!(t, Token::Eq), 
            |p| {
                let loc = p.loc();
                match p.peek() {
                    Some(Token::Ident(_)) => Ok(Var::Ident(p
                        .take_next()
                        .get_string())), 

                    Some(t) => Err(SyntaxError(format!(
                        "Invalid token found at {} while parsing data declaration type variables! \
                        Expected a token of variant `Ident`, but found `{}`", loc, t))),
                    None => Err(Self::unexpected_eof_while("parsing data declaration lhs type parameters", loc))
                }
            })?;
        self.eat(&Token::Eq)?;
        let variants = self.many_sep_by(
            Token::Pipe, Self::data_variant)?;
        let derives = if self.match_curr(&Token::Kw(Keyword::Derive)) {
            self.take_next();
            let loc = self.loc(); 
            match self.peek() {
                Some(Token::ParenL) => {
                    self.delimited(
                        Token::ParenL, Token::Comma, Token::ParenR, 
                        |p| {
                            let loc = p.loc();
                            match p.peek() {
                            Some(Token::Sym(_)) => {
                                Ok(Var::Cons(p.take_next().get_string()))
                            }
                            Some(t) => {
                                Err(SyntaxError(format!(
                                    "Invalid token type while parsing data declaration derive clause at {}. Expected a `Sym` but found `{:?}`", 
                                    loc, 
                                    t
                                )))
                            }
                            None => {
                                Err(Self::unexpected_eof_while("parsing data declaration derivations", loc))
                            }
                        }})
                }
                Some(Token::Sym(_)) => {
                    // safe to unwrap since we know `Token::Sym` successfully converts into a `Var::Cons`
                    Ok(vec![self.take_next().try_into().unwrap()])
                }
                Some(t) => {
                    Err(SyntaxError(format!(
                        "Invalid token type while parsing data declaration derive clause at {}. Expected a `Sym` but found `{:?}`", 
                        loc, 
                        t
                    )))
                }
                None => {
                    Err(Self::unexpected_eof_while(
                    "parsing data declaration derivations", 
                    loc))
                }
            }
        } else { Ok(vec![]) }?;
        Ok(Decl::Data { name, poly, variants, derives })
    }

    fn data_variant(&mut self) -> Result<DataVariant, SyntaxError> {
        todo!()
        // match self.peek() {
        //     Some(Token::Sym(_)) => {
        //         let ctor = self.take_next().get_string();
        //     }
        //     _ => {}
        // }
    }

    fn data_variant_args(&mut self) -> Result<Vec<TyPat<Var>>, SyntaxError> {
        todo!()
        // use Token::*; 
        // match self.peek() {
        //     Some(Pipe) => {}
        //     Some(CurlyL) => {
        //         // self.delimited(CurlyL, Comma, CurlyR, |p| )
        //     }
        //     Some(ParenL) => {}
        //     Some(Ident(_)) => {
        //         let ident = self.take_next();

        //     }
        //     Some(Sym()) => {}
        //     _ => {}
        // }
    }

    /// Reads a fixity keyword and precedence token to generate
    /// a fixity to apply to following operators
    fn fixity_spec(&mut self) -> Result<Fixity, SyntaxError> {
        // associativity rule from keyword token
        let assoc = self
            .take_next()
            .as_assoc_spec()
            .map_err(|err| SyntaxError(format!("{}", err)))?;
        // precedence
        let prec = self
            .take_next()
            .as_u8()
            .map_err(|err| SyntaxError(format!("{}", err)))?;
        Ok(Fixity { assoc, prec })
    }

    fn with_fixity(&mut self, fixity: Fixity) -> Result<Vec<Operator>, SyntaxError> {
        // let mut ops = vec![];
        // let loc = self.loc();

        self.many_while(|t| matches!(t, Token::Operator(_)), |p| {
            let loc = p.loc();
            let operator = p.take_next().as_operator();
            if let Some(op) = operator {
                if p.fixities.get(&op).is_some() {
                    return Err(SyntaxError(format!("Fixity declaration error! The operator `{}` at {} already has a defined fixity.", op, loc)))
                } else {
                    p.fixities.insert(op.clone(), fixity);
                }
                Ok(op)
            } else {
                Err(Self::unexpected_eof_while(
                    "parsing fixity spec at {}", loc))
            }
        }).and_then(|(ops, _)| Ok(ops))
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
    fn echo_exprs() {
        // let _ = echo_expr("\\a b -> a + b");
        // let _ = echo_expr("\\a -> \\b -> a + b");
        // let _ = echo_expr("\\(:A' a) -> a");
        // let _ = echo_expr("\\:A { a } -> a");
        let _ = echo_expr("(+) 1 2");
        // let _ = echo_expr("(+3)");
        // let _ = echo_expr("(c <| d / 9)");
        // let _ = echo_expr("3+");
        // let _ = echo_expr("let (a, b) = (1, 2) in a + b");

        let srcs = [
            "(+0)",
            "(+) 1 2", 
            "a |> b + c <| d", 
            "let (a, b) = (1, 2) in a + b",
            "case x + y of { (1, a) if a > 0 -> :True; _ -> :False }"
        ];

        for s in srcs {
            println!("source: `{}`", &s);
            let mut parser = Parser::new(s);
            let xs = parser.many(|p| {
                Ok((p.loc(), p.expression()?, p.loc(), p.peek().cloned().unwrap_or_else(|| Token::Eof)))
            });
            match xs {
                Ok((res, _)) => {
                    for (start, expr, end, tok) in res {
                        println!("start: {}, end: {}, ended on: {}", start, end, tok);
                        println!("{:#?}", expr);
                    }
                }
                Err(e) => println!("{}", e)
            }
        }

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

    macro_rules! __ {
        (:$ident:tt) => {
            Expr::Ident(Var::Ident(stringify!($ident).into()))
        };
    }

    #[test]
    fn test_tuple() {

        let pairs = [
            ( "(,)", Ok(__!(:,))),
            ("(a, b)", Ok(Expr::Tuple(vec![__!(:a), __!(:b)]))),
            ("(a b, c)", Ok(Expr::Tuple(vec![
                Expr::App { func: Box::new(__!(:a)), args: vec![__!(:b)] },
                __!(:c)
            ])))
        ];
        for (s, x) in pairs {
            let expr = Parser::new(s).expression();
            println!("{:?}", &expr);
            assert_eq!(expr, x)
        }
    }

    #[test]
    fn test_delimited() {
        use Token::{ParenL as L, Comma as C, ParenR as R};
        let src = "(1, 2, 3)";
        let mut parser = Parser::new(src);
        let expr = parser.delimited(L, C, R, &mut Parser::subexpression);
        println!("{:#?}", expr);
    }

    #[test]
    fn test_offside_rule() {
        use crate::compiler::parser::scan::Layout;
        use crate::compiler::syntax::NumFlag;

        // let sources = [
        //     "3 + \n   4",
        //     "a b c d",
        //     "let sum = \n  \\x y -> x + y"
        // ];

        let mut parser = Parser::new("(3 + \n   4)");
        let expr = parser
            .within_offside(Token::is_terminal, |p| 
                p.subexpression().and_then(|(_, x)| Ok(x)))
            .and_then(|res| 
                Ok(res.resolve(|x| x, |(func, args)| Expr::App {
                    func: Box::new(func), 
                    args 
                })));
        // println!("{:#?}", expr)
        assert_eq!(expr, Ok(
            Expr::Binary {
                infix: Operator::Reserved(BinOp::Plus),
                left: Box::new(
                    Expr::Lit(
                        Literal::Num { 
                            data: "3".into(), 
                            flag: NumFlag::Int 
                        }
                    )
                ),
                right: Box::new(
                    Expr::Lit(Literal::Num { 
                        data: "4".into(), 
                        flag: NumFlag::Int
                    })
                )}))
    }

    #[test]
    fn test_fixity() {
        let src = "infixr 9 </> \n\n3 + 4 </> 7";
        let mut parser = Parser::new(src);
        let decl = parser.declaration();
        println!("{:?}", parser.peek());
        let expr = parser.expression();
        println!("decl\n{:#?}\nexpr\n{:#?}", decl, expr)

    }
}
