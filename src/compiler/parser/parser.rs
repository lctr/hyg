use std::convert::{TryInto};

pub use crate::prelude::{
    span::{
        Location, Positioned
    },
    traits::Peek
};

use crate::{
    compiler::{
        lexer::{
            BinOp, Keyword, Lexer, Operator, Token,
        }, 
        syntax::{
            name::Name,
            expr::{
                Binding, Expr, Match, Section
            }, 
            decl::{
                Decl, DataVariant, Type, DataPat, TyParam
            },
            pattern::{Pat, Constraint}, 
            literal::Literal, 
            error::SyntaxError,
            fixity::{
                Fixity, FixityTable, Prec
            }, 
        }
    }, prelude::{either::Either, traits::Intern, symbol::{Symbol, Lexicon}}, 
};

pub use super::{
    scan::{Combinator, Consume, Layout},
    Comment,
};

impl<'t> Intern for Parser<'t> {
    type Key = Symbol;
    type Value = str;
    fn intern(&mut self, value: &Self::Value) -> Self::Key {
        self.lexer.intern_str(value)
    }
}

#[derive(Debug)]
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
            let sym = self.peek().and_then(|t| t.get_symbol());
            let curr = if let Some(s) = sym {
                (&self.lexer.lexicon[s]).to_string()
            } else {
                format!("{}", self.peek().unwrap_or_else(|| &Token::Eof))
            };
            Err(SyntaxError(format!(
                "Expected `{}`, but found `{}` at {}",
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

trait TokenLookup {
    fn lookup_str(&self, lexicon: &Lexicon) -> String;
}

impl TokenLookup for Token {
    fn lookup_str(&self, lexicon: &Lexicon) -> String {
        match self {
            Token::Upper(s) | Token::Lower(s) | Token::Operator(Operator::Custom(s)) => lexicon[*s].into(),
            _ => format!("{}", self)
        }
    }
}

impl TokenLookup for Option<&Token> {
    fn lookup_str(&self, lexicon: &Lexicon) -> String {
        self.and_then(|t| Some(t.lookup_str(lexicon))).unwrap()
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

    pub fn parse() -> Result<(), SyntaxError> {
        todo!()
    }

    //-------- DECLARATIONS

    /// After parsing top-level module stuff, parse declarations
    /// 
    /// A declaration may be any of the following:
    /// 
    /// * Fixity declaration
    ///     - 3 main nods
    ///         1.) `infixl` OR `infixr`, 
    ///         2.) `Num { flag: NumFlag::Int, ..}`
    ///         3.) AT LEAST ONE `Operator`
    ///     - must keep track of newly defined specs, as they will all need 
    ///       their own respective implementations and we want to flag to the 
    ///       user when said implementations are missing.
    /// * Data declaration
    /// * Type (alias) declaration
    /// * Class declaration
    /// * Function declaration
    /// * Type signature (?)
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
            Some(_) => {
                let sym = self.peek().and_then(|t| t.get_symbol());
                let tok = if let Some(s) = sym {
                    (&self.lexer.lexicon[s]).to_string()
                } else {
                    format!("{}", self.peek().unwrap_or_else(|| &Token::Eof))
                };
                Err(SyntaxError(format!(
                    "Invalid token `{}` at {} while parsing body declarations!", tok, loc
                )))
            }
            None => {
                Err(Self::unexpected_eof_while("parsing body declarations", loc))
            }
        }
    }

    /// Type-level syntax: basically patterns with only type constructors, type 
    /// variables, tuple/lists, and with the addition of low-precedence 
    /// right-associative arrows for functions.
    /// 
    /// * note: the function arrow `->` is a special kind of right 
    ///   associative where we basically give it precedence lower than 
    ///   possible.
    /// * consider the type signature of boolean `or`, which takes 
    ///   two `Bool` arguments and returns a `Bool` type:
    ///         
    ///         Bool -> Bool -> Bool
    ///       
    ///   Since functions are currried, the above is equivalent to
    ///         
    ///         Bool -> (Bool -> Bool)
    /// 
    /// * we need to support some convoluted things that can be boiled down
    ///   to a few rules:
    ///     * an atom must by any sort of identifier
    ///     * lists may only be of one type therefore must only have 1 node
    ///     * tuples can have any number of any different kind of types
    ///     * constructors are the default assumption for "apply"-like 
    ///       patterns, e.g., `a b c d`. 
    /// 
    /// 
    fn type_signature(&mut self) -> Result<Type, SyntaxError> {
        self.maybe_arrow_sep(|p| {
            // let first = p.type_atom()?;
            p.within_offside(
                Token::begins_pat, 
                Self::type_atom)
                .and_then(|res| 
                    res.resolve(
                        |ty| Ok(ty), 
                        |(head, tail)| 
                        Ok(Type::Apply(Box::new(head), tail))))
        })
    }

    fn maybe_arrow_sep<F>(&mut self, mut f: F) -> Result<Type, SyntaxError> 
    where
        F: FnMut(&mut Self) -> Result<Type, SyntaxError> 
    {
        let start = self.loc();
        let mut left = f(self)?;

        loop {
            if self.is_done() {break;} 
            if self.get_column() <= start.col {
                break;
            }
            if matches!(self.peek(), Some(Token::Kw(_) | Token::Pipe)) { 
                break; 
            }


            match self.peek() {
                Some(Token::Semi | Token::Pipe) => {self.take_next(); break;}
                Some(Token::ArrowR) => {
                    while self.match_curr(&Token::ArrowR) {
                        self.take_next();
                        let right = f(self)?;
                        left = Type::Arrow(Box::new(left), Box::new(right));
                    }
                }
                _ => break

            }
        }

        Ok(left)
    }

    fn grouped_types(&mut self) -> Result<Type, SyntaxError> {
        use Token::*;
        self.eat(&ParenL)?;
        if self.match_curr(&ParenR) {
            self.take_next();
            return Ok(Type::Unit)
        }

        let first_ty = self.type_signature()?;

        if self.match_curr(&ParenR) {
            self.take_next();
            return Ok(Type::Group(Box::new(first_ty)))
        }

        if self.match_curr(&Comma) {
            let mut rest = self.delimited(Comma, Comma, ParenR, Self::type_signature)?;
            rest.insert(0, first_ty);
            return Ok(Type::Tuple(rest))
        }

        let (rest_tys, _) = self
            .many_while(|t| !matches!(t, ParenR), 
            Self::type_signature)?;

        Ok(Type::Apply(Box::new(first_ty), rest_tys))
    }

    /// Helper for `type_sig`
    fn type_atom(&mut self) -> Result<Type, SyntaxError> {
        let loc = self.loc();
        use Token::*; 
        match self.peek() {
            Some(ParenL) => self.grouped_types(),
            Some(BrackL) => {
                self.eat(&BrackL)?;
                let ty = self.type_signature()?;
                self.eat(&BrackR)?;
                Ok(Type::List(Box::new(ty)))
            }
            Some(Lower(_)) => {
                let s = self.take_next().get_symbol().unwrap();
                Ok(Type::Var(Name::Ident(s)))
            }
            Some(Upper(_)) => {
                let s = self.take_next().get_symbol().unwrap();
                Ok(Type::TyCon(Name::Cons(s)))
            }
            Some(Underscore) => self.with_next(|_, _| Ok(Type::Anon)),

            Some(t) => Err(SyntaxError(format!(
                "Invalid type! Expected a token of type `Sym`, `Ident`, `(`, or `[`, but found `{:?}` at {}", t, loc
            ))),
            None => Err(Self::unexpected_eof_while("parsing type signature", loc))
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
            Some(Token::Upper(_)) => {
                let sym = self.take_next().get_symbol().unwrap();
                Ok(Name::Cons(sym))
            }
            t => {
                Err(SyntaxError(format!(
                    "Invalid name used for data declaration! Expected a `Sym` token, but instead found `{:?}` at {}", t, loc
                )))
            }
        }?;

        let mut constraints = vec![];
        // type parameters on the lhs of a data decl must be either:
        // Class name (`Sym`) + Type Variable (`Ident`)
        if self.match_curr(&Token::ParenL) {
            constraints = self.ty_constraints()?;
        };

        let poly = self.data_ty_vars()?;
        self.eat(&Token::Eq)?;

        let first = self.data_variant()?;
        let mut variants = vec![first];

        loop {
            if self.is_done() { break; }

            if matches!(self.peek(), Some(Token::Kw(_))) { break; }
            if !self.match_curr(&Token::Pipe) { break; }

            // if self.get_column() <= loc.col { break; }

            if self.match_curr(&Token::Pipe) {
                self.take_next();
                variants.push(self.data_variant()?);
            }
        }
        
        self.maybe_derive_clause(name, constraints, poly, variants)
    }

    fn expect_var_ident(&mut self) -> Result<Name, SyntaxError> {
        let loc = self.loc();
        if !self.is_done() {
            match self.peek() {
                Some(Token::Lower(_)) => {
                    let s = self.take_next().get_symbol().unwrap();
                    Ok(Name::Ident(s))
                },

                Some(t) => Err(SyntaxError(format!(
                    "Expected a type variable, but found `{:?}` at {}", t, loc
                ))),

                None => Err(Self::unexpected_eof_while(
                    "parsing type variables", loc))
            }
        } else {
            Err(Self::unexpected_eof_while("parsing type variables", loc))
        }
    }

    fn ty_constraints(&mut self) -> Result<Vec<Constraint<Name>>, SyntaxError> {
        let constraints = self.delimited(Token::ParenL, Token::Comma, Token::ParenR, 
                |parser| {
                    let loc = parser.loc();
                    match parser.peek() {
                        Some(Token::Upper(_)) => {
                            let cl = Name::Class(parser.take_next().get_symbol().unwrap());
                            let loc = parser.loc();

                            match parser.peek() {
                                Some(Token::Lower(_)) => {
                                    let tyvar = Name::Ident(parser.take_next().get_symbol().unwrap());
                                    Ok(Constraint(cl, tyvar))
                                }
                                t => {
                                    Err(SyntaxError(format!(
                                    "Unexpected token at {} in type constraint \
                                    position! Expected either either an \
                                    `Ident` or `Sym`, but found `{:?}`", 
                                    loc, t)))
                                }
                            }
                        }
                        Some(Token::Lower(_)) => {
                            let sym = parser.peek().and_then(|t| t.get_symbol());
                            let tok = sym.and_then(|s| Some(&parser.lexer.lexicon[s]));
                            Err(SyntaxError(format!(
                                "Invalid token type at {}! Expected an uppercase token of type `Upper` for the 
                                class constraint, but found type variable {}", loc, tok.unwrap_or_else(|| "")
                            )))
                        }
                        Some(t) => {
                            Err(SyntaxError(format!(
                                "Unexpected token at {} in type constraint \
                                position! Expected an uppercase token of \
                                type `Upper`, but found `{:?}`", 
                                loc, t
                                
                            )))
                        }
                        None => Err(Self::unexpected_eof_while("parsing type variables on left-hand side", loc))
                    }
        })?;
        self.eat(&Token::FatArrow)?;
        Ok(constraints)
    }

    fn data_ty_vars(&mut self) -> Result<Vec<Name>, SyntaxError> {
        let loc = self.loc();
        let mut poly = vec![];
        if self.match_curr(&Token::Eq) {
            return Ok(poly);
        }
    
        loop {
            if self.is_done() { break; }

            if matches!(self.peek(), Some(Token::Kw(_))) { break; }

            if self.get_column() <= loc.col && self.get_row() > loc.row { break; }

            if self.match_curr(&Token::Eq) { break; }

            poly.push(self.expect_var_ident()?);

            if self.match_curr(&Token::Eq) { break; }
        }
    
        Ok(poly)
    }

    fn maybe_derive_clause(&mut self, name: Name, constraints: Vec<Constraint<Name>>, poly: Vec<Name>, variants: Vec<DataVariant>) -> Result<Decl, SyntaxError> {
        let derives = if !self.match_curr(&Token::Kw(Keyword::Derive)) {
            Ok(vec![])
        } else {
            self.eat(&Token::Kw(Keyword::Derive))?;
            let loc = self.loc(); 
            match self.peek() {
                Some(Token::ParenL) => {
                    self.delimited(
                        Token::ParenL, Token::Comma, Token::ParenR, 
                        |p| {
                            let loc = p.loc();
                            match p.peek() {
                            Some(Token::Upper(_)) => {
                                let s = p.take_next().get_symbol().unwrap();
                                Ok(Name::Class(s))
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
                Some(Token::Upper(_)) => {
                    let s = self.take_next().get_symbol().unwrap();
                    Ok(vec![Name::Cons(s)])
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
        }?;

        Ok(Decl::Data { name, constraints, poly, variants, derives})
    }

    fn data_variant(&mut self) -> Result<DataVariant, SyntaxError> {
        let loc = self.loc();
        match self.peek() {
            Some(Token::Upper(_)) => {
                // safe to unwrap since we know it's got an interned symbol
                let ctor = Name::Cons(self.take_next().get_symbol().unwrap());

                if self.is_done() 
                || matches!(self.peek(), Some(&Token::Pipe 
                    | &Token::Kw(..))) {
                    return Ok(DataVariant { ctor, args: Either::Right(()) });
                }

                if matches!(self.peek(), Some(&Token::CurlyL)) {
                    return self
                        .data_record_fields()
                        .and_then(|fields| Ok(DataVariant { 
                            ctor, 
                            args: Either::Left(DataPat::Keys(fields))
                        }));
                }

                let mut args = vec![];
                loop {
                    if matches!(self.peek(), Some(Token::Kw(..))) {break; }

                    if self.match_curr(&Token::Pipe) {
                        self.take_next();
                    } 

                    if self.is_done()  { break; }
                    if self.get_column() <= loc.col { break; }

                    if matches!(self.peek().and_then(|t| Some(Token::begins_pat(t))), Some(true)) {
                        // args.push(self.type_signature()?);
                        args.push(self.data_variant_arg()?);
                    }
                    
                }
                
                Ok(DataVariant { ctor, args: Either::Left(DataPat::Args(args)) })
            }
            Some(t) => {
                Err(SyntaxError(format!(
                    "Invalid token found at {}! Expected a constructor, but found `{:?}`", loc, t
                )))
            }
            None => Err(Self::unexpected_eof_while("parsing data variants", loc))
        }
    }

    fn data_record_fields(&mut self) -> Result<Vec<(Name, Type)>, SyntaxError> {
        use Token::*;
        let loc = self.loc();
        let t = self.delimited(CurlyL, Comma, CurlyR, 
        |p| {
            let name = match p.peek() {
                Some(Lower(_)) => {
                    Ok(Name::Ident(p.take_next().get_symbol().unwrap()))
                }
                Some(t) => {
                    Err(SyntaxError(format!(
                        "Invalid token found at {}! Expected an identifier, but found `{:?}`", loc, t
                    )))
                }
                None => Err(Self::unexpected_eof_while("parsing data variant record fields", loc))
            }?;
            p.eat(&Colon2)?;
            let tysig = p.type_signature()?;
            Ok((name, tysig))
        })?;
        Ok(t)
    }

    fn data_variant_arg(&mut self) -> Result<Type, SyntaxError> {
        let loc = self.loc();
        use Token::*; 
        match self.peek() {
            Some(ParenL) => {
                self.take_next();

                if self.match_curr(&ParenR) {
                    self.take_next();
                    return Ok(Type::Tuple(vec![]))
                }

                let first = self.type_signature()?;

                if self.match_curr(&ParenR) {
                    self.take_next();
                    return Ok(first)
                }

                if self.match_curr(&Comma) {
                    let (mut items, _) = self.many_while(
                        |t| matches!(t, &Comma), 
                        Self::type_signature)?;
                    items.insert(0, first);
                    self.eat(&ParenR)?;
                    return Ok(Type::Tuple(items))
                }

                let (args, _) = self.many_while(
                    |t| !matches!(t, ParenR), 
                    Self::type_signature)?;

                Ok(Type::Apply(Box::new(first), args))
            }
            Some(Lower(_)) => {
                let s = self.take_next().get_symbol().unwrap();
                Ok(Type::Var(Name::Ident(s)))

            }
            Some(Upper(_)) => {
                let s = self.take_next().get_symbol().unwrap();
                Ok(Type::TyCon(Name::Cons(s)))
            }
            Some(t) => {
                Err(SyntaxError(format!(
                    "Invalid token found at {}! Expected a constructor, but found `{:?}`", loc, t
                )))
            }
            None => Err(Self::unexpected_eof_while("parsing data variants", loc))
        }
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
            .and_then(|p| Ok(p.into()))
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

    pub fn expression(&mut self) -> Result<Expr, SyntaxError> {
        let (start, expr) = self.subexpression()?;

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
        self.binary(Prec::LAST, &mut Self::terminal)
    }

    /// First saves the current location, and then parses an expressiob, 
    /// returning both the location and the parsed results. 
    pub fn subexpression(&mut self) -> Result<(Location, Expr), SyntaxError> {
        let loc = self.loc();
        let expr = self.binary(Prec::LAST, &mut Self::terminal)?;
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
            nodes.push(self.binary(Prec::LAST, &mut Self::terminal)?);
            if self.get_column() > start.col {
                row = self.get_row()
            }
        }
        Ok(Expr::App { func: Box::new(expr), args: nodes })
    }

    #[inline]
    fn peek_fixity(&mut self) -> Option<&Fixity> {
        let op = self.peek().and_then(|token| token.as_operator());
        if let Some(operator) = op {
            self.fixities.get(&operator)
        } else {
            None
        }
    }

    fn binary<F>(&mut self, min_prec: Prec, f: &mut F) -> Result<Expr, SyntaxError>
    where
        F: FnMut(&mut Self) -> Result<Expr, SyntaxError>,
    {
        let mut expr = f(self)?;
        while let Some(&Fixity { assoc, prec }) = self.peek_fixity() {
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

                    Some(t@Lower(_)) => Err(no_local_vars(loc, t, &binder)),

                    Some(Upper(_)) => {
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
                            .and_then(|alts| this
                                .eat(&ParenR)
                                .and_then(|_| Ok(Pat::Binder{ 
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

            Some(t@Lower(_)) => Err(no_local_vars(loc, t, &binder)),

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

            Some(Lower(_)) => {
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
                    Some(Lower(..) | Upper(..)) => {
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
            Token::Upper(_)
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
                Ok(Expr::Ident(Name::Ident(self.intern(&*Comma.to_string()))))
            }
            Some(Operator(..)) => {
                let loc = self.loc(); 
                let infix = self.take_next().as_operator().unwrap();
                if matches!(self.peek(), Some(ParenR)) {
                    self.eat(&ParenR)?;
                    return Ok(Expr::Ident(Name::Infix(infix)))
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
        use Token::{CurlyL as L, Comma as C, CurlyR as R, Eq as E};
        let fields = self.delimited(L, C, R, |p| {
            let left = p.take_next();
            let right = match p.peek() {
                Some(C) => None,
                Some(E) => {
                    p.take_next();
                    let (_, rhs) = p.subexpression()?;
                    Some(rhs)
                }
                _ => {
                    let (_, rhs) = p.subexpression()?;
                    Some(rhs)
                },
            };
            Ok((left, right))
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
    fn named_pats(&mut self, ) {
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

            Some(Token::Lower(_)) => {
                let s = self.take_next().get_symbol().unwrap();
                Ok(Expr::Ident(Name::Ident(s)))
            }

            Some(Token::Upper(_)) => {
                let s = self.take_next().get_symbol().unwrap();
                Ok(Expr::Ident(Name::Cons(s)))
            },

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
                    Some(Lower(_) | Upper(_)) => {
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
            Some(Upper(..) | Lower(..)) => {
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
            Some(Lower(_)) => Ok(Pat::Var(self.take_next())),

            Some(BrackL) => {
                let elements = self
                    .delimited(BrackL, Comma, BrackR, 
                        Self::lambda_arg)?;
                Ok(Pat::List(elements))
            }

            Some(Upper(..)) => {
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
}

mod tests {
    #![allow(unused)]
    extern crate test;
    use std::marker::PhantomData;

    use crate::{compiler::{syntax::name::Name, lexer::Span}, prelude::span::Spanned};

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
            test::black_box(Parser::new(source.as_str()).binary(0.into(), &mut Parser::expression))
        });
    }

    #[test]
    fn test_binary_expression() {
        assert_eq!(Prec::LAST, Prec::from(0));

        let source = "a + b / c - d";
        let mut parser = Parser::new(source);
        let syms = ["a", "b", "c", "d"].into_iter().map(|s| parser.intern(*s)).collect::<Vec<_>>();

        let expr = parser.binary(0.into(), &mut |p| {
            p.expression()
        });

        println!("{:#?}", &expr);
        assert_eq!(
            expr,
            Ok(Expr::Binary {
                infix: BinOp::Plus.into(),
                left: Box::new(Expr::Ident(Name::Ident(syms[0]))),
                right: Box::new(Expr::Binary {
                    infix: BinOp::Minus.into(),
                    left: Box::new(Expr::Binary {
                        infix: BinOp::Div.into(),
                        left: Box::new(Expr::Ident(Name::Ident(syms[1]))),
                        right: Box::new(Expr::Ident(Name::Ident(syms[2])))
                    }),
                    right: Box::new(Expr::Ident(Name::Ident(syms[3])))
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
        let _expr = parser.binary(Prec::LAST, &mut Parser::terminal);

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

    #[test]
    fn test_type_syntax() {
        let src = "(a -> b) -> [(a, b)]";
        let mut parser = Parser::new(src);
        let sig = parser
            .type_signature()
            .and_then(|ty| {
                println!("arity: {:?}", Type::arrow_arity_in(&ty)); 
                Ok(ty)
            });

        println!("{0:#?}\n\n{0}", &sig.unwrap());
    }

    #[test]
    fn test_record_fields() {
        let source = "\
        { name :: String\
        , age :: (Int, Int)\
        , free :: Free (f (Free f a))\
        , nicknamed :: String -> Bool \
        }";
        println!("input:");
        println!("{}", &source);
        let mut parser = Parser::new(source);
        let fields = parser.data_record_fields();
        let mut lexicon = parser.lexer.lexicon;
        let syms = [
            "name", "String", "age", 
            "Int", "free", "Free", 
            "f", "a", "nicknamed", "Bool"
            ].into_iter().map(|t| lexicon.intern(*t)).collect::<Vec<_>>();

        let expected = vec![
            (
                Name::Ident(syms[0]), 
                Type::TyCon(Name::Cons(syms[1]))
            ), 
            (
                Name::Ident(syms[2]),
                Type::Tuple(vec![
                    Type::TyCon(Name::Cons(syms[3])), 
                    Type::TyCon(Name::Cons(syms[3]))])

            ),
            (
                Name::Ident(syms[4]),
                Type::Apply(Box::new(Type::TyCon(Name::Cons(syms[5]))), 
                    vec![
                        Type::Group(Box::new(
                            Type::Apply(
                                Box::new(Type::Var(Name::Ident(syms[6]))),
                                    vec![
                                        Type::Group(
                                            Box::new(
                                                Type::Apply(
                                                    Box::new(
                                                    
                                                    Type::TyCon(Name::Cons(syms[5]))
                                                    ),
                                                    vec![
                                                        Type::Var(Name::Ident(syms[6])),
                                                        Type::Var(Name::Ident(syms[7]))
                                                    ]
                                                )
                                            )
                                        )
                                    ]
                                )
                            )
                        )
                    ])
            ),
            (
                Name::Ident(syms[8]),
                Type::Arrow(Box::new(Type::TyCon(Name::Cons(syms[1]))), Box::new(Type::TyCon(Name::Cons(syms[9]))))
            )
        ];

        assert_eq!(fields, Ok(expected))
    }

    #[test]
    fn test_data_decl() {
        let src = "
        data Bool = True | False
        
        data Womp (Show x) => x 
            = A { name :: x -> Bool -> Bool } 
            | B { names :: [x] -> [Bool]}
            deriving (Eq, Show)
        
        data Free f a
            = Pure a 
            | Free (f (Free f a))
        ";
        let mut parser = Parser::new(src);
        (0..3).for_each(|_| {
            println!("> {:?}", parser.peek());
            let decl = parser.declaration();
            println!("{:#?}", decl);
            println!(">> {:?}", parser.peek());
        });
    }
}
