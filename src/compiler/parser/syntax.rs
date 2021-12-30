#![allow(unused)]
use std::borrow::Cow;

use crate::{
    compiler::lexer::NumFlag,
    prelude::{either::Either, traits::Newtype},
};

use super::{fixity::Fixity, Operator, Token};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    Char(char),
    Str(String),
    Bytes(Vec<u8>),
    // identical to internal data for a `Token` of variant `Num`. Not read until evaluation as they may be overloaded, e.g., the flags `Dec` and `Sci` may become f32 or f64.
    Num { data: String, flag: NumFlag },
    // used for constants/data constructor names
    Sym(String),
}

impl Literal {
    /// Note: this method is primarily used for pattern matching purposes,
    /// hence this matches `Token`s whose variants coincide with those of
    /// `Literal`, EXCLUDING the `Literal::Sym` variant.
    /// This is because the `Literal::Sym` variant plays a role in matching data constructors as well as record field keys
    pub fn is_token_literal(token: &Token) -> bool {
        use Token::*;
        matches!(token, Char(_) | Bytes(_) | Str(_) | Num { .. })
    }

    /// Takes a `Token` and, if its variant corresponds to a `Literal` variant,
    /// returns it wrapped in a `Some` variant.
    /// Returns `None` otherwise.
    ///
    /// *Note:* Unlike the `is_token_literal` method, this method ACCEPTS `Sym` `Token` variants.
    pub fn from_token(token: Token) -> Option<Self> {
        match token {
            Token::Sym(s) => Some(Self::Sym(s)),
            Token::Char(c) => Some(Self::Char(c)),
            Token::Str(s) => Some(Self::Str(s)),
            Token::Bytes(bs) => Some(Self::Bytes(bs)),
            Token::Num { data, flag } => Some(Self::Num { data, flag }),
            _ => None,
        }
    }
}

impl From<Literal> for Pat {
    fn from(lit: Literal) -> Self {
        Pat::Lit(lit)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pat {
    Lit(Literal),
    /// Identifier. Matches anything.
    Var(Token),
    /// `(x, y)`
    Tuple(Vec<Pat>),
    /// `[x, y]`
    List(Vec<Pat>),
    /// Data constructor pattern. The pattern `(:Foo x)` matches an instance (bound by `x`) of a given data constructor `Foo :: X -> Foo X`
    Ctor(Token, Vec<Pat>),
    /// `x { y, z }`
    Record {
        name: Token,
        fields: Vec<(Token, Pat)>,
    },
    /// Wildcard pattern `_`, matches with (but implies no binding to) anything.
    Wild,
    /// IN CASE EXPRESSION BRANCH PATTERNS: Equivalent to using the `Wild` subpattern for the rest of the pattern. For example, the pattern `(..)` matches every tuple, and the pattern `_ { .. }` matches every record.
    /// IN LAMBDA ARG PATTERNS: to discard the rest of the arguments provided
    Rest,
    /// VALID ONLY FOR CASE EXPRESSION BRANCH PATTERNS
    Binder {
        binder: Token,
        pattern: Vec<Pat>,
    },
    /// VALID ONLY FOR CASE EXPRESSION BRANCH PATTERNS
    Union(Vec<Pat>),
}

/// Represent the patterns being matched on in a case expression
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Match {
    pub pattern: Pat,
    /// On pattern matching, is the pattern re-bound to a (local) variable?
    pub bound: bool,
    /// Does a pattern have to satisfy an expression in order to successfully match? If so, then the `pattern` field *must* be a `Pat::Binder` variant.
    /// a if a > 0 -> a - 1
    pub guard: Option<Expr>,
    /// Are there any other patterns alternatives to this one?
    ///
    /// *Note:* It is a syntax error to have alternatives consisting of `Var` patterns such as `x`, regardless as to whether `x` as a binding locally in scope. Therefore, a pattern like `x | y` is always illegal.
    ///
    /// Instead, to match a pattern that requires a constructor, the sigil `:` is prepended to the constructor token, e.g., `:MyConstructor a b -> ...`
    ///
    /// When a match pattern has alternatives and is also bound, it corresponds to binding *any one of* the available patterns.
    /// For example, the pattern `a@([1, 2] | [2, 1])` either `[1, 2]` or `[2, 1]` and binds the resulting match to the variable `a`.
    ///
    /// It is illegal for distinct binders to apply to the same element in a match, i.e., `a @ 1 | b @ 2`, as this would lead to the introduction of unbound variables.
    /// Hence, the pattern binder `@` has lower precedence than `|`, such that `a @ 1 | 2` is equivalent to `a @ (1 | 2)`
    pub alts: Vec<Pat>,
}

impl Default for Match {
    fn default() -> Self {
        Match {
            pattern: Pat::Wild,
            bound: false,
            guard: None,
            alts: vec![],
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Arity(i32);

impl Newtype for Arity {
    type Inner = i32;
    fn get(&self) -> Self::Inner {
        self.0
    }

    fn get_ref(&self) -> &Self::Inner {
        &self.0
    }

    fn get_mut(&mut self) -> &mut Self::Inner {
        &mut self.0
    }
}

impl std::ops::Add for Arity {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for Arity {
    fn add_assign(&mut self, rhs: Self) {
        (*self).0 = self.0 + rhs.0;
    }
}

impl From<i32> for Arity {
    fn from(n: i32) -> Self {
        Arity(n)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Binding {
    pat: Pat,
    expr: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Var {
    Ident(String),
    Infix(Operator),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    Lit(Literal),
    Ident(Var),
    /// Operator section. See [`Section`] for more details.
    Section(Section),
    Tuple(Vec<Expr>),
    /// While there exists a `List` variant for `Pat`, which would match this expression, an `Array` variant corresponds to a literal expression parsed from `[a, b, c]` where the elements (in this case `a`, `b`, `c`) are all individually parsed and collected, i.e., list literal.
    ///
    /// Another way of thinking about it is that `Expr::Array`s are treated eagerly, may not grow, and have a size known at compile time. The same cannot be said for `Expr::List`s.
    Array(Vec<Expr>),
    /// On the other hand, a `List` variant for `Expr` is a list comprehension
    /// and inherently lazy.
    ///
    /// Example: If we assume `is'even` is a predicate on numbers, then the
    /// input `[a + 1 | a <- 0..10, is'even a ]` generates a list of even
    /// numbers between 1 and 9.
    ///
    /// `Expr::List`s are equivalent to iterating a lambda expression through a range of arguments specified by its *generators* (or *binders*, based on the field name `bind`) and filtered according to its *predicates* (field name `pred`).
    List {
        expr: Box<Expr>,
        bind: Vec<Binding>,
        pred: Vec<Expr>,
    },
    Lam {
        arg: Pat,
        arity: Arity,
        body: Box<Expr>,
    },
    App {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    Unary {
        prefix: Operator,
        right: Box<Expr>,
    },
    Binary {
        infix: Operator,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Case {
        expr: Box<Expr>,
        arms: Vec<(Match, Expr)>,
    },
    Cond {
        cond: Box<Expr>,
        then: Box<Expr>,
        other: Box<Expr>,
    },
    /// A `let` expression is equivalent to an immediately applied lambda
    Let {
        bind: Vec<Binding>,
        body: Box<Expr>,
    },
    /// Record expressions are calls or dictionary applications, *not*
    /// declarations, and hence may not need the RHS of its field,  **if and
    /// only if** the `Token` is an `Ident` variant pointing to a local
    /// binding *in scope*, where RHS corrresponds to the `Option<Expr>` of
    /// the `fields` field's elements.
    ///
    /// * The `proto` field refers to the *constructor* used to build
    ///   the record. If the `Token` in the `proto` field is an `Ident`
    ///   pointing to a binding in scope that is a record type (i.e., a
    ///   reference to another record), then that record will be copied
    ///   (or shared or cloned, depending on analysis and evaluation
    ///   strategy) as default values for record fields not indicated in
    ///   the record call expression.
    ///
    /// Otherwise, the LHS of each field accepts a `Token::Ident` or a
    /// `Token::Sym` and requires a `Some` variant containing the RHS
    /// expression. This operationally builds a record with a(*n associated?*)
    /// type from `proto` with fields `fields`.
    Record {
        proto: Token,
        fields: Vec<(Token, Option<Expr>)>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Section {
    Left { infix: Operator, left: Box<Expr> },
    Right { infix: Operator, right: Box<Expr> },
}

/// Trivial implementation since `Expr::Section` is a tuple variant with a single `Section` field
impl From<Section> for Expr {
    fn from(section: Section) -> Self {
        Expr::Section(section)
    }
}

/// Transform a pair of (Expr, Section) and (Section, Expr) into an Expr.
/// *Note:* the order of the pairs should be irrelevant.
/// TODO: Confirm this ^ later.
impl From<(Expr, Section)> for Expr {
    fn from((expr, section): (Expr, Section)) -> Self {
        match section {
            Section::Left { infix, left } => Expr::Binary {
                infix,
                left,
                right: Box::new(expr),
            },
            Section::Right { infix, right } => Expr::Binary {
                infix,
                left: Box::new(expr),
                right,
            },
        }
    }
}

impl From<(Section, Expr)> for Expr {
    fn from((section, expr): (Section, Expr)) -> Self {
        match section {
            Section::Left { infix, left } => Expr::Binary {
                infix,
                left,
                right: Box::new(expr),
            },
            Section::Right { infix, right } => Expr::Binary {
                infix,
                left: Box::new(expr),
                right,
            },
        }
    }
}

// TODO!!!!
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
// defining data structures.
pub enum Decl {
    // using Spanish *tipo* for `type` since it's a reserved kw and I don't wanna do `r#type` or other variations i'll forget to be consistent with
    Sig {
        name: Token,
        tipo: Type,
    },
    Data {
        name: Token,
        forms: Vec<(Token, Vec<Pat>)>,
    },
    Func {
        name: Token,
    },
    Class {
        name: Token,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Path {
    Single(Token),
    Flat(Vec<Token>),
    Nest(Vec<Token>, Vec<Path>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stmt {
    Expr(Expr),
    Decl(Decl),
    Import {
        items: Vec<()>,
        path: Vec<Token>,
        alias: Token,
    },
    Export {
        items: Vec<()>,
        path: Vec<Token>,
        alias: Token,
    },
}

pub struct Module {
    module: Token,
    imports: Vec<()>,
    exports: Vec<()>,
    fixities: Vec<(Operator, Fixity)>,
    exprs: Vec<Expr>,
    decls: Vec<Decl>,
}

// for generating unique identifiers
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fresh(u64);
