// extern crate pretty;

use std::convert::TryInto;

use crate::prelude::pretty::punctuate;

use super::{
    literal::Literal,
    name::Name,
    pattern::{Morpheme, Pat},
    Newtype, Operator, Token,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Binding {
    pub pat: Pat,
    pub expr: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Generator<L, R> {
    pub lhs: L,
    pub rhs: R,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    Lit(Literal),
    Ident(Name),
    /// Operator section. See [`Section`] for more details.
    Section(Section),
    Tuple(Vec<Expr>),
    /// While there exists a `List` variant for `Pat`, which would match this expression, an `Array` variant corresponds to a literal expression parsed from `[a, b, c]` where the elements (in this case `a`, `b`, `c`) are all individually parsed and collected, i.e., list literal.
    ///
    /// Another way of thinking about it is that `Expr::Array`s are treated eagerly, may not grow, and have a size known at compile time. The same cannot be said for `Expr::List`s.
    Array(Vec<Expr>),
    // TODO!!!!
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
        binds: Vec<Binding>,
        preds: Vec<Expr>,
    },
    // unsure whether it's best to curry all lambda's during parsing
    Lam {
        arg: Pat,
        body: Box<Expr>,
    },
    #[allow(unused)]
    Lambda {
        args: Vec<Pat>,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Section {
    Left { infix: Operator, left: Box<Expr> },
    Right { infix: Operator, right: Box<Expr> },
}

impl Section {
    pub fn get_infix(&self) -> &Operator {
        match self {
            Section::Left { infix, .. }
            | Section::Right { infix, .. } => infix,
        }
    }
}

impl std::fmt::Display for Section {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Section::Left { infix, left } => {
                write!(f, "({}{})", left, infix)
            }
            Section::Right { infix, right } => {
                write!(f, "({}{})", infix, right)
            }
        }
    }
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
            Section::Right { infix, right } => {
                Expr::Binary {
                    infix,
                    left: Box::new(expr),
                    right,
                }
            }
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
            Section::Right { infix, right } => {
                Expr::Binary {
                    infix,
                    left: Box::new(expr),
                    right,
                }
            }
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Expr::Lit(l) => write!(f, "{}", l),
            Expr::Ident(x) => write!(f, "{}", x),
            Expr::Section(s) => write!(f, "{}", s),
            Expr::Tuple(xs) => {
                write!(f, "(")?;
                punctuate(',', xs, f)?;
                write!(f, ")")
            }
            Expr::Array(xs) => {
                write!(f, "[")?;
                punctuate(',', xs, f)?;
                write!(f, "]")
            }
            Expr::List { expr, binds, preds } => {
                write!(f, "[ ")?;
                write!(f, "{} |", expr)?;
                if !binds.is_empty() {
                    // punctuate(",", bind, f)?;
                }
                if !preds.is_empty() {
                    punctuate(",", preds, f)?;
                }
                write!(f, " ]")
            }
            Expr::Lam { arg, body } => todo!(),
            Expr::Lambda { args, body } => todo!(),
            Expr::App { func, args } => {
                write!(f, "(")?;
                punctuate("", args, f)?;
                write!(f, ")")
            }
            Expr::Unary { prefix, right } => todo!(),
            Expr::Binary { infix, left, right } => todo!(),
            Expr::Case { expr, arms } => todo!(),
            Expr::Cond { cond, then, other } => todo!(),
            Expr::Let { bind, body } => todo!(),
            Expr::Record { proto, fields } => todo!(),
        }
    }
}

// fn print_expr(indent: usize, expr: &Expr) {}
