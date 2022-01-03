use std::convert::TryFrom;

use crate::prelude::traits::Identity;

use super::{literal::Literal, Token};

impl From<Literal> for Pat {
    fn from(lit: Literal) -> Self {
        Pat::Lit(lit)
    }
}

pub type Pat = Morpheme<Literal, Token>;

pub enum Field {
    /// A record field getter. May be either a
    Access,
    Alias,
    Update,
}

/// Generic enum to represent the skeleton of *patterns*.
/// The nodes generated by the `Parser` for *patterns* and *NOT*
/// expressions: `Pat`s, are instances of `Morpheme<Literal, Token>.
/// This is because this enum will be reused through various passes,
/// e.g., renaming.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Morpheme<L, T> {
    Lit(L),
    /// Identifier. Matches anything.
    Var(T),
    /// `(x, y)`
    Tuple(Vec<Self>),
    /// `[x, y]`
    List(Vec<Self>),
    /// Data constructor pattern. The pattern `(:Foo x)` matches an instance (bound by `x`) of a given data constructor `Foo :: X -> Foo X`
    Ctor(T, Vec<Self>),
    /// Data constructor record pattern. `x { .y y, :z z }`
    /// Note, if no field rhs in lambda arguments, a `Var` pattern is internall
    /// added. In particular, `x { :y }` = `x { y }`.
    Record {
        ctor: T,
        fields: Vec<(T, Option<Self>)>,
    },
    /// Wildcard pattern `_`, matches with (but implies no binding to) anything.
    Wild,
    /// IN CASE EXPRESSION BRANCH PATTERNS: Equivalent to using the `Wild` subpattern for the rest of the pattern. For example, the pattern `(..)` matches every tuple, and the pattern `_ { .. }` matches every record.
    ///
    /// IN LAMBDA ARG PATTERNS: to discard the rest of the arguments provided
    Rest,
    /// VALID ONLY FOR CASE EXPRESSION BRANCH PATTERNS
    ///
    /// A pattern bound to a variable, e.g., `a @ 1`
    Binder {
        binder: T,
        pattern: Box<Self>,
    },
    /// VALID ONLY FOR CASE EXPRESSION BRANCH PATTERNS
    ///
    /// A pattern as a collection of subpatterns. A `Union` pattern, such as
    /// `1 | 2`, matches if any of its subpatterns match.
    Union(Vec<Self>),
}

impl<L, T> Morpheme<L, T> {
    pub fn map_lit<F, X>(self, f: F) -> Morpheme<X, T>
    where
        F: Fn(L) -> X,
    {
        match self {
            Morpheme::Lit(l) => Morpheme::Lit(f(l)),
            Morpheme::Var(t) => Morpheme::Var(t),
            Morpheme::Tuple(ms) => Morpheme::Tuple(
                ms.into_iter()
                    .map(|m| m.map_lit(|l| f(l)))
                    .collect::<Vec<_>>(),
            ),

            Morpheme::List(ms) => Morpheme::List(
                ms.into_iter()
                    .map(|m| m.map_lit(|l| f(l)))
                    .collect::<Vec<_>>(),
            ),
            Morpheme::Ctor(t, ms) => Morpheme::Ctor(
                t,
                ms.into_iter()
                    .map(|m| m.map_lit(|l| f(l)))
                    .collect::<Vec<_>>(),
            ),
            Morpheme::Record { ctor: name, fields } => {
                Morpheme::Record {
                    ctor: name,
                    fields: fields
                        .into_iter()
                        .map(|(t, m)| {
                            (
                                t,
                                m.map(|m| {
                                    m.map_lit(|l| f(l))
                                }),
                            )
                        })
                        .collect::<Vec<_>>(),
                }
            }
            Morpheme::Wild => Morpheme::Wild,
            Morpheme::Rest => Morpheme::Rest,
            Morpheme::Binder { binder, pattern } => {
                Morpheme::Binder {
                    binder,
                    pattern: Box::new(pattern.map_lit(f)),
                }
            }
            Morpheme::Union(ms) => Morpheme::Union(
                ms.into_iter()
                    .map(|m| m.map_lit(|l| f(l)))
                    .collect::<Vec<_>>(),
            ),
        }
    }
}

impl std::convert::TryFrom<Morpheme<Literal, Token>>
    for Token
{
    type Error = &'static str;
    fn try_from(
        value: Morpheme<Literal, Token>,
    ) -> Result<Self, Self::Error> {
        match value {
            Morpheme::Lit(Literal::Bytes(x)) => Ok(Token::Bytes(x)),
            Morpheme::Lit(Literal::Char(c)) => Ok(Token::Char(c)),
            Morpheme::Lit(Literal::Num { data, flag }) => Ok(Token::Num{data,flag}),
            Morpheme::Lit(Literal::Str(s)) => Ok(Token::Str(s)),
            Morpheme::Lit(Literal::Sym(s)) => Ok(Token::Sym(s)),
            Morpheme::Var(t) => Ok(t),
            Morpheme::Wild => Ok(Token::Underscore),
            Morpheme::Rest => Ok(Token::Dot2),
            _ => Err("Invalid Token extraction from Morpheme<Literal, Token>! Only single-token morphemes/patterns may be successfully converted into a Token.")
        }
    }
}

pub enum Matcher {
    /// Refuses literals, unions, guards and binders
    Lambda,
    /// Allows literals
    Case,
}

// trait
