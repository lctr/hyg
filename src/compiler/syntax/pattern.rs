use std::convert::TryInto;

// use crate::prelude::traits::Identity;

use crate::prelude::symbol::Symbol;

use super::{
    decl::Type, literal::Literal, name::Name, Token,
};

impl From<Literal> for Pat {
    fn from(lit: Literal) -> Self {
        Pat::Lit(lit)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Constraint<T>(pub(crate) T, pub(crate) T);

impl<T> std::hash::Hash for Constraint<T>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl<T> std::fmt::Display for Constraint<T>
where
    T: std::fmt::Display,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{} => {}", &(self.0), &(self.1))
    }
}

/// Patterns for type signatures
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SigPat<T> {
    /// Unit -- alias for `()`, which is effectively an empty tuple
    Unit,
    /// Type variables -- equivalent to generics or type parameters,
    /// corresponding to lowercase identifiers, i.e., `Token::Lower`
    Var(T),
    /// Type constructors/constants -- correspond to `Token::Upper`
    TyCon(T),
    /// Parentheized to prevent ambiguity, in particular when grouping
    /// functions
    Group(Box<Self>),
    /// Tuple type -- may contain any **finite** number of
    /// (comma-separated) types.
    Tuple(Vec<Self>),
    /// List or "vector" type -- may only be of one type
    List(Box<Self>),
    /// Syntactically identical to application in expressions, around which we
    /// can arbitrarily add a single parenthesis (and in many cases, may
    /// need to).
    ///
    /// **Ex:** `Result Char Int` is equivalent to `Result<char, i32>`
    Apply(Box<Self>, Vec<Self>),
    /// Function type
    Arrow(Box<Self>, Box<Self>),
    /// Don't know if this would ever be necessary, or even detrimental,
    /// but anonymous type? Perhaps for internal use
    Anon,
    /// Constraint or type context
    Given(Constraint<T>),
}

impl<T> SigPat<T> {
    /// Replaces any `Group` instance by its corresponding inner variant.
    pub fn degroup(self) -> Self {
        match self {
            SigPat::Unit
            | SigPat::Anon
            | SigPat::Var(_)
            | SigPat::TyCon(_)
            | SigPat::Given(_) => self,
            SigPat::Group(inner) => *inner,
            SigPat::Tuple(ps) => SigPat::Tuple(
                ps.into_iter().map(Self::degroup).collect(),
            ),
            SigPat::List(t) => {
                SigPat::List(Box::new((*t).degroup()))
            }
            SigPat::Apply(x, ys) => SigPat::Apply(
                Box::new((*x).degroup()),
                ys.into_iter().map(Self::degroup).collect(),
            ),
            SigPat::Arrow(x, y) => SigPat::Arrow(
                Box::new((*x).degroup()),
                Box::new((*y).degroup()),
            ),
        }
    }

    /// Given a pattern of arrows representing functions, computes its
    /// number of arguments
    pub fn arrow_arity_in(sigpat: &Self) -> Option<usize> {
        match sigpat {
            Self::Arrow(left, _right) => {
                let mut arity = 1_usize;
                let mut temp = left;
                loop {
                    match temp.as_ref() {
                        SigPat::Arrow(ln, _) => {
                            arity += 1;
                            temp = ln;
                        }
                        _ => break,
                    }
                }
                Some(arity)
            }
            _ => None,
        }
    }
}

// impl From<SigPat<T>> for SigPat<Name> where T: From<Name> {}

pub type Pat = Morpheme<Literal, Token>;
pub type Pattern = Morpheme<Literal, Symbol>;

/// Generic enum to represent the skeleton of *patterns*.
/// The nodes generated by the `Parser` for *patterns* and *NOT*
/// expressions: `Pat`s, are instances of `Morpheme<Literal, Token>.
/// This is because this enum will be reused through various passes,
/// e.g., renaming.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Morpheme<L, T> {
    /// Shortcut for empty tuple pattern `()`
    Unit,
    Lit(L),
    /// Identifier. Matches anything.
    Var(T),
    /// `(x, y)`
    Tuple(Vec<Self>),
    /// `[x, y]`
    List(Vec<Self>),
    /// `(x)`, a pattern must first be a `Grouped` pattern before becoming
    /// either a `tuple` pattern (if followed by a comma `,`) or a `Ctor`
    /// pattern (if n > 0 patterns exist to the right of `x`)
    Grouped(Box<Self>),
    /// Data constructor pattern. The pattern `(:Foo x)` matches an instance (bound by `x`) of a given data constructor `Foo :: X -> Foo X`
    Ctor(T, Vec<Self>),
    /// ACCESS pattern for record data. `x { .y y, :z z }`
    /// Note, if no field rhs in lambda arguments, a `Var` pattern is internall
    /// added. In particular, `x { :y }` = `x { y }`.
    /// Data declarations require type signatures, and therefore use a separate
    /// variant
    Record {
        ctor: T,
        fields: Vec<(T, Option<Self>)>,
    },
    Dict {
        ctor: T,
        fields: Vec<(T, Type)>,
    },
    /// Wildcard pattern `_`, matches with (but implies no binding to) anything.
    Wild,
    /// IN CASE EXPRESSION BRANCH PATTERNS: Equivalent to using the `Wild` subpattern for the rest of the pattern. For example, the pattern `(..)` matches every tuple, and the pattern `_ { .. }` matches every record.
    ///
    /// Rest pattern `..`, syntactic sugar for repeated wildcards, discarding
    /// the rest of the possible patterns for a given expression.
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
    /// Annotated pattern `P :: T`
    Annot(Box<Self>, T),
    /// Matches two patterns separated by an infix operator. Used by function declarations, and perhaps in case expressions IFF subpatterns are all `Literal` ?
    Infix {
        infix: T,
        left: Box<Self>,
        right: Box<Self>,
    },
}

impl<L, T> Morpheme<L, T> {
    /// Apply a closure F :: T -> X to all data of type `T` contained by
    /// a given `Morpheme<L, T>` instance, returning a `Morpheme<L, X>`.
    pub fn map_t<F, X>(self, f: F) -> Morpheme<L, X>
    where
        F: Fn(T) -> X,
    {
        match self {
            Morpheme::Unit => Morpheme::Unit,
            Morpheme::Lit(l) => Morpheme::Lit(l),
            Morpheme::Var(t) => Morpheme::Var(f(t)),
            Morpheme::Tuple(ts) => Morpheme::Tuple(
                ts.into_iter()
                    .map(|m| m.map_t(|t| f(t)))
                    .collect::<Vec<_>>(),
            ),
            Morpheme::List(ts) => Morpheme::List(
                ts.into_iter()
                    .map(|m| m.map_t(|t| f(t)))
                    .collect::<Vec<_>>(),
            ),
            Morpheme::Grouped(x) => {
                Morpheme::Grouped(Box::new((*x).map_t(f)))
            }
            Morpheme::Ctor(t, ms) => Morpheme::Ctor(
                f(t),
                ms.into_iter()
                    .map(|m| m.map_t(|l| f(l)))
                    .collect::<Vec<_>>(),
            ),
            Morpheme::Record { ctor: name, fields } => {
                Morpheme::Record {
                    ctor: f(name),
                    fields: fields
                        .into_iter()
                        .map(|(t, m)| {
                            (
                                f(t),
                                m.map(|m| {
                                    m.map_t(|l| f(l))
                                }),
                            )
                        })
                        .collect::<Vec<_>>(),
                }
            }
            Morpheme::Dict { ctor: name, fields } => {
                Morpheme::Dict {
                    ctor: f(name),
                    fields: fields
                        .into_iter()
                        .map(|(t, m)| (f(t), m))
                        .collect::<Vec<_>>(),
                }
            }
            Morpheme::Wild => Morpheme::Wild,
            Morpheme::Rest => Morpheme::Rest,
            Morpheme::Binder { binder, pattern } => {
                Morpheme::Binder {
                    binder: f(binder),
                    pattern: Box::new(pattern.map_t(f)),
                }
            }
            Morpheme::Union(ms) => Morpheme::Union(
                ms.into_iter()
                    .map(|m| m.map_t(|t| f(t)))
                    .collect::<Vec<_>>(),
            ),
            Morpheme::Annot(m, t) => Morpheme::Annot(
                Box::new((*m).map_t(|t| f(t))),
                f(t),
            ),
            Morpheme::Infix { infix, left, right } => {
                Morpheme::Infix {
                    infix: f(infix),
                    left: Box::new((*left).map_t(|t| f(t))),
                    right: Box::new(
                        (*right).map_t(|t| f(t)),
                    ),
                }
            }
        }
    }

    pub fn map_lit<F, X>(self, f: F) -> Morpheme<X, T>
    where
        F: Fn(L) -> X,
    {
        match self {
            Morpheme::Unit => Morpheme::Unit,
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
            Morpheme::Grouped(x) => {
                Morpheme::Grouped(Box::new((*x).map_lit(f)))
            }
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
            Morpheme::Dict { ctor: name, fields } => {
                Morpheme::Dict { ctor: name, fields }
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
            Morpheme::Annot(m, t) => Morpheme::Annot(
                Box::new((*m).map_lit(f)),
                t,
            ),
            Morpheme::Infix { infix, left, right } => {
                Morpheme::Infix {
                    infix,
                    left: Box::new(
                        (*left).map_lit(|l| f(l)),
                    ),
                    right: Box::new(
                        (*right).map_lit(|l| f(l)),
                    ),
                }
            }
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
            Morpheme::Var(t) => Ok(t),
            Morpheme::Wild => Ok(Token::Underscore),
            Morpheme::Rest => Ok(Token::Dot2),
            _ => Err("Invalid Token extraction from Morpheme! Only single-token morphemes/patterns may be successfully converted into a Token.")
        }
    }
}

impl std::convert::TryFrom<Type> for Token {
    type Error = &'static str;
    fn try_from(
        value: SigPat<Name>,
    ) -> Result<Self, Self::Error> {
        let err = Result::<Self, Self::Error>::Err("Unable to convert `SigPat` into a single token!");
        match value {
            SigPat::Var(Name::Ident(t))
            | SigPat::TyCon(Name::Ident(t)) => {
                Ok(Token::Lower(t))
            }
            SigPat::Var(Name::Cons(t))
            | SigPat::TyCon(Name::Cons(t)) => {
                Ok(Token::Upper(t))
            }
            SigPat::Group(t) => match *t {
                // recurse only after going in once
                SigPat::Var(Name::Ident(t))
                | SigPat::TyCon(Name::Ident(t)) => {
                    Ok(Token::Lower(t))
                }
                SigPat::Var(Name::Cons(t))
                | SigPat::TyCon(Name::Cons(t)) => {
                    Ok(Token::Upper(t))
                }
                SigPat::Anon => Ok(Token::Underscore),
                SigPat::Group(s) => (*s).try_into(),
                _ => err,
            },
            SigPat::Anon => Ok(Token::Underscore),
            _ => err,
        }
    }
}

impl<T> std::fmt::Display for SigPat<T>
where
    T: std::fmt::Display,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            SigPat::Unit => write!(f, "()"),
            SigPat::Var(t) | SigPat::TyCon(t) => {
                write!(f, "{}", t)
            }
            SigPat::Group(t) => write!(f, "({})", t),
            SigPat::Tuple(ms) => write!(
                f,
                "({})",
                ms.iter()
                    .map(|m| format!("{}", m))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            SigPat::List(m) => write!(f, "[{}]", m),
            SigPat::Apply(x, ys) => write!(
                f,
                "{} {}",
                x,
                ys.iter()
                    .map(|y| format!("{}", y))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            SigPat::Arrow(l, r) => {
                write!(f, "{} -> {}", l, r)
            }
            SigPat::Anon => write!(f, "_"),
            SigPat::Given(x) => write!(f, "{}", x),
        }
    }
}

impl<L, T> std::fmt::Display for Morpheme<L, T>
where
    L: std::fmt::Display,
    T: std::fmt::Display,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Morpheme::Unit => write!(f, "()"),
            Morpheme::Lit(l) => write!(f, "{}", l),
            Morpheme::Var(t) => write!(f, "{}", t),
            Morpheme::Tuple(ms) => {
                write!(
                    f,
                    "({})",
                    ms.iter()
                        .map(|m| format!("{}", m))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Morpheme::List(ms) => write!(
                f,
                "[{}]",
                ms.iter()
                    .map(|m| format!("{}", m))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Morpheme::Grouped(m) => write!(f, "{}", *m),
            Morpheme::Ctor(t, ms) => {
                if ms.is_empty() {
                    write!(f, "{}", t)
                } else if ms.len() == 1 {
                    write!(f, "({} {})", t, ms[0])
                } else {
                    write!(
                        f,
                        "({} {})",
                        t,
                        ms.iter()
                            .map(|m| format!("{}", m))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                }
            }
            Morpheme::Record { ctor, fields } => write!(
                f,
                "{} {{{}}}",
                ctor,
                fields
                    .iter()
                    .map(|(t, fl)| match fl {
                        Some(field) => {
                            format!("{} :: {}", t, field)
                        }
                        None => format!("{}", t),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Morpheme::Dict { ctor, fields } => write!(
                f,
                "{} {{{}}}",
                ctor,
                fields
                    .iter()
                    .map(|(t, fl)| format!(
                        "{} :: {}",
                        t, fl
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Morpheme::Wild => write!(f, "_"),
            Morpheme::Rest => write!(f, ".."),
            Morpheme::Binder { binder, pattern } => {
                write!(f, "{} @ {}", binder, pattern)
            }
            Morpheme::Union(ms) => write!(
                f,
                "({})",
                ms.iter()
                    .map(|m| format!("{}", m))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Morpheme::Annot(m, t) => {
                write!(f, "({} :: {})", m, t)
            }
            Morpheme::Infix { infix, left, right } => {
                write!(f, "{} {} {}", left, infix, right)
            }
        }
    }
}
