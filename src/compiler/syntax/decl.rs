use crate::prelude::either::Either;

use super::expr::Expr;
use super::fixity::Fixity;
use super::name::Name;
use super::pattern::{Constraint, Pat, SigPat};
use super::Operator;

/*

infix'l n + - |>
infix'r n ^^ ** <|

type (->) x y = x -> y

data Name x y = Id x | Var y

class Thing x where
    f1 :: x -> x
    f2 :: x


uncurry f x y = \(x, y) -> f x y



*/

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Decl {
    Infix {
        fixity: Fixity,
        infixes: Vec<Operator>,
    },
    Data {
        name: Name,
        constraints: Vec<Constraint<Name>>,
        poly: Vec<Name>,
        variants: Vec<DataVariant>,
        derives: Vec<Name>,
    },
    /// Type alias. E.g., `type A b = C [b]`
    Alias {
        name: Name,
        poly: Vec<Name>,
        rhs: Type,
    },
    Class {
        name: Name,
        constraints: Vec<Constraint<Name>>,
        defs: Vec<Clause<Pat, Expr, Decl>>,
    },
    Function {
        name: Name,
        /// Each definition corresponds to an equation wherein
        defs: Vec<Clause<Pat, Expr, Decl>>,
    },
    Annotation {
        name: Name,
        tipo: Type,
    },
    Instance {
        who: Type,
        constraints: Vec<Constraint<Name>>,
        defs: Vec<Clause<Pat, Expr, Decl>>,
    },
}


#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stmt {
    /// Bind an expression to a pattern. `p <- x`
    Bind {},
}

/// `[`:pats`]` = `:body` where `:decls`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Clause<P, E, D> {
    /// The tail of the left-hand side of an equation, i.e., all terms
    /// to the left of the `=` sign **except** the first.
    pub pats: Vec<P>,
    pub body: E,
    pub decls: Vec<D>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DataVariant {
    pub ctor: Name,
    pub args: Either<DataPat, ()>,
}

pub type FieldPat = (Name, Type);
pub type Type = SigPat<Name>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DataPat {
    Args(Vec<Type>),
    Keys(Vec<FieldPat>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TyParam<T> {
    Just(T),
    Given(Constraint<T>),
}
