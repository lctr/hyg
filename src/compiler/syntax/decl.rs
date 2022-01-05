use crate::prelude::either::Either;

use super::{
    expr::{Expr, Var},
    fixity::Fixity,
    literal::Literal,
    pattern::{Constraint, Morpheme, Pat, SigPat},
    Operator, Token,
};

/*




infix'l n + - |>
infix'r n ^^ ** <|

type (->) x y = x -> y

data Name x y = Id x | Var y

class Thing x where
    f1 :: x -> x
    f2 :: x


uncurry f x y = \(x, y) -> f x y
              = |x y| f x y         <- syntactic sugar?


*/

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Decl {
    Infix {
        fixity: Fixity,
        infixes: Vec<Operator>,
    },
    Data {
        name: Var,
        constraints: Vec<Constraint<Var>>,
        poly: Vec<Var>,
        variants: Vec<DataVariant>,
        derives: Vec<Var>,
    },
    // `
    Where {
        pat: Pat,
        body: Vec<Expr>,
        decls: Vec<Decl>,
    },
    Function {
        name: Var,
        defs: Vec<Clause>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stmt {
    /// Bind an expression to a pattern. `p <- x`
    Bind { pat: Pat, expr: Expr },
}

/// `[`:pats`]` = `:body` where `:decls`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Clause {
    pub pats: Vec<Pat>,
    pub body: Vec<Expr>,
    pub decls: Vec<Decl>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DataVariant {
    pub ctor: Var,
    pub args: Either<DataPat, ()>,
}

pub type FieldPat = (Var, Type);
pub type Type = SigPat<Var>;

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
