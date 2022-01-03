use super::{
    expr::{Expr, Var},
    fixity::Fixity,
    pattern::{Morpheme, Pat},
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

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum Decl {
    Infix {
        fixity: Fixity,
        infixes: Vec<Operator>,
    },
    Data {
        name: Var,
        poly: Vec<Var>,
        variants: Vec<DataVariant>,
        derives: Vec<Var>,
    },
    Function {
        name: Var,
        defs: Vec<(Vec<Pat>, Expr)>,
    },
    Module {
        name: Var,
    },
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct FnEquation {
    lhs: Vec<Pat>,
    rhs: Expr,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct DataVariant {
    ctor: Var,
    args: Vec<TyPat<Var>>,
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum TyPat<T> {
    Var(T),
    Ctor(T),
    Tuple(Vec<Self>),
    List(Vec<Self>),
    Fields(Vec<(T, Self)>),
    Group(T, Vec<Self>),
    Arrow(Box<Self>, Box<Self>),
}
